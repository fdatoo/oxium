//! Per-voxel light propagation (sky + emissive block sources).
//!
//! Two BFS flood-fills, run on a worker thread:
//!
//! - **Sky light** drops 15 from the world ceiling. Air is lossless,
//!   translucent solids attenuate, and water costs 3 per vertical step.
//! - **Block light** spreads outward from blocks with `info.emission > 0`.
//!
//! Both are *recompute-on-dirty*: an edit reseeds and re-runs the BFS instead
//! of doing an incremental update. The recompute approach trades a few extra
//! milliseconds of worker time for ~10× less code complexity — see the design
//! spec's "Why recompute over incremental" table.

use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::{ChunkLightInputs, DenseChunk, Neighbors, pack_rgb, unpack_rgb};
use crate::voxel::coords::{CHUNK_DIM_U, LocalPos};
use glam::UVec3;
use std::collections::VecDeque;

pub mod sky_sources;

pub use sky_sources::{ChunkSkyLightSources, NO_SOURCE_FLOOR};

/// Chunk side length as a signed integer (mirrors `D` in the mesher).
const D: i32 = CHUNK_DIM_U as i32;

/// Recompute *both* sky and block light for one chunk, in place.
///
/// `neighbors` is read-only context: the sky pass uses the +Y neighbour to
/// continue light columns through the chunk's top face. The BFS itself
/// only spreads *within* this chunk — cross-boundary leaks are picked up
/// later by the streaming system, which marks the bordering chunk as
/// `light_dirty` and queues another recompute.
pub fn recompute_chunk(chunk: &mut DenseChunk, neighbors: &Neighbors<'_>, reg: &BlockRegistry) {
    sky_light(chunk, neighbors, None, reg);
    block_rgb(chunk, neighbors, reg);
}

pub fn recompute_chunk_with_inputs(
    chunk: &mut DenseChunk,
    neighbors: &Neighbors<'_>,
    inputs: &ChunkLightInputs,
    reg: &BlockRegistry,
) {
    sky_light(chunk, neighbors, Some(inputs), reg);
    block_rgb(chunk, neighbors, reg);
}

/// Compute sky light: each column drops `15` straight down until it hits an
/// opaque block. Air is lossless in the vertical drop; water costs 3 and
/// other transparent non-air blocks cost 1. A BFS pass then spreads light
/// horizontally so overhangs receive
/// the correct gradient — and is seeded both from the vertical drop and
/// from the four lateral chunk neighbours' boundary cells, so a tunnel
/// dug across a chunk seam keeps a smooth light gradient instead of
/// hard-switching to black at the boundary.
fn sky_light(
    chunk: &mut DenseChunk,
    neighbors: &Neighbors<'_>,
    inputs: Option<&ChunkLightInputs>,
    reg: &BlockRegistry,
) {
    chunk.sky_light.iter_mut().for_each(|v| *v = 0);

    // The chunk-above neighbour, if loaded — its bottom row tells us how
    // much sky light enters this chunk from the top. Without an above
    // neighbour we assume full sun: every column starts at 15.
    let above = neighbors.chunks[crate::mesher::Face::PosY as usize];

    for z in 0..D {
        for x in 0..D {
            let mut light = match above {
                Some(a) => a.sky_light[LocalPos(UVec3::new(x as u32, 0, z as u32)).to_index()],
                None => inputs.map_or(15, |i| i.top_sky_at(x as u32, z as u32)),
            };
            for y in (0..D).rev() {
                let idx = LocalPos(UVec3::new(x as u32, y as u32, z as u32)).to_index();
                let info = reg.info(chunk.blocks[idx]);
                if info.opaque {
                    // Light is absorbed; nothing reaches the next cell down.
                    light = 0;
                }
                chunk.sky_light[idx] = light;
                if light > 0 && !info.opaque {
                    let cost = match chunk.blocks[idx] {
                        Block::Air => 0,
                        Block::Water => 3,
                        _ => 1,
                    };
                    light = light.saturating_sub(cost);
                }
            }
        }
    }

    // Lateral boundary inflow: for each ±X / ±Z neighbour, copy its
    // *mirror* boundary cells into our cells along that face, minus one
    // attenuation step (the cost of crossing the seam). This is what
    // makes a tunnel that crosses chunk boundaries keep its gradient —
    // without it, the next chunk along the tunnel starts the BFS with
    // no seeds and stays uniformly dark. We skip +Y because the
    // vertical column drop above already consumed it.
    seed_from_neighbors(chunk, neighbors, BfsChannel::Sky);

    // BFS: seed every cell currently >= 2 (anything lower will be reached
    // by spreading from a higher cell, so no need to enqueue it now).
    let mut q: VecDeque<(i32, i32, i32, u8)> = VecDeque::new();
    for z in 0..D {
        for y in 0..D {
            for x in 0..D {
                let idx = LocalPos(UVec3::new(x as u32, y as u32, z as u32)).to_index();
                if chunk.sky_light[idx] >= 2 {
                    q.push_back((x, y, z, chunk.sky_light[idx]));
                }
            }
        }
    }
    bfs_spread_sky(&mut q, chunk, reg);
}

#[derive(Copy, Clone)]
enum BfsChannel {
    Sky,
    BlockRgb,
}

/// Seed the chunk's boundary cells from each face neighbour's mirror
/// boundary, less one attenuation step (the cost of crossing the
/// seam). Called by both sky-light and block-light passes after their
/// in-chunk seeding. The BFS that runs afterward picks up these
/// boundary values and spreads them inward.
///
/// The neighbour's cell at the mirror position represents whatever the
/// neighbour already knows about that location's light. If the
/// neighbour was just regenerated and has high light at its boundary
/// (e.g., the lit end of a tunnel), this seeds *our* boundary cells
/// so the BFS continues the gradient from there.
fn seed_from_neighbors(chunk: &mut DenseChunk, neighbors: &Neighbors<'_>, channel: BfsChannel) {
    use crate::mesher::Face;
    for face in Face::all() {
        if matches!(channel, BfsChannel::Sky) {
            match face {
                // +Y inflow is handled by the column drop above.
                //
                // -Y is deliberately not a sky source. Importing sky upward
                // from the chunk below lets stale lower chunks keep a sealed
                // shaft alive after an opaque edit, and can create feedback
                // where isolated cave chunks relight one another. Sunlight
                // enters downward through +Y, then spreads horizontally.
                Face::PosY | Face::NegY => continue,
                _ => {}
            }
        }
        let Some(n) = neighbors.chunks[face as usize] else {
            continue;
        };
        for v in 0..D {
            for u in 0..D {
                let (our_lp, their_lp) = mirror_boundary(face, u, v);
                let our_idx = our_lp.to_index();
                let their_idx = their_lp.to_index();
                match channel {
                    BfsChannel::Sky => {
                        let seeded = n.sky_light[their_idx].saturating_sub(1);
                        if seeded > chunk.sky_light[our_idx] {
                            chunk.sky_light[our_idx] = seeded;
                        }
                    }
                    BfsChannel::BlockRgb => {
                        let (tr, tg, tb) = unpack_rgb(n.block_rgb[their_idx]);
                        let (or, og, ob) = unpack_rgb(chunk.block_rgb[our_idx]);
                        let new_r = or.max(tr.saturating_sub(1));
                        let new_g = og.max(tg.saturating_sub(1));
                        let new_b = ob.max(tb.saturating_sub(1));
                        if new_r != or || new_g != og || new_b != ob {
                            chunk.block_rgb[our_idx] = pack_rgb(new_r, new_g, new_b);
                        }
                    }
                }
            }
        }
    }
}

/// Snapshot the chunk's per-face boundary lighting (`sky_light` and
/// `block_rgb` channels) into one `Vec<u8>` per face, ordered by
/// [`crate::mesher::Face`] discriminant. Used by the relight worker
/// to detect which faces' boundary values actually changed after the
/// BFS — the Relit handler then invalidates only the neighbours that
/// would consume the changed values.
///
/// Each face's `Vec` is `D² × 4` bytes: sky, R, G, B per cell,
/// walked in the same `(u, v)` order as `mirror_boundary`. The
/// per-byte comparison is cheap (~12 KB total per chunk) and exact —
/// no hash collisions to worry about.
pub fn snapshot_face_boundaries(chunk: &crate::voxel::chunk::DenseChunk) -> [Vec<u8>; 6] {
    use crate::mesher::Face;
    std::array::from_fn(|face_i| {
        let face = match face_i {
            0 => Face::PosX,
            1 => Face::NegX,
            2 => Face::PosY,
            3 => Face::NegY,
            4 => Face::PosZ,
            5 => Face::NegZ,
            _ => unreachable!(),
        };
        let mut out = Vec::with_capacity((D * D * 4) as usize);
        for v in 0..D {
            for u in 0..D {
                let (our_lp, _) = mirror_boundary(face, u, v);
                let idx = our_lp.to_index();
                let (r, g, b) = unpack_rgb(chunk.block_rgb[idx]);
                out.push(chunk.sky_light[idx]);
                out.push(r);
                out.push(g);
                out.push(b);
            }
        }
        out
    })
}

/// Return `(our_boundary_cell, neighbour_mirror_cell)` for a given
/// face's `(u, v)` boundary coordinate. `(u, v)` covers the 2D slice
/// in the two axes orthogonal to the face's normal; `face` decides
/// which axis is `u` vs `v` and which extreme of the chunk dimension
/// the boundary sits on.
pub fn mirror_boundary(face: crate::mesher::Face, u: i32, v: i32) -> (LocalPos, LocalPos) {
    use crate::mesher::Face;
    let last = D as u32 - 1;
    let (ours, theirs) = match face {
        // PosX: our boundary at x = D-1, neighbour mirror at x = 0.
        Face::PosX => (
            UVec3::new(last, v as u32, u as u32),
            UVec3::new(0, v as u32, u as u32),
        ),
        Face::NegX => (
            UVec3::new(0, v as u32, u as u32),
            UVec3::new(last, v as u32, u as u32),
        ),
        Face::PosY => (
            UVec3::new(u as u32, last, v as u32),
            UVec3::new(u as u32, 0, v as u32),
        ),
        Face::NegY => (
            UVec3::new(u as u32, 0, v as u32),
            UVec3::new(u as u32, last, v as u32),
        ),
        Face::PosZ => (
            UVec3::new(u as u32, v as u32, last),
            UVec3::new(u as u32, v as u32, 0),
        ),
        Face::NegZ => (
            UVec3::new(u as u32, v as u32, 0),
            UVec3::new(u as u32, v as u32, last),
        ),
    };
    (LocalPos(ours), LocalPos(theirs))
}

/// Compute RGB block light: seed each emissive block with its three-channel
/// emission, BFS outward, per-channel attenuation by 1 per air step (cost-1
/// extra in water). Boundary cells are also seeded from neighbour chunks so
/// colored sources continue to glow into adjacent chunks rather than
/// hard-cutting at the seam.
fn block_rgb(chunk: &mut DenseChunk, neighbors: &Neighbors<'_>, reg: &BlockRegistry) {
    chunk.block_rgb.iter_mut().for_each(|v| *v = 0);

    // Queue entry: (x, y, z, packed u16). Channels propagate together so
    // the BFS visits each cell once for all three.
    let mut q: VecDeque<(i32, i32, i32, u16)> = VecDeque::new();
    for z in 0..D {
        for y in 0..D {
            for x in 0..D {
                let idx = LocalPos(UVec3::new(x as u32, y as u32, z as u32)).to_index();
                let [er, eg, eb] = reg.info(chunk.blocks[idx]).emission;
                if er > 0 || eg > 0 || eb > 0 {
                    let cell = pack_rgb(er, eg, eb);
                    chunk.block_rgb[idx] = cell;
                    q.push_back((x, y, z, cell));
                }
            }
        }
    }
    seed_from_neighbors(chunk, neighbors, BfsChannel::BlockRgb);
    // Re-enqueue any boundary cell the seed-pass bumped to non-zero so
    // the BFS picks them up.
    for z in 0..D {
        for y in 0..D {
            for x in 0..D {
                let idx = LocalPos(UVec3::new(x as u32, y as u32, z as u32)).to_index();
                let on_boundary =
                    x == 0 || y == 0 || z == 0 || x == D - 1 || y == D - 1 || z == D - 1;
                let cell = chunk.block_rgb[idx];
                if on_boundary && cell != 0 {
                    let (r, g, b) = unpack_rgb(cell);
                    // Only enqueue if any channel can still propagate (>= 2).
                    if r >= 2 || g >= 2 || b >= 2 {
                        q.push_back((x, y, z, cell));
                    }
                }
            }
        }
    }
    bfs_spread_rgb(&mut q, chunk, reg);
}

/// Sky BFS step. Spreads sky light to neighbours within this chunk only;
/// cross-chunk spread is the streaming system's responsibility.
fn bfs_spread_sky(
    q: &mut VecDeque<(i32, i32, i32, u8)>,
    chunk: &mut DenseChunk,
    reg: &BlockRegistry,
) {
    use crate::mesher::Face;
    while let Some((x, y, z, level)) = q.pop_front() {
        if level <= 1 {
            continue;
        }
        let next = level - 1;
        for face in Face::all() {
            let [dx, dy, dz] = face.normal();
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            if nx < 0 || ny < 0 || nz < 0 || nx >= D || ny >= D || nz >= D {
                continue; // chunk boundary; handled by the dirty system
            }
            let idx = LocalPos(UVec3::new(nx as u32, ny as u32, nz as u32)).to_index();
            let info = reg.info(chunk.blocks[idx]);
            if info.opaque {
                continue;
            }
            // Water attenuates light an extra 2 per step (so an additional
            // `cost - 1 = 2` is consumed). Air and other transparent blocks
            // cost 1 like the BFS default.
            let cost: u8 = if chunk.blocks[idx] == Block::Water {
                3
            } else {
                1
            };
            let prop = next.saturating_sub(cost.saturating_sub(1));
            if prop > chunk.sky_light[idx] {
                chunk.sky_light[idx] = prop;
                q.push_back((nx, ny, nz, prop));
            }
        }
    }
}

/// RGB BFS step. Spreads all three channels simultaneously to neighbours
/// within this chunk only; cross-chunk spread is the streaming system's
/// responsibility (via `light_dirty` cascade).
fn bfs_spread_rgb(
    q: &mut VecDeque<(i32, i32, i32, u16)>,
    chunk: &mut DenseChunk,
    reg: &BlockRegistry,
) {
    use crate::mesher::Face;
    while let Some((x, y, z, cell)) = q.pop_front() {
        let (lr, lg, lb) = unpack_rgb(cell);
        if lr <= 1 && lg <= 1 && lb <= 1 {
            continue;
        }
        for face in Face::all() {
            let [dx, dy, dz] = face.normal();
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            if nx < 0 || ny < 0 || nz < 0 || nx >= D || ny >= D || nz >= D {
                continue;
            }
            let idx = LocalPos(UVec3::new(nx as u32, ny as u32, nz as u32)).to_index();
            let info = reg.info(chunk.blocks[idx]);
            if info.opaque {
                continue;
            }
            let cost: u8 = if chunk.blocks[idx] == Block::Water {
                3
            } else {
                1
            };
            let attenuation = cost.saturating_sub(1);
            // Per-channel propagation: each channel attenuates independently.
            let prop_r = lr.saturating_sub(1).saturating_sub(attenuation);
            let prop_g = lg.saturating_sub(1).saturating_sub(attenuation);
            let prop_b = lb.saturating_sub(1).saturating_sub(attenuation);
            let (cur_r, cur_g, cur_b) = unpack_rgb(chunk.block_rgb[idx]);
            let new_r = cur_r.max(prop_r);
            let new_g = cur_g.max(prop_g);
            let new_b = cur_b.max(prop_b);
            if new_r != cur_r || new_g != cur_g || new_b != cur_b {
                chunk.block_rgb[idx] = pack_rgb(new_r, new_g, new_b);
                q.push_back((nx, ny, nz, chunk.block_rgb[idx]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::DenseChunk;
    use glam::IVec3;

    fn empty_neighbors() -> Neighbors<'static> {
        Neighbors { chunks: [None; 6] }
    }

    #[test]
    fn all_air_has_full_sky_light() {
        let mut c = DenseChunk::empty();
        let r = BlockRegistry::new();
        recompute_chunk(&mut c, &empty_neighbors(), &r);
        assert!(c.sky_light.iter().all(|&v| v == 15));
    }

    #[test]
    fn full_stone_has_no_sky_light() {
        let mut c = DenseChunk::new_filled(Block::Stone);
        let r = BlockRegistry::new();
        recompute_chunk(&mut c, &empty_neighbors(), &r);
        assert!(c.sky_light.iter().all(|&v| v == 0));
    }

    #[test]
    fn torch_emits_block_light_with_falloff() {
        let mut c = DenseChunk::empty();
        c.set(LocalPos(UVec3::new(16, 16, 16)), Block::Torch);
        let r = BlockRegistry::new();
        recompute_chunk(&mut c, &empty_neighbors(), &r);
        let center = LocalPos(UVec3::new(16, 16, 16)).to_index();
        let adj = LocalPos(UVec3::new(17, 16, 16)).to_index();
        let far = LocalPos(UVec3::new(20, 16, 16)).to_index();
        let (cr, cg, cb) = unpack_rgb(c.block_rgb[center]);
        let (ar, ag, ab) = unpack_rgb(c.block_rgb[adj]);
        let (fr, fg, fb) = unpack_rgb(c.block_rgb[far]);
        assert_eq!(cr, 13);
        assert_eq!(cg, 13);
        assert_eq!(cb, 13);
        assert!(ar >= 11, "adj R too low: {}", ar);
        assert_eq!(ag, ar, "G should match R for a uniformly-emissive torch");
        assert_eq!(ab, ar, "B should match R for a uniformly-emissive torch");
        assert!(fr < ar);
        assert_eq!(fg, fr);
        assert_eq!(fb, fr);
    }

    #[test]
    fn opaque_block_blocks_sky() {
        let mut c = DenseChunk::empty();
        // A 1-block-thick stone slab at y=20 across the whole chunk.
        for z in 0..32 {
            for x in 0..32 {
                c.set(LocalPos(UVec3::new(x, 20, z)), Block::Stone);
            }
        }
        let r = BlockRegistry::new();
        recompute_chunk(&mut c, &empty_neighbors(), &r);
        assert_eq!(c.sky_light[LocalPos(UVec3::new(10, 25, 10)).to_index()], 15);
        assert_eq!(c.sky_light[LocalPos(UVec3::new(10, 19, 10)).to_index()], 0);
    }

    #[test]
    fn vertical_water_column_attenuates_sky() {
        let mut c = DenseChunk::empty();
        for z in 0..32 {
            for x in 0..32 {
                for y in 20..=31 {
                    c.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
                }
            }
        }
        let r = BlockRegistry::new();
        recompute_chunk(&mut c, &empty_neighbors(), &r);
        assert_eq!(c.sky_light[LocalPos(UVec3::new(10, 31, 10)).to_index()], 15);
        assert_eq!(c.sky_light[LocalPos(UVec3::new(10, 26, 10)).to_index()], 0);
        assert_eq!(c.sky_light[LocalPos(UVec3::new(10, 19, 10)).to_index()], 0);
    }

    #[test]
    fn manifest_controls_missing_above_sky_inflow() {
        let mut c = DenseChunk::empty();
        let r = BlockRegistry::new();
        let inputs =
            ChunkLightInputs::from_dense(&c, crate::voxel::coords::ChunkCoord(IVec3::ZERO), &r);
        recompute_chunk_with_inputs(&mut c, &empty_neighbors(), &inputs, &r);
        assert!(
            c.sky_light.iter().all(|&v| v == 0),
            "missing +Y with no worldgen sky hint must not assume full daylight"
        );
    }

    #[test]
    fn manifest_open_sky_hint_lights_missing_above() {
        let mut c = DenseChunk::empty();
        let r = BlockRegistry::new();
        let inputs = ChunkLightInputs::from_dense_with_surface(
            &c,
            crate::voxel::coords::ChunkCoord(IVec3::ZERO),
            &r,
            |_, _| Some(-1),
        );
        recompute_chunk_with_inputs(&mut c, &empty_neighbors(), &inputs, &r);
        assert!(c.sky_light.iter().all(|&v| v == 15));
    }

    #[test]
    fn sky_does_not_seed_upward_from_below_neighbor() {
        let mut c = DenseChunk::empty();
        let mut below = DenseChunk::empty();
        below.sky_light.iter_mut().for_each(|v| *v = 15);
        let r = BlockRegistry::new();
        let inputs =
            ChunkLightInputs::from_dense(&c, crate::voxel::coords::ChunkCoord(IVec3::ZERO), &r);
        let neighbors = Neighbors {
            chunks: [None, None, None, Some(&below), None, None],
        };

        recompute_chunk_with_inputs(&mut c, &neighbors, &inputs, &r);

        assert!(
            c.sky_light.iter().all(|&v| v == 0),
            "stale lower chunks must not keep sealed sky light alive"
        );
    }
}
