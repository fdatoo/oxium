//! Per-voxel light propagation (sky + emissive block sources).
//!
//! Two BFS flood-fills, run on a worker thread:
//!
//! - **Sky light** drops 15 from the world ceiling and falls off by 1 per
//!   non-opaque step (water costs 3).
//! - **Block light** spreads outward from blocks with `info.emission > 0`.
//!
//! Both are *recompute-on-dirty*: an edit reseeds and re-runs the BFS instead
//! of doing an incremental update. The recompute approach trades a few extra
//! milliseconds of worker time for ~10× less code complexity — see the design
//! spec's "Why recompute over incremental" table.

use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::{DenseChunk, Neighbors};
use crate::voxel::coords::{LocalPos, CHUNK_DIM_U};
use glam::UVec3;
use std::collections::VecDeque;

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
    sky_light(chunk, neighbors, reg);
    block_light(chunk, reg);
}

/// Compute sky light: each column drops `15` straight down until it hits an
/// opaque block; non-opaque non-air blocks (e.g. leaves, water) cost 1 per
/// step. A BFS pass then spreads light horizontally so overhangs receive
/// the correct gradient.
fn sky_light(chunk: &mut DenseChunk, neighbors: &Neighbors<'_>, reg: &BlockRegistry) {
    chunk.sky_light.iter_mut().for_each(|v| *v = 0);

    // The chunk-above neighbour, if loaded — its bottom row tells us how
    // much sky light enters this chunk from the top. Without an above
    // neighbour we assume full sun: every column starts at 15.
    let above = neighbors.chunks[crate::mesher::Face::PosY as usize];

    for z in 0..D {
        for x in 0..D {
            let mut light = match above {
                Some(a) => a.sky_light[LocalPos(UVec3::new(x as u32, 0, z as u32)).to_index()],
                None => 15,
            };
            for y in (0..D).rev() {
                let idx = LocalPos(UVec3::new(x as u32, y as u32, z as u32)).to_index();
                let info = reg.info(chunk.blocks[idx]);
                if info.opaque {
                    // Light is absorbed; nothing reaches the next cell down.
                    light = 0;
                }
                chunk.sky_light[idx] = light;
                // Drop 1 step of attenuation when continuing through
                // non-opaque (but not perfectly transparent: air doesn't
                // attenuate, leaves do — we encode that as the BFS step
                // cost below, not in the vertical drop).
                if light > 0 && !info.opaque && chunk.blocks[idx] != Block::Air {
                    light = light.saturating_sub(1);
                }
            }
        }
    }

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
    bfs_spread(&mut q, chunk, reg, /* is_sky */ true);
}

/// Compute block light: seed with every emissive block and BFS outward.
fn block_light(chunk: &mut DenseChunk, reg: &BlockRegistry) {
    chunk.block_light.iter_mut().for_each(|v| *v = 0);

    let mut q: VecDeque<(i32, i32, i32, u8)> = VecDeque::new();
    for z in 0..D {
        for y in 0..D {
            for x in 0..D {
                let idx = LocalPos(UVec3::new(x as u32, y as u32, z as u32)).to_index();
                let info = reg.info(chunk.blocks[idx]);
                if info.emission > 0 {
                    chunk.block_light[idx] = info.emission;
                    q.push_back((x, y, z, info.emission));
                }
            }
        }
    }
    bfs_spread(&mut q, chunk, reg, /* is_sky */ false);
}

/// Generic BFS step. Spreads the front to neighbours within this chunk
/// only — cross-chunk spread is the streaming system's responsibility.
fn bfs_spread(
    q: &mut VecDeque<(i32, i32, i32, u8)>,
    chunk: &mut DenseChunk,
    reg: &BlockRegistry,
    is_sky: bool,
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
            let cost: u8 = if chunk.blocks[idx] == Block::Water { 3 } else { 1 };
            let prop = next.saturating_sub(cost.saturating_sub(1));
            let prev = if is_sky {
                chunk.sky_light[idx]
            } else {
                chunk.block_light[idx]
            };
            if prop > prev {
                if is_sky {
                    chunk.sky_light[idx] = prop;
                } else {
                    chunk.block_light[idx] = prop;
                }
                q.push_back((nx, ny, nz, prop));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::DenseChunk;

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
        assert_eq!(c.block_light[center], 13);
        // 1-step falloff = emission - 1 = 12.
        assert!(
            c.block_light[adj] >= 11,
            "adj light too low: {}",
            c.block_light[adj]
        );
        assert!(c.block_light[far] < c.block_light[adj]);
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
}
