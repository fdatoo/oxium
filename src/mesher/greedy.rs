//! Greedy meshing: merge coplanar adjacent same-appearance face cells into
//! single rectangle quads.
//!
//! The naive mesher (in `naive.rs`) emits one quad per visible block face.
//! Greedy meshing instead works in three steps **per face direction**:
//!
//! 1. **Slice scan.** For each axis-perpendicular slice, build a 2D mask
//!    where each cell describes the *visible* face at that block (its block
//!    kind, its baked AO, its light).
//! 2. **Greedy merge.** Sweep the mask left-to-right, top-to-bottom; for
//!    each unvisited cell extend right as far as the cells match, then
//!    extend down as far as the row matches, marking all cells visited.
//!    Emit a single quad spanning the merged rectangle.
//! 3. **AO-aware flip.** When the four corner AO values are anisotropic,
//!    flip the quad's triangle split so the AO shading interpolates along
//!    the brighter axis — fixes the classic "twisted corner" artifact.
//!
//! Result: a 32³ chunk that the naive mesher emits as ~thousands of quads
//! collapses into ~tens to ~hundreds of quads, depending on terrain
//! complexity. The visual output is *identical* to the naive output — we
//! just send less data to the GPU per frame.

use crate::mesher::ao::corner_ao_at;
use crate::mesher::{ChunkMesh, Face, Vertex, UNTEXTURED_TILE};
use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{LocalPos, CHUNK_DIM_U};
use glam::UVec3;

/// Chunk side length as a `usize` — used in array sizes below.
const D: usize = CHUNK_DIM_U as usize;

/// Per-mask-cell data used during greedy merging. Two cells can only merge
/// when they are bit-for-bit equal, so anything that *visually* distinguishes
/// adjacent quads (block kind, AO corners, packed light) lives here.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
struct Cell {
    /// `Block` discriminant, or 0 for "no visible face here".
    block: u16,
    /// Four-corner AO values (0..=3 each), in the same order as `face_corners`.
    ao: [u8; 4],
    /// Packed `(sky << 4) | block` light byte, sampled from the *outside*
    /// neighbour (M5 will populate the real sky/block channels).
    light: u8,
}

/// The "no face here" sentinel. Cells equal to this are skipped by the
/// merger and never produce geometry.
const EMPTY_CELL: Cell = Cell {
    block: 0,
    ao: [0; 4],
    light: 0,
};

/// Mesh a chunk with optional neighbour information using greedy merging.
/// Drop-in replacement for `naive::mesh_chunk_with_neighbors` — same
/// signature and visual output.
pub fn mesh_greedy(
    chunk: &DenseChunk,
    neighbors: &[Option<&DenseChunk>; 6],
    reg: &BlockRegistry,
) -> ChunkMesh {
    let mut mesh = ChunkMesh::empty();

    // Helper closure: read a block at `(x, y, z)`, looking it up in the
    // appropriate neighbour chunk when exactly one axis is one step out of
    // range. Multi-axis corner samples (used by AO only) and unloaded
    // single-axis neighbours both return `None`, meaning "unknown".
    //
    // AO samplers handle `None` as *unoccluded* — the gentler artefact at
    // chunk corners than wrong-AO would be. The visibility-check caller
    // unwraps the same value as `Air`, conservatively emitting boundary
    // faces when neighbour info isn't available. That conservative emit
    // can leave visible chunk-bottom shelves at the *permanent* bottom
    // of the loaded vertical range; we deliberately accept that trade
    // because the alternative (treat None as opaque) hides *all* side
    // faces of edge chunks, leaving large sky-visible holes in the
    // foreground.
    // Read the packed `(sky << 4) | block` light byte at the cell one
    // step outside our chunk's face. Mirrors `block_at`'s neighbour-
    // resolution logic but for light arrays.
    //
    // The previous version of the mesher hard-coded a `0xFF` fallback
    // here (sky=15, block=15) for any out-of-range cell. That worked
    // by coincidence during the day — the sky channel was scaled by
    // `sun_intensity ≈ 1` in the shader, and full-bright sky matched
    // what most exposed surfaces would have anyway. At night
    // `sun_intensity = 0` kills the sky channel, leaving the
    // block=15 fallback to render every chunk-boundary vertex as a
    // fully torch-lit pixel — visible as scattered orange specks on
    // every tree, dug-out cave face, and chunk-seam edge. The proper
    // fix is to read from the actual neighbouring chunk when one is
    // loaded, and fall back to a sensible direction-specific default
    // only when no neighbour data exists.
    let light_at = |x: i32, y: i32, z: i32| -> u8 {
        let dim = D as i32;
        let in_range = x >= 0 && y >= 0 && z >= 0 && x < dim && y < dim && z < dim;
        if in_range {
            let idx = LocalPos(UVec3::new(x as u32, y as u32, z as u32)).to_index();
            let brightness = crate::voxel::chunk::rgb_brightness(chunk.block_rgb[idx]);
            return (chunk.sky_light[idx] & 0x0F) << 4 | (brightness & 0x0F);
        }
        let out_x = (x < 0) as i32 + (x >= dim) as i32;
        let out_y = (y < 0) as i32 + (y >= dim) as i32;
        let out_z = (z < 0) as i32 + (z >= dim) as i32;
        if out_x + out_y + out_z >= 2 {
            // Multi-axis corner — no single neighbour owns this cell.
            // Light's only consumed on face cells (1-axis-out), so
            // this branch shouldn't fire in practice; pick the
            // gentlest default just in case.
            return 0x00;
        }
        let (face_idx, lx, ly, lz) = if x < 0 {
            (Face::NegX as usize, (dim - 1) as u32, y as u32, z as u32)
        } else if x >= dim {
            (Face::PosX as usize, 0u32, y as u32, z as u32)
        } else if y < 0 {
            (Face::NegY as usize, x as u32, (dim - 1) as u32, z as u32)
        } else if y >= dim {
            (Face::PosY as usize, x as u32, 0u32, z as u32)
        } else if z < 0 {
            (Face::NegZ as usize, x as u32, y as u32, (dim - 1) as u32)
        } else {
            (Face::PosZ as usize, x as u32, y as u32, 0u32)
        };
        if let Some(n) = neighbors[face_idx] {
            let idx = LocalPos(UVec3::new(lx, ly, lz)).to_index();
            let brightness = crate::voxel::chunk::rgb_brightness(n.block_rgb[idx]);
            return (n.sky_light[idx] & 0x0F) << 4 | (brightness & 0x0F);
        }
        // Truly no neighbour data (chunk not loaded yet). Pick a
        // direction-specific default:
        //   - +Y face → assume open sky above, full sky-light. Worst
        //     case the surface temporarily reads bright during
        //     stream-in; the chunk re-meshes once the +Y neighbour
        //     arrives.
        //   - All other faces → 0 (dark). Cave faces a player digs
        //     out and chunk-boundary side faces no longer bloom
        //     bright at night.
        if face_idx == Face::PosY as usize { 0xF0 } else { 0x00 }
    };

    let block_at = |x: i32, y: i32, z: i32| -> Option<Block> {
        let dim = D as i32;
        let in_range = x >= 0 && y >= 0 && z >= 0 && x < dim && y < dim && z < dim;
        if in_range {
            return Some(chunk.get(LocalPos(UVec3::new(x as u32, y as u32, z as u32))));
        }
        let out_x = (x < 0) as i32 + (x >= dim) as i32;
        let out_y = (y < 0) as i32 + (y >= dim) as i32;
        let out_z = (z < 0) as i32 + (z >= dim) as i32;
        if out_x + out_y + out_z >= 2 {
            // Multi-axis corner — no neighbour chunk holds this voxel.
            return None;
        }
        let (face, lx, ly, lz) = if x < 0 {
            (Face::NegX, (dim - 1) as u32, y as u32, z as u32)
        } else if x >= dim {
            (Face::PosX, 0u32, y as u32, z as u32)
        } else if y < 0 {
            (Face::NegY, x as u32, (dim - 1) as u32, z as u32)
        } else if y >= dim {
            (Face::PosY, x as u32, 0u32, z as u32)
        } else if z < 0 {
            (Face::NegZ, x as u32, y as u32, (dim - 1) as u32)
        } else {
            (Face::PosZ, x as u32, y as u32, 0u32)
        };
        let n = neighbors[face as usize]?;
        Some(n.get(LocalPos(UVec3::new(lx, ly, lz))))
    };

    for face in Face::all() {
        greedy_one_face(face, chunk, &block_at, &light_at, reg, &mut mesh);
    }
    emit_water_tops_per_block(chunk, &block_at, &light_at, reg, &mut mesh);
    // Single linear pass over the 32³ dense grid to set the water flag.
    // The renderer reads it to decide whether the planar-reflection pass
    // can be skipped. Cost is negligible compared to the mesh itself, and
    // it only runs at mesh time — not per frame.
    mesh.has_water = chunk.blocks.iter().any(|b| *b == Block::Water);
    mesh
}

/// Walk every (x, y, z) and emit a 1-block-per-quad top face for any
/// water column whose neighbour above is non-opaque (the same
/// visibility rule the greedy pass would use). Per-block resolution
/// is what enables the water shader's vertex wave displacement to
/// produce visible undulation without creating chunk-boundary
/// hairlines.
fn emit_water_tops_per_block<F, L>(
    chunk: &DenseChunk,
    block_at: &F,
    light_at: &L,
    reg: &BlockRegistry,
    mesh: &mut ChunkMesh,
) where
    F: Fn(i32, i32, i32) -> Option<Block>,
    L: Fn(i32, i32, i32) -> u8,
{
    for y in 0..D as i32 {
        for z in 0..D as i32 {
            for x in 0..D as i32 {
                let here = chunk.get(LocalPos(UVec3::new(x as u32, y as u32, z as u32)));
                if here != Block::Water {
                    continue;
                }
                let above = block_at(x, y + 1, z).unwrap_or(Block::Air);
                // Only emit the topmost water block's top face — the
                // ones below have water above them and would be
                // visibility-culled by the greedy rule anyway.
                if reg.info(above).opaque || above == Block::Water {
                    continue;
                }
                let ao = corner_ao_at(Face::PosY, |dx, dy, dz| {
                    block_at(x + dx, y + dy, z + dz)
                });
                let light = light_at(x, y + 1, z);
                let cell = Cell {
                    block: Block::Water as u16,
                    ao,
                    light,
                };
                // Emit a 1×1 quad at slice y, anchored at (u=x, v=z)
                // in PosY's axis mapping. Reuses the existing
                // greedy-quad emitter for vertex packing consistency.
                emit_greedy_quad(
                    mesh,
                    Face::PosY,
                    y as u8, // slice == y for PosY
                    x as u8, // u_axis == X for PosY
                    z as u8, // v_axis == Z for PosY
                    1,
                    1,
                    1, // n_axis = Y
                    0, // u_axis = X
                    2, // v_axis = Z
                    cell,
                    reg,
                );
            }
        }
    }
}

/// Sweep slices for one face direction and emit greedy-merged quads.
fn greedy_one_face<F, L>(
    face: Face,
    chunk: &DenseChunk,
    block_at: &F,
    light_at: &L,
    reg: &BlockRegistry,
    mesh: &mut ChunkMesh,
) where
    F: Fn(i32, i32, i32) -> Option<Block>,
    L: Fn(i32, i32, i32) -> u8,
{
    // Axis mapping per face. `n_axis` is the axis perpendicular to the
    // face's plane; `u_axis`/`v_axis` are the in-plane axes. `normal_sign`
    // points in the face's normal direction.
    let (n_axis, u_axis, v_axis, normal_sign) = match face {
        Face::PosX => (0u8, 2u8, 1u8, 1i32),
        Face::NegX => (0, 1, 2, -1),
        Face::PosY => (1, 0, 2, 1),
        Face::NegY => (1, 2, 0, -1),
        Face::PosZ => (2, 1, 0, 1),
        Face::NegZ => (2, 0, 1, -1),
    };

    // Mask is a `D × D` grid of `Cell`. Reused across slices to avoid
    // allocating per slice.
    let mut mask = [EMPTY_CELL; D * D];

    for slice in 0..D as i32 {
        // (1) Build the mask for this slice.
        for v in 0..D as i32 {
            for u in 0..D as i32 {
                let (x, y, z) = unmap(slice, u, v, n_axis, u_axis, v_axis);
                let here = chunk.get(LocalPos(UVec3::new(x as u32, y as u32, z as u32)));
                let neighbor_pos = step_along(x, y, z, n_axis, normal_sign);
                let neighbor = block_at(neighbor_pos.0, neighbor_pos.1, neighbor_pos.2)
                    .unwrap_or(Block::Air);

                // A face is visible when the block is not air, its neighbour
                // is not opaque, and either we're an opaque block (so the
                // face shows colour) or the two blocks differ visually.
                let mut visible = here != Block::Air
                    && !reg.info(neighbor).opaque
                    && (reg.info(here).opaque || here != neighbor);

                // Water TOP faces are excluded from greedy merging —
                // they get re-emitted as 1-block quads by
                // `emit_water_tops_per_block` after this pass. The
                // vertex shader displaces each quad's corners by the
                // wave height, so at 1-block resolution the
                // chunk-boundary slope discontinuities that plagued
                // single-greedy-quad water surfaces become *part of
                // the wave detail* instead of looking like seams.
                if face == Face::PosY && here == Block::Water {
                    visible = false;
                }

                let cell = if visible {
                    let ao = corner_ao_at(face, |dx, dy, dz| {
                        block_at(x + dx, y + dy, z + dz)
                    });
                    // Sample the air-side neighbour's light (sky in
                    // the high nibble, block in the low nibble).
                    // `light_at` handles in-chunk reads, cross-chunk
                    // reads via the `neighbors` array, and the truly-
                    // no-data fallback.
                    let light = light_at(neighbor_pos.0, neighbor_pos.1, neighbor_pos.2);
                    Cell {
                        block: here as u16,
                        ao,
                        light,
                    }
                } else {
                    EMPTY_CELL
                };

                mask[(v as usize) * D + u as usize] = cell;
            }
        }

        // (2) Greedy-merge the mask into rectangles.
        let mut visited = [false; D * D];
        for vi in 0..D {
            let mut ui = 0;
            while ui < D {
                let idx0 = vi * D + ui;
                if visited[idx0] || mask[idx0] == EMPTY_CELL {
                    ui += 1;
                    continue;
                }
                let cell = mask[idx0];

                // Extend in +u as far as identical, unvisited cells exist.
                let mut w = 1;
                while ui + w < D && !visited[idx0 + w] && mask[idx0 + w] == cell {
                    w += 1;
                }

                // Extend in +v as far as every cell in the row matches.
                let mut h = 1;
                'outer: while vi + h < D {
                    for k in 0..w {
                        let i = (vi + h) * D + (ui + k);
                        if visited[i] || mask[i] != cell {
                            break 'outer;
                        }
                    }
                    h += 1;
                }

                // Mark visited and emit a single greedy quad.
                for dv in 0..h {
                    for du in 0..w {
                        visited[(vi + dv) * D + (ui + du)] = true;
                    }
                }
                emit_greedy_quad(
                    mesh,
                    face,
                    slice as u8,
                    ui as u8,
                    vi as u8,
                    w as u8,
                    h as u8,
                    n_axis,
                    u_axis,
                    v_axis,
                    cell,
                    reg,
                );
                ui += w;
            }
        }
    }
}

/// Translate `(slice, u, v)` back into a world `(x, y, z)`.
fn unmap(slice: i32, u: i32, v: i32, n_axis: u8, u_axis: u8, v_axis: u8) -> (i32, i32, i32) {
    let mut out = [0i32; 3];
    out[n_axis as usize] = slice;
    out[u_axis as usize] = u;
    out[v_axis as usize] = v;
    (out[0], out[1], out[2])
}

/// Step one block along `n_axis` in the given direction.
fn step_along(x: i32, y: i32, z: i32, n_axis: u8, sign: i32) -> (i32, i32, i32) {
    let mut o = [x, y, z];
    o[n_axis as usize] += sign;
    (o[0], o[1], o[2])
}

/// Emit a single greedy-merged quad of size `w × h` at slice `slice` and
/// in-plane position `(ui, vi)`.
#[allow(clippy::too_many_arguments)]
fn emit_greedy_quad(
    mesh: &mut ChunkMesh,
    face: Face,
    slice: u8,
    ui: u8,
    vi: u8,
    w: u8,
    h: u8,
    n_axis: u8,
    u_axis: u8,
    v_axis: u8,
    cell: Cell,
    reg: &BlockRegistry,
) {
    let block = Block::from_repr(cell.block).unwrap_or(Block::Stone);
    let info = reg.info(block);
    // Grass-style top-tint.
    let color = match face {
        Face::PosY if info.top_color.is_some() => info.top_color.unwrap(),
        _ => info.color,
    };
    let color_u8 = [
        (color[0] * 255.0) as u8,
        (color[1] * 255.0) as u8,
        (color[2] * 255.0) as u8,
        (color[3] * 255.0) as u8,
    ];
    // Which atlas tile does this (block, face) pair sample from?
    // `None` ⇒ `UNTEXTURED_TILE` sentinel ⇒ shader skips the sample
    // and shades the vertex colour directly (used for `Torch`, etc.).
    let tile_index = info
        .tile_for_face(face)
        .map(|t| t.index())
        .unwrap_or(UNTEXTURED_TILE);

    // Positive faces sit on the high side of the slice (e.g. PosY of block
    // at y=k lives at y=k+1); negative faces sit at the low side.
    let normal_pos = matches!(face, Face::PosX | Face::PosY | Face::PosZ);
    let s = if normal_pos { slice + 1 } else { slice };

    // Four corners in mesher (u, v) order, used to compute the 3D
    // vertex positions. This ordering is what the winding fix in
    // `order` below relies on — DO NOT reshuffle it without also
    // re-verifying the cross-product test in
    // `tests::every_face_winds_outward`.
    let corner_pos_uv: [(u8, u8); 4] = [
        (0, 0),
        (w, 0),
        (w, h),
        (0, h),
    ];
    // Map each cell-relative (u, v) back to a 3D voxel-local position
    // by adding the quad's `(ui, vi)` origin.
    let mut positions = [[0u8; 3]; 4];
    for (i, (u, v)) in corner_pos_uv.iter().enumerate() {
        let mut p = [0u8; 3];
        p[n_axis as usize] = s;
        p[u_axis as usize] = ui + *u;
        p[v_axis as usize] = vi + *v;
        positions[i] = p;
    }

    // Texture UVs in *tile units* per corner. The shader does `fract`
    // on these, so integer-aligned corner values give per-block tile
    // repetition automatically across greedy w×h quads.
    //
    // The texture must read RIGHT-SIDE UP on every face: PNG-image V=0
    // (top of the image) needs to land at high world Y (top of the
    // rendered face). The mesher's (u, v) axes don't agree with world
    // (X, Y, Z) the same way on every face, so a single UV table can't
    // be right for all of them:
    //
    //   - **PosY / NegY** (top + bottom): no world-Y component on
    //     the face, so V can map to either horizontal world axis.
    //   - **PosX / NegZ**: mesher-V *is* world Y; flip V so high
    //     mesher-V (top of face) lands at low texture V (top of tile).
    //   - **NegX / PosZ**: mesher-*U* is world Y; swap U/V so texture
    //     V follows world Y, and flip the new V the same way.
    //
    // Without this dispatch, NegX / PosZ render textures rotated 90°
    // and PosX / NegZ render them upside-down (the visible grass strip
    // ends up at the bottom of the block instead of the top).
    let corner_tex_uv: [(u8, u8); 4] = match face {
        Face::PosY | Face::NegY => [(0, 0), (w, 0), (w, h), (0, h)],
        Face::PosX | Face::NegZ => [(0, h), (w, h), (w, 0), (0, 0)],
        Face::NegX | Face::PosZ => [(0, w), (0, 0), (h, 0), (h, w)],
    };

    // Triangle-corner order. The corner_uv layout above gives all six
    // faces correct outward-pointing normals (verified by hand-running
    // the cross product (v1-v0) × (v2-v0) against each axis mapping)
    // *only* when we always use [0, 3, 2, 1]. The previous code branched
    // on `normal_pos` and used [0, 1, 2, 3] for negative-normal faces;
    // that flipped their winding, so NegX/NegY/NegZ ended up back-face
    // culled. From the player's POV that meant the west/bottom/south
    // sides of every cube were invisible — you only ever saw the east,
    // top, and north sides. (The hand-traced fix is documented in the
    // commit message.)
    let order: [usize; 4] = [0, 3, 2, 1];

    let base = mesh.vertices.len() as u32;
    let ao = cell.ao;
    let light = cell.light;
    for i in 0..4 {
        let (u_tile, v_tile) = corner_tex_uv[order[i]];
        mesh.vertices.push(Vertex {
            pos: positions[order[i]],
            ao: ao[order[i]],
            color: color_u8,
            normal_face: face as u8,
            light,
            _pad: [0; 2],
            tile_index,
            u_tile,
            v_tile,
            _pad2: 0,
        });
    }

    // Anisotropic AO: when the two diagonal corner-pair sums differ, the
    // default 0-1-2 / 0-2-3 triangle split produces a noticeable seam.
    // Flipping to 1-2-3 / 1-3-0 hides it by routing the interpolation
    // along the brighter diagonal — a small but visually important detail
    // that's free to add.
    let flip =
        ao[order[0]] as u32 + ao[order[2]] as u32 > ao[order[1]] as u32 + ao[order[3]] as u32;
    if flip {
        mesh.indices
            .extend_from_slice(&[base + 1, base + 2, base + 3, base + 1, base + 3, base]);
    } else {
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::block::BlockRegistry;
    use crate::voxel::chunk::DenseChunk;

    #[test]
    fn empty_chunk_produces_no_quads() {
        let c = DenseChunk::empty();
        let r = BlockRegistry::new();
        let n: [Option<&DenseChunk>; 6] = [None; 6];
        assert_eq!(mesh_greedy(&c, &n, &r).vertices.len(), 0);
    }

    #[test]
    fn single_block_produces_six_quads() {
        let mut c = DenseChunk::empty();
        c.set(LocalPos(UVec3::new(10, 10, 10)), Block::Stone);
        let r = BlockRegistry::new();
        let n: [Option<&DenseChunk>; 6] = [None; 6];
        let mesh = mesh_greedy(&c, &n, &r);
        assert_eq!(mesh.vertices.len(), 24, "6 faces × 4 verts");
    }

    #[test]
    fn solid_chunk_produces_six_merged_quads() {
        // A fully-solid chunk with no neighbours: each of the 6 boundary
        // faces is greedy-merged into a single 32×32 quad → 6 × 4 verts.
        // (Unloaded neighbours are treated as Air for the visibility
        // check, so the boundary faces are conservatively emitted.)
        let c = DenseChunk::new_filled(Block::Stone);
        let r = BlockRegistry::new();
        let n: [Option<&DenseChunk>; 6] = [None; 6];
        let mesh = mesh_greedy(&c, &n, &r);
        assert_eq!(mesh.vertices.len(), 24, "expected 6 merged 32x32 quads");
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn every_face_winds_outward() {
        // Regression guard for the greedy mesher's negative-face
        // winding bug: each emitted triangle's geometric normal must
        // match the face's declared outward normal, otherwise back-face
        // culling drops the wrong direction. Build a single-block
        // chunk and check every emitted triangle.
        let mut c = DenseChunk::empty();
        c.set(LocalPos(UVec3::new(10, 10, 10)), Block::Stone);
        let r = BlockRegistry::new();
        let n: [Option<&DenseChunk>; 6] = [None; 6];
        let mesh = mesh_greedy(&c, &n, &r);
        assert_eq!(mesh.vertices.len(), 24, "6 faces × 4 verts");

        // Walk every triangle and confirm `(v1-v0) × (v2-v0)` aligns
        // with the face's outward normal.
        for tri in mesh.indices.chunks_exact(3) {
            let v0 = mesh.vertices[tri[0] as usize];
            let v1 = mesh.vertices[tri[1] as usize];
            let v2 = mesh.vertices[tri[2] as usize];
            let face = match v0.normal_face {
                0 => Face::PosX,
                1 => Face::NegX,
                2 => Face::PosY,
                3 => Face::NegY,
                4 => Face::PosZ,
                5 => Face::NegZ,
                _ => panic!("bad face index"),
            };
            let expected = face.normal();
            let a = [
                v1.pos[0] as i32 - v0.pos[0] as i32,
                v1.pos[1] as i32 - v0.pos[1] as i32,
                v1.pos[2] as i32 - v0.pos[2] as i32,
            ];
            let b = [
                v2.pos[0] as i32 - v0.pos[0] as i32,
                v2.pos[1] as i32 - v0.pos[1] as i32,
                v2.pos[2] as i32 - v0.pos[2] as i32,
            ];
            let cross = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            // cross should be a positive scalar multiple of `expected`.
            let dot = cross[0] * expected[0] + cross[1] * expected[1] + cross[2] * expected[2];
            assert!(
                dot > 0,
                "{:?} triangle wound wrong: cross={:?} expected normal={:?}",
                face,
                cross,
                expected
            );
        }
    }
}
