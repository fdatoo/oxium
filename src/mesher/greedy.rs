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
use crate::mesher::{ChunkMesh, Face, Vertex};
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
        greedy_one_face(face, chunk, &block_at, reg, &mut mesh);
    }
    mesh
}

/// Sweep slices for one face direction and emit greedy-merged quads.
fn greedy_one_face<F>(
    face: Face,
    chunk: &DenseChunk,
    block_at: &F,
    reg: &BlockRegistry,
    mesh: &mut ChunkMesh,
) where
    F: Fn(i32, i32, i32) -> Option<Block>,
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
                let visible = here != Block::Air
                    && !reg.info(neighbor).opaque
                    && (reg.info(here).opaque || here != neighbor);

                let cell = if visible {
                    let ao = corner_ao_at(face, |dx, dy, dz| {
                        block_at(x + dx, y + dy, z + dz)
                    });
                    // Light is read from the *outside* (the air-side
                    // neighbour). For now sky/block default to fully bright
                    // when out of range; M5 will populate the real values.
                    let n = neighbor_pos;
                    let light = if n.0 >= 0
                        && n.1 >= 0
                        && n.2 >= 0
                        && n.0 < D as i32
                        && n.1 < D as i32
                        && n.2 < D as i32
                    {
                        let idx = LocalPos(UVec3::new(n.0 as u32, n.1 as u32, n.2 as u32))
                            .to_index();
                        (chunk.sky_light[idx] & 0x0F) << 4 | (chunk.block_light[idx] & 0x0F)
                    } else {
                        0xFF
                    };
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

    // Positive faces sit on the high side of the slice (e.g. PosY of block
    // at y=k lives at y=k+1); negative faces sit at the low side.
    let normal_pos = matches!(face, Face::PosX | Face::PosY | Face::PosZ);
    let s = if normal_pos { slice + 1 } else { slice };

    // Four corners in (u, v) order, untransformed.
    let corner_uv: [(u8, u8); 4] = [
        (ui, vi),
        (ui + w, vi),
        (ui + w, vi + h),
        (ui, vi + h),
    ];
    // Map each (u, v) back to a 3D voxel-local position.
    let mut positions = [[0u8; 3]; 4];
    for (i, (u, v)) in corner_uv.iter().enumerate() {
        let mut p = [0u8; 3];
        p[n_axis as usize] = s;
        p[u_axis as usize] = *u;
        p[v_axis as usize] = *v;
        positions[i] = p;
    }

    // Pick a winding so the resulting triangle's normal matches the face's
    // outward normal. With our u/v axis mappings, positive-normal faces
    // happen to come out CW from outside under the natural order; we
    // reverse to get them CCW, matching the back-face-cull setup. (Verified
    // by hand against each of the six axis mappings; see commit message.)
    let order: [usize; 4] = if normal_pos { [0, 3, 2, 1] } else { [0, 1, 2, 3] };

    let base = mesh.vertices.len() as u32;
    let ao = cell.ao;
    let light = cell.light;
    for i in 0..4 {
        mesh.vertices.push(Vertex {
            pos: positions[order[i]],
            ao: ao[order[i]],
            color: color_u8,
            normal_face: face as u8,
            light,
            _pad: [0; 2],
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
}
