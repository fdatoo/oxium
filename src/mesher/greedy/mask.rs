//! Slice-mask construction and rectangle merging.

use crate::mesher::ao::corner_ao_at;
use crate::mesher::{ChunkMesh, Face};
use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::LocalPos;
use glam::UVec3;

use super::D;
use super::emit::emit_greedy_quad;
use super::sampling::Sampler;

/// Per-mask-cell data used during greedy merging.
///
/// Two cells merge only when they are bit-for-bit equal, so anything that
/// visually distinguishes adjacent quads lives here.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(super) struct Cell {
    /// `Block` discriminant, or 0 for "no visible face here".
    pub(super) block: u16,
    /// Four-corner AO values (0..=3 each), in emitter corner order.
    pub(super) ao: [u8; 4],
    /// Packed `(sky << 4) | block` light byte sampled from the outside cell.
    pub(super) light: u8,
}

/// The "no face here" sentinel.
pub(super) const EMPTY_CELL: Cell = Cell {
    block: 0,
    ao: [0; 4],
    light: 0,
};

struct AxisMap {
    n_axis: u8,
    u_axis: u8,
    v_axis: u8,
    normal_sign: i32,
}

/// Sweep slices for one face direction and emit greedy-merged quads.
pub(super) fn greedy_one_face(
    face: Face,
    chunk: &DenseChunk,
    sampler: &Sampler<'_>,
    reg: &BlockRegistry,
    mesh: &mut ChunkMesh,
) {
    let axes = axis_map(face);
    let mut mask = [EMPTY_CELL; D * D];

    for slice in 0..D as i32 {
        build_mask_slice(face, chunk, sampler, reg, &mut mask, slice, &axes);
        merge_mask_slice(face, reg, mesh, &mask, slice, &axes);
    }
}

fn build_mask_slice(
    face: Face,
    chunk: &DenseChunk,
    sampler: &Sampler<'_>,
    reg: &BlockRegistry,
    mask: &mut [Cell; D * D],
    slice: i32,
    axes: &AxisMap,
) {
    for v in 0..D as i32 {
        for u in 0..D as i32 {
            let (x, y, z) = unmap(slice, u, v, axes.n_axis, axes.u_axis, axes.v_axis);
            let here = chunk.get(LocalPos(UVec3::new(x as u32, y as u32, z as u32)));
            let neighbor_pos = step_along(x, y, z, axes.n_axis, axes.normal_sign);
            let neighbor = sampler.block_at(neighbor_pos.0, neighbor_pos.1, neighbor_pos.2);

            let mut visible = if neighbor.is_none()
                && (face == Face::NegY || (here == Block::Water && face != Face::PosY))
            {
                // Missing below-neighbor chunks are usually just the streaming
                // wavefront. Water also must not treat missing side neighbours
                // as air: doing so emits chunk-sized transparent curtains.
                false
            } else {
                let neighbor = neighbor.unwrap_or(Block::Air);
                here != Block::Air
                    && !reg.info(neighbor).opaque
                    && (reg.info(here).opaque || here != neighbor)
            };

            // Water top faces are emitted at one-block resolution after the
            // greedy pass so vertex waves have enough geometry to displace.
            if face == Face::PosY && here == Block::Water {
                visible = false;
            }

            let cell = if visible {
                let ao = corner_ao_at(face, |dx, dy, dz| sampler.block_at(x + dx, y + dy, z + dz));
                let light = sampler.light_at(neighbor_pos.0, neighbor_pos.1, neighbor_pos.2);
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
}

fn merge_mask_slice(
    face: Face,
    reg: &BlockRegistry,
    mesh: &mut ChunkMesh,
    mask: &[Cell; D * D],
    slice: i32,
    axes: &AxisMap,
) {
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

            let mut w = 1;
            while ui + w < D && !visited[idx0 + w] && mask[idx0 + w] == cell {
                w += 1;
            }

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
                axes.n_axis,
                axes.u_axis,
                axes.v_axis,
                cell,
                reg,
            );
            ui += w;
        }
    }
}

fn axis_map(face: Face) -> AxisMap {
    match face {
        Face::PosX => AxisMap {
            n_axis: 0,
            u_axis: 2,
            v_axis: 1,
            normal_sign: 1,
        },
        Face::NegX => AxisMap {
            n_axis: 0,
            u_axis: 1,
            v_axis: 2,
            normal_sign: -1,
        },
        Face::PosY => AxisMap {
            n_axis: 1,
            u_axis: 0,
            v_axis: 2,
            normal_sign: 1,
        },
        Face::NegY => AxisMap {
            n_axis: 1,
            u_axis: 2,
            v_axis: 0,
            normal_sign: -1,
        },
        Face::PosZ => AxisMap {
            n_axis: 2,
            u_axis: 1,
            v_axis: 0,
            normal_sign: 1,
        },
        Face::NegZ => AxisMap {
            n_axis: 2,
            u_axis: 0,
            v_axis: 1,
            normal_sign: -1,
        },
    }
}

/// Translate `(slice, u, v)` back into a chunk-local `(x, y, z)`.
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
