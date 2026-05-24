//! Water-specific top-face emission.

use crate::mesher::ao::corner_ao_at;
use crate::mesher::{ChunkMesh, Face};
use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::LocalPos;
use glam::UVec3;

use super::D;
use super::emit::emit_greedy_quad;
use super::mask::Cell;
use super::sampling::Sampler;

/// Emit 1-block-per-quad water top faces.
///
/// Greedy water tops look efficient, but one 32-wide transparent sheet gives
/// the vertex-wave shader too little geometry and creates chunk-boundary
/// hairlines. Per-block tops keep waves visually continuous.
pub(super) fn emit_water_tops_per_block(
    chunk: &DenseChunk,
    sampler: &Sampler<'_>,
    reg: &BlockRegistry,
    mesh: &mut ChunkMesh,
) {
    for y in 0..D as i32 {
        for z in 0..D as i32 {
            for x in 0..D as i32 {
                let here = chunk.get(LocalPos(UVec3::new(x as u32, y as u32, z as u32)));
                if here != Block::Water {
                    continue;
                }
                let Some(above) = sampler.block_at(x, y + 1, z) else {
                    // Unknown +Y may be an internal seam inside a water column.
                    continue;
                };
                if reg.info(above).opaque || above == Block::Water {
                    continue;
                }

                let ao = corner_ao_at(Face::PosY, |dx, dy, dz| {
                    sampler.block_at(x + dx, y + dy, z + dz)
                });
                let light = sampler.light_at(x, y + 1, z);
                let cell = Cell {
                    block: Block::Water as u16,
                    ao,
                    light,
                };
                emit_greedy_quad(
                    mesh,
                    Face::PosY,
                    y as u8,
                    x as u8,
                    z as u8,
                    1,
                    1,
                    1,
                    0,
                    2,
                    cell,
                    reg,
                );
            }
        }
    }
}
