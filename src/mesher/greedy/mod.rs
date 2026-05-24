//! Greedy meshing: merge coplanar adjacent same-appearance face cells into
//! single rectangle quads.
//!
//! The algorithm runs per face direction:
//!
//! 1. [`mask`] scans each slice into a 32 x 32 grid of visible face cells.
//! 2. Adjacent equal cells merge into a maximal rectangle.
//! 3. [`emit`] writes one quad, flipping the triangle split when AO requires it.
//! 4. [`water`] adds one-block water-top quads after the greedy pass.
//!
//! The public entry point stays [`mesh_greedy`]; submodules exist only to keep
//! the sampling, mask-building, and vertex-emission invariants readable.

mod emit;
mod mask;
mod sampling;
mod water;

use crate::mesher::{ChunkMesh, Face};
use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::CHUNK_DIM_U;

use sampling::Sampler;

/// Chunk side length as a `usize`, used for fixed-size masks.
pub(super) const D: usize = CHUNK_DIM_U as usize;

/// Mesh a chunk with optional neighbour information using greedy merging.
///
/// Drop-in replacement for `naive::mesh_chunk_with_neighbors`: same signature,
/// same visual output, fewer quads for large coplanar surfaces.
pub fn mesh_greedy(
    chunk: &DenseChunk,
    neighbors: &[Option<&DenseChunk>; 6],
    reg: &BlockRegistry,
) -> ChunkMesh {
    let mut mesh = ChunkMesh::empty();
    let sampler = Sampler::new(chunk, neighbors);

    for face in Face::all() {
        mask::greedy_one_face(face, chunk, &sampler, reg, &mut mesh);
    }
    water::emit_water_tops_per_block(chunk, &sampler, reg, &mut mesh);

    // Single linear pass over the 32^3 dense grid to set the water flag. The
    // renderer reads it to decide whether the planar-reflection pass can skip.
    mesh.has_water = chunk.blocks.contains(&Block::Water);
    mesh
}

#[cfg(test)]
mod tests;
