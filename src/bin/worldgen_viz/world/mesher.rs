//! Per-chunk meshing for the viz: positions + paint-pass colors.
//!
//! Reuses oxium::mesher::Face for face direction enumeration but emits the
//! viz Vertex (position + color) instead of the game's atlas-aware vertex.
//! Color comes from a `PaintContext` (paint.rs) — Block mode reads the
//! voxel directly, other modes read precomputed per-column data.

use crate::paint::PaintContext;
use crate::render::scene::Vertex;
use glam::Vec3;
use oxium::mesher::Face;
use oxium::voxel::block::Block;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::{ChunkCoord, LocalPos, CHUNK_DIM_U};

#[derive(Default)]
pub struct VizMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub fn mesh_chunk(coord: ChunkCoord, chunk: &DenseChunk, paint: &PaintContext) -> VizMesh {
    let dim = CHUNK_DIM_U as i32;
    let origin = Vec3::new(
        (coord.0.x * dim) as f32,
        (coord.0.y * dim) as f32,
        (coord.0.z * dim) as f32,
    );
    let world_origin = glam::IVec3::new(coord.0.x * dim, coord.0.y * dim, coord.0.z * dim);
    let mut mesh = VizMesh::default();
    for lz in 0..dim {
        for ly in 0..dim {
            for lx in 0..dim {
                let local = LocalPos(glam::UVec3::new(lx as u32, ly as u32, lz as u32));
                let block = chunk.get(local);
                if !is_solid(block) {
                    continue;
                }
                let p = origin + Vec3::new(lx as f32, ly as f32, lz as f32);
                let voxel = world_origin + glam::IVec3::new(lx, ly, lz);
                for face in Face::all() {
                    let [nx, ny, nz] = face.normal();
                    let (nlx, nly, nlz) = (lx + nx, ly + ny, lz + nz);
                    let neighbor_solid = if nlx < 0 || nlx >= dim || nly < 0 || nly >= dim || nlz < 0 || nlz >= dim {
                        false
                    } else {
                        let nl = LocalPos(glam::UVec3::new(nlx as u32, nly as u32, nlz as u32));
                        is_solid(chunk.get(nl))
                    };
                    if !neighbor_solid {
                        let color = paint.color_for(voxel.x, voxel.y, voxel.z, face, block);
                        emit_face(p, color, face, &mut mesh);
                    }
                }
            }
        }
    }
    mesh
}

fn emit_face(p: Vec3, color: [f32; 3], face: Face, mesh: &mut VizMesh) {
    let base = mesh.vertices.len() as u32;
    let quad: [Vec3; 4] = match face {
        Face::PosX => [
            Vec3::new(1., 0., 0.),
            Vec3::new(1., 0., 1.),
            Vec3::new(1., 1., 1.),
            Vec3::new(1., 1., 0.),
        ],
        Face::NegX => [
            Vec3::new(0., 0., 1.),
            Vec3::new(0., 0., 0.),
            Vec3::new(0., 1., 0.),
            Vec3::new(0., 1., 1.),
        ],
        Face::PosY => [
            Vec3::new(0., 1., 0.),
            Vec3::new(1., 1., 0.),
            Vec3::new(1., 1., 1.),
            Vec3::new(0., 1., 1.),
        ],
        Face::NegY => [
            Vec3::new(0., 0., 1.),
            Vec3::new(1., 0., 1.),
            Vec3::new(1., 0., 0.),
            Vec3::new(0., 0., 0.),
        ],
        Face::PosZ => [
            Vec3::new(1., 0., 1.),
            Vec3::new(0., 0., 1.),
            Vec3::new(0., 1., 1.),
            Vec3::new(1., 1., 1.),
        ],
        Face::NegZ => [
            Vec3::new(0., 0., 0.),
            Vec3::new(1., 0., 0.),
            Vec3::new(1., 1., 0.),
            Vec3::new(0., 1., 0.),
        ],
    };
    for v in &quad {
        mesh.vertices.push(Vertex {
            position: (p + *v).into(),
            color,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn is_solid(b: Block) -> bool {
    !matches!(b, Block::Air | Block::Water)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::{PaintContext, PaintMode};
    use glam::IVec3;

    /// Test helper: a Block-mode paint context (no Generator needed).
    fn block_paint(coord: ChunkCoord) -> PaintContext {
        // Build directly without a Generator since Block mode needs no columns.
        let dim = CHUNK_DIM_U as i32;
        PaintContext::without_columns(
            PaintMode::Block,
            coord.0.x * dim,
            coord.0.z * dim,
        )
    }

    #[test]
    fn empty_chunk_meshes_to_no_vertices() {
        let chunk = DenseChunk::empty();
        let coord = ChunkCoord(IVec3::ZERO);
        let m = mesh_chunk(coord, &chunk, &block_paint(coord));
        assert_eq!(m.vertices.len(), 0);
        assert_eq!(m.indices.len(), 0);
    }

    #[test]
    fn single_stone_at_origin_has_six_faces() {
        let mut chunk = DenseChunk::empty();
        chunk.set(LocalPos(glam::UVec3::new(0, 0, 0)), Block::Stone);
        let coord = ChunkCoord(IVec3::ZERO);
        let m = mesh_chunk(coord, &chunk, &block_paint(coord));
        assert_eq!(m.vertices.len(), 6 * 4, "expected 6 quads = 24 verts");
        assert_eq!(m.indices.len(), 6 * 6, "expected 6 quads = 36 indices");
    }

    #[test]
    fn adjacent_solids_hide_shared_face() {
        let mut chunk = DenseChunk::empty();
        chunk.set(LocalPos(glam::UVec3::new(0, 0, 0)), Block::Stone);
        chunk.set(LocalPos(glam::UVec3::new(1, 0, 0)), Block::Stone);
        let coord = ChunkCoord(IVec3::ZERO);
        let m = mesh_chunk(coord, &chunk, &block_paint(coord));
        assert_eq!(m.vertices.len(), 10 * 4);
    }

    #[test]
    fn boundary_face_emitted_when_chunk_edge() {
        let mut chunk = DenseChunk::empty();
        let edge = LocalPos(glam::UVec3::new(31, 0, 0));
        chunk.set(edge, Block::Stone);
        let coord = ChunkCoord(IVec3::ZERO);
        let m = mesh_chunk(coord, &chunk, &block_paint(coord));
        assert_eq!(m.vertices.len(), 6 * 4);
    }
}
