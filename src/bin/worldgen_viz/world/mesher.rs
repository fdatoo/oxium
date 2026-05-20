//! Per-chunk meshing for the viz: positions + baked block-paint colors.
//!
//! Reuses oxium::mesher::Face for face direction enumeration but emits the
//! viz Vertex (position + color) instead of the game's atlas-aware vertex.

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

pub fn mesh_chunk(coord: ChunkCoord, chunk: &DenseChunk) -> VizMesh {
    let dim = CHUNK_DIM_U as i32;
    let origin = Vec3::new(
        (coord.0.x * dim) as f32,
        (coord.0.y * dim) as f32,
        (coord.0.z * dim) as f32,
    );
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
                let color = color_for(block);
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
                        emit_face(p, color, face, &mut mesh);
                    }
                }
            }
        }
    }
    mesh
}

/// Per-face brightness multiplier. Approximates ambient + sun shading
/// without paying for real normals or a light buffer: top faces full
/// bright, bottom faces deep shadow, sides graduated so XZ-facing
/// cliffs read as cliffs and the user gets a sense of elevation. This
/// is the cheapest readable-height trick; PR 3 swaps it for a real
/// normal-aware shader once paint modes need that anyway.
fn face_tint(face: Face) -> f32 {
    match face {
        Face::PosY => 1.00,            // top — full sun
        Face::PosX | Face::NegZ => 0.82, // sun-side walls
        Face::NegX | Face::PosZ => 0.66, // shadow-side walls
        Face::NegY => 0.40,            // underside
    }
}

fn emit_face(p: Vec3, color: [f32; 3], face: Face, mesh: &mut VizMesh) {
    let base = mesh.vertices.len() as u32;
    let tint = face_tint(face);
    let color = [color[0] * tint, color[1] * tint, color[2] * tint];
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

fn color_for(b: Block) -> [f32; 3] {
    match b {
        Block::Stone => [0.55, 0.55, 0.55],
        Block::Dirt => [0.50, 0.32, 0.18],
        Block::Grass => [0.30, 0.65, 0.25],
        Block::Sand => [0.92, 0.85, 0.62],
        Block::Snow => [0.95, 0.95, 0.97],
        Block::Lava => [1.0, 0.45, 0.08],
        _ => [0.4, 0.4, 0.4],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    #[test]
    fn empty_chunk_meshes_to_no_vertices() {
        let chunk = DenseChunk::empty();
        let m = mesh_chunk(ChunkCoord(IVec3::ZERO), &chunk);
        assert_eq!(m.vertices.len(), 0);
        assert_eq!(m.indices.len(), 0);
    }

    #[test]
    fn single_stone_at_origin_has_six_faces() {
        let mut chunk = DenseChunk::empty();
        chunk.set(LocalPos(glam::UVec3::new(0, 0, 0)), Block::Stone);
        let m = mesh_chunk(ChunkCoord(IVec3::ZERO), &chunk);
        assert_eq!(m.vertices.len(), 6 * 4, "expected 6 quads = 24 verts");
        assert_eq!(m.indices.len(), 6 * 6, "expected 6 quads = 36 indices");
    }

    #[test]
    fn adjacent_solids_hide_shared_face() {
        let mut chunk = DenseChunk::empty();
        chunk.set(LocalPos(glam::UVec3::new(0, 0, 0)), Block::Stone);
        chunk.set(LocalPos(glam::UVec3::new(1, 0, 0)), Block::Stone);
        let m = mesh_chunk(ChunkCoord(IVec3::ZERO), &chunk);
        // 2 cubes share one internal face; each cube has 6 faces, minus 2
        // hidden = 10 visible quads.
        assert_eq!(m.vertices.len(), 10 * 4);
    }

    #[test]
    fn boundary_face_emitted_when_chunk_edge() {
        // Block at local (31, 0, 0): the +X face of this block is at the
        // chunk's outer +X edge; we emit it (no neighbor info in PR 1).
        let mut chunk = DenseChunk::empty();
        let edge = LocalPos(glam::UVec3::new(31, 0, 0));
        chunk.set(edge, Block::Stone);
        let m = mesh_chunk(ChunkCoord(IVec3::ZERO), &chunk);
        assert_eq!(m.vertices.len(), 6 * 4);
    }
}
