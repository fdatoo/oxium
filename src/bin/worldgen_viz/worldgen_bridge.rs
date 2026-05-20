//! Worldgen → mesh adapter for the visualizer.
//!
//! Regenerates a small fixed-size chunk region from a WorldgenConfig
//! and meshes it. Returns vertex + index buffers in the visualizer's
//! `scene::Vertex` format.

use crate::scene::Vertex;
use glam::Vec3;
use oxium::voxel::block::Block;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::{ChunkCoord, CHUNK_DIM_U};
use oxium::worldgen::config::{ConfigHolder, WorldgenConfig};
use oxium::worldgen::Generator;

/// Visualizer-fixed region. At 32^3 voxels per chunk, 2×2 horizontal ×
/// 4 vertical = 16 chunks = ~500k voxels — meshes in ~30-50ms.
pub const REGION_CHUNKS_X: i32 = 3;
pub const REGION_CHUNKS_Z: i32 = 3;
/// Include enough vertical range to cover the actual terrain
/// surface (typical Oxium heights are 50-110 in land biomes), plus
/// a couple chunks below for cave visibility. Y range = -32..127.
pub const REGION_CHUNKS_Y_MIN: i32 = -1;
pub const REGION_CHUNKS_Y_MAX: i32 = 4; // exclusive

/// Generate the region and return mesh vertices + indices for the
/// visualizer's debug shader.
pub fn regen_region_mesh(seed: u64, config: &WorldgenConfig) -> (Vec<Vertex>, Vec<u32>) {
    let holder = ConfigHolder::new(config.clone());
    let generator = Generator::with_config(seed, holder);

    let mut verts = Vec::with_capacity(4096);
    let mut idxs = Vec::with_capacity(6144);

    for cy in REGION_CHUNKS_Y_MIN..REGION_CHUNKS_Y_MAX {
        for cz in 0..REGION_CHUNKS_Z {
            for cx in 0..REGION_CHUNKS_X {
                let mut chunk = DenseChunk::empty();
                let coord = ChunkCoord(glam::IVec3::new(cx, cy, cz));
                generator.fill_chunk(coord, &mut chunk);
                emit_chunk_faces(&chunk, cx, cy, cz, &mut verts, &mut idxs);
            }
        }
    }
    (verts, idxs)
}

fn emit_chunk_faces(
    chunk: &DenseChunk,
    cx: i32,
    cy: i32,
    cz: i32,
    verts: &mut Vec<Vertex>,
    idxs: &mut Vec<u32>,
) {
    let dim = CHUNK_DIM_U as i32;
    let origin = Vec3::new((cx * dim) as f32, (cy * dim) as f32, (cz * dim) as f32);
    for lz in 0..dim {
        for ly in 0..dim {
            for lx in 0..dim {
                let local = oxium::voxel::coords::LocalPos(glam::UVec3::new(
                    lx as u32, ly as u32, lz as u32,
                ));
                let block = chunk.blocks[local.to_index()];
                if !is_solid(block) {
                    continue;
                }
                let p = origin + Vec3::new(lx as f32, ly as f32, lz as f32);
                let color = color_for(block);
                for (nx, ny, nz, face) in &[
                    (1i32, 0, 0, Face::PosX),
                    (-1, 0, 0, Face::NegX),
                    (0, 1, 0, Face::PosY),
                    (0, -1, 0, Face::NegY),
                    (0, 0, 1, Face::PosZ),
                    (0, 0, -1, Face::NegZ),
                ] {
                    let nlx = lx + nx;
                    let nly = ly + ny;
                    let nlz = lz + nz;
                    let neighbor_solid = if nlx < 0
                        || nlx >= dim
                        || nly < 0
                        || nly >= dim
                        || nlz < 0
                        || nlz >= dim
                    {
                        false // chunk-edge: render as if neighbor empty
                    } else {
                        let nl = oxium::voxel::coords::LocalPos(glam::UVec3::new(
                            nlx as u32, nly as u32, nlz as u32,
                        ));
                        is_solid(chunk.blocks[nl.to_index()])
                    };
                    if !neighbor_solid {
                        emit_face(p, color, *face, verts, idxs);
                    }
                }
            }
        }
    }
}

#[derive(Copy, Clone)]
enum Face {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

fn emit_face(p: Vec3, color: [f32; 3], face: Face, verts: &mut Vec<Vertex>, idxs: &mut Vec<u32>) {
    let base = verts.len() as u32;
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
        verts.push(Vertex {
            position: (p + *v).into(),
            color,
        });
    }
    idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
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
        _ => [0.4, 0.4, 0.4],
    }
}
