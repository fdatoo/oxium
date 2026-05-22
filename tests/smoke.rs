//! End-to-end smoke test.
//!
//! Builds a small world, generates a handful of chunks with the real
//! `Generator`, runs the lighting BFS on each, compresses them with the
//! palette format and stores them in a `World` — then asserts every
//! produced light value is in the legal `0..=15` range.
//!
//! Catches regressions where one pure module's API change silently
//! breaks another's expectations (e.g. a vertex layout shift that
//! confuses the mesher, a packed-array bug that emits 0xFF lights).

use glam::IVec3;
use oxium::lighting;
use oxium::mesher::greedy::mesh_greedy;
use oxium::voxel::block::BlockRegistry;
use oxium::voxel::chunk::{DenseChunk, Neighbors, PalettedChunk, rgb_brightness};
use oxium::voxel::coords::ChunkCoord;
use oxium::voxel::world::{ChunkSlot, World};
use oxium::worldgen::Generator;

#[test]
fn generate_light_mesh_round_trip() {
    let reg = BlockRegistry::new();
    let generator = Generator::new(42);

    // A 5×3×5 region of chunks centred around the origin's surface
    // band — enough variety that worldgen produces a mix of stone,
    // dirt, grass, water, and air.
    let mut world = World::new(42);
    for x in -2..=2 {
        for z in -2..=2 {
            for y in 1..=3 {
                let coord = ChunkCoord(IVec3::new(x, y, z));
                let mut dense = DenseChunk::empty();
                generator.fill_chunk(coord, &mut dense);
                let neighbors = Neighbors { chunks: [None; 6] };
                lighting::recompute_chunk(&mut dense, &neighbors, &reg);
                let pchunk = PalettedChunk::compress(&dense);
                world.insert(coord, pchunk);
            }
        }
    }

    // Every chunk's light arrays must round-trip through compress /
    // decompress with values in `0..=15`. Out-of-range values mean a
    // packed-array bug stomped on a neighbouring nibble.
    for slot in world.chunks.values() {
        if let ChunkSlot::Stored { data, .. } = slot {
            let d = data.decompress();
            for &v in d.sky_light.iter() {
                assert!(v <= 15, "sky light out of range: {v}");
            }
            for &cell in d.block_rgb.iter() {
                let v = rgb_brightness(cell);
                assert!(v <= 15, "block light out of range: {v}");
            }
        }
    }
}

#[test]
fn greedy_mesh_a_generated_chunk() {
    // Mesh one generated chunk and assert we produce *some* geometry —
    // a chunk with terrain in it must yield at least one quad.
    let reg = BlockRegistry::new();
    let generator = Generator::new(42);
    let mut dense = DenseChunk::empty();
    generator.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut dense);
    let neighbors: [Option<&DenseChunk>; 6] = [None; 6];
    let mesh = mesh_greedy(&dense, &neighbors, &reg);
    assert!(
        !mesh.vertices.is_empty(),
        "expected greedy meshing of a real chunk to produce geometry"
    );
    // Triangles are 3 indices each, so the count must be divisible by 3.
    assert_eq!(mesh.indices.len() % 3, 0);
}
