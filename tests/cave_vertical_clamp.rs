//! Integration test: vertical-run clamp ensures no XZ column has more
//! than MAX_VERTICAL_AIR_RUN consecutive air voxels.

use glam::UVec3;
use oxium::voxel::block::Block;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::{ChunkCoord, LocalPos, CHUNK_DIM_U};
use oxium::worldgen::Generator;

#[test]
fn chunk_has_no_vertical_air_run_over_six() {
    // Generate a chunk known to have caves at depth.
    let generator = Generator::new(42);
    let coord = ChunkCoord(glam::IVec3::new(0, -2, 0)); // Y=-64..-32, underground
    let mut chunk = DenseChunk::empty();
    generator.fill_chunk(coord, &mut chunk);

    let dim = CHUNK_DIM_U as u32;
    for x in 0..dim {
        for z in 0..dim {
            let mut run = 0i32;
            let mut longest = 0i32;
            for y in 0..dim {
                let pos = LocalPos(UVec3::new(x, y, z));
                if chunk.get(pos) == Block::Air {
                    run += 1;
                    if run > longest {
                        longest = run;
                    }
                } else {
                    run = 0;
                }
            }
            assert!(
                longest <= 6,
                "column ({x},{z}) at chunk {coord:?} has air run of {longest}"
            );
        }
    }
}
