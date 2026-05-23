//! Integration test: cave breakthroughs at the heightmap have correct
//! surface materials on the actual cave floor, not on the cave ceiling.

use glam::UVec3;
use oxium::voxel::block::Block;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::{CHUNK_DIM_U, ChunkCoord, LocalPos};
use oxium::worldgen::Generator;

#[test]
fn surface_breakthrough_places_grass_on_cave_floor() {
    let generator = Generator::new(42);

    let mut found_floor_grass = false;
    let mut found_ceiling_grass = false;

    let dim = CHUNK_DIM_U as u32;

    // Scan an 8x8 grid of surface-level chunks across chunk Y 0..=3.
    // Cave breakthroughs are probabilistic; a wider scan provides more
    // chance of encountering one without becoming slow.
    for cx in -4i32..=4 {
        for cz in -4i32..=4 {
            for cy in 0i32..=3 {
                let coord = ChunkCoord(glam::IVec3::new(cx, cy, cz));
                let mut chunk = DenseChunk::empty();
                generator.fill_chunk(coord, &mut chunk);

                for lx in 0..dim {
                    for lz in 0..dim {
                        for ly in 1..(dim - 1) {
                            let cur = LocalPos(UVec3::new(lx, ly, lz));
                            let above = LocalPos(UVec3::new(lx, ly + 1, lz));
                            let below = LocalPos(UVec3::new(lx, ly - 1, lz));

                            // Ceiling grass: grass with air both above AND below.
                            if chunk.get(cur) == Block::Grass
                                && chunk.get(above) == Block::Air
                                && chunk.get(below) == Block::Air
                            {
                                found_ceiling_grass = true;
                            }

                            // Cave-floor grass: grass on top of stone with an
                            // open voxel immediately above; AND open at +2 (so
                            // it's not just a hillside, but actually a cave
                            // breach). Planned water can occupy breached floors
                            // after surface fixing, so water counts as open.
                            if ly + 2 < dim {
                                let above2 = LocalPos(UVec3::new(lx, ly + 2, lz));
                                if chunk.get(cur) == Block::Grass
                                    && matches!(chunk.get(above), Block::Air | Block::Water)
                                    && chunk.get(below) == Block::Stone
                                    && matches!(chunk.get(above2), Block::Air | Block::Water)
                                {
                                    found_floor_grass = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(
        !found_ceiling_grass,
        "found grass on a cave ceiling (grass with air above AND below)"
    );
    assert!(
        found_floor_grass,
        "no cave-floor grass found in 9x9x4 chunk scan — possibly no cave breaches in this region (sanity check)"
    );
}
