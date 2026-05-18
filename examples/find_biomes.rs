use glam::{IVec3, UVec3};
use oxium::voxel::block::Block;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::{ChunkCoord, LocalPos};
use oxium::worldgen::Generator;

fn main() {
    let g = Generator::new(42);
    let mut by_height: Vec<(i32, i32, i32)> = Vec::new(); // (height, x, z) for stone tops > 75
    let mut deserts: Vec<(i32, i32, i32)> = Vec::new();
    let mut d = DenseChunk::empty();
    for cx in -12..=12 {
        for cz in -12..=12 {
            for cy in 2..=4 {
                let coord = ChunkCoord(IVec3::new(cx, cy, cz));
                g.fill_chunk(coord, &mut d);
                let origin = coord.origin().0;
                for lz in 0..32 {
                    for lx in 0..32 {
                        for ly in (0..32).rev() {
                            let lp = LocalPos(UVec3::new(lx, ly, lz));
                            let b = d.blocks[lp.to_index()];
                            if b != Block::Air {
                                let y = origin.y + ly as i32;
                                let wx = origin.x + lx as i32;
                                let wz = origin.z + lz as i32;
                                if b == Block::Stone && y > 75 {
                                    by_height.push((y, wx, wz));
                                }
                                if b == Block::Sand && y > 64 {
                                    deserts.push((y, wx, wz));
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    by_height.sort_by(|a, b| b.0.cmp(&a.0));
    println!("Total mountain peaks (stone > y=75): {}", by_height.len());
    for &(h, x, z) in by_height.iter().take(5) {
        println!("  peak: y={} at ({}, {})", h, x, z);
    }
    println!("Total desert columns: {}", deserts.len());
    for &(h, x, z) in deserts.iter().take(5) {
        println!("  desert: y={} at ({}, {})", h, x, z);
    }
}
