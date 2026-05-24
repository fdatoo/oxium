use super::*;
use crate::worldgen::region::{CavePool, ChamberRadius};
use glam::{IVec3, Vec3};

/// Build a chunk of solid Stone with a spherical air cavity carved out.
fn stone_chunk_with_cavity(center_local: (u32, u32, u32), radius: u32) -> DenseChunk {
    let mut chunk = DenseChunk::empty();
    for x in 0..CHUNK_DIM_U {
        for y in 0..CHUNK_DIM_U {
            for z in 0..CHUNK_DIM_U {
                chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Stone);
            }
        }
    }
    let (cx, cy, cz) = center_local;
    let r2 = (radius * radius) as i64;
    for x in 0..CHUNK_DIM_U {
        for y in 0..CHUNK_DIM_U {
            for z in 0..CHUNK_DIM_U {
                let dx = x as i64 - cx as i64;
                let dy = y as i64 - cy as i64;
                let dz = z as i64 - cz as i64;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Air);
                }
            }
        }
    }
    chunk
}

#[test]
fn cave_pool_fills_air_inside_ellipsoid_up_to_surface_y() {
    // Chunk at world Y -64..-33 (chunk Y = -2).
    let coord = ChunkCoord(IVec3::new(0, -2, 0));
    let planner = FluidPlanner::new(42, coord);
    let origin_y = coord.origin().0.y; // -64

    // Chamber centered at local (16, 16, 16) → world (16, -48, 16).
    let center_local = (16u32, 16u32, 16u32);
    let radius = 8u32;
    let mut chunk = stone_chunk_with_cavity(center_local, radius);

    let world_center = Vec3::new(
        coord.origin().0.x as f32 + center_local.0 as f32,
        origin_y as f32 + center_local.1 as f32,
        coord.origin().0.z as f32 + center_local.2 as f32,
    );
    let radii = ChamberRadius(Vec3::splat(radius as f32));
    // Pool surface sits 30% up from the floor of the cavity.
    let floor_y = (world_center.y - radii.0.y).floor() as i32;
    let height = (radii.0.y * 2.0) as i32;
    let surface_y = floor_y + (height as f32 * 0.30) as i32;

    let pool = CavePool {
        center: world_center,
        radii,
        surface_y,
        bed_y: floor_y,
        kind: FluidBodyKind::CavePool,
    };

    planner.apply_cave_pools(&mut chunk, &[&pool]);

    // Every Air voxel inside the sphere at or below surface_y should be Water.
    let mut water_count = 0u32;
    let mut above_surface_water = 0u32;
    for x in 0..CHUNK_DIM_U {
        for y in 0..CHUNK_DIM_U {
            for z in 0..CHUNK_DIM_U {
                let wy = origin_y + y as i32;
                let block = chunk.get(LocalPos(UVec3::new(x, y, z)));
                if block == Block::Water {
                    water_count += 1;
                    if wy > surface_y {
                        above_surface_water += 1;
                    }
                }
            }
        }
    }
    assert!(water_count > 0, "no water placed in cave pool");
    assert_eq!(above_surface_water, 0, "water placed above surface_y");
}

#[test]
fn cave_pool_does_not_overwrite_solid_blocks() {
    let coord = ChunkCoord(IVec3::new(0, -2, 0));
    let planner = FluidPlanner::new(42, coord);
    // All-stone chunk — pool should write nothing.
    let mut chunk = DenseChunk::empty();
    for x in 0..CHUNK_DIM_U {
        for y in 0..CHUNK_DIM_U {
            for z in 0..CHUNK_DIM_U {
                chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Stone);
            }
        }
    }
    let pool = CavePool {
        center: Vec3::new(16.0, -48.0, 16.0),
        radii: ChamberRadius(Vec3::splat(8.0)),
        surface_y: -46,
        bed_y: -56,
        kind: FluidBodyKind::CavePool,
    };
    planner.apply_cave_pools(&mut chunk, &[&pool]);
    for x in 0..CHUNK_DIM_U {
        for y in 0..CHUNK_DIM_U {
            for z in 0..CHUNK_DIM_U {
                assert_eq!(
                    chunk.get(LocalPos(UVec3::new(x, y, z))),
                    Block::Stone,
                    "solid block overwritten at ({x},{y},{z})"
                );
            }
        }
    }
}

#[test]
fn lava_pool_kind_places_lava_block() {
    let coord = ChunkCoord(IVec3::new(0, -4, 0)); // deep chunk: Y -128..-97
    let planner = FluidPlanner::new(42, coord);
    let origin_y = coord.origin().0.y;

    let mut chunk = stone_chunk_with_cavity((16, 16, 16), 8);
    let world_center_y = origin_y as f32 + 16.0;
    let pool = CavePool {
        center: Vec3::new(16.0, world_center_y, 16.0),
        radii: ChamberRadius(Vec3::splat(8.0)),
        surface_y: world_center_y as i32 - 2,
        bed_y: world_center_y as i32 - 8,
        kind: FluidBodyKind::LavaPool,
    };
    planner.apply_cave_pools(&mut chunk, &[&pool]);

    let mut lava_count = 0u32;
    for x in 0..CHUNK_DIM_U {
        for y in 0..CHUNK_DIM_U {
            for z in 0..CHUNK_DIM_U {
                if chunk.get(LocalPos(UVec3::new(x, y, z))) == Block::Lava {
                    lava_count += 1;
                }
            }
        }
    }
    assert!(lava_count > 0, "no lava placed for LavaPool kind");
}
