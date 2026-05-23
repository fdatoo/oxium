//! Static terrain-aware fluid planning for chunk generation.
//!
//! Runtime fluid simulation is deliberately out of scope here. This
//! planner writes deterministic source bodies after terrain and cave
//! carving have decided which voxels are solid or empty.

use crate::voxel::block::Block;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{CHUNK_DIM_U, ChunkCoord, LocalPos};
use crate::worldgen::ColumnData;
use crate::worldgen::region::CavePool;
use glam::UVec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidBodyKind {
    Ocean,
    River,
    Lake,
    CavePool,
    LavaPool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidReason {
    OceanConnected,
    RiverChannel,
    LakeBasin,
    CaveBasin,
    LavaBasin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FluidCell {
    pub kind: FluidBodyKind,
    pub block: Block,
    pub surface_y: i32,
    pub bed_y: i32,
    pub reason: FluidReason,
}

pub struct FluidPlanner {
    pub(crate) coord: ChunkCoord,
}

impl FluidPlanner {
    pub fn new(_seed: u64, coord: ChunkCoord) -> Self {
        Self { coord }
    }

    /// Fill surface water bodies for all columns in the chunk.
    ///
    /// Uses the unified `ColumnData::water_surface_y` field (plate-driven
    /// ocean, sink-fill lake, or river) instead of the old chunk-local BFS
    /// ocean mask. Any column with `water_surface_y == Some(w)` gets Water
    /// stamped into every Air voxel in `(col.height, w]`.
    pub fn apply_surface_fluids(&self, chunk: &mut DenseChunk, columns: &[ColumnData]) {
        debug_assert_eq!(columns.len(), (CHUNK_DIM_U as usize).pow(2));

        let dim = CHUNK_DIM_U as usize;
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let col = columns[z as usize * dim + x as usize];
                let Some(water_y) = col.water_surface_y else {
                    continue;
                };
                let cell = FluidCell {
                    kind: FluidBodyKind::Ocean, // placeholder; kind is informational only
                    block: Block::Water,
                    surface_y: water_y,
                    bed_y: col.height,
                    reason: FluidReason::OceanConnected,
                };
                // Fill start: normally one above the terrain floor, but
                // clamped to water_y so the range is never empty when
                // 3D density has pushed terrain above the river surface.
                // Step 6 (force-flood) will pre-clear the water column
                // to Air so this becomes equivalent to fill_column_air_range.
                let fill_from = (col.height + 1).min(water_y);
                self.fill_column_replace_range(chunk, x, z, cell, fill_from, water_y);
            }
        }
    }

    /// Stamp fluid into every Air voxel inside a cave pool ellipsoid at
    /// or below the pool's `surface_y`. Writes only to `Block::Air` so
    /// aquifer-placed lava and terrain solids are never overwritten.
    ///
    /// The pool ellipsoid is defined by `(center, radii)` in world space.
    /// For each voxel `p` inside the chunk, membership is tested as:
    ///   `((p - center) / radii)² ≤ 1 && wy ≤ pool.surface_y`
    pub fn apply_cave_pools(&self, chunk: &mut DenseChunk, pools: &[&CavePool]) {
        if pools.is_empty() {
            return;
        }
        let origin = self.coord.origin().0;
        for z in 0..CHUNK_DIM_U {
            for y in 0..CHUNK_DIM_U {
                for x in 0..CHUNK_DIM_U {
                    let wy = origin.y + y as i32;
                    let wx = origin.x + x as i32;
                    let wz = origin.z + z as i32;
                    let pos = LocalPos(UVec3::new(x, y, z));
                    if chunk.get(pos) != Block::Air {
                        continue;
                    }
                    for pool in pools {
                        if wy > pool.surface_y {
                            continue;
                        }
                        let d = glam::Vec3::new(
                            (wx as f32 - pool.center.x) / pool.radii.x,
                            (wy as f32 - pool.center.y) / pool.radii.y,
                            (wz as f32 - pool.center.z) / pool.radii.z,
                        );
                        if d.length_squared() <= 1.0 {
                            let block = match pool.kind {
                                FluidBodyKind::LavaPool => Block::Lava,
                                _ => Block::Water,
                            };
                            chunk.set(pos, block);
                            break;
                        }
                    }
                }
            }
        }
    }

    fn fill_column_replace_range(
        &self,
        chunk: &mut DenseChunk,
        x: u32,
        z: u32,
        cell: FluidCell,
        bed_y: i32,
        surface_y: i32,
    ) {
        let origin_y = self.coord.origin().0.y;
        for y in 0..CHUNK_DIM_U {
            let wy = origin_y + y as i32;
            if wy < bed_y || wy > surface_y {
                continue;
            }
            let pos = LocalPos(UVec3::new(x, y, z));
            if !matches!(chunk.get(pos), Block::Lava) {
                chunk.set(pos, cell.block);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::region::CavePool;
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
        let radii = Vec3::splat(radius as f32);
        // Pool surface sits 30% up from the floor of the cavity.
        let floor_y = (world_center.y - radii.y).floor() as i32;
        let height = (radii.y * 2.0) as i32;
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
            radii: Vec3::splat(8.0),
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
            radii: Vec3::splat(8.0),
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

}
