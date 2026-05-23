//! Static terrain-aware fluid planning for chunk generation.
//!
//! Runtime fluid simulation is deliberately out of scope here. This
//! planner writes deterministic fluid source blocks after terrain and cave
//! carving have decided which voxels are solid or empty. There are two
//! entry points:
//!
//! - [`FluidPlanner::apply_surface_fluids`]: stamps Water into any Air voxel
//!   in `(col.height, water_surface_y]` for columns where a surface water
//!   body (ocean, lake, river) has been identified by the hydrology pass.
//! - [`FluidPlanner::apply_cave_pools`]: stamps Water or Lava into Air
//!   voxels inside cave chamber ellipsoids that qualified for a pool during
//!   region build (see `caves::derive_cave_pools`).
//!
//! Both passes run after the main voxel fill loop and after the surface-block
//! fixer, so they stamp into already-correctly-surfaced terrain.
//!
//! See `docs/book/content/part-4-chunk-fill/4.8-fluid-settle.mdx`.

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
#[path = "fluid_tests.rs"]
mod tests;
