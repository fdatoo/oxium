//! Per-chunk sky-source heightmap. For each `(x, z)` column in the chunk,
//! records the world-Y of the lowest cell that is a sky light source —
//! i.e., a non-opaque cell with nothing opaque above it (within this
//! chunk).
//!
//! The current recompute lighting path does not need this for correctness,
//! but the metadata is kept alongside chunks for diagnostics and future
//! sky-source optimizations.

use crate::voxel::block::BlockRegistry;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{CHUNK_DIM, CHUNK_DIM_U, ChunkCoord, LocalPos};
use glam::UVec3;

/// Sentinel meaning "no opaque block in this column inside this chunk."
/// The whole column passed through as transparent; the real source floor
/// (if any) lives in a chunk above or below.
pub const NO_SOURCE_FLOOR: i32 = i32::MIN;

/// Per-chunk record of where the sky-source floor sits for each `(x, z)`
/// column. Each entry is a **world-Y coordinate**: the y of the lowest
/// cell that is itself a sky source (level 15 from sky).
///
/// Cells at world-Y `>= lowest_source_y(x, z)` within this chunk are
/// sources. Cells below receive sky through the recompute pass.
#[derive(Debug, Clone)]
pub struct ChunkSkyLightSources {
    lowest_source_y: Box<[i32; (CHUNK_DIM_U * CHUNK_DIM_U) as usize]>,
}

impl Default for ChunkSkyLightSources {
    /// An empty heightmap — every column reports `NO_SOURCE_FLOOR`.
    /// Used for chunks that haven't had `build_from_dense` called yet.
    fn default() -> Self {
        Self {
            lowest_source_y: Box::new([NO_SOURCE_FLOOR; (CHUNK_DIM_U * CHUNK_DIM_U) as usize]),
        }
    }
}

impl ChunkSkyLightSources {
    /// World-Y of the lowest sky-source cell in column `(lx, lz)`. Returns
    /// `NO_SOURCE_FLOOR` if the column has no opaque block in this chunk.
    pub fn lowest_source_y(&self, lx: u32, lz: u32) -> i32 {
        debug_assert!(lx < CHUNK_DIM_U && lz < CHUNK_DIM_U);
        self.lowest_source_y[Self::idx(lx, lz)]
    }

    #[inline]
    fn idx(lx: u32, lz: u32) -> usize {
        (lx + lz * CHUNK_DIM_U) as usize
    }

    /// Scan a chunk's blocks top-down per column. For each column, the
    /// first opaque cell defines the source floor; everything above is a
    /// source. If no opaque cell is found in the column within this
    /// chunk, the entry is `NO_SOURCE_FLOOR`.
    ///
    /// `chunk_coord` is needed to translate the local-Y of the topmost
    /// opaque cell into a world-Y.
    pub fn build_from_dense(
        dense: &DenseChunk,
        chunk_coord: ChunkCoord,
        registry: &BlockRegistry,
    ) -> Self {
        let mut out = Self::default();
        let chunk_bottom_y = chunk_coord.0.y * CHUNK_DIM;
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                let mut floor: i32 = NO_SOURCE_FLOOR;
                for ly in (0..CHUNK_DIM_U).rev() {
                    let pos = LocalPos(UVec3::new(lx, ly, lz));
                    let block = dense.blocks[pos.to_index()];
                    if registry.info(block).opaque {
                        // First opaque cell from the top — source floor
                        // is the cell immediately above it.
                        floor = chunk_bottom_y + ly as i32 + 1;
                        break;
                    }
                }
                out.lowest_source_y[Self::idx(lx, lz)] = floor;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::block::Block;
    use crate::voxel::chunk::DenseChunk;
    use glam::IVec3;

    #[test]
    fn default_is_all_no_source_floor() {
        let s = ChunkSkyLightSources::default();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(s.lowest_source_y(lx, lz), NO_SOURCE_FLOOR);
            }
        }
    }

    #[test]
    fn all_air_chunk_has_no_source_floor_anywhere() {
        let dense = DenseChunk::empty();
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(&dense, ChunkCoord(IVec3::ZERO), &reg);
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(
                    s.lowest_source_y(lx, lz),
                    NO_SOURCE_FLOOR,
                    "column ({lx},{lz}) should have no floor in all-air chunk",
                );
            }
        }
    }

    #[test]
    fn all_stone_chunk_floor_is_just_above_chunk_top() {
        // A chunk at chunk_coord (0, 0, 0) spans world-Y 0..32. Topmost
        // opaque cell in every column is at local y=31, world y=31. The
        // source floor is the cell above, world y=32.
        let dense = DenseChunk::new_filled(Block::Stone);
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(&dense, ChunkCoord(IVec3::ZERO), &reg);
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(s.lowest_source_y(lx, lz), 32);
            }
        }
    }

    #[test]
    fn stone_layer_at_local_y_20_gives_floor_at_world_y_21() {
        // 1-block stone layer at y=20 across the whole chunk;
        // everything else air. Floor = local y+1 = 21 in world space
        // (chunk_coord.y = 0).
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 20, lz)), Block::Stone);
            }
        }
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(&dense, ChunkCoord(IVec3::ZERO), &reg);
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(s.lowest_source_y(lx, lz), 21);
            }
        }
    }

    #[test]
    fn floor_uses_topmost_opaque_when_multiple_layers_exist() {
        // Two opaque layers at y=10 and y=20. Topmost (y=20) defines
        // the floor: world y=21.
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 10, lz)), Block::Stone);
                dense.set(LocalPos(UVec3::new(lx, 20, lz)), Block::Stone);
            }
        }
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(&dense, ChunkCoord(IVec3::ZERO), &reg);
        assert_eq!(s.lowest_source_y(5, 5), 21);
    }

    #[test]
    fn floor_is_per_column_independent() {
        // Stone at y=15 only in column (0,0); rest of chunk is air.
        let mut dense = DenseChunk::empty();
        dense.set(LocalPos(UVec3::new(0, 15, 0)), Block::Stone);
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(&dense, ChunkCoord(IVec3::ZERO), &reg);
        assert_eq!(
            s.lowest_source_y(0, 0),
            16,
            "column (0,0) has floor at world y=16"
        );
        assert_eq!(
            s.lowest_source_y(1, 0),
            NO_SOURCE_FLOOR,
            "column (1,0) has no floor — should be NO_SOURCE_FLOOR",
        );
        assert_eq!(s.lowest_source_y(0, 1), NO_SOURCE_FLOOR);
    }

    #[test]
    fn world_y_translation_respects_chunk_coord() {
        // Same stone layer at local y=10, but chunk is at chunk_coord
        // (0, 3, 0) which spans world-Y 96..128. Local y=10 → world y=106.
        // Source floor = world y=107.
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 10, lz)), Block::Stone);
            }
        }
        let reg = BlockRegistry::new();
        let s =
            ChunkSkyLightSources::build_from_dense(&dense, ChunkCoord(IVec3::new(0, 3, 0)), &reg);
        assert_eq!(s.lowest_source_y(5, 5), 107);
    }

    #[test]
    fn negative_chunk_y_translates_correctly() {
        // Chunk at chunk_coord (0, -1, 0) spans world-Y -32..0.
        // Stone at local y=5 → world y=-27. Floor = world y=-26.
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 5, lz)), Block::Stone);
            }
        }
        let reg = BlockRegistry::new();
        let s =
            ChunkSkyLightSources::build_from_dense(&dense, ChunkCoord(IVec3::new(0, -1, 0)), &reg);
        assert_eq!(s.lowest_source_y(5, 5), -26);
    }

    #[test]
    fn non_opaque_non_air_does_not_create_floor() {
        // Water and leaves are non-opaque; they should NOT define a
        // source floor (sky still propagates through them, attenuated).
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 20, lz)), Block::Water);
            }
        }
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(&dense, ChunkCoord(IVec3::ZERO), &reg);
        assert_eq!(
            s.lowest_source_y(5, 5),
            NO_SOURCE_FLOOR,
            "water is non-opaque; should not create a source floor",
        );
    }
}
