//! Mapping between chunk coordinates, region filenames, and header slots.

use crate::voxel::coords::ChunkCoord;
use std::path::PathBuf;

use super::format::REGION_DIM;

/// Number of slots in one region file (4096).
pub const REGION_SLOTS: usize = (REGION_DIM as usize).pow(3);

/// Region-grid coordinate for the 16 x 16 x 16 file that owns a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Header slot inside one region file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionSlot(usize);

impl RegionSlot {
    /// Raw slot table index, in `0..REGION_SLOTS`.
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

/// Path of the region file that owns `chunk_coord`.
///
/// The file is in `saves_dir/regions/r.{rx}.{ry}.{rz}.bin`; each component is
/// the region grid coordinate derived by floor-dividing the chunk coord by 16.
pub fn region_path(saves_dir: &std::path::Path, chunk_coord: ChunkCoord) -> PathBuf {
    let coord = region_coord(chunk_coord);
    saves_dir
        .join("regions")
        .join(format!("r.{}.{}.{}.bin", coord.x, coord.y, coord.z))
}

/// Header slot for a given chunk coord.
///
/// Includes Y so two chunks at the same XZ but different Y do not collide on
/// disk. The local coordinate is `rem_euclid(16)` so negative chunk coords map
/// into the correct owning file and slot.
pub fn region_slot(chunk_coord: ChunkCoord) -> RegionSlot {
    let lx = chunk_coord.0.x.rem_euclid(REGION_DIM) as usize;
    let ly = chunk_coord.0.y.rem_euclid(REGION_DIM) as usize;
    let lz = chunk_coord.0.z.rem_euclid(REGION_DIM) as usize;
    RegionSlot((lx * (REGION_DIM as usize) + ly) * (REGION_DIM as usize) + lz)
}

/// Compatibility helper for callers that still need the raw slot index.
#[inline]
pub fn slot_index(chunk_coord: ChunkCoord) -> usize {
    region_slot(chunk_coord).index()
}

/// The `(rx, ry, rz)` region grid coordinate that owns `chunk_coord`.
pub fn region_coord(chunk_coord: ChunkCoord) -> RegionCoord {
    RegionCoord {
        x: chunk_coord.0.x.div_euclid(REGION_DIM),
        y: chunk_coord.0.y.div_euclid(REGION_DIM),
        z: chunk_coord.0.z.div_euclid(REGION_DIM),
    }
}
