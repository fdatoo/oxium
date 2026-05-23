//! Lake rim lookup for per-column chunk fill.
//!
//! `lake_rim_at` is the only public function here. It translates a world
//! column `(wx, wz)` to the fine-cell index in the region and returns the
//! sink-fill rim elevation if the cell is tagged as a lake.
//!
//! See `docs/book/content/part-3-region-build/3.5-rivers-lakes.mdx`.

use crate::worldgen::region::{FineRegion, bitset_get};
use crate::worldgen::tuning::*;

/// Look up the lake rim at world `(wx, wz)`. Returns `Some(rim_y)` if
/// the column is inside a lake; `None` otherwise. Used by chunk fill
/// to flood lake water above the original heightmap.
pub fn lake_rim_at(wx: i32, wz: i32, region: &FineRegion) -> Option<i32> {
    let region_origin = (
        region.coord.x * FINE_REGION_SIZE,
        region.coord.z * FINE_REGION_SIZE,
    );
    let lx = wx - region_origin.0;
    let lz = wz - region_origin.1;
    let ix = lx.div_euclid(FINE_CELL);
    let iz = lz.div_euclid(FINE_CELL);
    if ix < 0 || iz < 0 || ix >= FINE_CELLS_PER_REGION || iz >= FINE_CELLS_PER_REGION {
        return None;
    }
    let i = (iz * FINE_CELLS_PER_REGION + ix) as usize;
    if bitset_get(&region.is_lake, i) {
        Some(region.lake_rim[i] as i32)
    } else {
        None
    }
}
