//! Region coordinate types: `RegionCoord` and `MacroRegionCoord`.
//!
//! These are pure coordinate wrappers with no dependencies on the rest of
//! the region module. Everything that needs to locate a region in the
//! infinite world grid starts here.

use crate::worldgen::tuning::{FINE_REGION_SIZE, MACRO_REGION_SIZE};

// ── Fine region coordinates ───────────────────────────────────────────

/// Grid coordinate of a fine region (512 × 512 blocks).
///
/// `x` and `z` equal `floor(world_coord / FINE_REGION_SIZE)`. Negative
/// values are valid — the world grid extends in all directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionCoord {
    pub x: i32,
    pub z: i32,
}

impl RegionCoord {
    /// The fine region containing `(wx, wz)`.
    pub fn containing(wx: i32, wz: i32) -> Self {
        Self {
            x: wx.div_euclid(FINE_REGION_SIZE),
            z: wz.div_euclid(FINE_REGION_SIZE),
        }
    }

    /// World coordinate of the region's south-west corner.
    pub fn origin(self) -> (i32, i32) {
        (self.x * FINE_REGION_SIZE, self.z * FINE_REGION_SIZE)
    }
}

// ── Macro region coordinates ──────────────────────────────────────────

/// Grid coordinate of a macro region (8192 × 8192 blocks).
///
/// The macro grid covers the same infinite plane as fine regions but at
/// 16× coarser granularity. One macro region covers 16 × 16 fine regions.
/// Used by the trunk-river pass to provide long-distance drainage context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroRegionCoord {
    pub x: i32,
    pub z: i32,
}

impl MacroRegionCoord {
    pub fn containing(wx: i32, wz: i32) -> Self {
        Self {
            x: wx.div_euclid(MACRO_REGION_SIZE),
            z: wz.div_euclid(MACRO_REGION_SIZE),
        }
    }

    pub fn origin(self) -> (i32, i32) {
        (self.x * MACRO_REGION_SIZE, self.z * MACRO_REGION_SIZE)
    }
}
