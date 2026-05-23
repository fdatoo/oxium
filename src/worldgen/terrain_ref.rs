//! [`TerrainRef`] — a grouped borrow of the noise and config used whenever
//! `h_pre`, `is_cliff`, or the 3D density volume is sampled at a world
//! coordinate.
//!
//! The three fields always travel together: [`HeightmapNoise::h_pre`] requires
//! all three to evaluate the climate-spline surface height. Grouping them into
//! one `Copy` struct eliminates the three-argument repetition that otherwise
//! appears at every call site in `hydrology/` and `caves/`.
//!
//! `TerrainRef` is **not** an abstraction boundary — it is a typed argument
//! cluster. Callers are free to destructure it inline when that is clearer.

use crate::worldgen::config::{ClimateConfig, DensityConfig};
use crate::worldgen::density::HeightmapNoise;

/// A grouped borrow of the noise and configuration used for terrain-height,
/// cliff, and 3D-density evaluation at any world coordinate.
///
/// - `heightmap` provides `h_pre` (spline-based surface height) and
///   `is_cliff`.
/// - `climate` and `density` are the configs those splines read from.
///
/// All three are required by every `h_pre` call — grouping them avoids
/// repeating the triplet at every function signature in `hydrology/` and
/// `caves/`.
///
/// `TerrainRef<'a>` is `Copy` (three shared references, ≤ 8 bytes each),
/// so it can be passed into closures and recursive helpers without cloning.
#[derive(Clone, Copy)]
pub(crate) struct TerrainRef<'a> {
    /// The FBM + domain-warp noise that produces `h_pre` and `is_cliff`.
    pub heightmap: &'a HeightmapNoise,
    /// Climate spline config — temperature/humidity/continentalness curves.
    pub climate: &'a ClimateConfig,
    /// 3D density config — Y extents, relief blend weights.
    pub density: &'a DensityConfig,
}
