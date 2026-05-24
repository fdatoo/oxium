//! Shared context types threaded through the cave system builder helpers.
//!
//! [`CaveCtx`] bundles the three values that namespace every hash roll inside
//! the builder: `(seed, region_coord, system_idx)`. Passing it as a single
//! `Copy` argument instead of three separate values makes function signatures
//! readable and prevents accidentally mixing up `seed` and `system_idx`.
//!
//! [`StyleParams`] bundles the per-style numeric ranges extracted from the
//! hot-reloadable [`crate::worldgen::config::CaveStyleTable`]. Both
//! `sample_chambers` and `connect_chambers_mst` consume a `&StyleParams`.
//!
//! Both types are `pub(super)` — they are implementation details of the
//! `caves` submodule and should not be visible outside it.

use crate::worldgen::region::RegionCoord;

/// Identifies one cave system within a region.
///
/// All hash rolls inside the cave builder are namespaced by
/// `(seed, coord, system_idx)` so the result is byte-deterministic from
/// `(seed, coord)`. Bundling all three avoids repeating the triple at
/// every helper call site.
///
/// `CaveCtx` is `Copy` (≈16 bytes on 64-bit) — pass by value freely.
#[derive(Clone, Copy)]
pub(super) struct CaveCtx {
    pub(super) seed: u64,
    pub(super) coord: RegionCoord,
    pub(super) system_idx: i32,
}

/// Per-style numeric ranges extracted from [`crate::worldgen::config::CaveStyleTable`].
///
/// Extracted once per system by [`crate::worldgen::caves::system::style_params`]
/// and passed into `sample_chambers` and `connect_chambers_mst` so neither
/// function needs to carry the full cave config.
pub(super) struct StyleParams {
    /// `(min, max)` chamber count for this style.
    pub(super) chamber_count: (u32, u32),
    /// `(min, max)` XZ semi-axis range (blocks).
    pub(super) r_xz: (f32, f32),
    /// `(min, max)` Y semi-axis range (blocks).
    pub(super) r_y: (f32, f32),
    /// `(min, max)` tunnel capsule radius range (blocks).
    pub(super) tunnel_r: (f32, f32),
}
