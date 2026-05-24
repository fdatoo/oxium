//! Procedural world generation: a pure `(seed, ChunkCoord) -> DenseChunk` map.
//!
//! "Pure" matters: terrain output must depend only on the seed and chunk
//! coordinate so chunks can be regenerated from disk-free state and so unit
//! tests can pin output with golden hashes. To make that compatible with
//! the more expensive geography the new pipeline produces, intermediate
//! data is memoised in two LRU caches keyed on region coords (see
//! `region.rs`). The function is still pure in `(seed, coord)` — caches
//! are just memoisation.
//!
//! ### Pipeline (per chunk)
//!
//! 1. **Pre-fetch regions.** `gather_chunk_regions` populates a 3 × 3
//!    grid of fine regions around the chunk; per-column queries read
//!    from this grid without re-locking the cache.
//! 2. **Per-column terrain.** For each column:
//!    * `heightmap::h_pre` evaluates `SEA_LEVEL + plate_shelf +
//!      plate_ridge_lift + warped_fbm * plate_roughness` from the
//!      Voronoi plate decomposition (`plates.rs`) and the
//!      domain-warped FBM relief.
//!    * `slope_at` on `h_pre` decides cliff exposure (slope-driven,
//!      replaces v1's `MOUNTAIN_ROCK_LINE` line).
//!    * `valley_carve` queries the region's river segments for a
//!      perpendicular-distance U-profile carve depth, subtracting
//!      it from `h_pre` to get `h_final`.
//!    * `climate.rs`-driven biome classifier picks Tundra /
//!      SnowyForest / Plains / Forest / Desert / Tropical via
//!      threshold-perturbed temperature & humidity noise.
//! 3. **Surface block selection.** Cliff → Stone; beach band → Sand;
//!    snow line / cold biome → Snow; Desert → Sand; otherwise Grass,
//!    with a stochastic sand-transition band on the grass side of the
//!    desert boundary.
//! 4. **Caves.** Graph-based cave systems (`caves.rs`) deposit
//!    chambers + spline tunnels into the chunk. The `CAVE_SURFACE_BUFFER`
//!    rule preserves the grass cap except where an explicit entrance
//!    (sinkhole / cliff mouth / skylight) punches through. Cheese + pillar
//!    noise carvers provide ambient density variation underground.
//! 5. **Water flood.** Sea-level flood + per-column lake-rim flood
//!    (the latter from sink-filled basins in the hydrology pass)
//!    turn any air cell below the appropriate water level into Water.
//! 6. **Trees.** Per-cell deterministic placement (`tree_in_cell`)
//!    stamps Oak round-canopy or Palm spreading-fronds shapes
//!    depending on the column's biome.
//!
//! ### Layer dependencies
//!
//! Plates → continental mask + ridge lift → heightmap → flow
//! accumulation (fine + macro hierarchical) → valley carve →
//! cave systems / cheese+pillar noise → biomes / surface materials / trees.
//! Each layer is in its own module; this file is the public entry
//! point that wires them together.
//!
//! See `docs/book/content/part-2-overview/2.1-big-picture.mdx` for an
//! illustrated overview, `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`
//! for the architectural spec, and `docs/superpowers/specs/2026-05-19-worldgen-3d-design.md`
//! for the 3D density + climate multi-noise design.

use crate::voxel::coords::ChunkCoord;
use noise::{Fbm, Simplex};

// ── Module declarations ───────────────────────────────────────────────────
//
// Leaf modules (small, no further splits): hash, spline, flat_cache,
// noise_channel, config, plates, climate, carver, aquifer.
//
// Submodule directories: density/, region/, hydrology/, caves/.
// Each directory has a mod.rs //! header that orients readers.
//
// Generator impl modules: each adds an `impl Generator { ... }` block.
// Child modules can see parent-module private items (including private
// methods and pub(crate) fields on Generator), so helpers stay accessible
// without widening visibility.
pub mod aquifer;
pub mod biome;
pub mod carver;
pub mod caves;
pub mod climate;
pub mod columns;
pub mod columns_impl;
pub mod config;
pub mod density;
pub mod fill_chunk_impl;
pub mod flat_cache;
pub mod fluid;
pub mod generator_impl;
pub mod hash;
pub mod hydrology;
pub mod noise_channel;
pub mod pipeline;
pub mod plates;
pub mod probe;
pub mod probe_impl;
pub mod region;
pub mod spline;
pub mod surface;
pub mod surface_fixer;
pub(crate) mod terrain_ref;
pub mod trees;
pub mod trees_impl;
pub mod tuning;

/// World-space Y at which the sea surface sits. Re-exported here so
/// callers outside the worldgen module (renderer, persistence, tests)
/// don't have to import `tuning::SEA_LEVEL` directly.
pub use crate::worldgen::tuning::SEA_LEVEL;
// Re-export Biome and TreeKind so callers at `worldgen::Biome` continue
// to work after the type moved to `biome.rs`.
pub use biome::Biome;
pub use columns::ColumnData;

// Compatibility shim: `oxium::worldgen::heightmap::HeightmapNoise` is
// used by `tests/worldgen_fingerprint.rs`. Keep this alias until
// the fingerprint test is updated to use `worldgen::density::HeightmapNoise`.
// Internal callers (hydrology, caves) now use `crate::worldgen::density::*` directly.
pub mod heightmap {
    pub use crate::worldgen::density::density_3d::DensityNoise;
    pub use crate::worldgen::density::heightmap::HeightmapNoise;
    pub use crate::worldgen::density::math::{
        offset_to_world_y, peaks_and_valleys, plate_roughness_bias, signed_continentalness, slide,
        smooth_plate_contribution,
    };
}

/// Pre-built noise fields for one world seed.
///
/// The struct exists mainly so the noise fields are constructed *once*: the
/// `Fbm` builder is comparatively expensive, and chunk generation calls
/// `get` thousands of times per chunk.
pub struct Generator {
    /// PR 2: plate-driven heightmap (continental shelf + ridges +
    /// domain-warped FBM relief). Owns the FBM/warp noise fields.
    pub(crate) heightmap: density::heightmap::HeightmapNoise,
    /// 3D density evaluator (PR A): height-bias term combined with a
    /// 3D relief FBM. Drives the per-voxel solid/air decision in
    /// `fill_chunk` so moderate slopes don't read as clean
    /// chevron stripes.
    pub(crate) density: density::density_3d::DensityNoise,
    /// Temperature map (large-period 2D noise). Drives the cold/warm
    /// axis of the biome R-tree lookup. Negative values are colder
    /// (Tundra / SnowyForest), positive warmer (Desert / Tropical).
    pub(crate) temperature_map: Fbm<Simplex>,
    /// Humidity map (large-period 2D noise). Drives the wet/dry axis;
    /// wetter columns earn denser tree cover, drier columns read as
    /// sparser plains.
    pub(crate) humidity_map: Fbm<Simplex>,
    /// PR 4: weirdness 2D noise (mid-frequency). Mirrors MC's
    /// "ridge" axis as a biome-table input — lets the same
    /// (temperature, humidity) climate produce both base biomes
    /// and variant biomes (ice spikes / sunflower plains analogues
    /// — future work; for now the biome list doesn't distinguish).
    pub(crate) weirdness_noise: Fbm<Simplex>,
    /// PR 4: pre-built biome R-tree. Constructed once at Generator
    /// init from the bundled config's `BiomesConfig::entries`.
    /// Hot-reloading the biome table requires a Generator restart.
    pub(crate) biome_list: std::sync::Arc<climate::ParameterList>,
    // PR 6: surface rules are read straight from the live ConfigHolder
    // (`cfg.surface`) at chunk-fill time so config swaps from the viz /
    // file watcher take effect on the next regen. There used to be a
    // cached `surface_system` field here, but it baked the rules at
    // construction and silently ignored every config swap.
    /// Legacy aquifer sampler retained for visualizer compatibility.
    /// Chunk fill now uses `fluid::FluidPlanner` instead of per-voxel
    /// aquifer pressure placement.
    pub(crate) aquifer: aquifer::AquiferSystem,
    /// PR 8: cheese / pillar noise channels. Built once per Generator
    /// from `WorldgenConfig::cave`. Read per-voxel in `fill_chunk`
    /// to compose with the graph cave SDFs.
    pub(crate) noise_carvers: caves::NoiseCarvers,
    /// MC-style procedural carver tunnel cache. Per-chunk LRU keyed
    /// on origin chunk coord; each entry is the deterministic list
    /// of tunnels seeded by that chunk. Filled lazily as `fill_chunk`
    /// gathers neighbours' carvers to rasterise into the local mask.
    pub(crate) carver_cache: std::sync::Arc<
        std::sync::Mutex<lru::LruCache<ChunkCoord, std::sync::Arc<Vec<carver::CarverTunnel>>>>,
    >,
    pub(crate) seed: u64,
    /// LRU cache of pre-built fine regions. Consulted per chunk fill
    /// to evaluate valley carve and lake water; built on first touch.
    pub(crate) fine_cache: region::FineCache,
    /// LRU cache of macro regions feeding the trunk-river injection
    /// into fine flow accumulation.
    pub(crate) macro_cache: region::MacroCache,
    /// Hot-reloadable worldgen config (RON-backed). Read once per
    /// chunk via [`Self::config_snapshot`] to keep chunk gen
    /// deterministic even if the file watcher swaps mid-generation.
    pub(crate) config: config::ConfigHolder,
}

// ── Generator impl modules ────────────────────────────────────────────────
//
// Each `*_impl.rs` file adds an `impl Generator { ... }` block. They are
// declared as child modules here so the Rust module system makes them part
// of the worldgen module tree. Child modules can access every item in this
// file (including private ones on Generator) without widening visibility.
//
//   generator_impl.rs — new/with_config/config_snapshot, build_fine_region,
//                        gather_chunk_regions, build_carver_mask
//   columns_impl.rs   — column_data, column_data_with
//   fill_chunk_impl.rs — fill_chunk, light_inputs_for_chunk
//   trees_impl.rs     — add_trees, tree_in_cell_with_regions, stamp_tree
//   probe_impl.rs     — probe_column, paint_column, sample_stage,
//                        evaluate_density_breakdown

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
