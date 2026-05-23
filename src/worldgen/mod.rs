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
use noise::{Fbm, MultiFractal, Simplex};

// New worldgen modules. Most of them are PR 1 stubs that get filled
// in by later PRs (PR 2 implements heightmap, PR 3 hydrology, PR 4
// caves, PR 5 biomes/surface/trees). Two are fully implemented now
// because everything else builds on them:
//
//   * `tuning`  — central constants table.
//   * `plates`  — Voronoi plate decomposition.
//   * `hash`    — deterministic mixer used by plates / trees / caves.
//   * `region`  — LRU caches + region data structs.
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
use pipeline::ChunkRegions;

// Compatibility shim: `oxium::worldgen::heightmap::HeightmapNoise` is
// used by `tests/worldgen_fingerprint.rs`. Keep this alias until
// the fingerprint test is updated to use `worldgen::density::HeightmapNoise`.
// Internal callers (hydrology, caves) now use `crate::worldgen::density::*` directly.
pub mod heightmap {
    pub use crate::worldgen::density::heightmap::{DensityNoise, HeightmapNoise};
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
    pub(crate) density: density::heightmap::DensityNoise,
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

// ── §1 Generator construction ─────────────────────────────────────────────
impl Generator {
    /// Build a `Generator` with the given world seed and the bundled
    /// default config. Equivalent to [`Self::with_config`] passing a
    /// freshly-constructed `ConfigHolder` from
    /// [`config::WorldgenConfig::bundled_default`].
    pub fn new(seed: u64) -> Self {
        let cfg =
            config::WorldgenConfig::bundled_default().expect("bundled default.ron must parse");
        let holder = config::ConfigHolder::new(cfg);
        Self::with_config(seed, holder)
    }

    /// Build a `Generator` with the given world seed and an externally
    /// owned `ConfigHolder`. The application typically owns the holder
    /// (and the file watcher); the Generator reads from it via
    /// [`Self::config_snapshot`].
    pub fn with_config(seed: u64, config: config::ConfigHolder) -> Self {
        let mut g = Self::new_internal(seed);
        g.config = config;
        g
    }

    /// Cheap atomic read of the current config. Holds an
    /// `Arc<WorldgenConfig>` snapshot — call once per chunk and reuse
    /// across the chunk's lifetime to avoid mid-chunk drift if a
    /// hot-reload races chunk gen.
    pub fn config_snapshot(&self) -> std::sync::Arc<config::WorldgenConfig> {
        self.config.load()
    }

    /// Internal constructor. Builds all the noise fields but leaves
    /// `config` set to the bundled default; [`Self::with_config`]
    /// overwrites it.
    ///
    /// Each noise field is seeded with a different per-axis salt so they
    /// don't produce correlated patterns (mountain-noise lining up with
    /// height-noise would just amplify existing hills instead of adding
    /// new geographic features).
    fn new_internal(seed: u64) -> Self {
        // Heightmap noise: 4 octaves, ~96-block period at octave 0.
        // PR 2: the plate-driven heightmap owns its own FBM + warp
        // noise fields. The old `height_noise`, `mountain_noise`, and
        // `mountainness_map` are gone — plate geometry replaces them.
        // Load the bundled default once so all noise fields share a
        // consistent initial config (the file watcher can later swap
        // values, but the noise *frequencies* baked here stay).
        let bundled =
            config::WorldgenConfig::bundled_default().expect("bundled default.ron must parse");
        let heightmap = density::heightmap::HeightmapNoise::new(seed, &bundled.climate);
        let density = density::heightmap::DensityNoise::new(seed, &bundled.density);
        // Climate maps. Large period so a
        // single climate cell covers many chunks — players walk for
        // a while between biome bands instead of crossing one every
        // few steps. Independently seeded so temperature and
        // humidity drift apart and combine into all four corners of
        // the cold/warm × dry/wet square.
        let temperature_map = Fbm::<Simplex>::new(seed.wrapping_add(8) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 512.0)
            .set_persistence(0.5);
        let humidity_map = Fbm::<Simplex>::new(seed.wrapping_add(9) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 512.0)
            .set_persistence(0.5);
        // PR 4: weirdness noise — mid-frequency 2D Fbm. Used as the
        // 6th biome-lookup axis (variant biomes within the same
        // T/H/C region).
        let weirdness_noise = Fbm::<Simplex>::new(seed.wrapping_add(501) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / bundled.biomes.weirdness_period as f64)
            .set_persistence(0.5);
        // Build the biome R-tree once from the bundled entries.
        let biome_list =
            std::sync::Arc::new(climate::ParameterList::new(bundled.biomes.entries.clone()));
        // Legacy aquifer diagnostics. Fluid generation itself is handled
        // by `fluid::FluidPlanner` after terrain and caves are resolved.
        let aquifer = aquifer::AquiferSystem::new(seed, bundled.aquifer.clone());
        // PR 8: noise carvers (cheese / pillar Fbm channels) built
        // from the bundled cave config. Channel topology is fixed in
        // Rust; only tunable values re-read live.
        let noise_carvers = caves::NoiseCarvers::new(seed, &bundled.cave);
        // Carver cache: cap chosen so a chunk-fill's 11×5×11
        // neighbour query has comfortable headroom for adjacent
        // chunks' fills to reuse hot entries.
        let carver_cache = std::sync::Arc::new(std::sync::Mutex::new(lru::LruCache::new(
            std::num::NonZeroUsize::new(2048).unwrap(),
        )));
        Self {
            heightmap,
            density,
            temperature_map,
            humidity_map,
            weirdness_noise,
            biome_list,
            aquifer,
            noise_carvers,
            carver_cache,
            seed,
            fine_cache: region::fresh_fine_cache(),
            macro_cache: region::fresh_macro_cache(),
            config: config::ConfigHolder::new(bundled),
        }
    }

    // ── §2 Region / carver helpers ────────────────────────────────────────

    /// Fetch (build if missing) the carver tunnels rooted at the
    /// given chunk. Pure in `(seed, coord)`; cached so neighbour
    /// chunk fills don't rebuild it.
    fn get_carver_tunnels(&self, coord: ChunkCoord) -> std::sync::Arc<Vec<carver::CarverTunnel>> {
        let mut cache = self.carver_cache.lock().unwrap();
        if let Some(t) = cache.get(&coord) {
            return t.clone();
        }
        let t = std::sync::Arc::new(carver::build_tunnels_for_chunk(self.seed, coord));
        cache.put(coord, t.clone());
        t
    }

    /// Build the per-chunk carver mask by rasterising every tunnel
    /// from neighbour chunks within reach. Returned as a flat
    /// `CHUNK_DIM³` bool array indexed `lx + DIM·ly + DIM²·lz`.
    fn build_carver_mask(&self, coord: ChunkCoord) -> Vec<bool> {
        let dim = crate::voxel::coords::CHUNK_DIM as usize;
        let mut mask = vec![false; dim * dim * dim];
        let origin = coord.0 * crate::voxel::coords::CHUNK_DIM;
        // Carver max reach is ~130 blocks horizontally — that's 5
        // chunks at 32 each. Vertical extent is much smaller
        // (tunnels rarely drift more than ±15 blocks in Y), but
        // give a small margin.
        let r_xz: i32 = 5;
        let r_y: i32 = 2;
        for dx in -r_xz..=r_xz {
            for dy in -r_y..=r_y {
                for dz in -r_xz..=r_xz {
                    let nc = ChunkCoord(glam::IVec3::new(
                        coord.0.x + dx,
                        coord.0.y + dy,
                        coord.0.z + dz,
                    ));
                    let tunnels = self.get_carver_tunnels(nc);
                    if tunnels.is_empty() {
                        continue;
                    }
                    for tunnel in tunnels.iter() {
                        carver::rasterize_into_mask(tunnel, origin, &mut mask);
                    }
                }
            }
        }
        mask
    }

    /// Build the fine region at `coord` from noise (heightmap +
    /// hydrology). The result is byte-deterministic in
    /// `(seed, coord)`; this method is invoked at most once per
    /// region per cache lifetime (rebuilds happen on eviction).
    fn build_fine_region(&self, coord: region::RegionCoord) -> region::FineRegion {
        let mut r = region::FineRegion::empty(coord);
        r.coord = coord;
        let cfg = self.config.load();
        let terrain = terrain_ref::TerrainRef {
            heightmap: &self.heightmap,
            climate: &cfg.climate,
            density: &cfg.density,
        };
        hydrology::build_fine_hydro(
            self.seed,
            coord,
            terrain,
            &self.macro_cache,
            &self.fine_cache,
            &mut r,
        );
        caves::build_systems_for_region(self.seed, coord, terrain, &mut r, &cfg.cave);
        r
    }

    /// Pre-fetch the 3 × 3 grid of regions centered on the chunk's
    /// origin region. Used by `fill_chunk` so the per-column hot path
    /// doesn't hammer the cache mutex 9 × 1024 times.
    fn gather_chunk_regions(&self, coord: ChunkCoord) -> ChunkRegions {
        let origin = coord.origin().0;
        let center = region::RegionCoord::containing(origin.x, origin.z);
        let mut grid: [[Option<std::sync::Arc<region::FineRegion>>; 3]; 3] = Default::default();
        for dz in -1..=1i32 {
            for dx in -1..=1i32 {
                let c = region::RegionCoord {
                    x: center.x + dx,
                    z: center.z + dz,
                };
                grid[(dz + 1) as usize][(dx + 1) as usize] =
                    Some(region::get_fine(&self.fine_cache, c, || {
                        self.build_fine_region(c)
                    }));
            }
        }
        ChunkRegions { center, grid }
    }

    // ── §3 Column data (terrain + biome per (wx, wz)) ────────────────────
    // column_data and column_data_with live in columns_impl.rs.
    // Defined there via `impl Generator`; `column_data_with` is
    // `pub(super)` so sibling child modules can call it through `self`.
    //
    // See also: src/worldgen/columns_impl.rs

    // ── §4 Probe / visualizer debug methods ─────────────────────────────────
    // Moved to probe_impl.rs (probe_column, paint_column, sample_stage,
    // evaluate_density_breakdown). Defined there via `impl Generator` in the
    // probe_impl child module; Generator fields are pub(crate) so the child
    // module can access them directly.
    //
    // See also: src/worldgen/probe_impl.rs

    // ── §5 Chunk fill pipeline & lighting inputs ─────────────────────────────

    /// Return the world seed this generator was constructed with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    // fill_chunk and light_inputs_for_chunk are in fill_chunk_impl.rs.
    // Defined there via `impl Generator` in a child module;
    // Generator fields are pub(crate) so the child module accesses them
    // directly. Private helpers (gather_chunk_regions, column_data_with,
    // build_carver_mask, add_trees) are accessible because child modules can
    // see parent-module private items in Rust.
    //
    // See also: src/worldgen/fill_chunk_impl.rs

    // ── §6 Tree placement ─────────────────────────────────────────────────────
    // add_trees, tree_in_cell_with_regions, and stamp_tree are in trees_impl.rs.
    // Static helpers (Tree struct, tree_hash, try_set_air) remain in trees.rs.
    //
    // See also: src/worldgen/trees_impl.rs
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
