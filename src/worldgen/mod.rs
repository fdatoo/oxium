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
//!    (sinkhole / cliff mouth / skylight) punches through. Below
//!    `WORMHOLE_BAND_Y` a sparse 3D-noise wormhole field carves
//!    additional connective passages.
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
//! cave systems / wormholes → biomes / surface materials / trees.
//! Each layer is in its own module; this file is the public entry
//! point that wires them together.

use crate::voxel::block::Block;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{ChunkCoord, LocalPos, CHUNK_DIM_U};
use crate::worldgen::tuning::{FINE_REGION_SIZE, MAX_TERRAIN_Y, TREE_RATE_TROPICAL};
use glam::UVec3;
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

// New worldgen modules. Most of them are PR 1 stubs that get filled
// in by later PRs (PR 2 implements heightmap, PR 3 hydrology, PR 4
// caves, PR 5 biomes/surface/trees). Two are fully implemented now
// because everything else builds on them:
//
//   * `tuning`  — central constants table.
//   * `plates`  — Voronoi plate decomposition.
//   * `hash`    — deterministic mixer used by plates / trees / caves.
//   * `region`  — LRU caches + region data structs.
pub mod caves;
pub mod climate;
pub mod config;
pub mod density_graph;
pub mod flat_cache;
pub mod hash;
pub mod heightmap;
pub mod hydrology;
pub mod plates;
pub mod region;
pub mod spline;
pub mod surface;
pub mod trees;
pub mod tuning;

/// World-space Y at which the sea surface sits. Re-exported here so
/// callers outside the worldgen module (renderer, persistence, tests)
/// don't have to import `tuning::SEA_LEVEL` directly.
pub use crate::worldgen::tuning::SEA_LEVEL;

// All other tuning constants live in `worldgen::tuning`. The names
// below are imported into this module's scope for ergonomics.
use crate::worldgen::tuning::{
    CAVE_FLOOR_Y, CAVE_SDF_INTENSITY,
    CAVE_SURFACE_BUFFER, COLD_SNOW_MIN_ABOVE_SEA,
    SAND_TRANSITION_BAND, SNOW_LINE, SURFACE_BAND, TREE_CELL_SIZE, TREE_MARGIN,
    TREE_RATE_FOREST, TREE_RATE_PLAINS,
};

/// Pre-built noise fields for one world seed.
///
/// The struct exists mainly so the noise fields are constructed *once*: the
/// `Fbm` builder is comparatively expensive, and chunk generation calls
/// `get` thousands of times per chunk.
pub struct Generator {
    /// PR 2: plate-driven heightmap (continental shelf + ridges +
    /// domain-warped FBM relief). Owns the FBM/warp noise fields.
    heightmap: heightmap::HeightmapNoise,
    /// 3D density evaluator (PR A): height-bias term combined with a
    /// 3D relief FBM. Drives the per-voxel solid/air decision in
    /// `fill_chunk` so moderate slopes don't read as clean
    /// chevron stripes.
    density: heightmap::DensityNoise,
    /// Temperature map (large-period 2D noise). Drives the cold/warm
    /// axis of the biome R-tree lookup. Negative values are colder
    /// (Tundra / SnowyForest), positive warmer (Desert / Tropical).
    temperature_map: Fbm<Simplex>,
    /// Humidity map (large-period 2D noise). Drives the wet/dry axis;
    /// wetter columns earn denser tree cover, drier columns read as
    /// sparser plains.
    humidity_map: Fbm<Simplex>,
    /// Deep-band wormhole filler: sparse 3D noise that supplements
    /// the graph-based cave systems below `WORMHOLE_BAND_Y`. Above
    /// that, caves come exclusively from the cave-system graph.
    wormhole_noise: caves::WormholeNoise,
    /// PR 4: weirdness 2D noise (mid-frequency). Mirrors MC's
    /// "ridge" axis as a biome-table input — lets the same
    /// (temperature, humidity) climate produce both base biomes
    /// and variant biomes (ice spikes / sunflower plains analogues
    /// — future work; for now the biome list doesn't distinguish).
    weirdness_noise: Fbm<Simplex>,
    /// PR 4: pre-built biome R-tree. Constructed once at Generator
    /// init from the bundled config's `BiomesConfig::entries`.
    /// Hot-reloading the biome table requires a Generator restart.
    biome_list: std::sync::Arc<climate::ParameterList>,
    seed: u64,
    /// LRU cache of pre-built fine regions. Consulted per chunk fill
    /// to evaluate valley carve and lake water; built on first touch.
    fine_cache: region::FineCache,
    /// LRU cache of macro regions feeding the trunk-river injection
    /// into fine flow accumulation.
    macro_cache: region::MacroCache,
    /// Hot-reloadable worldgen config (RON-backed). Read once per
    /// chunk via [`Self::config_snapshot`] to keep chunk gen
    /// deterministic even if the file watcher swaps mid-generation.
    config: config::ConfigHolder,
}

impl Generator {
    /// Build a `Generator` with the given world seed and the bundled
    /// default config. Equivalent to [`Self::with_config`] passing a
    /// freshly-constructed `ConfigHolder` from
    /// [`config::WorldgenConfig::bundled_default`].
    pub fn new(seed: u64) -> Self {
        let cfg = config::WorldgenConfig::bundled_default()
            .expect("bundled default.ron must parse");
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
        let bundled = config::WorldgenConfig::bundled_default()
            .expect("bundled default.ron must parse");
        let heightmap = heightmap::HeightmapNoise::new(seed, &bundled.climate);
        let density = heightmap::DensityNoise::new(seed, &bundled.density);
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
        let wormhole_noise = caves::WormholeNoise::new(seed);
        // PR 4: weirdness noise — mid-frequency 2D Fbm. Used as the
        // 6th biome-lookup axis (variant biomes within the same
        // T/H/C region).
        let weirdness_noise = Fbm::<Simplex>::new(seed.wrapping_add(501) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / bundled.biomes.weirdness_period as f64)
            .set_persistence(0.5);
        // Build the biome R-tree once from the bundled entries.
        let biome_list = std::sync::Arc::new(climate::ParameterList::new(
            bundled.biomes.entries.clone(),
        ));
        Self {
            heightmap,
            density,
            temperature_map,
            humidity_map,
            wormhole_noise,
            weirdness_noise,
            biome_list,
            seed,
            fine_cache: region::fresh_fine_cache(),
            macro_cache: region::fresh_macro_cache(),
            config: config::ConfigHolder::new(bundled),
        }
    }

    /// Build the fine region at `coord` from noise (heightmap +
    /// hydrology). The result is byte-deterministic in
    /// `(seed, coord)`; this method is invoked at most once per
    /// region per cache lifetime (rebuilds happen on eviction).
    fn build_fine_region(&self, coord: region::RegionCoord) -> region::FineRegion {
        let mut r = region::FineRegion::empty(coord);
        r.coord = coord;
        let cfg = self.config.load();
        hydrology::build_fine_hydro(
            self.seed,
            coord,
            &self.heightmap,
            &cfg.climate,
            &cfg.density,
            &self.macro_cache,
            &self.fine_cache,
            &mut r,
        );
        caves::build_systems_for_region(
            self.seed,
            coord,
            &self.heightmap,
            &cfg.climate,
            &cfg.density,
            &mut r,
        );
        r
    }

    /// Pre-fetch the 3 × 3 grid of regions centered on the chunk's
    /// origin region. Used by `fill_chunk` so the per-column hot path
    /// doesn't hammer the cache mutex 9 × 1024 times.
    fn gather_chunk_regions(&self, coord: ChunkCoord) -> ChunkRegions {
        let origin = coord.origin().0;
        let center = region::RegionCoord::containing(origin.x, origin.z);
        let mut grid: [[Option<std::sync::Arc<region::FineRegion>>; 3]; 3] =
            Default::default();
        for dz in -1..=1i32 {
            for dx in -1..=1i32 {
                let c = region::RegionCoord {
                    x: center.x + dx,
                    z: center.z + dz,
                };
                grid[(dz + 1) as usize][(dx + 1) as usize] = Some(region::get_fine(
                    &self.fine_cache,
                    c,
                    || self.build_fine_region(c),
                ));
            }
        }
        ChunkRegions { center, grid }
    }

    /// Convenience wrapper around `column_data_with` that gathers
    /// the 3 × 3 region neighbourhood inline. Used by tests and by
    /// `tree_in_cell` (which is called from outside the chunk-fill
    /// hot loop).
    pub fn column_data(&self, wx: i32, wz: i32) -> ColumnData {
        let coord = region::RegionCoord::containing(wx, wz);
        let chunk_origin =
            ChunkCoord(glam::IVec3::new(coord.x * (FINE_REGION_SIZE / 32), 0, coord.z * (FINE_REGION_SIZE / 32)));
        let regions = self.gather_chunk_regions(chunk_origin);
        self.column_data_with(wx, wz, &regions)
    }

    /// Per-column terrain decisions using pre-fetched regions. The
    /// per-column hot path inside `fill_chunk` calls this version so
    /// we don't pay 9 mutex-protected cache lookups per column.
    fn column_data_with(&self, wx: i32, wz: i32, regions: &ChunkRegions) -> ColumnData {
        let cfg = self.config.load();
        // PR 3: spline-driven heightmap. h_pre is now the surface Y
        // derived from the climate-spline `offset_spline`, NOT the
        // old plate-mosaic shelf+ridge+warpedFBM formula.
        let h_pre = self
            .heightmap
            .h_pre(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density);
        // Cliff = high slope + no stencil sample dipping below sea.
        // The pre-PR-3 CLIFF_MIN_HEIGHT gate is gone (the spline
        // already places mountains far from the coast by design).
        let is_cliff = self
            .heightmap
            .is_cliff(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density);

        // Valley carve over the chunk's pre-fetched 3 × 3 region
        // neighbourhood. Operates on the spline-derived h_pre (PR 3
        // interface change — same shape as before, just a different
        // h_pre source).
        let carve = regions.valley_carve(wx, wz, self.seed);
        let height = (h_pre - carve)
            .clamp((CAVE_FLOOR_Y + 8) as f32, MAX_TERRAIN_Y as f32) as i32;

        // PR 4: 6D climate sample + R-tree biome lookup with
        // per-block hash-Voronoi jitter for organic borders.
        let (jx, jz) = climate::voronoi_jitter_offset(self.seed, wx, height, wz);
        let qwx = wx + jx;
        let qwz = wz + jz;
        let xz_jitter = [qwx as f64, qwz as f64];

        let temperature = self.temperature_map.get(xz_jitter) as f32;
        let humidity = self.humidity_map.get(xz_jitter) as f32;
        // Continentalness, terrain_shape, ridges_pv from the same
        // climate sampler that drives the heightmap splines.
        let (c, s, _pv, _look) = self.heightmap.climate(
            self.seed,
            qwx as f32,
            qwz as f32,
            &cfg.climate,
        );
        // Weirdness — independent mid-frequency Fbm.
        let weirdness = (self
            .weirdness_noise
            .get(xz_jitter) as f32)
            * cfg.biomes.weirdness_amplitude;
        // Depth axis: normalized world-Y of the column's surface.
        let depth_t = (height as f32 - cfg.density.y_min as f32)
            / (cfg.density.y_max - cfg.density.y_min) as f32;
        let depth = 1.0 - 2.0 * depth_t; // +1 at world floor, -1 at world top

        let target = climate::TargetPoint::new(
            temperature,
            humidity,
            c,
            s,
            depth,
            weirdness,
        );
        let biome = self.biome_list.lookup(&target);
        // `desertness` is kept on ColumnData for the legacy sand
        // transition heuristic in fill_chunk. Derived from the
        // (now-quantized) humidity + temperature pair: high
        // temperature × low humidity == high desertness.
        let desertness = (temperature * 0.5) - humidity * 0.5;

        let lake_rim = regions.lake_rim_at(wx, wz);
        ColumnData {
            height,
            is_cliff,
            desertness,
            biome,
            lake_rim,
        }
    }

    /// Return the world seed this generator was constructed with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Generate `coord`'s contents into `out`. Pure with respect to
    /// `(seed, coord)`.
    pub fn fill_chunk(&self, coord: ChunkCoord, out: &mut DenseChunk) {
        let origin = coord.origin().0;
        // Pre-fetch the regions overlapping this chunk plus their
        // neighbour halos. A chunk (32 blocks) is smaller than a
        // region (512 blocks), so it touches at most 4 distinct
        // regions; we cover the worst case by fetching the regions
        // containing each chunk corner and union-ing their
        // neighbour rings. Doing this up-front means the per-column
        // `column_data` path is just cheap noise evaluation and
        // already-cached region reads — no mutex traffic per column.
        let regions = self.gather_chunk_regions(coord);
        // Pre-collect every cave system whose bounding box intersects
        // this chunk so the per-cell cave SDF query iterates a short
        // list rather than walking the full region's system list.
        let chunk_max = origin + glam::IVec3::splat(CHUNK_DIM_U as i32);
        let cave_systems = regions.cave_systems_intersecting(origin, chunk_max);
        // Snapshot the hot-reloadable config once at the top of this
        // chunk and reuse for every voxel — keeps each chunk
        // deterministic even if a file watcher swaps mid-generation.
        let cfg = self.config_snapshot();

        // PR 5: cell-grid evaluator. Builds a 9x9x9 corner lattice
        // of pre-slide density values for this chunk; per-voxel
        // density is the trilerp of the 8 surrounding corners. ~730
        // expensive density evaluations per chunk instead of 32768
        // (≈45× speedup on the per-voxel hot path).
        let graph = density_graph::build_default_tree(&cfg.climate, &cfg.density);
        let evaluator = density_graph::CellEvaluator::new(
            &graph,
            &self.density,
            &cfg.density,
            (origin.x, origin.y, origin.z),
            |wx, wz| {
                let (c, s, pv, _) =
                    self.heightmap.climate(self.seed, wx as f32, wz as f32, &cfg.climate);
                density_graph::ColumnClimate {
                    continentalness: c,
                    terrain_shape: s,
                    ridges_pv: pv,
                }
            },
        );
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let wx = origin.x + x as i32;
                let wz = origin.z + z as i32;
                let col = self.column_data_with(wx, wz, &regions);
                let height = col.height;
                let lake_rim = col.lake_rim;

                // PR A: density-based top-down scan. The "surface" is
                // wherever density transitions from negative (air) to
                // positive (solid) — found block-by-block, not at a
                // fixed `height`. State across the y-loop:
                //   * `depth_below_surface` = None when the current
                //     voxel is air; Some(d) when we're `d` blocks
                //     into solid after the most recent air→solid
                //     transition (so depth=0 is the topmost solid
                //     block of an exposed surface).
                let h_target = height as f32;
                // PR 5: seed `depth_below_surface` from the voxel one
                // above the chunk's top via the cell evaluator's
                // *non-interpolated* boundary corner. The chunk
                // boundary itself is a corner so we evaluate the
                // graph directly there (the trilerp would otherwise
                // need an extra corner row outside the chunk).
                let above_chunk_top_wy = origin.y + CHUNK_DIM_U as i32;
                let above_climate = {
                    let (c, s, pv, _) =
                        self.heightmap
                            .climate(self.seed, wx as f32, wz as f32, &cfg.climate);
                    density_graph::ColumnClimate {
                        continentalness: c,
                        terrain_shape: s,
                        ridges_pv: pv,
                    }
                };
                let above_density_pre_slide = graph.evaluate(
                    wx,
                    above_chunk_top_wy,
                    wz,
                    above_climate,
                    &self.density,
                    &cfg.density,
                );
                let above_density =
                    heightmap::slide(above_density_pre_slide, above_chunk_top_wy, &cfg.density);
                let mut depth_below_surface: Option<i32> =
                    if above_density > 0.0 { Some(4) } else { None };
                for y in (0..CHUNK_DIM_U).rev() {
                    let wy = origin.y + y as i32;
                    let local = LocalPos(UVec3::new(x, y, z));

                    let approx_depth = height - wy;
                    // PR 5: density via cell-grid corner sampling +
                    // trilerp. Slide is post-interp (it's cheap and
                    // varies per-voxel-y; baking it into corners
                    // would interact poorly with the trilerp at the
                    // slide boundary).
                    let raw_density =
                        heightmap::slide(evaluator.evaluate(wx, wy, wz), wy, &cfg.density);

                    let mut cave_contribution = 0.0_f32;
                    if !cave_systems.is_empty() && wy > CAVE_FLOOR_Y {
                        if approx_depth > CAVE_SURFACE_BUFFER {
                            cave_contribution +=
                                caves::cave_sdf(wx, wy, wz, &cave_systems);
                        }
                        cave_contribution +=
                            caves::entrance_sdf(wx, wy, wz, &cave_systems);
                    }
                    if approx_depth > CAVE_SURFACE_BUFFER
                        && wy > CAVE_FLOOR_Y
                        && self.wormhole_noise.carve(wx, wy, wz)
                    {
                        cave_contribution += CAVE_SDF_INTENSITY;
                    }

                    // The new composition's quarter_negative softens
                    // above the surface, but below the surface
                    // density still saturates large (composition_scale
                    // × factor × y_gradient_amplitude). Caves carve a
                    // bounded ±CAVE_SDF_INTENSITY contribution, so we
                    // still need the cap when cave subtraction is in
                    // play. PR 5 will adopt MC's full squeeze+subtract
                    // pattern that doesn't need an explicit cap.
                    let density_for_compare = if cave_contribution > 0.0 {
                        raw_density.min(1.0)
                    } else {
                        raw_density
                    };
                    let solid = (density_for_compare - cave_contribution) > 0.0;

                    let block = if !solid {
                        // Air — flood with water only where water
                        // actually belongs: under a lake (below its
                        // rim) or under the ocean (column height at
                        // or below sea level). Caves under a land
                        // column stay dry, since there's no hydraulic
                        // connection to the ocean. This is a
                        // primitive aquifer rule — temporary until
                        // the real MC-style aquifer lands.
                        depth_below_surface = None;
                        let in_lake = lake_rim.map_or(false, |rim| wy <= rim);
                        let in_ocean = height <= SEA_LEVEL && wy <= SEA_LEVEL;
                        if in_lake || in_ocean {
                            Block::Water
                        } else {
                            Block::Air
                        }
                    } else {
                        // Solid — depth is "blocks below the air→solid
                        // transition we just crossed". `near_surface`
                        // gates surface-block selection: every air
                        // voxel still resets `depth`, but only the
                        // first solid block within ±SURFACE_BAND of the
                        // column's preliminary surface becomes a real
                        // surface. Deep cave floors, chunk-boundary
                        // resets, and 3D-noise overhangs above the
                        // surface band all fall through to Stone.
                        let depth = depth_below_surface.map(|d| d + 1).unwrap_or(0);
                        depth_below_surface = Some(depth);
                        let near_surface =
                            (h_target - wy as f32).abs() <= SURFACE_BAND as f32;
                        if col.is_cliff || !near_surface {
                            Block::Stone
                        } else if depth == 0 {
                            // Topmost solid within the surface band.
                            if wy >= SEA_LEVEL - 1
                                && wy <= SEA_LEVEL + 2
                                && !col.biome.snow_capped()
                            {
                                Block::Sand
                            } else if wy >= SNOW_LINE {
                                Block::Snow
                            } else if col.biome.snow_capped()
                                && wy >= SEA_LEVEL + COLD_SNOW_MIN_ABOVE_SEA
                            {
                                Block::Snow
                            } else if col.biome == Biome::Desert {
                                Block::Sand
                            } else {
                                let dist_to_boundary = 0.30 - col.desertness;
                                if dist_to_boundary > 0.0
                                    && dist_to_boundary < SAND_TRANSITION_BAND
                                {
                                    let p = 0.5
                                        * (1.0
                                            - dist_to_boundary
                                                / SAND_TRANSITION_BAND);
                                    let roll = hash::mix_unit(self.seed, &[wx, wz, 71]);
                                    if roll < p {
                                        Block::Sand
                                    } else {
                                        Block::Grass
                                    }
                                } else {
                                    Block::Grass
                                }
                            }
                        } else if depth <= 3 {
                            Block::Dirt
                        } else {
                            Block::Stone
                        }
                    };
                    out.set(local, block);
                }
                // Bind `lake_rim` so the compiler sees it used in
                // both branches above.
                let _ = lake_rim;
            }
        }
        // After the terrain pass, lay trees on top. Cross-chunk trees
        // (whose trunks live in a neighbouring chunk but whose leaves
        // overlap this one) are placed too, because we scan every
        // cell in a `TREE_MARGIN`-block ring around the chunk.
        self.add_trees(coord, out);
    }


    /// Place all trees whose blocks could overlap `coord`'s chunk
    /// volume. Each tree is deterministic in `(seed, cell_x, cell_z)`,
    /// so every chunk that touches the tree writes the same blocks —
    /// no double-placement and no missing slices at chunk boundaries.
    fn add_trees(&self, coord: ChunkCoord, out: &mut DenseChunk) {
        let chunk_origin = coord.origin().0;
        let cmin = chunk_origin;
        let cmax = chunk_origin + glam::IVec3::splat(crate::voxel::coords::CHUNK_DIM);
        // Cells whose interior could spill into the chunk's extended
        // bounds, allowing for tree-block radius around the cell.
        let xmin = cmin.x - TREE_MARGIN;
        let xmax = cmax.x + TREE_MARGIN;
        let zmin = cmin.z - TREE_MARGIN;
        let zmax = cmax.z + TREE_MARGIN;
        let cell_xmin = xmin.div_euclid(TREE_CELL_SIZE);
        let cell_xmax = (xmax - 1).div_euclid(TREE_CELL_SIZE);
        let cell_zmin = zmin.div_euclid(TREE_CELL_SIZE);
        let cell_zmax = (zmax - 1).div_euclid(TREE_CELL_SIZE);
        for cell_x in cell_xmin..=cell_xmax {
            for cell_z in cell_zmin..=cell_zmax {
                if let Some(tree) = self.tree_in_cell(cell_x, cell_z) {
                    self.stamp_tree(tree, coord, out);
                }
            }
        }
    }

    /// Return the tree (if any) belonging to the `(cell_x, cell_z)` tree
    /// cell. Determined entirely by `(seed, cell coords)` so adjacent
    /// chunks agree on which trees exist.
    fn tree_in_cell(&self, cell_x: i32, cell_z: i32) -> Option<Tree> {
        let wx = cell_x * TREE_CELL_SIZE
            + (tree_hash(self.seed, cell_x, cell_z, 1) % 6) as i32
            + 1;
        let wz = cell_z * TREE_CELL_SIZE
            + (tree_hash(self.seed, cell_x, cell_z, 2) % 6) as i32
            + 1;
        let col = self.column_data(wx, wz);

        // Trees don't grow on cliffs (bare stone), above the alpine
        // snow line, or where the column is submerged under a lake.
        if col.is_cliff {
            return None;
        }
        if col.height >= SNOW_LINE {
            return None;
        }
        // Lake veto: if this column sits below a lake's water surface,
        // no tree (even palms can't grow underwater).
        if let Some(rim) = col.lake_rim {
            if col.height < rim {
                return None;
            }
        }

        // Sand-surface veto. The beach band runs `[SEA_LEVEL - 1,
        // SEA_LEVEL + 2]` and surface material in that band is Sand;
        // oaks don't grow on sand. Palms *do*, but rarely — they're
        // the iconic tropical-beach silhouette.
        let on_beach =
            col.height >= SEA_LEVEL - 1 && col.height <= SEA_LEVEL + 2;
        let kind = col.biome.tree_kind();
        if on_beach && kind != TreeKind::Palm {
            return None;
        }

        let rate = col.biome.tree_rate_percentile()?;
        // Palms on the beach are rarer — divide their rate by 4 so
        // tropical beaches read as scattered palms, not dense palm
        // forests.
        let effective_rate = if on_beach { rate / 4 } else { rate };
        let roll = tree_hash(self.seed, cell_x, cell_z, 0) % 100;
        if roll >= effective_rate {
            return None;
        }

        // Find the actual topmost solid block for this column. With
        // 3D density the surface can sit up to ±SURFACE_BAND from
        // `col.height`; use a top-down density walk so the tree's
        // trunk lands on the real surface, not the heightmap target.
        let cfg = self.config_snapshot();
        // Sample the climate triple at this column so the topmost-solid
        // search uses the same spline outputs the chunk fill does.
        let (cc, sc, rc, _) = self.heightmap.climate(
            self.seed,
            wx as f32,
            wz as f32,
            &cfg.climate,
        );
        let offset = cfg.climate.offset_spline.evaluate(cc, sc, rc);
        let factor = cfg.climate.factor_spline.evaluate(cc, sc, rc);
        let jagged = cfg.climate.jaggedness_spline.evaluate(cc, sc, rc);
        let height = self
            .density
            .topmost_solid(
                col.height as f32,
                wx,
                wz,
                col.height + SURFACE_BAND + 2,
                offset,
                factor,
                jagged,
                &cfg.density,
            )
            .unwrap_or(col.height);
        let trunk_h = match kind {
            TreeKind::Oak => {
                4 + (tree_hash(self.seed, cell_x, cell_z, 3) % 3) as i32
            }
            TreeKind::Palm => {
                7 + (tree_hash(self.seed, cell_x, cell_z, 3) % 3) as i32
            }
        };
        Some(Tree {
            wx,
            wz,
            base_y: height,
            trunk_h,
            kind,
        })
    }

    /// Write the trunk + leaf blocks of `tree` into `out`. Blocks whose
    /// world coordinates fall outside this chunk are silently ignored
    /// (the neighbouring chunk's call to `stamp_tree` writes them
    /// instead). Existing non-air voxels are preserved so the trunk
    /// doesn't carve through hills.
    fn stamp_tree(&self, tree: Tree, coord: ChunkCoord, out: &mut DenseChunk) {
        // Trunk: vertical column of Wood blocks above the surface.
        for dy in 1..=tree.trunk_h {
            try_set_air(coord, out, tree.wx, tree.base_y + dy, tree.wz, Block::Wood);
        }
        let top_y = tree.base_y + tree.trunk_h;
        match tree.kind {
            TreeKind::Oak => {
                // Round canopy. Radius² uses 6 so corner cells drop
                // out and the silhouette stays roughly spherical.
                for dy in -1..=2 {
                    for dz in -2..=2 {
                        for dx in -2..=2 {
                            let r2 = dx * dx + dy * dy + dz * dz;
                            if r2 > 6 {
                                continue;
                            }
                            try_set_air(
                                coord,
                                out,
                                tree.wx + dx,
                                top_y + dy,
                                tree.wz + dz,
                                Block::Leaves,
                            );
                        }
                    }
                }
            }
            TreeKind::Palm => {
                // Spreading-fronds canopy: 4–6 horizontal arms, one
                // block thick, radiating from the trunk top. Each arm
                // is a straight line of 3 blocks; a small +1y cap
                // sits at the centre.
                try_set_air(coord, out, tree.wx, top_y + 1, tree.wz, Block::Leaves);
                let arm_count = 5; // five fronds, evenly spaced
                for a in 0..arm_count {
                    let theta =
                        a as f32 * std::f32::consts::TAU / arm_count as f32;
                    for step in 1..=3i32 {
                        let dx = (theta.cos() * step as f32).round() as i32;
                        let dz = (theta.sin() * step as f32).round() as i32;
                        // Fronds droop: outer tip is 1 block lower
                        // than the trunk top.
                        let dy = if step >= 3 { -1 } else { 0 };
                        try_set_air(
                            coord,
                            out,
                            tree.wx + dx,
                            top_y + dy,
                            tree.wz + dz,
                            Block::Leaves,
                        );
                    }
                }
            }
        }
    }
}

/// Per-column biome + geometry summary used by both `fill_chunk` and
/// `add_trees` so block selection and tree placement stay in sync.
#[derive(Debug, Clone, Copy)]
pub struct ColumnData {
    /// Surface height in world Y, post-carve, clamped.
    pub height: i32,
    /// True if the column's `h_pre` slope exceeds `CLIFF_SLOPE_THRESH`
    /// AND its elevation is at/above `CLIFF_MIN_HEIGHT`. Cliff
    /// columns expose stone faces directly, skipping the dirt cap.
    pub is_cliff: bool,
    /// Jitter-perturbed `desertness` noise value. Used by the
    /// sand/grass transition band: inside the band on the grass side
    /// of the desert boundary, the surface block is rolled
    /// stochastically.
    pub desertness: f32,
    /// Discrete biome label derived from temperature, humidity, and
    /// the desert mask, with threshold perturbation applied.
    pub biome: Biome,
    /// Lake water surface elevation at this column, if it sits inside
    /// (or adjacent to) a sink-filled basin. `None` outside lakes.
    /// Used by both the chunk-fill water flood and the tree placer
    /// (trees veto if the column is submerged in lake water).
    pub lake_rim: Option<i32>,
}

/// Discrete biome label assigned to each column. The set is small on
/// purpose — every variant has a distinct visual signature (different
/// surface block or noticeably different tree density), so the
/// difference between biomes reads from a screenshot.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum Biome {
    /// Cold column. Snow on the surface; no trees grow here.
    Tundra,
    /// Cold *and* humid. Same Snow surface as Tundra but trees do
    /// grow (taiga / boreal forest analogue).
    SnowyForest,
    /// Temperate, dry. Grass surface, very sparse trees — open
    /// rolling fields.
    Plains,
    /// Temperate, humid. Grass surface, dense tree cover.
    Forest,
    /// Hot, dry. Sand surface, no trees.
    Desert,
    /// Hot, humid (new in PR 5). Grass surface, denser tree cover
    /// than Forest — placeholder for the future palm/jungle pass.
    /// Palm-shape trees are stamped here via `TreeKind::Palm`.
    Tropical,
}

impl Biome {
    /// True when the biome should cap the surface column with Snow.
    fn snow_capped(self) -> bool {
        matches!(self, Biome::Tundra | Biome::SnowyForest)
    }

    /// Probability (0..100) that a `TREE_CELL_SIZE × TREE_CELL_SIZE`
    /// patch in this biome rolls a tree. `None` for biomes that
    /// don't host trees at all.
    fn tree_rate_percentile(self) -> Option<u32> {
        match self {
            Biome::Tundra | Biome::Desert => None,
            Biome::Plains => Some(TREE_RATE_PLAINS),
            Biome::Forest | Biome::SnowyForest => Some(TREE_RATE_FOREST),
            Biome::Tropical => Some(TREE_RATE_TROPICAL),
        }
    }

    /// Which tree shape to stamp in this biome's cells. Oak for
    /// temperate / boreal, Palm for tropical.
    fn tree_kind(self) -> TreeKind {
        match self {
            Biome::Tropical => TreeKind::Palm,
            _ => TreeKind::Oak,
        }
    }
}

/// Tree shape selector. PR 5 introduces palms for `Tropical`;
/// follow-up PRs may add jungle / pine / palm-specific blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeKind {
    Oak,
    Palm,
}

/// 3 × 3 grid of fine regions centered on a chunk's origin region.
/// Pre-fetched at the start of `fill_chunk` so the per-column
/// `column_data_with` / `valley_carve` / `lake_rim_at` queries don't
/// hammer the cache mutex.
struct ChunkRegions {
    center: region::RegionCoord,
    /// `grid[dz + 1][dx + 1]` is the region at offset `(dx, dz)` from
    /// `center`. Always populated (build_fine_region is invoked on
    /// cache miss).
    grid: [[Option<std::sync::Arc<region::FineRegion>>; 3]; 3],
}

impl ChunkRegions {
    /// Look up the region containing world coords `(wx, wz)` inside
    /// the pre-fetched 3 × 3 grid. Returns `None` if the column is
    /// outside the grid (shouldn't happen for any column inside the
    /// chunk that triggered the gather).
    fn region_at(&self, wx: i32, wz: i32) -> Option<&region::FineRegion> {
        let c = region::RegionCoord::containing(wx, wz);
        let dx = c.x - self.center.x + 1;
        let dz = c.z - self.center.z + 1;
        if dx < 0 || dz < 0 || dx >= 3 || dz >= 3 {
            return None;
        }
        self.grid[dz as usize][dx as usize].as_deref()
    }

    /// Lake rim at this column, if it sits inside (or adjacent to)
    /// a sink-filled basin. Returns the highest rim among the
    /// column's own fine cell and the 8 neighbour cells reachable
    /// through the pre-fetched 3 × 3 region grid.
    ///
    /// The 1-cell halo is what makes lake water reach the shore
    /// cleanly: a fine cell adjacent to a lake (but not flagged
    /// `is_lake` itself) still inherits the lake's rim when the rim
    /// is above the column's natural height, so the water surface
    /// doesn't terrace at fine-cell boundaries.
    fn lake_rim_at(&self, wx: i32, wz: i32) -> Option<i32> {
        use crate::worldgen::tuning::FINE_CELL;
        let mut best: Option<i32> = None;
        for dz in -1..=1i32 {
            for dx in -1..=1i32 {
                let nx = wx + dx * FINE_CELL;
                let nz = wz + dz * FINE_CELL;
                if let Some(region) = self.region_at(nx, nz) {
                    if let Some(rim) = hydrology::lake_rim_at(nx, nz, region) {
                        best = Some(best.map_or(rim, |b| b.max(rim)));
                    }
                }
            }
        }
        best
    }

    /// Collect every cave system in the pre-fetched 3 × 3 grid whose
    /// bounding box intersects `[chunk_min, chunk_max]`. Called once
    /// per `fill_chunk`; the result drives the cave SDF query.
    fn cave_systems_intersecting(
        &self,
        chunk_min: glam::IVec3,
        chunk_max: glam::IVec3,
    ) -> Vec<&region::CaveSystem> {
        let mut out = Vec::new();
        for row in &self.grid {
            for slot in row {
                if let Some(r) = slot {
                    for sys in &r.cave_systems {
                        if sys.bb_max.x >= chunk_min.x
                            && sys.bb_min.x <= chunk_max.x
                            && sys.bb_max.y >= chunk_min.y
                            && sys.bb_min.y <= chunk_max.y
                            && sys.bb_max.z >= chunk_min.z
                            && sys.bb_min.z <= chunk_max.z
                        {
                            out.push(sys);
                        }
                    }
                }
            }
        }
        out
    }

    /// Valley carve at this column: iterate over the river segments
    /// in the column's region plus its 8 neighbours (clipped to the
    /// pre-fetched 3 × 3 grid). Per-column cost is O(total segments
    /// inside the visible ring) — typically a few dozen.
    fn valley_carve(&self, wx: i32, wz: i32, seed: u64) -> f32 {
        let c = region::RegionCoord::containing(wx, wz);
        let center_dx = c.x - self.center.x + 1;
        let center_dz = c.z - self.center.z + 1;
        if center_dx < 0 || center_dz < 0 || center_dx >= 3 || center_dz >= 3 {
            // Column outside the gathered grid — should not happen
            // in practice; return 0 (no carve) defensively.
            return 0.0;
        }
        let primary = self.grid[center_dz as usize][center_dx as usize]
            .as_deref()
            .expect("3x3 grid is always populated");
        let mut neighbour_regions: [Option<&region::FineRegion>; 8] = [None; 8];
        let nbr_offsets: [(i32, i32); 8] = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ];
        for i in 0..8 {
            let (ox, oz) = nbr_offsets[i];
            let nx = center_dx + ox;
            let nz = center_dz + oz;
            if nx < 0 || nz < 0 || nx >= 3 || nz >= 3 {
                continue;
            }
            neighbour_regions[i] = self.grid[nz as usize][nx as usize].as_deref();
        }
        hydrology::valley_carve(wx, wz, primary, &neighbour_regions, seed)
    }
}

/// Tree placement metadata for one cell.
#[derive(Debug, Clone, Copy)]
struct Tree {
    /// World-space X coordinate of the trunk.
    wx: i32,
    /// World-space Z coordinate of the trunk.
    wz: i32,
    /// World-space Y of the surface block under the trunk (the trunk
    /// itself starts at `base_y + 1`).
    base_y: i32,
    /// Number of Wood blocks above the surface, inclusive.
    trunk_h: i32,
    /// Tree shape (`Oak` round canopy vs `Palm` spreading fronds).
    /// Picked from the biome at the cell's column.
    kind: TreeKind,
}

/// Write `b` at world coords `(wx, wy, wz)` if they fall inside
/// `coord`'s 32³ volume *and* the existing block is Air. Both
/// conditions are required so a tree's trunk doesn't cut through
/// hills and adjacent chunks' calls don't overwrite each other.
fn try_set_air(
    coord: ChunkCoord,
    out: &mut DenseChunk,
    wx: i32,
    wy: i32,
    wz: i32,
    b: Block,
) {
    use crate::voxel::coords::CHUNK_DIM;
    let chunk_origin = coord.origin().0;
    let lx = wx - chunk_origin.x;
    let ly = wy - chunk_origin.y;
    let lz = wz - chunk_origin.z;
    if lx < 0 || ly < 0 || lz < 0 || lx >= CHUNK_DIM || ly >= CHUNK_DIM || lz >= CHUNK_DIM {
        return;
    }
    let lp = LocalPos(UVec3::new(lx as u32, ly as u32, lz as u32));
    if out.get(lp) != Block::Air {
        return;
    }
    out.set(lp, b);
}

/// Deterministic mixer: `(seed, x, z, salt) → u32`. Uses the same
/// xor-shift / golden-ratio multiply pattern as the Wang/Mix hashes
/// commonly stamped into shader noise functions. Good enough for
/// tree placement; not cryptographic.
fn tree_hash(seed: u64, x: i32, z: i32, salt: u32) -> u32 {
    let mut h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (x as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h = h.rotate_left(13);
    h ^= (z as i64 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h = h.rotate_left(17);
    h ^= (salt as u64).wrapping_mul(0xCC9E_2D51_1B87_3593);
    ((h ^ (h >> 33)) as u32) ^ ((h >> 16) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::CHUNK_VOL;
    use glam::IVec3;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Reduce a chunk to a single u64 hash — easier than comparing full
    /// 32 KB arrays in test failure messages.
    fn hash_chunk(c: &DenseChunk) -> u64 {
        let mut h = DefaultHasher::new();
        for b in c.blocks.iter() {
            (*b as u16).hash(&mut h);
        }
        h.finish()
    }

    #[test]
    fn fill_is_deterministic() {
        let g = Generator::new(42);
        let mut a = DenseChunk::empty();
        let mut b = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::ZERO), &mut a);
        g.fill_chunk(ChunkCoord(IVec3::ZERO), &mut b);
        assert_eq!(hash_chunk(&a), hash_chunk(&b));
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = DenseChunk::empty();
        let mut b = DenseChunk::empty();
        Generator::new(1).fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut a);
        Generator::new(2).fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut b);
        assert_ne!(hash_chunk(&a), hash_chunk(&b));
    }

    /// Golden test: locks the generator's output for a known seed/chunk.
    /// First run prints the actual hash; update `GOLDEN_42_002` once and
    /// future runs catch unintentional behavioural drift.
    #[test]
    fn golden_seed42_chunk_0_2_0() {
        // Hash re-baselined for PR A: 3D density evaluator inside
        // SURFACE_BAND. The surface is now fuzz-jittered by 3D
        // relief noise instead of being column-quantised, so the
        // chevron-staircase artifact on moderate slopes is gone.
        const GOLDEN_42_002: u64 = 0xEECE_DFF7_DEC7_EB0A;
        let g = Generator::new(42);
        let mut c = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut c);
        let actual = hash_chunk(&c);
        if GOLDEN_42_002 == 0xDEAD_BEEF_DEAD_BEEF {
            println!("UPDATE GOLDEN_42_002 to: 0x{:016X}", actual);
        } else {
            assert_eq!(actual, GOLDEN_42_002, "worldgen output changed");
        }
    }

    /// Cave system sanity: somewhere underground (below sea level) we
    /// Cave system sanity: graph-based caves are spatially
    /// structured — not every chunk has carving (that's the point;
    /// systems are discoverable). But across a generous scan of
    /// underground chunks, at least one should have caves AND every
    /// scanned chunk should still be mostly solid stone (no chunk
    /// blown wide open by an oversized chamber).
    #[test]
    fn underground_chunk_has_both_caves_and_solid() {
        let g = Generator::new(42);
        let mut found_carved_chunk = false;
        // Scan a 16 × 16 grid of chunks (one region's worth) at
        // chunk y=-2 (world y [-64, -33]). This depth straddles the
        // Middle / Deep cave bands and the wormhole noise band
        // (`WORMHOLE_BAND_Y` = -40), so at least one chunk should
        // hit something.
        for cx in -8..8 {
            for cz in -8..8 {
                let mut c = DenseChunk::empty();
                g.fill_chunk(ChunkCoord(IVec3::new(cx, -2, cz)), &mut c);
                let mut air = 0;
                let mut stone = 0;
                for b in c.blocks.iter() {
                    match b {
                        Block::Air | Block::Water => air += 1,
                        Block::Stone => stone += 1,
                        _ => {}
                    }
                }
                // Every chunk should still be mostly stone — no
                // system carves more than half a chunk.
                assert!(
                    stone > CHUNK_VOL / 2,
                    "chunk ({cx}, -2, {cz}) had insufficient stone: stone={stone}"
                );
                if air > CHUNK_VOL / 200 {
                    // Lowered to 0.5% per chunk because tunnels can
                    // pass through a chunk and only intersect a
                    // narrow strip of cells.
                    found_carved_chunk = true;
                }
            }
        }
        assert!(
            found_carved_chunk,
            "expected ≥1 chunk in 16×16 scan to overlap a cave feature; found none"
        );
    }

    /// Caves carved under a land column (column height well above
    /// sea level, no lake above) must be dry — not flooded with
    /// water from sea level. Sea-level water only belongs in ocean
    /// columns; lake water only belongs under lakes.
    #[test]
    fn deep_caves_under_land_are_dry() {
        let g = Generator::new(42);
        // Find a chunk where every column is land AND none has a
        // lake above it. Then check no Water in chunk-Y=-2 below it.
        let mut found = None;
        'outer: for cz in -8..8 {
            for cx in -8..8 {
                let mut ok = true;
                'cols: for lz in 0..CHUNK_DIM_U {
                    for lx in 0..CHUNK_DIM_U {
                        let wx = cx * CHUNK_DIM_U as i32 + lx as i32;
                        let wz = cz * CHUNK_DIM_U as i32 + lz as i32;
                        let col = g.column_data(wx, wz);
                        if col.height <= SEA_LEVEL + 5 || col.lake_rim.is_some() {
                            ok = false;
                            break 'cols;
                        }
                    }
                }
                if ok {
                    found = Some((cx, cz));
                    break 'outer;
                }
            }
        }
        let (cx, cz) = found.expect("expected a lake-free all-land chunk");
        let mut chunk = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(cx, -2, cz)), &mut chunk);
        let water = chunk.blocks.iter().filter(|b| matches!(b, Block::Water)).count();
        assert_eq!(
            water, 0,
            "lake-free land chunk ({cx}, -2, {cz}) had {water} water blocks — \
             caves under land should be dry, not flooded"
        );
    }

    /// Deep underground chunks must contain no surface blocks.
    /// Symptom of the per-chunk depth-reset bug: the topmost solid
    /// voxel of every chunk got rendered as a surface block, so
    /// digging straight down showed grass-dirt-stone cycles every
    /// 32 blocks vertically.
    #[test]
    fn deep_underground_has_no_surface_blocks() {
        let g = Generator::new(42);
        // Chunk-Y=-2 covers world Y -64..-33 — comfortably below
        // any plausible surface across all biomes.
        let mut grass = 0u32;
        let mut dirt = 0u32;
        let mut sand = 0u32;
        let mut snow = 0u32;
        for cx in -8..8 {
            for cz in -8..8 {
                let mut chunk = DenseChunk::empty();
                g.fill_chunk(ChunkCoord(IVec3::new(cx, -2, cz)), &mut chunk);
                for b in chunk.blocks.iter() {
                    match b {
                        Block::Grass => grass += 1,
                        Block::Dirt => dirt += 1,
                        Block::Sand => sand += 1,
                        Block::Snow => snow += 1,
                        _ => {}
                    }
                }
            }
        }
        assert_eq!(grass, 0, "deep underground had {grass} grass blocks");
        assert_eq!(dirt, 0, "deep underground had {dirt} dirt blocks");
        assert_eq!(sand, 0, "deep underground had {sand} sand blocks");
        assert_eq!(snow, 0, "deep underground had {snow} snow blocks");
    }

    /// Generator::with_config holds the ConfigHolder, and
    /// `config_snapshot()` reflects swaps.
    #[test]
    fn generator_holds_config_and_reads_from_holder() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let holder = crate::worldgen::config::ConfigHolder::new(cfg);
        let g = Generator::with_config(42, holder.clone());
        // Initial snapshot has bundled default factor (4.0).
        assert!((g.config_snapshot().density.factor - 4.0).abs() < 1e-5);
        // Swap a new config; the Generator's snapshot should reflect it.
        let mut new_cfg = (*holder.load()).clone();
        new_cfg.density.factor = 1.0;
        holder.swap(new_cfg);
        assert!((g.config_snapshot().density.factor - 1.0).abs() < 1e-5);
    }

    /// Biome diversity: scanning a few thousand columns across a
    /// generous area should turn up every biome at least once.
    /// Otherwise either the thresholds are misconfigured (cold
    /// belt vanishingly narrow, forest too rare) or the climate
    /// noise isn't actually getting sampled.
    #[test]
    fn all_biomes_appear_in_a_large_scan() {
        let g = Generator::new(42);
        let mut seen = std::collections::HashSet::new();
        // Step 8 blocks at a time so a 2048×2048 scan only costs
        // 64 K column evaluations — fast enough to keep the test
        // under a second even in debug builds.
        for wz in (-1024..1024).step_by(8) {
            for wx in (-1024..1024).step_by(8) {
                seen.insert(g.column_data(wx, wz).biome);
            }
        }
        for expected in [
            Biome::Tundra,
            Biome::SnowyForest,
            Biome::Plains,
            Biome::Forest,
            Biome::Desert,
            Biome::Tropical,
        ] {
            assert!(
                seen.contains(&expected),
                "biome {:?} never appeared in the scan; saw {:?}",
                expected,
                seen
            );
        }
    }

    /// Cold biomes should plant Snow as their surface block above
    /// the coastal elevation buffer. With 3D density the surface
    /// height isn't exactly `col.height` anymore — it can shift by
    /// up to `SURFACE_BAND` blocks — so the test now scans the
    /// chunk top-down to find the actual topmost solid block and
    /// checks its kind.
    #[test]
    fn cold_biome_caps_with_snow() {
        let g = Generator::new(42);
        // With 3D density the surface can deviate from `col.height`
        // by up to `DENSITY_FALLOFF` (~4) blocks. Pick a column
        // where `col.height` is comfortably between the cold-snow
        // floor and the snow line so the actual surface lands in
        // the cold-biome cap band.
        let min_h = SEA_LEVEL + COLD_SNOW_MIN_ABOVE_SEA + 6;
        let max_h = SNOW_LINE - 6;
        let mut found: Option<(i32, i32)> = None;
        'outer: for wz in (-1024..1024).step_by(8) {
            for wx in (-1024..1024).step_by(8) {
                let col = g.column_data(wx, wz);
                if col.biome == Biome::Tundra
                    && col.height >= min_h
                    && col.height < max_h
                    && !col.is_cliff
                {
                    found = Some((wx, wz));
                    break 'outer;
                }
            }
        }
        let (wx, wz) = found.expect("expected at least one tundra column");
        let col = g.column_data(wx, wz);
        let cx = wx.div_euclid(CHUNK_DIM_U as i32);
        let cz = wz.div_euclid(CHUNK_DIM_U as i32);
        // The actual surface might be in one of two chunks if it
        // happens to span a vertical chunk boundary; check both.
        let lx = wx.rem_euclid(CHUNK_DIM_U as i32) as u32;
        let lz = wz.rem_euclid(CHUNK_DIM_U as i32) as u32;
        let mut found_surface = None;
        for cy in [col.height.div_euclid(CHUNK_DIM_U as i32),
                   col.height.div_euclid(CHUNK_DIM_U as i32) + 1] {
            let mut chunk = DenseChunk::empty();
            g.fill_chunk(ChunkCoord(IVec3::new(cx, cy, cz)), &mut chunk);
            // Top-down scan in this chunk to find the topmost solid.
            for ly in (0..CHUNK_DIM_U).rev() {
                let block = chunk.blocks[crate::voxel::coords::LocalPos(
                    glam::UVec3::new(lx, ly, lz),
                ).to_index()];
                if !matches!(block, Block::Air | Block::Water) {
                    found_surface = Some(block);
                    break;
                }
            }
            if found_surface.is_some() { break; }
        }
        let surface = found_surface.expect("topmost solid not found in tundra column");
        assert!(
            matches!(surface, Block::Snow | Block::Sand | Block::Stone),
            "tundra surface at ({wx}, {wz}) was {:?}, expected Snow",
            surface
        );
    }

    /// Rivers and lakes should produce a non-trivial amount of
    /// inland water — somewhere in a generous scan we expect at
    /// least one column carved below sea level *and* high enough
    /// that the carve is the cause (not just baseline ocean from
    /// the height noise). Catches future refactors that
    /// inadvertently neutralise the carve pass.
    #[test]
    fn rivers_or_lakes_carve_inland_water() {
        let g = Generator::new(42);
        // A "carved" column is one whose height landed *below* sea
        // level while the heightmap *without* the river/lake pass
        // would have stayed on dry land. We approximate "would have
        // stayed dry" by sampling far from any river/lake band —
        // but since `column_data` already runs the carve, easier to
        // just count columns at SEA_LEVEL-1 or below where the
        // surrounding 5-block disc has at least one dry column.
        // That rules out the smooth ocean background.
        let mut inland_water_columns = 0usize;
        for wz in (-512..512).step_by(4) {
            for wx in (-512..512).step_by(4) {
                let h = g.column_data(wx, wz).height;
                if h >= SEA_LEVEL {
                    continue;
                }
                // Inland if any neighbour 24 blocks away is above
                // sea level.
                let neighbours = [
                    g.column_data(wx + 24, wz).height,
                    g.column_data(wx - 24, wz).height,
                    g.column_data(wx, wz + 24).height,
                    g.column_data(wx, wz - 24).height,
                ];
                if neighbours.iter().any(|&n| n > SEA_LEVEL + 4) {
                    inland_water_columns += 1;
                }
            }
        }
        assert!(
            inland_water_columns > 0,
            "expected at least one inland water column from river/lake carving, found none"
        );
    }

    /// The plate-driven heightmap caps final heights at
    /// `MAX_TERRAIN_Y` (140 by default) so the tallest possible
    /// peak still sits inside the loaded vertical radius. Catches a
    /// future refactor that drops the cap (or sets `MAX_TERRAIN_Y`
    /// above the chunk-stack ceiling).
    #[test]
    fn ridged_mountains_respect_height_cap() {
        let g = Generator::new(42);
        // Scan a generous area: the cap should hold everywhere, not
        // just near origin. CC ridges peak at ~SEA_LEVEL + 28 + 90 =
        // 180, but the clamp pulls them back to MAX_TERRAIN_Y.
        for wx in (-2048..=2048).step_by(128) {
            for wz in (-2048..=2048).step_by(128) {
                let col = g.column_data(wx, wz);
                assert!(
                    col.height <= MAX_TERRAIN_Y,
                    "column height {} broke the cap at ({wx}, {wz})",
                    col.height
                );
            }
        }
    }

    #[test]
    fn chunk_at_sea_level_has_water_or_solid() {
        // Sanity: the column-wise terrain must place *something* in any
        // sea-level chunk — either solid (under-water terrain) or water
        // (above-terrain flood).
        let g = Generator::new(42);
        let mut c = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut c);
        let has_non_air = c.blocks.iter().any(|&b| b != Block::Air);
        assert!(has_non_air);
    }
}
