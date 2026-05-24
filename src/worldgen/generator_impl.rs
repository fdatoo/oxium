//! [`Generator`] construction and per-chunk infrastructure helpers.
//!
//! Lives in a separate file from `mod.rs` so the struct definition,
//! re-exports, and module declarations can be read without scrolling past
//! ~200 lines of constructor noise-field wiring.
//!
//! ## What lives here
//!
//! - **Public API** — `new`, `with_config`, `config_snapshot`, `seed`.
//! - **Region helpers** — `build_fine_region` and `gather_chunk_regions`:
//!   the two functions that populate the per-chunk 3×3 `ChunkRegions` grid.
//! - **Carver helpers** — `get_carver_tunnels` + `build_carver_mask`: the
//!   MC-style sphere-capsule worm tunnel pre-pass (separate from the graph
//!   cave system).
//!
//! All helpers are `fn` or `pub(crate) fn` on `Generator`. They are private
//! to the module tree but callable from every sibling child module
//! (`fill_chunk_impl`, `trees_impl`, `probe_impl`, `columns_impl`) because
//! child modules can see parent-module private items.
//!
//! See `docs/book/content/part-4-chunk-fill/4.1-generator-construction.mdx`.

use super::Generator;
use super::pipeline::ChunkRegions;
use crate::voxel::coords::{CHUNK_DIM, ChunkCoord};
use crate::worldgen::{
    aquifer, carver, caves, climate, config, density, hydrology, region, terrain_ref,
};
use glam::IVec3;
use noise::{Fbm, MultiFractal, Simplex};

impl Generator {
    // ── Public API ────────────────────────────────────────────────────────

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

    /// Cheap atomic read of the current config snapshot.
    ///
    /// Returns an `Arc<WorldgenConfig>` — call once per chunk and reuse
    /// across the chunk's lifetime to avoid mid-chunk drift if a
    /// hot-reload races chunk generation.
    pub fn config_snapshot(&self) -> std::sync::Arc<config::WorldgenConfig> {
        self.config.load()
    }

    /// Return the world seed this generator was constructed with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    // ── Internal construction ─────────────────────────────────────────────

    /// Internal constructor. Builds all noise fields but leaves `config`
    /// set to the bundled default; [`Self::with_config`] overwrites it.
    ///
    /// Each noise field is seeded with a different per-axis salt so they
    /// don't produce correlated patterns (mountain-noise lining up with
    /// height-noise would just amplify existing hills instead of adding
    /// new geographic features).
    fn new_internal(seed: u64) -> Self {
        // Load the bundled default once so all noise fields share a
        // consistent initial config. Hot-reloading can later swap knob
        // values, but noise *frequencies* are baked at construction —
        // changing `terrain_shape_period` requires a generator restart.
        let bundled =
            config::WorldgenConfig::bundled_default().expect("bundled default.ron must parse");
        let heightmap = density::heightmap::HeightmapNoise::new(seed, &bundled.climate);
        let density = density::density_3d::DensityNoise::new(seed, &bundled.density);
        // Climate maps. Large period (~512 blocks) so biome bands are wide
        // enough that players walk for a while between them. Independently
        // seeded so temperature and humidity drift apart and span all four
        // corners of the cold/warm × dry/wet square.
        let temperature_map = Fbm::<Simplex>::new(seed.wrapping_add(8) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 512.0)
            .set_persistence(0.5);
        let humidity_map = Fbm::<Simplex>::new(seed.wrapping_add(9) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 512.0)
            .set_persistence(0.5);
        // Weirdness noise — mid-frequency 2D FBM. The 6th biome-lookup axis
        // for variant biomes within the same (T, H, C) region (ice spikes,
        // sunflower plains analogues). Period comes from config so it can
        // be tuned without a code change.
        let weirdness_noise = Fbm::<Simplex>::new(seed.wrapping_add(501) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / bundled.biomes.weirdness_period as f64)
            .set_persistence(0.5);
        // Build the biome R-tree once from the bundled entries. Hot-reloading
        // the biome table requires a Generator restart (the R-tree is immutable).
        let biome_list =
            std::sync::Arc::new(climate::ParameterList::new(bundled.biomes.entries.clone()));
        // Legacy aquifer diagnostics. Fluid generation itself is handled by
        // `fluid::FluidPlanner` after terrain and caves are resolved.
        let aquifer = aquifer::AquiferSystem::new(seed, bundled.aquifer.clone());
        // Noise carvers (cheese / pillar FBM channels). Channel topology is
        // fixed in Rust; per-chunk evaluations re-read the live config for
        // tunable amplitude values.
        let noise_carvers = caves::NoiseCarvers::new(seed, &bundled.cave);
        // Carver cache cap: a chunk-fill's 11×5×11 neighbour query (for
        // rasterising carver tunnels from surrounding chunks) generates up to
        // 605 distinct chunk keys. 2048 entries gives ample headroom so
        // adjacent fill calls share hot entries without evicting each other.
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

    // ── Region helpers ────────────────────────────────────────────────────

    /// Build the fine region at `coord` from noise (heightmap + hydrology +
    /// cave systems). Pure in `(seed, coord)` — the region cache calls this
    /// at most once per region per cache lifetime; rebuilds happen on eviction.
    ///
    /// Steps:
    /// 1. Initialise an empty `FineRegion`.
    /// 2. `build_fine_hydro` — sink-fill + D8 flow → river segments + lake rims.
    /// 3. `build_systems_for_region` — Poisson-disk chambers + MST tunnels +
    ///    entrance rollers.
    pub(super) fn build_fine_region(&self, coord: region::RegionCoord) -> region::FineRegion {
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

    /// Pre-fetch the 3 × 3 grid of fine regions centered on the chunk.
    ///
    /// A chunk (32 blocks) is smaller than a region (512 blocks), so it
    /// touches at most 4 distinct regions. The 3×3 grid is the minimal
    /// axis-aligned box of regions that always fully contains the chunk plus
    /// a 1-cell halo (needed by `lake_rim_at`'s 8-neighbour search).
    ///
    /// Pre-fetching here means the per-column hot path is pure arithmetic —
    /// no mutex locks, no cache misses.
    pub(super) fn gather_chunk_regions(&self, coord: ChunkCoord) -> ChunkRegions {
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

    // ── Carver helpers ────────────────────────────────────────────────────

    /// Fetch (or build) the carver tunnels rooted at `coord`.
    ///
    /// Each chunk deterministically generates a list of MC-style sphere-capsule
    /// worm tunnels (`CarverTunnel`) based on `(seed, coord)`. These are cached
    /// because a single chunk fill reads tunnels from up to 605 surrounding
    /// chunks — without the cache, each of those would rebuild every frame.
    fn get_carver_tunnels(&self, coord: ChunkCoord) -> std::sync::Arc<Vec<carver::CarverTunnel>> {
        let mut cache = self.carver_cache.lock().unwrap();
        if let Some(t) = cache.get(&coord) {
            return t.clone();
        }
        let t = std::sync::Arc::new(carver::build_tunnels_for_chunk(self.seed, coord));
        cache.put(coord, t.clone());
        t
    }

    /// Build the per-chunk carver mask by rasterising every tunnel from
    /// neighbouring chunks within reach. Returns a flat `CHUNK_DIM³` bool
    /// array indexed `lx + DIM·ly + DIM²·lz`.
    ///
    /// Carver max reach is ~130 blocks horizontally (5 chunks at 32 each)
    /// and ~60 blocks vertically; the loop covers `11×5×11 = 605` chunks.
    pub(crate) fn build_carver_mask(&self, coord: ChunkCoord) -> Vec<bool> {
        let dim = CHUNK_DIM as usize;
        let mut mask = vec![false; dim * dim * dim];
        let origin = coord.0 * CHUNK_DIM;
        let r_xz: i32 = 5;
        let r_y: i32 = 2;
        for dx in -r_xz..=r_xz {
            for dy in -r_y..=r_y {
                for dz in -r_xz..=r_xz {
                    let nc = ChunkCoord(IVec3::new(coord.0.x + dx, coord.0.y + dy, coord.0.z + dz));
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
}
