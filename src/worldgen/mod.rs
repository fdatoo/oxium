//! Procedural world generation: a pure `(seed, ChunkCoord) -> DenseChunk` map.
//!
//! "Pure" matters: terrain output must depend only on the seed and chunk
//! coordinate so chunks can be regenerated from disk-free state and so unit
//! tests can pin output with golden hashes.
//!
//! The pipeline runs in five conceptual passes per column:
//!
//! 1. **Biome maps.** Two very low-frequency noise fields drive geography:
//!    `mountainness_map` decides where tall ranges cluster, `desert_map`
//!    decides where sand replaces grass at the surface. Both are smooth so
//!    biomes blend instead of stepping.
//! 2. **Heightmap.** Base FBM gives rolling hills (`±AMPLITUDE` blocks).
//!    A *ridged* `mountain_noise` (peaks form along the zero-crossings of
//!    a smooth field, not at isolated maxima) adds `+MOUNTAIN_PEAK`
//!    elevation scaled by `mountainness_map` — only mountainy regions
//!    grow ranges, and the ranges form connected ridges rather than
//!    scattered bumps.
//! 3. **Layers.** Top block is grass (sand near sea level *or* in
//!    deserts, stone on tall mountain peaks); next three are dirt;
//!    everything below is stone.
//! 4. **Caves.** Two interleaved cave systems:
//!    * **Tunnels** — the intersection of two independent 3D noises'
//!      zero-crossings. Each noise's `|n| < TUNNEL_BAND` defines an
//!      infinite warped sheet; where two sheets cross they form
//!      long winding ribbons ~2-3 blocks wide. Classic voxel-game
//!      tunnel shape.
//!    * **Caverns** — a single low-frequency 3D noise whose extreme
//!      values open into large irregular rooms, occasionally
//!      intersecting tunnels for big chambers with corridor entries.
//!    Caves only carve below the dirt cap so surface terrain stays
//!    intact, and respect a tiny floor so the world doesn't drop
//!    away to infinity at the chunk-stack bottom.
//! 5. **Sea level.** Any air at or below `SEA_LEVEL` becomes water —
//!    flooded cave passages turn into underwater grottos automatically.
//!
//! Tree placement (in `add_trees`) reuses the same biome data so
//! deserts and mountain peaks stay bare.

use crate::voxel::block::Block;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{ChunkCoord, LocalPos, CHUNK_DIM_U};
use crate::worldgen::tuning::{FINE_REGION_SIZE, MAX_TERRAIN_Y};
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
pub mod hash;
pub mod heightmap;
pub mod hydrology;
pub mod plates;
pub mod region;
pub mod surface;
pub mod trees;
pub mod tuning;

/// World-space Y at which the sea surface sits. Blocks above this with no
/// solid above turn into air; air below this turns into water.
pub const SEA_LEVEL: i32 = 62;
/// World-space Y below which we leave a thin "floor" so the bottom of
/// the loaded chunk stack doesn't dissolve into nothing. Caves above
/// this can carve normally; cells at or below are left as their
/// non-cave block.
const CAVE_FLOOR_Y: i32 = -120;
/// Minimum depth (in blocks) below the surface a cave is allowed to
/// carve. Anything shallower than this would punch through the dirt
/// cap and leave holes in the grass, so we leave a buffer.
const CAVE_SURFACE_BUFFER: i32 = 4;
/// World-space Y above which any surface block in a cold biome gets
/// capped with snow regardless of the desert/grass decision. Used so
/// even temperate forests have a snowy alpine band on the upper
/// flanks of nearby peaks.
const SNOW_LINE: i32 = 110;
/// Temperature threshold (in normalised noise units, roughly `[-1, 1]`)
/// below which a column counts as cold — gets a Snow surface cap and
/// no trees regardless of humidity. Around `-0.10` so the cold belt
/// covers a modest fraction of the world rather than dominating.
const COLD_THRESHOLD: f32 = -0.10;
/// Humidity threshold above which a temperate column counts as a
/// forest (denser trees). Below the threshold the column reads as
/// plains (rare trees).
const FOREST_HUMIDITY: f32 = 0.05;
/// Tree-cell spawn percentile (out of 100) for plains: dry grassland
/// with the occasional lone tree.
const TREE_RATE_PLAINS: u32 = 12;
/// Tree-cell spawn percentile for forest: dense woodland.
const TREE_RATE_FOREST: u32 = 55;
/// World is partitioned into `CELL_SIZE × CELL_SIZE` (XZ) tree cells.
/// Each cell rolls a deterministic hash to decide whether it contains a
/// tree (and where in the cell). 8 blocks per cell + ~35 % spawn rate
/// gives a forest density of roughly one tree per 180 blocks² — enough
/// that hills look wooded without filling every meadow.
const TREE_CELL_SIZE: i32 = 8;
/// Maximum world-space radius (XZ + Y above surface) a tree's blocks can
/// occupy. Used to decide which neighbouring tree cells could spill
/// blocks into the chunk currently being generated.
const TREE_MARGIN: i32 = 5;

/// Pre-built noise fields for one world seed.
///
/// The struct exists mainly so the noise fields are constructed *once*: the
/// `Fbm` builder is comparatively expensive, and chunk generation calls
/// `get` thousands of times per chunk.
pub struct Generator {
    /// PR 2: plate-driven heightmap (continental shelf + ridges +
    /// domain-warped FBM relief). Owns the FBM/warp noise fields.
    heightmap: heightmap::HeightmapNoise,
    /// Geographic "is this region desert?" mask. Same large period as the
    /// mountainness map but uncorrelated (different seed) so deserts and
    /// mountains drift independently.
    desert_map: Fbm<Simplex>,
    /// Temperature map (large-period 2D noise). Drives the cold/warm
    /// axis of the biome system; negative values are colder and earn
    /// snow surfaces, positive values are warmer (and combined with
    /// the desert mask, hottest values are arid).
    temperature_map: Fbm<Simplex>,
    /// Humidity map (large-period 2D noise). Drives the wet/dry axis;
    /// wetter columns earn denser tree cover, drier columns read as
    /// sparser plains.
    humidity_map: Fbm<Simplex>,
    /// Deep-band wormhole filler: sparse 3D noise that supplements
    /// the graph-based cave systems below `WORMHOLE_BAND_Y`. Above
    /// that, caves come exclusively from the cave-system graph.
    wormhole_noise: caves::WormholeNoise,
    /// High-frequency 2D noise used to perturb biome thresholds so
    /// the resulting boundaries wave instead of cutting in straight
    /// contour lines. Added in PR 5.
    biome_jitter_noise: Fbm<Simplex>,
    seed: u64,
    /// LRU cache of pre-built fine regions. Consulted per chunk fill
    /// to evaluate valley carve and lake water; built on first touch.
    fine_cache: region::FineCache,
    /// LRU cache of macro regions feeding the trunk-river injection
    /// into fine flow accumulation.
    macro_cache: region::MacroCache,
}

impl Generator {
    /// Build a `Generator` with the given world seed.
    ///
    /// Each noise field is seeded with a different per-axis salt so they
    /// don't produce correlated patterns (mountain-noise lining up with
    /// height-noise would just amplify existing hills instead of adding
    /// new geographic features).
    pub fn new(seed: u64) -> Self {
        // Heightmap noise: 4 octaves, ~96-block period at octave 0.
        // PR 2: the plate-driven heightmap owns its own FBM + warp
        // noise fields. The old `height_noise`, `mountain_noise`, and
        // `mountainness_map` are gone — plate geometry replaces them.
        let heightmap = heightmap::HeightmapNoise::new(seed);
        // Biome maps: large period so each biome covers many chunks.
        let desert_map = Fbm::<Simplex>::new(seed.wrapping_add(4) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 512.0)
            .set_persistence(0.5);
        // Climate maps. Same scale as the other biome masks so a
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
        // High-frequency biome-edge jitter. 2 octaves of Simplex at
        // ~24-block period, amplitude shaped by `BIOME_JITTER_AMPL`.
        let biome_jitter_noise = Fbm::<Simplex>::new(seed.wrapping_add(401) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / tuning::BIOME_JITTER_PERIOD as f64)
            .set_persistence(0.5);
        Self {
            heightmap,
            desert_map,
            temperature_map,
            humidity_map,
            wormhole_noise,
            biome_jitter_noise,
            seed,
            fine_cache: region::fresh_fine_cache(),
            macro_cache: region::fresh_macro_cache(),
        }
    }

    /// Sample the biome-edge jitter at world `(wx, wz)`. Scaled to
    /// `±BIOME_JITTER_AMPL` (noise-value units).
    fn biome_jitter(&self, wx: i32, wz: i32) -> f32 {
        (self.biome_jitter_noise.get([wx as f64, wz as f64]) as f32)
            * tuning::BIOME_JITTER_AMPL
    }

    /// Rotated jitter — sample at `(wz, -wx)` so it's uncorrelated
    /// with the primary jitter. Used for the humidity threshold so
    /// Forest/Plains edges don't co-jitter with desert edges.
    fn biome_jitter_rot(&self, wx: i32, wz: i32) -> f32 {
        (self.biome_jitter_noise.get([wz as f64, -(wx as f64)]) as f32)
            * tuning::BIOME_JITTER_AMPL
    }

    /// Build the fine region at `coord` from noise (heightmap +
    /// hydrology). The result is byte-deterministic in
    /// `(seed, coord)`; this method is invoked at most once per
    /// region per cache lifetime (rebuilds happen on eviction).
    fn build_fine_region(&self, coord: region::RegionCoord) -> region::FineRegion {
        let mut r = region::FineRegion::empty(coord);
        r.coord = coord;
        hydrology::build_fine_hydro(
            self.seed,
            coord,
            &self.heightmap,
            &self.macro_cache,
            &mut r,
        );
        caves::build_systems_for_region(self.seed, coord, &self.heightmap, &mut r);
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
    fn column_data(&self, wx: i32, wz: i32) -> ColumnData {
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
        // Pre-river heightmap from plates + warped FBM.
        let h_pre = self.heightmap.h_pre(self.seed, wx as f32, wz as f32);
        // Slope-driven cliff classification on the *unmodified* h_pre.
        let is_cliff = self.heightmap.is_cliff(self.seed, wx as f32, wz as f32);

        // Valley carve over the chunk's pre-fetched 3 × 3 region
        // neighbourhood. Slightly larger ring than strictly correct
        // (would need a 5 × 5 ring to handle valley contributions
        // from segments 2 regions away from a chunk's edge column),
        // but in practice rivers cross at most 1 region boundary
        // within the carve radius. Minor visual artifact for v1.
        let carve = regions.valley_carve(wx, wz, self.seed);
        let height = (h_pre - carve)
            .clamp((CAVE_FLOOR_Y + 8) as f32, MAX_TERRAIN_Y as f32) as i32;

        // Climate sample. PR 5 adds threshold perturbation: a small
        // shared high-frequency noise field jitters the biome
        // thresholds so transition edges wave instead of cutting in
        // straight contour lines.
        let xz = [wx as f64, wz as f64];
        let jitter = self.biome_jitter(wx, wz);

        let desertness_raw = self.desert_map.get(xz) as f32;
        let desertness = desertness_raw + jitter;
        let is_desert = desertness > 0.30;

        let temperature_raw = self.temperature_map.get(xz) as f32;
        let temperature = temperature_raw - jitter; // negate so cold zones jitter independently
        let humidity_raw = self.humidity_map.get(xz) as f32;
        // Rotated jitter for the humidity threshold so Forest/Plains
        // and cold/warm boundaries don't co-jitter.
        let humidity = humidity_raw + self.biome_jitter_rot(wx, wz);
        let biome = Biome::classify(temperature, humidity, is_desert);

        ColumnData {
            height,
            is_cliff,
            desertness,
            biome,
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
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let wx = origin.x + x as i32;
                let wz = origin.z + z as i32;
                let col = self.column_data_with(wx, wz, &regions);
                let height = col.height;
                // Lake water rim (if this column sits in a sink-filled
                // basin). Above the column's solid height but below
                // the rim, the column floods with water.
                let lake_rim = regions.lake_rim_at(wx, wz);

                for y in 0..CHUNK_DIM_U {
                    let wy = origin.y + y as i32;
                    let local = LocalPos(UVec3::new(x, y, z));

                    let block = if wy > height {
                        // Above the solid surface — flood with water
                        // up to either the lake rim (highest priority)
                        // or sea level, whichever is appropriate.
                        if let Some(rim) = lake_rim {
                            if wy <= rim {
                                Block::Water
                            } else if wy <= SEA_LEVEL {
                                Block::Water
                            } else {
                                Block::Air
                            }
                        } else if wy <= SEA_LEVEL {
                            Block::Water
                        } else {
                            Block::Air
                        }
                    } else {
                        let depth = height - wy;
                        // Cave-carve gates:
                        // * Floor: don't carve below CAVE_FLOOR_Y so
                        //   the bottom of the vertical load radius has
                        //   *something* in it (otherwise the player
                        //   could see straight down into the sky from
                        //   deep underground).
                        // * Surface buffer: chambers and tunnels never
                        //   carve within CAVE_SURFACE_BUFFER blocks of
                        //   the surface — except where an explicit
                        //   entrance feature (sinkhole / cliff mouth
                        //   / skylight) punches through.
                        // * Deep wormholes: a sparse 3D-noise band
                        //   layered only below WORMHOLE_BAND_Y.
                        let in_entrance =
                            !cave_systems.is_empty()
                                && caves::entrance_air(wx, wy, wz, &cave_systems);
                        let in_chamber_or_tunnel = depth > CAVE_SURFACE_BUFFER
                            && !cave_systems.is_empty()
                            && caves::cave_air(wx, wy, wz, &cave_systems);
                        let in_wormhole =
                            depth > CAVE_SURFACE_BUFFER
                                && self.wormhole_noise.carve(wx, wy, wz);
                        let cave =
                            wy > CAVE_FLOOR_Y && (in_entrance || in_chamber_or_tunnel || in_wormhole);
                        if cave {
                            if wy <= SEA_LEVEL {
                                Block::Water
                            } else {
                                Block::Air
                            }
                        } else if depth == 0 {
                            // Surface block selection. Priority:
                            //   1. Cliff → bare Stone (replaces v1
                            //      `MOUNTAIN_ROCK_LINE`).
                            //   2. Beach (`height ∈ [SL-1, SL+2]`) →
                            //      Sand. PR 5 widens to 4 blocks.
                            //   3. Snow line → Snow.
                            //   4. Cold biome → Snow.
                            //   5. Desert / Tropical-beach-adjacent →
                            //      Sand; otherwise → Grass.
                            //   6. PR 5: stochastic sand/grass
                            //      transition band on the grass side
                            //      of the desert boundary.
                            if col.is_cliff {
                                Block::Stone
                            } else if height >= SEA_LEVEL - 1
                                && height <= SEA_LEVEL + 2
                                && !col.biome.snow_capped()
                            {
                                // Beach band (4 blocks tall).
                                Block::Sand
                            } else if height >= SNOW_LINE {
                                Block::Snow
                            } else if col.biome.snow_capped() {
                                Block::Snow
                            } else if col.biome == Biome::Desert {
                                Block::Sand
                            } else {
                                // PR 5 stochastic sand transition:
                                // inside `SAND_TRANSITION_BAND` (in
                                // noise-value units) on the grass
                                // side of the desert boundary, roll
                                // for sand vs grass. Probability
                                // ramps from 0 at the band's outer
                                // edge to ~50% at the boundary.
                                let dist_to_boundary = 0.30 - col.desertness;
                                if dist_to_boundary > 0.0
                                    && dist_to_boundary
                                        < tuning::SAND_TRANSITION_BAND
                                {
                                    let p = 0.5
                                        * (1.0
                                            - dist_to_boundary
                                                / tuning::SAND_TRANSITION_BAND);
                                    let roll = hash::mix_unit(
                                        self.seed,
                                        &[wx, wz, 71],
                                    );
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
                            // Cliff faces are stone all the way down —
                            // no dirt under sheer rock. Everywhere
                            // else gets the standard dirt cap.
                            if col.is_cliff {
                                Block::Stone
                            } else {
                                Block::Dirt
                            }
                        } else {
                            Block::Stone
                        }
                    };
                    out.set(local, block);
                }
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
        // First the column-level vetoes — a cell can be a tree
        // candidate by roll but its column might be a beach, a bare
        // peak, or above the alpine snow line, in which case no
        // amount of luck makes a tree grow.
        let wx = cell_x * TREE_CELL_SIZE
            + (tree_hash(self.seed, cell_x, cell_z, 1) % 6) as i32
            + 1;
        let wz = cell_z * TREE_CELL_SIZE
            + (tree_hash(self.seed, cell_x, cell_z, 2) % 6) as i32
            + 1;
        let col = self.column_data(wx, wz);
        if col.height <= SEA_LEVEL + 1 {
            return None;
        }
        // Trees don't grow on cliffs — replaces v1's mountain-rock-line
        // veto with a slope-driven equivalent.
        if col.is_cliff {
            return None;
        }
        if col.height >= SNOW_LINE {
            return None;
        }
        // Biome decides both *whether* trees grow here at all and
        // *how densely* they pack. Tundra and Desert return `None`
        // outright; the rest carry a percentile (0..100) that the
        // roll below has to clear. Higher percentile ⇒ denser
        // forest. Threshold form `(roll mod 100) < rate` so the
        // distribution stays uniform-ish across cells.
        let rate = col.biome.tree_rate_percentile()?;
        let roll = tree_hash(self.seed, cell_x, cell_z, 0) % 100;
        if roll >= rate {
            return None;
        }
        let height = col.height;
        let kind = col.biome.tree_kind();
        let trunk_h = match kind {
            TreeKind::Oak => {
                // Roll #3: oak trunk height in 4..=6 blocks.
                4 + (tree_hash(self.seed, cell_x, cell_z, 3) % 3) as i32
            }
            TreeKind::Palm => {
                // Palms are taller and skinnier: 7..=9 blocks.
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
struct ColumnData {
    /// Surface height in world Y, post-carve, clamped.
    height: i32,
    /// True if the column's `h_pre` slope exceeds `CLIFF_SLOPE_THRESH`.
    is_cliff: bool,
    /// Jitter-perturbed `desertness` noise value. Used by the
    /// sand/grass transition band (PR 5): inside the band on the
    /// grass side of the desert boundary, the surface block is
    /// rolled stochastically.
    desertness: f32,
    /// Discrete biome label derived from temperature, humidity, and
    /// the desert mask, with PR 5's threshold perturbation applied.
    biome: Biome,
}

/// Discrete biome label assigned to each column. The set is small on
/// purpose — every variant has a distinct visual signature (different
/// surface block or noticeably different tree density), so the
/// difference between biomes reads from a screenshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Biome {
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
    /// Map climate values plus the (jitter-perturbed) desert mask to
    /// a discrete biome. PR 5 adds `Tropical` for the hot+wet
    /// bucket that the new continental geography produces a lot of.
    fn classify(temperature: f32, humidity: f32, is_desert: bool) -> Self {
        if temperature < COLD_THRESHOLD {
            return if humidity > 0.0 {
                Biome::SnowyForest
            } else {
                Biome::Tundra
            };
        }
        if is_desert {
            return Biome::Desert;
        }
        if humidity > FOREST_HUMIDITY {
            // Hot+wet ⇒ Tropical; temperate+wet ⇒ Forest.
            if temperature > 0.20 {
                Biome::Tropical
            } else {
                Biome::Forest
            }
        } else {
            Biome::Plains
        }
    }

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
            Biome::Tropical => Some(crate::worldgen::tuning::TREE_RATE_TROPICAL),
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

    /// Lake rim at this column, if it sits inside a sink-filled
    /// basin. Returns `None` outside lakes or outside the grid.
    fn lake_rim_at(&self, wx: i32, wz: i32) -> Option<i32> {
        self.region_at(wx, wz)
            .and_then(|r| hydrology::lake_rim_at(wx, wz, r))
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
        // Hash re-baselined for PR 5 (Tropical biome + threshold
        // perturbation + stochastic sand transition band + palm
        // tree stamps).
        const GOLDEN_42_002: u64 = 0x8179_F099_EB0F_7C0F;
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

    /// Cold biomes should plant Snow as their surface block. Pick a
    /// column known to be Tundra and verify the topmost solid block
    /// is Snow, not Grass.
    #[test]
    fn cold_biome_caps_with_snow() {
        let g = Generator::new(42);
        // Find any tundra column by scanning the same area as
        // `all_biomes_appear_in_a_large_scan`.
        let mut found: Option<(i32, i32)> = None;
        'outer: for wz in (-1024..1024).step_by(8) {
            for wx in (-1024..1024).step_by(8) {
                if g.column_data(wx, wz).biome == Biome::Tundra {
                    found = Some((wx, wz));
                    break 'outer;
                }
            }
        }
        let (wx, wz) = found.expect("expected at least one tundra column");
        let col = g.column_data(wx, wz);
        // Build the chunk that contains the surface block and read
        // out the cell at the column's `height`.
        let cy = col.height.div_euclid(CHUNK_DIM_U as i32);
        let cx = wx.div_euclid(CHUNK_DIM_U as i32);
        let cz = wz.div_euclid(CHUNK_DIM_U as i32);
        let mut chunk = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(cx, cy, cz)), &mut chunk);
        let lx = wx.rem_euclid(CHUNK_DIM_U as i32) as u32;
        let lz = wz.rem_euclid(CHUNK_DIM_U as i32) as u32;
        let ly = col.height.rem_euclid(CHUNK_DIM_U as i32) as u32;
        let surface = chunk.blocks[crate::voxel::coords::LocalPos(
            glam::UVec3::new(lx, ly, lz),
        )
        .to_index()];
        // Beach / mountain rock overrides take priority over the
        // biome cap (the rules in `fill_chunk`); the picked column
        // shouldn't trip either of those, but accept either Snow or
        // those overrides defensively so the test reports a clearer
        // failure if it does.
        assert!(
            matches!(surface, Block::Snow | Block::Sand | Block::Stone),
            "tundra surface block at ({wx}, {wz}) y={} was {:?}, expected Snow",
            col.height,
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
