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
use crate::worldgen::tuning::MAX_TERRAIN_Y;
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
/// Half-width of the "near-zero" band around each tunnel noise's
/// zero-crossing surface. Tunnels appear where BOTH
/// [`Generator::tunnel_a`] and [`Generator::tunnel_b`] sit inside this
/// band — geometrically, that's the intersection of two warped sheets
/// in 3D, which traces out long winding ribbons. Wider band → fatter
/// and more frequent tunnels; narrower → sparse capillaries.
const TUNNEL_BAND: f64 = 0.08;
/// Threshold for the cavern noise: cells where the noise exceeds this
/// open into a cavern. Higher value → rarer / smaller rooms. The
/// 3D noise's amplitude is roughly `[-1, 1]` so 0.62 keeps caverns
/// uncommon enough that they read as discoveries, not Swiss cheese.
const CAVERN_THRESH: f64 = 0.62;
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
/// Half-width of the "near-zero" band on the river noise. Columns
/// whose `river_noise` value sits inside `±RIVER_BAND` get carved
/// down toward the river bed; the smaller the band the narrower the
/// rivers (and the more often they pinch into thin streams). 0.045
/// produces 3-5 block wide rivers at our river-noise frequency.
const RIVER_BAND: f64 = 0.045;
/// Depth below [`SEA_LEVEL`] that the *centre* of a river column
/// gets carved to. Edges of the band smoothly interpolate up to the
/// natural heightmap so a river meandering through a mountain valley
/// reads as a carved bed, not a sheer drop.
const RIVER_CARVE: i32 = 3;
/// Lower bound on the lake noise above which the column belongs to a
/// lake. Values above `LAKE_THRESH + LAKE_RAMP` are fully inside; the
/// `LAKE_RAMP` window in between smoothly interpolates so shorelines
/// taper rather than terracing.
const LAKE_THRESH: f64 = 0.50;
const LAKE_RAMP: f64 = 0.12;
/// Depth below [`SEA_LEVEL`] that a fully-inside lake column carves
/// to. Lakes are slightly deeper than rivers so they read as wider
/// bodies of standing water rather than fattened streams.
const LAKE_CARVE: i32 = 5;

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
    /// Single-octave 2D noise whose zero-crossings define river
    /// centerlines. Smooth (1 octave) so the rivers meander as
    /// continuous curves instead of jagging back on themselves
    /// every few blocks.
    river_noise: Fbm<Simplex>,
    /// 2D noise whose high-value regions define lake basins. Higher
    /// period than the river noise so lakes are larger and less
    /// frequent — they punctuate the landscape rather than tiling it.
    lake_noise: Fbm<Simplex>,
    /// First of two 3D noise fields whose zero-crossings intersect to
    /// form cave tunnels. By itself this would carve a single warped
    /// sheet through the world; combined with [`Self::tunnel_b`] only
    /// the tube along the intersection survives.
    tunnel_a: Fbm<Simplex>,
    /// Second tunnel noise — seeded independently from [`Self::tunnel_a`]
    /// so the two sheets cross at random angles rather than running
    /// parallel.
    tunnel_b: Fbm<Simplex>,
    /// Lower-frequency 3D noise that opens into large cavern rooms
    /// where its value runs hot. Layered on top of the tunnel system
    /// so a tunnel occasionally widens into a chamber.
    cavern_noise: Fbm<Simplex>,
    seed: u64,
    /// LRU cache of pre-built fine regions. PR 1: present but not yet
    /// consumed by `fill_chunk`; PRs 2–4 fill in the per-region
    /// computation that chunk fill consults.
    #[allow(dead_code)]
    fine_cache: region::FineCache,
    /// LRU cache of pre-built macro regions for the trunk-river pass.
    /// PR 1: present but unused; PR 3 fills it.
    #[allow(dead_code)]
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
        // River noise: 1 octave for smooth, gently curving zero
        // crossings — extra octaves would give the river a jagged
        // bank profile. Period ~150 blocks so rivers feel like
        // walkable distances, not micro-streams or world-spanning
        // canals.
        let river_noise = Fbm::<Simplex>::new(seed.wrapping_add(10) as u32)
            .set_octaves(1)
            .set_frequency(1.0 / 150.0);
        // Lake noise: longer period than the river so lake basins
        // are the "rare, large" feature. 2 octaves so the shoreline
        // has some shape instead of being a pure smooth blob.
        let lake_noise = Fbm::<Simplex>::new(seed.wrapping_add(11) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 280.0)
            .set_persistence(0.5);
        // Tunnel system: two independent 3D noises at the same frequency.
        // 2 octaves keeps the surfaces relatively smooth — too many
        // octaves and the tunnel walls turn into ragged stair-steps.
        // Period ~40 blocks ⇒ tunnels meander on a scale of a few
        // chunks, comfortably explorable.
        let tunnel_a = Fbm::<Simplex>::new(seed.wrapping_add(5) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 40.0)
            .set_persistence(0.5);
        let tunnel_b = Fbm::<Simplex>::new(seed.wrapping_add(6) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 40.0)
            .set_persistence(0.5);
        // Caverns: a single low-frequency 3D field. Lower frequency than
        // tunnels (period ~80 blocks) so each "hot" region is large
        // enough to read as a room rather than a wider patch of tunnel.
        let cavern_noise = Fbm::<Simplex>::new(seed.wrapping_add(7) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 80.0)
            .set_persistence(0.5);
        Self {
            heightmap,
            desert_map,
            temperature_map,
            humidity_map,
            river_noise,
            lake_noise,
            tunnel_a,
            tunnel_b,
            cavern_noise,
            seed,
            fine_cache: region::fresh_fine_cache(),
            macro_cache: region::fresh_macro_cache(),
        }
    }

    /// Decide whether the cell at world-space `(wx, wy, wz)` should be
    /// carved away as cave.
    ///
    /// Returns `true` if either:
    /// * Both [`Self::tunnel_a`] and [`Self::tunnel_b`] sit inside
    ///   `±TUNNEL_BAND` (a tunnel passes through here), OR
    /// * [`Self::cavern_noise`] exceeds [`CAVERN_THRESH`] (a cavern
    ///   opens here).
    ///
    /// The caller is responsible for additional gates (surface buffer,
    /// floor depth) — this method only answers "does the cave noise
    /// say air?" so the layer pass can compose it with its own rules.
    fn is_cave(&self, wx: i32, wy: i32, wz: i32) -> bool {
        let p = [wx as f64, wy as f64, wz as f64];
        let a = self.tunnel_a.get(p);
        let b = self.tunnel_b.get(p);
        // Tunnel: each noise field's zero level set is a smooth warped
        // sheet through the world. The intersection of two such sheets
        // is a 1D curve — the actual tunnel centre. Widening each band
        // from "zero" to "near-zero" thickens the sheets into slabs,
        // and their intersection thickens from a curve to a tube of
        // roughly TUNNEL_BAND × TUNNEL_BAND cross-section. With our
        // band of 0.08 that's tubes ~2-3 blocks across.
        if a.abs() < TUNNEL_BAND && b.abs() < TUNNEL_BAND {
            return true;
        }
        // Cavern: a single noise's high-value region. Lower frequency
        // means the region is larger when it occurs, producing a
        // "room" instead of a patch of tunnel.
        self.cavern_noise.get(p) > CAVERN_THRESH
    }

    /// Per-column terrain decisions: surface height + biome + slope
    /// flag. Used by both `fill_chunk` and `add_trees` so a single
    /// noise evaluation per column drives every geographic choice.
    ///
    /// PR 2: heightmap is now plate-driven `h_pre` (continental
    /// shelf + plate-edge ridges + warped-FBM relief). Cliff
    /// detection comes from the slope of `h_pre`, replacing the v1
    /// `MOUNTAIN_ROCK_LINE` rule. Rivers and lakes still use legacy
    /// noise carve here (PR 3 replaces this with the real river
    /// network).
    fn column_data(&self, wx: i32, wz: i32) -> ColumnData {
        let xz = [wx as f64, wz as f64];
        // Pre-river heightmap from plates + warped FBM.
        let mut height = self.heightmap.h_pre(self.seed, wx as f32, wz as f32);
        // Slope-driven cliff classification on the *unmodified* h_pre
        // — measuring slope after the river carve would falsely flag
        // every valley side as a cliff.
        let is_cliff = self.heightmap.is_cliff(self.seed, wx as f32, wz as f32);

        // Legacy river / lake carve. PR 3 will replace this with the
        // real flow-accumulation network; for now the existing noise
        // carve operates on h_pre so the world has at least the
        // current generation's water bodies while we work.
        let r_noise = self.river_noise.get(xz) as f32;
        let river_strength =
            (1.0 - (r_noise.abs() / RIVER_BAND as f32)).clamp(0.0, 1.0);
        let l_noise = self.lake_noise.get(xz) as f32;
        let lake_strength = smoothstep(
            LAKE_THRESH as f32,
            (LAKE_THRESH + LAKE_RAMP) as f32,
            l_noise,
        );
        let carve = river_strength.max(lake_strength);
        if carve > 0.0 {
            let bed = if lake_strength > river_strength {
                (SEA_LEVEL - LAKE_CARVE) as f32
            } else {
                (SEA_LEVEL - RIVER_CARVE) as f32
            };
            height = height * (1.0 - carve) + bed * carve;
        }
        let height = height.clamp(
            (CAVE_FLOOR_Y + 8) as f32,
            MAX_TERRAIN_Y as f32,
        ) as i32;

        let desertness = self.desert_map.get(xz) as f32;
        // Hard cutoff (no transition smoothing) so the desert/grass
        // boundary stays crisp and recognisable.
        let is_desert = desertness > 0.30;

        // Climate axes drive the biome system. The biome itself is a
        // discrete derivation of (temperature, humidity, desertness)
        // — see `Biome::classify` — so consumers don't have to repeat
        // the threshold logic. Computed up front for both the layer
        // pass and the tree placer.
        let temperature = self.temperature_map.get(xz) as f32;
        let humidity = self.humidity_map.get(xz) as f32;
        let biome = Biome::classify(temperature, humidity, is_desert);

        ColumnData {
            height,
            is_cliff,
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
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let wx = origin.x + x as i32;
                let wz = origin.z + z as i32;
                let col = self.column_data(wx, wz);
                let height = col.height;

                for y in 0..CHUNK_DIM_U {
                    let wy = origin.y + y as i32;
                    let local = LocalPos(UVec3::new(x, y, z));

                    let block = if wy > height {
                        if wy <= SEA_LEVEL {
                            Block::Water
                        } else {
                            Block::Air
                        }
                    } else {
                        let depth = height - wy;
                        // Cave-carve gates:
                        // * Surface buffer: never carve through the
                        //   topmost CAVE_SURFACE_BUFFER blocks, so the
                        //   grass cap stays intact.
                        // * Floor: don't carve below CAVE_FLOOR_Y so
                        //   the bottom of the vertical load radius has
                        //   *something* in it (otherwise the player
                        //   could see straight down into the sky
                        //   colour from deep underground).
                        // * Noise: tunnel-intersection OR cavern (see
                        //   `is_cave` for the geometry of each).
                        let cave = depth > CAVE_SURFACE_BUFFER
                            && wy > CAVE_FLOOR_Y
                            && self.is_cave(wx, wy, wz);
                        if cave {
                            if wy <= SEA_LEVEL {
                                Block::Water
                            } else {
                                Block::Air
                            }
                        } else if depth == 0 {
                            // Surface block selection. Priority order:
                            //   1. Cliff (slope > CLIFF_SLOPE_THRESH) →
                            //      bare Stone. Wins over beach so cliffed
                            //      coastlines read as rock faces, not
                            //      sand strips. Replaces v1's
                            //      `MOUNTAIN_ROCK_LINE` rule.
                            //   2. Beach (column at/below sea level + 1
                            //      and not a cliff) — coastline sand.
                            //   3. Snow line — alpine snow cap, biome-
                            //      independent.
                            //   4. Cold biome — surface snow at any
                            //      elevation.
                            //   5. Desert → Sand; everything else → Grass.
                            if col.is_cliff {
                                Block::Stone
                            } else if height <= SEA_LEVEL + 1 {
                                Block::Sand
                            } else if height >= SNOW_LINE {
                                Block::Snow
                            } else if col.biome.snow_capped() {
                                Block::Snow
                            } else if col.biome == Biome::Desert {
                                Block::Sand
                            } else {
                                Block::Grass
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
        // Roll #3: trunk height in 4..=6 blocks.
        let trunk_h = 4 + (tree_hash(self.seed, cell_x, cell_z, 3) % 3) as i32;
        Some(Tree {
            wx,
            wz,
            base_y: height,
            trunk_h,
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
        // Leaves: a thick disc + slight cap around the top of the trunk.
        let top_y = tree.base_y + tree.trunk_h;
        // Round canopy. Radius² uses 6 so the corner cells are dropped
        // and the silhouette stays roughly spherical instead of cubic.
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
}

/// Per-column biome + geometry summary used by both `fill_chunk` and
/// `add_trees` so block selection and tree placement stay in sync.
#[derive(Debug, Clone, Copy)]
struct ColumnData {
    /// Surface height in world Y, post-carve, clamped.
    height: i32,
    /// True if the column's `h_pre` slope exceeds `CLIFF_SLOPE_THRESH`.
    /// Drives bare-rock surface exposure and prevents trees from
    /// taking root on sheer faces. Replaces the v1 `mountain_weight`
    /// + `MOUNTAIN_ROCK_LINE` combo.
    is_cliff: bool,
    /// Discrete biome label derived from temperature, humidity, and
    /// the desert mask.
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
    /// Hot, dry. Sand surface, no trees. Existing desert biome
    /// preserved here so the rest of the system has a single
    /// vocabulary.
    Desert,
}

impl Biome {
    /// Map raw climate noise + the desert mask to a discrete biome.
    ///
    /// The model is the classic two-axis temperature × humidity grid
    /// boiled down to five buckets: anything below
    /// [`COLD_THRESHOLD`] is cold (Tundra or SnowyForest depending on
    /// humidity), anything that the legacy `desert_map` marks as
    /// desert beats out the warm/humid bucket, and the remaining
    /// temperate region splits on [`FOREST_HUMIDITY`].
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
            Biome::Forest
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
    /// don't host trees at all — saves the placement loop a noise
    /// evaluation per cell.
    fn tree_rate_percentile(self) -> Option<u32> {
        match self {
            Biome::Tundra | Biome::Desert => None,
            Biome::Plains => Some(TREE_RATE_PLAINS),
            Biome::Forest | Biome::SnowyForest => Some(TREE_RATE_FOREST),
        }
    }
}

/// GLSL/WGSL-style smoothstep. We re-implement it (Rust has nothing in
/// std and we don't want a dep for one function) because both `column_data`
/// and the mountain falloff want a smooth Hermite ramp from `edge0` to
/// `edge1`.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
        // Hash re-baselined for PR 2 (plate-driven heightmap +
        // warped FBM + cliff exposure). Refresh again whenever an
        // intentional generator change lands.
        const GOLDEN_42_002: u64 = 0xBA8B_6AAA_8597_30AD;
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
    /// should see a non-trivial amount of carved-out air, with both
    /// solid stone present *and* air present in the same chunk. The
    /// previous Swiss-cheese carving could go either too-empty (huge
    /// blob caves) or too-solid (sparse pock-marks); this asserts the
    /// middle ground where caves read as a tunnel network.
    #[test]
    fn underground_chunk_has_both_caves_and_solid() {
        let g = Generator::new(42);
        // Chunk at y=-1 ⇒ world y range [-32, -1], comfortably below
        // sea level and above CAVE_FLOOR_Y, so cave carving is
        // unconditional aside from the noise check.
        let mut c = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(0, -1, 0)), &mut c);
        let mut air = 0;
        let mut stone = 0;
        for b in c.blocks.iter() {
            match b {
                Block::Air | Block::Water => air += 1,
                Block::Stone => stone += 1,
                _ => {}
            }
        }
        assert!(stone > CHUNK_VOL / 4, "expected mostly stone, got {stone}");
        assert!(
            air > CHUNK_VOL / 100,
            "expected at least 1% carved air to prove caves carve anywhere, got {air}"
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
