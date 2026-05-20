//! All tunable worldgen constants in one place.
//!
//! The new worldgen has enough knobs that scattering them across modules
//! would make tuning a treasure hunt. Anything you might want to nudge to
//! reshape the world lives here.

// ── Plates ────────────────────────────────────────────────────────────

/// Edge length of one plate seed cell. Plate seed points live on a
/// jittered grid; each cell contains exactly one seed. Smaller → more
/// plates per region → finer-grained continental shapes (and shorter
/// mountain spines). Larger → continents the size of whole player
/// excursions before any plate boundary is crossed.
pub const PLATE_CELL_SIZE: i32 = 1024;

/// Fraction of plates classified as continental (the rest are oceanic).
/// Combined with the smooth shelf blend this determines the world's
/// land:ocean ratio.
pub const CONTINENTAL_RATIO: f32 = 0.45;

/// Per-continental-plate base elevation, sampled uniformly from this
/// range. Sea-level frame (continental_base + SEA_LEVEL = column floor
/// before relief). Higher → continents stand prouder above sea level.
pub const CONTINENTAL_BASE_RANGE: (f32, f32) = (18.0, 28.0);
/// Per-oceanic-plate base elevation. Negative — oceans sit below sea
/// level by this amount (before relief). Deep enough that the
/// continental shelf taper produces shallow coastal water.
pub const OCEANIC_BASE_RANGE: (f32, f32) = (-50.0, -20.0);
/// Per-plate amplitude multiplier on the warped-FBM relief. Multiplies
/// the base relief so some continents read as flat steppe and others
/// as rolling hills.
pub const ROUGHNESS_RANGE: (f32, f32) = (0.7, 1.4);

/// Boundary-intensity threshold below which a plate boundary lifts
/// ridges. `t = (d_b - d_a) / (d_b + d_a)` — `0` exactly on a
/// boundary, `1` deep inside a plate.
///
/// Widened from 0.12 → 0.30 in the cliff-strip fix: a narrower band
/// concentrates the ridge falloff into a short distance, making the
/// peak slope at the band midpoint steep enough to trigger cliff
/// exposure in a straight strip parallel to the plate boundary. A
/// wider band spreads the rise out so the peak slope stays below
/// `CLIFF_SLOPE_THRESH`. Trade-off: ridges are broader / less needle-
/// sharp, which reads more like a real range silhouette anyway.
pub const BOUNDARY_RIDGE_WIDTH: f32 = 0.30;

/// Peak ridge height (blocks) at a continental–continental boundary.
/// Lowered from 90 → 60 in the post-overhaul polish pass: the
/// previous value combined with `+28` continental base elevation
/// pegged `MAX_TERRAIN_Y` constantly, which clipped peaks into the
/// flat-topped mesa silhouette.
pub const RIDGE_PEAK_CC: f32 = 60.0;
/// Peak ridge height at a continental–oceanic boundary (Andes-style
/// coastal range).
pub const RIDGE_PEAK_CO: f32 = 40.0;
/// Peak ridge height at an oceanic–oceanic boundary (island arc
/// archipelago).
pub const RIDGE_PEAK_OO: f32 = 24.0;

// ── Heightmap ─────────────────────────────────────────────────────────

/// World-space Y at which the sea surface sits.
pub const SEA_LEVEL: i32 = 62;
/// Amplitude (blocks) of the domain-warp vector field applied to the
/// height FBM input coordinates. Bigger → more dramatic finger-ridges
/// and curved depressions.
pub const WARP_AMPLITUDE: f32 = 40.0;
/// Spatial period of the warp vector field.
pub const WARP_PERIOD: f32 = 400.0;
/// Slope (in blocks-per-block) above which a column is exposed as a
/// cliff: surface block becomes `Stone` and the dirt sub-surface is
/// skipped. Measured over an ±4-block stencil so small-scale FBM
/// jitter doesn't register.
pub const CLIFF_SLOPE_THRESH: f32 = 2.2;
/// Minimum `h_pre` (world Y) for cliff exposure to apply at all.
/// Reserved for genuinely tall terrain: any column with `h_pre <
/// SEA_LEVEL + 28` gets its biome surface material regardless of
/// slope, so coastal bluffs and low ridges never read as continuous
/// stone walls. Cliff exposure then becomes a feature of high
/// mountains only.
pub const CLIFF_MIN_HEIGHT: i32 = SEA_LEVEL + 28;
/// Hard upper cap on final terrain Y. Keeps tallest peaks inside the
/// vertical chunk-load radius.
pub const MAX_TERRAIN_Y: i32 = 140;

// ── 3D density (PR A: surface-band 3D; PR B: unbounded) ─────────────

/// Scale for the height-bias contribution to density: `bias = (h_target
/// - wy) / DENSITY_FALLOFF`. Small values pin the surface tightly to
/// the heightmap (almost no 3D fuzz); large values let the noise push
/// the surface around by many blocks (dramatic overhangs and
/// floating spurs). `4.0` is the de-stairs sweet spot — enough fuzz
/// to break the chevron pattern, not so much that terrain feels
/// chaotic.
pub const DENSITY_FALLOFF: f32 = 4.0;
/// Amplitude of the 3D relief noise contribution to density. Compared
/// against the unit-scale bias; `1.0` lets the noise push the
/// surface by ~`DENSITY_FALLOFF` blocks at maximum.
pub const RELIEF_AMP: f32 = 1.0;
/// Spatial period of the 3D relief noise's base octave. Smaller →
/// bumpier surface; larger → smoother.
pub const RELIEF_PERIOD: f32 = 32.0;
/// Half-width of the band around `h_target` used by the
/// `topmost_solid` helper that finds tree-trunk anchor points.
/// `fill_chunk` no longer short-circuits density evaluation by this
/// band (PR B evaluates 3D density at every voxel so overhangs can
/// appear anywhere); only the tree placer uses it as a search bound.
pub const SURFACE_BAND: i32 = 16;
/// Peak intensity of the cave SDF carve. The 3D density (bias +
/// noise) ranges roughly in `[-2, +2]`; setting the SDF intensity
/// to `4.0` means cave interiors definitely carve to air and
/// chamber walls soften gracefully where the SDF tapers to zero.
pub const CAVE_SDF_INTENSITY: f32 = 4.0;

// ── Rivers (fine) ─────────────────────────────────────────────────────

/// Coarse-cell edge length used by the fine flow-accumulation grid.
/// Smaller → finer-grained river paths (and more memory per region).
pub const FINE_CELL: i32 = 8;
/// Fine region edge length (blocks). Determines the sink-fill horizon
/// (window = `(1 + 2 * halo) * region_size`) and the LRU entry size.
pub const FINE_REGION_SIZE: i32 = 512;
/// Number of fine-region halos on each side when building. 2 → window
/// of 2560 blocks → ~2.5 km sink-fill horizon.
pub const FINE_HALO_REGIONS: i32 = 2;
/// Flow-accumulation threshold for a fine cell to be tagged as a river.
pub const RIVER_THRESH: u32 = 50;
/// Width-law scale: `width = clamp(sqrt(acc) * scale, min, max)`.
pub const RIVER_WIDTH_SCALE: f32 = 0.30;
/// Minimum river width at threshold drainage.
pub const MIN_RIVER_WIDTH: f32 = 1.5;
/// Maximum river width near major mouths.
pub const MAX_RIVER_WIDTH: f32 = 32.0;
/// Depth (blocks below the natural heightmap) of the river bed at the
/// centerline of a wide river.
pub const RIVER_BED_DEPTH: i32 = 4;
/// Valley half-width as a multiple of river width. 3× means a 6-block
/// river has a 36-block-wide valley taper on each side.
pub const VALLEY_HALF_WIDTH_MULT: f32 = 3.0;
/// Lateral meander amplitude per unit of river width.
pub const MEANDER_AMP_PER_WIDTH: f32 = 0.4;
/// Hard cap on meander amplitude regardless of width.
pub const MAX_MEANDER_AMP: f32 = 16.0;
/// Width multiplier applied at ocean mouths so deltas flare out.
pub const MOUTH_FLARE_MULT: f32 = 1.6;

// ── Rivers (macro / trunk pass) ──────────────────────────────────────

/// Macro coarse-cell edge length. 8× FINE_CELL so each macro cell
/// covers 8×8 fine cells.
pub const MACRO_CELL: i32 = 64;
/// Macro region edge length. 16× FINE_REGION_SIZE.
pub const MACRO_REGION_SIZE: i32 = 8192;
/// Number of macro-region halos when building. 1 → ~24 km horizon.
pub const MACRO_HALO_REGIONS: i32 = 1;
/// Flow-accumulation threshold (in macro cells) above which a cell is
/// flagged as a trunk river.
pub const MACRO_RIVER_THRESH: u32 = 500;

// ── Caves ────────────────────────────────────────────────────────────

/// Inclusive range of cave systems rolled per fine region.
pub const CAVE_SYSTEMS_PER_REGION: (u32, u32) = (1, 4);
/// Vertical band (inclusive both ends) for Shallow systems.
pub const CAVE_BAND_SHALLOW: (i32, i32) = (10, 50);
/// Vertical band for Middle systems.
pub const CAVE_BAND_MIDDLE: (i32, i32) = (-40, 30);
/// Vertical band for Deep systems.
pub const CAVE_BAND_DEEP: (i32, i32) = (-110, -30);
/// Inclusive range of chambers per system.
pub const CHAMBERS_PER_SYSTEM: (u32, u32) = (4, 8);
/// Range of ellipsoid semi-axis lengths for chambers, in blocks.
pub const CHAMBER_RADIUS_RANGE: (f32, f32) = (6.0, 14.0);
/// Poisson-disk minimum spacing between chamber centers, as a
/// multiple of chamber radius.
pub const POISSON_MIN_SPACING_MULT: f32 = 3.0;
/// Inclusive range of extra MST edges (loops) to add beyond the
/// minimum spanning tree.
pub const MST_EXTRA_LOOPS: (u32, u32) = (1, 2);
/// Tunnel cross-section radius, in blocks. With the carve threshold
/// at cap=1 / intensity=4, the effective carved tunnel radius is 75%
/// of this — so (3.0, 4.5) gives navigable 4.5..7 block-wide tunnels.
pub const TUNNEL_RADIUS: (f32, f32) = (3.0, 4.5);

/// Probability a chamber in the Shallow band tries to expose to the
/// surface (sinkhole / cliff mouth / skylight).
pub const ENTRANCE_PROB_SHALLOW: f32 = 0.40;
pub const ENTRANCE_PROB_MIDDLE: f32 = 0.15;
pub const ENTRANCE_PROB_DEEP: f32 = 0.05;
/// Maximum gap (blocks) between chamber top and surface for a
/// sinkhole to be geometrically possible.
pub const SINKHOLE_DEPTH_MAX: i32 = 8;
/// Maximum horizontal distance (blocks) from a chamber to a steep-
/// gradient column for a cliff mouth to be possible.
pub const CLIFF_ENTRANCE_DIST: i32 = 30;
/// Top-of-terrain buffer (blocks). Non-entrance carving is forbidden
/// inside this depth so the grass cap stays intact except at
/// deliberate openings.
pub const CAVE_SURFACE_BUFFER: i32 = 4;
/// Floor (world Y) below which caves stop carving. Keeps the loaded
/// chunk-stack bottom solid.
pub const CAVE_FLOOR_Y: i32 = -120;
/// Y below which sparse 3D-noise wormholes are layered in addition to
/// the graph systems. Wormholes only operate in the deep band so
/// shallow caves stay coherent.
pub const WORMHOLE_BAND_Y: i32 = -40;
/// Half-width of the near-zero band on the wormhole noise. Wider →
/// thicker / more frequent wormholes.
pub const WORMHOLE_BAND: f64 = 0.05;

// ── Biomes & surface ─────────────────────────────────────────────────

/// Temperature noise value below which a column is cold (Tundra /
/// SnowyForest). The climate noise is roughly distributed in
/// `[-1, 1]` and centred near 0, so `-0.30` puts ~20% of the world
/// in the cold belt — down from ~45% at the previous `-0.10` value,
/// which produced too much surface snow.
pub const COLD_THRESHOLD: f32 = -0.30;
/// Minimum height (blocks above sea level) for cold-biome
/// surface-snow to apply. Below this elevation, cold biomes still
/// get their grass/dirt surface so coastal cold regions don't put
/// a strip of snow directly against the ocean.
pub const COLD_SNOW_MIN_ABOVE_SEA: i32 = 8;
/// Humidity noise value above which a temperate column is Forest /
/// Tropical (otherwise Plains / Desert).
pub const FOREST_HUMIDITY: f32 = 0.05;
/// Amplitude of the threshold-perturbation noise used to break up
/// straight biome edges. Sampled in noise-value units.
pub const BIOME_JITTER_AMPL: f32 = 0.05;
/// Spatial period of the threshold-perturbation noise.
pub const BIOME_JITTER_PERIOD: f32 = 24.0;
/// Width (in noise-value units, not blocks) of the stochastic
/// sand/grass transition band on the grass side of a desert
/// boundary. Roughly 4–8 blocks in spatial width depending on local
/// desert-noise gradient.
pub const SAND_TRANSITION_BAND: f32 = 0.05;
/// Spatial width (blocks) over which tree density is interpolated
/// across a biome boundary.
pub const TREE_BLEND_WIDTH: f32 = 12.0;
/// World Y at and above which surface columns are capped with Snow
/// regardless of biome (alpine snow line).
pub const SNOW_LINE: i32 = 110;

// ── Trees ────────────────────────────────────────────────────────────

pub const TREE_RATE_PLAINS: u32 = 12;
pub const TREE_RATE_FOREST: u32 = 55;
pub const TREE_RATE_SNOWY_FOREST: u32 = 55;
pub const TREE_RATE_TROPICAL: u32 = 65;
pub const TREE_CELL_SIZE: i32 = 8;
pub const TREE_MARGIN: i32 = 5;

// ── Region cache caps ────────────────────────────────────────────────

pub const FINE_CACHE_CAP: usize = 256;
pub const MACRO_CACHE_CAP: usize = 32;

// ── Derived helpers ──────────────────────────────────────────────────

/// Number of fine cells along one edge of a fine region.
pub const FINE_CELLS_PER_REGION: i32 = FINE_REGION_SIZE / FINE_CELL;
/// Number of macro cells along one edge of a macro region.
pub const MACRO_CELLS_PER_REGION: i32 = MACRO_REGION_SIZE / MACRO_CELL;
/// Number of fine cells per macro cell along one axis.
pub const FINE_PER_MACRO: i32 = MACRO_CELL / FINE_CELL;
