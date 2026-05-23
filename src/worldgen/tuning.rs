//! All tunable worldgen constants in one place.
//!
//! The new worldgen has enough knobs that scattering them across modules
//! would make tuning a treasure hunt. Anything you might want to nudge to
//! reshape the world lives here.
//!
//! ### The tuning.rs / default.ron boundary
//!
//! Two configuration surfaces exist and they serve different audiences:
//!
//! - **`default.ron`** (`assets/worldgen/default.ron`) holds
//!   *hot-reloadable world-character knobs*: terrain splines, biome
//!   hyperbox entries, noise channel amplitudes, cave style weights.
//!   A designer can edit this file while the engine is running; the file
//!   watcher swaps the `Arc<WorldgenConfig>` and newly generated chunks
//!   pick up the change. Put a new knob in `default.ron` if a designer
//!   could tune it live without recompiling.
//! - **`tuning.rs`** (this file) holds *compile-time architectural
//!   invariants*: constants that size data structures, drive coordinate
//!   calculations, or set hard physical limits. These values affect
//!   cache layouts and algorithm correctness; changing them requires a
//!   recompile and may invalidate the fingerprint hash. Put a value here
//!   if changing it at runtime could cause undefined behaviour, cache
//!   corruption, or determinism violations.
//!
//! When in doubt: if the value is "how many chunks fit in an LRU" or
//! "how many blocks per region" it belongs here. If it's "how steep
//! does a slope have to be to be a cliff" it belongs in `default.ron`.
//!
//! See `docs/superpowers/specs/2026-05-20-worldgen-docs-design.md` for
//! the full rationale.

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

/// Per-plate variation used as an additive bias on the climate
/// `terrain_shape` channel (post-PR-3). Range preserved so existing
/// plate rolls stay deterministic; `heightmap::plate_roughness_bias`
/// maps this range to `ClimateConfig::plate_roughness_bias_range`.
pub const ROUGHNESS_RANGE: (f32, f32) = (0.7, 1.4);

// ── Heightmap ─────────────────────────────────────────────────────────

/// World-space Y at which the sea surface sits.
pub const SEA_LEVEL: i32 = 62;
/// Slope (in blocks-per-block) above which a column is exposed as a
/// cliff: surface block becomes `Stone` and the dirt sub-surface is
/// skipped. Measured over an ±4-block stencil so small-scale noise
/// jitter doesn't register.
pub const CLIFF_SLOPE_THRESH: f32 = 2.2;
/// Hard upper cap on final terrain Y. Keeps tallest peaks inside the
/// vertical chunk-load radius.
pub const MAX_TERRAIN_Y: i32 = 140;

// ── 3D density ──────────────────────────────────────────────────────

/// Half-width of the band around `h_target` used by the
/// `topmost_solid` helper that finds tree-trunk anchor points.
/// `fill_chunk` no longer short-circuits density evaluation by this
/// band (3D density is evaluated at every voxel so overhangs can
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
pub const RIVER_THRESH: u32 = 24;
/// Width-law scale: `width = clamp(sqrt(acc) * scale, min, max)`.
pub const RIVER_WIDTH_SCALE: f32 = 0.30;
/// Minimum river width at threshold drainage.
pub const MIN_RIVER_WIDTH: f32 = 3.0;
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
/// Minimum guaranteed carve depth below water surface for lake beds.
/// Ensures at least this many blocks of open water above the terrain floor.
pub const MIN_LAKE_BED_DROP: i32 = 3;
/// Minimum natural basin depth (h_fill − h, in fine-cell samples) required
/// to classify a sink-fill cell as a lake. 1-block-deep "scratch" basins
/// from sink-fill are excluded; they look like unnatural flat puddles and
/// the Step-5 carving would make them visibly deep even though the basin
/// is topographically insignificant.
pub const LAKE_MIN_NATURAL_DEPTH: i32 = 2;

// ── Cave pools ────────────────────────────────────────────────────────

/// Minimum XZ semi-axis (blocks) for a chamber to qualify for a pool.
/// Chambers smaller than this are too tight to look like a real pool.
pub const POOL_MIN_RADIUS_XZ: f32 = 5.0;
/// Fraction of the chamber's vertical span filled by the pool. 0.30
/// means the pool surface sits 30% of the way up from the chamber floor,
/// leaving headroom above.
pub const POOL_SURFACE_FRACTION: f32 = 0.30;
/// Minimum gap (blocks) between the pool surface and the surface of the
/// world above. Pools must be fully underground.
pub const POOL_TOP_CLEARANCE: i32 = 4;
/// Probability a qualifying deep chamber becomes a lava pool rather than
/// a water pool.
pub const POOL_LAVA_PROB: f32 = 0.35;

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

// ── Fall-hazard clamp ─────────────────────────────────────────────────

/// Maximum consecutive vertical air voxels per XZ column before a stone
/// "ledge" is inserted by the post-density-fill pass. Eliminates
/// fall-to-death drops.
///
/// **Temporary value:** raised from 6 to 256 to verify that the 8-block
/// stone-ledge artifact seen in early cave testing was caused by this
/// clamp (it was — the clamp was firing inside shallow cave entrances and
/// pasting stone slabs across open chambers). The correct fix is to
/// tighten the surface-buffer gating in the cave carver rather than
/// disabling the safety net entirely. Once the over-carving root cause is
/// confirmed fixed, this should be restored to a small value (6–16 blocks)
/// so truly vertical free-fall shafts still get a ledge inserted.
pub const MAX_VERTICAL_AIR_RUN: i32 = 256;

// ── Caves ────────────────────────────────────────────────────────────

/// Inclusive range of cave systems rolled per fine region.
/// Was: (0, 0) — graph caves disabled.
/// Now: (0, 3) per cave-overhaul spec defaults. The configured upper bound
/// is also exposed in CaveConfig.systems_per_region_max for hot reload.
pub const CAVE_SYSTEMS_PER_REGION: (u32, u32) = (0, 3);
/// Vertical band (inclusive both ends) for Shallow systems.
pub const CAVE_BAND_SHALLOW: (i32, i32) = (10, 50);
/// Vertical band for Middle systems.
pub const CAVE_BAND_MIDDLE: (i32, i32) = (-40, 30);
/// Vertical band for Deep systems.
pub const CAVE_BAND_DEEP: (i32, i32) = (-110, -30);
/// Inclusive range of chambers per system. Small clusters feel
/// like discrete rooms-connected-by-passages instead of mega
/// dungeons.
pub const CHAMBERS_PER_SYSTEM: (u32, u32) = (2, 4);
/// Range of ellipsoid semi-axis lengths for chambers, in blocks.
/// 3.5..5.5 → 7..11-block-diameter chambers — MC-parity room sizes.
pub const CHAMBER_RADIUS_RANGE: (f32, f32) = (3.5, 5.5);
/// Poisson-disk minimum spacing between chamber centers, as a
/// multiple of chamber radius.
pub const POISSON_MIN_SPACING_MULT: f32 = 3.0;
/// Inclusive range of extra MST edges (loops) to add beyond the
/// minimum spanning tree.
pub const MST_EXTRA_LOOPS: (u32, u32) = (1, 2);
/// Tunnel cross-section radius, in blocks. (2.0, 3.0) gives
/// 4-6 block-wide passages — walkable corridors connecting rooms.
pub const TUNNEL_RADIUS: (f32, f32) = (2.0, 3.0);

/// Probability a chamber in the Shallow band tries to expose to the
/// surface (sinkhole / cliff mouth / skylight).
pub const ENTRANCE_PROB_SHALLOW: f32 = 0.70;
pub const ENTRANCE_PROB_MIDDLE: f32 = 0.25;
pub const ENTRANCE_PROB_DEEP: f32 = 0.05;
/// Maximum gap (blocks) between chamber top and surface for a
/// sinkhole to be geometrically possible.
pub const SINKHOLE_DEPTH_MAX: i32 = 8;
/// Maximum horizontal distance (blocks) from a chamber to a steep-
/// gradient column for a cliff mouth to be possible.
pub const CLIFF_ENTRANCE_DIST: i32 = 30;
/// Top-of-terrain buffer (blocks). Cheese, terasology, chamber, and trunk
/// carvers fire only at depth > BUFFER below the heightmap. The entrance SDF
/// (sinkholes, skylights, cliff mouths) is not gated by this value — it uses
/// its own `wy <= height + SURFACE_BAND` gate so intentional cave openings
/// still reach the surface regardless of this setting.
/// Was -1 (surface-tube openings allowed) until the entrance SDF was given
/// its own gate; raised to 8 so ambient noise can't eat through the surface.
pub const CAVE_SURFACE_BUFFER: i32 = 8;
/// Floor (world Y) below which caves stop carving. Keeps the loaded
/// chunk-stack bottom solid.
pub const CAVE_FLOOR_Y: i32 = -120;
/// Lateral spread (in blocks) for cave-surface block displacement when
/// a cave breaches the heightmap. Terasology default is 3.
pub const SURFACE_SPREAD: i32 = 3;

// ── Biomes & surface ─────────────────────────────────────────────────

// PR 4: COLD_THRESHOLD, FOREST_HUMIDITY, BIOME_JITTER_*,
// SAND_TRANSITION_BAND are gone — the if-else biome classifier
// they parameterised is replaced by the R-tree lookup in
// climate.rs, and the boundaries are softened by per-block
// hash-Voronoi jitter (climate::voronoi_jitter_offset). The
// `BiomesConfig::entries` table in default.ron stakes biome
// claims directly.

/// Minimum height (blocks above sea level) for cold-biome
/// surface-snow to apply. Below this elevation, cold biomes still
/// get their grass/dirt surface so coastal cold regions don't put
/// a strip of snow directly against the ocean.
pub const COLD_SNOW_MIN_ABOVE_SEA: i32 = 8;
// PR 6: SAND_TRANSITION_BAND is gone. The stochastic sand/grass
// transition is now expressible as `SandTransitionRoll` inside the
// surface rule tree (`assets/worldgen/default.ron`); the default
// tree doesn't use it, but the DSL primitive exists.
/// Spatial width (blocks) over which tree density is interpolated
/// across a biome boundary.
pub const TREE_BLEND_WIDTH: f32 = 12.0;
/// World Y at and above which surface columns are capped with Snow
/// regardless of biome (alpine snow line).
pub const SNOW_LINE: i32 = 110;

// ── Trees ────────────────────────────────────────────────────────────

/// Trees per 1000 cells in plains biome (sparse, open feel).
pub const TREE_RATE_PLAINS: u32 = 12;
/// Trees per 1000 cells in temperate forest (dense canopy).
pub const TREE_RATE_FOREST: u32 = 55;
/// Trees per 1000 cells in cold/snowy forest.
pub const TREE_RATE_SNOWY_FOREST: u32 = 55;
/// Trees per 1000 cells in tropical biome.
pub const TREE_RATE_TROPICAL: u32 = 65;
/// XZ edge length of one tree placement cell (blocks). Each cell
/// independently rolls whether to host a tree; smaller → denser
/// but also more hash queries per chunk fill.
pub const TREE_CELL_SIZE: i32 = 8;
/// Inset margin (blocks) from the cell boundary inside which the
/// trunk position is jittered. Keeps trunks away from cell edges
/// so cross-chunk leaf stampings don't extend farther than the
/// scan radius accounts for.
pub const TREE_MARGIN: i32 = 5;

// ── Region cache caps ────────────────────────────────────────────────

/// Maximum number of `FineRegion` entries retained in the LRU.
/// At ~80 KB per entry this is ~20 MB peak fine-cache footprint. The
/// cap is sized to cover the typical streaming radius (16 chunks ≈ 512
/// blocks ≈ 1 fine region diameter) with comfortable headroom for the
/// 3×3 neighbourhood each chunk fill prefetches.
pub const FINE_CACHE_CAP: usize = 256;
/// Maximum number of `MacroRegion` entries retained in the LRU.
/// Macro regions are 16× larger in area but the cache only needs a
/// handful of entries — the macro horizon is 1 halo (24 km), so a
/// single player session rarely needs more than a few macro regions.
pub const MACRO_CACHE_CAP: usize = 32;

// ── Derived helpers ──────────────────────────────────────────────────

/// Number of fine cells along one edge of a fine region.
pub const FINE_CELLS_PER_REGION: i32 = FINE_REGION_SIZE / FINE_CELL;
/// Number of macro cells along one edge of a macro region.
pub const MACRO_CELLS_PER_REGION: i32 = MACRO_REGION_SIZE / MACRO_CELL;
/// Number of fine cells per macro cell along one axis.
pub const FINE_PER_MACRO: i32 = MACRO_CELL / FINE_CELL;
