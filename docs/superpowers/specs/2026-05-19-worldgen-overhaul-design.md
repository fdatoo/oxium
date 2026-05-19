# Worldgen overhaul — Design Spec

**Date:** 2026-05-19
**Status:** Approved, ready for implementation planning
**Branch:** TBD (will be the implementation branch for PR 1; subsequent PRs branch from the previous merge)

## Summary

Replace the single-file `worldgen/mod.rs` with a layered pipeline driven by a deterministic, lazy **region cache**. The new system produces:

- **Continental geography with archipelagos.** A Voronoi plate decomposition assigns each world region to a continental or oceanic plate; continent/ocean mask and mountain spine placement derive from the plate model. Continents cover ~30–45% of the world; oceanic plate boundaries produce island arcs.
- **Real river networks.** D8 flow accumulation on a coarse heightmap at two resolutions (fine 8 m cells, macro 64 m cells) produces drainage from headwaters to coasts. Width scales with `√drainage_area`; small streams are 2 blocks, trunk rivers near continental mouths reach ~30 blocks. Rivers carve U-profile valleys into the heightmap.
- **Coherent cave systems.** Each region rolls 1–4 cave systems; chambers connected by an MST plus 1–2 loops; tunnels are domain-warped spline curves. Three classes of surface entrance (sinkhole, cliff mouth, skylight) expose select systems to daylight. Sparse 3D-noise "wormholes" supplement only in the deep band (`y < −40`).
- **Climate-driven biomes with smooth edges.** Six biomes (existing five plus `Tropical`); threshold perturbation jitters biome boundaries; sand/grass edges get a stochastic transition band; tree-rate interpolation across boundaries removes density steps.

Rollout: full in-place replacement of the existing `worldgen` module across six PRs. Existing saves continue to load — chunks already on disk stay in v1 terrain; newly-streamed chunks adopt v2 geography (a one-time terrain seam at the boundary of previously-explored areas, accepted).

A small **save manifest** addition (`saves/<name>/world.toml`, storing `seed`, `worldgen_version`, `created_at`) lands as part of PR 1, replacing the hard-coded `seed = 42` in `app.rs`.

## Goals

1. Continents, oceans, and island archipelagos — driven coherently from one underlying structure (Voronoi plates).
2. Mountain ranges along continental plate boundaries; chains, not bumps.
3. River network that truly flows downhill, with broad meandering trunk rivers near continental coasts and narrow headwaters near mountains.
4. River-carved valleys (continuous, not noise-decorative).
5. Cave systems with explicit chamber-and-tunnel topology and surface entrances; not Swiss cheese.
6. Smooth biome edges; coastlines that read as coastlines.

## Non-goals

- Backward compatibility with v1 terrain in pre-existing saves (one-time terrain seam accepted).
- Erosion simulation (we approximate via river-carved valleys and warped FBM relief; no iterative process).
- Configurable world bounds (world is still effectively infinite — the macro cache extends as the player moves).
- Multiplayer / shared-server worlds.
- Networked save synchronisation.
- Mods / data-driven biome or block configuration.
- New ores, structures (villages, dungeons), or ambient creatures.

## Architecture

### Module layout

```
src/worldgen/
├── mod.rs          # public API: Generator::new, fill_chunk, SEA_LEVEL.
├── region.rs       # RegionCoord, FineRegion, MacroRegionCoord, MacroRegion, LruCaches.
├── plates.rs       # Voronoi plate decomposition. plate_at(seed, world_xz) -> (PlateId, PlateId, boundary_t).
├── climate.rs      # Temperature/humidity noise, Biome classifier with threshold perturbation.
├── heightmap.rs    # h_pre (plates + ridges + warped FBM) and final height (post-river-carve), plus cliff detection.
├── hydrology.rs    # D8 flow at fine + macro resolution. River segments, lake rims, valley carve.
├── caves.rs        # CaveSystem (chamber MST + spline tunnels), surface entrances, carve composition.
├── surface.rs      # Per-column surface block selection: cliffs, beaches, snow, biome materials.
├── trees.rs        # Tree placement (per-cell deterministic). TreeKind { Oak, Palm }.
└── tuning.rs       # All constants. One module so tuning is one file.
```

`Generator` is the only public type. It holds:

```rust
pub struct Generator {
    seed: u64,
    fine_cache:  Arc<Mutex<LruCache<RegionCoord,      Arc<FineRegion>>>>,
    macro_cache: Arc<Mutex<LruCache<MacroRegionCoord, Arc<MacroRegion>>>>,
    // noise fields and per-axis salt offsets behind it (private)
}

impl Generator {
    pub fn new(seed: u64) -> Self;
    pub fn fill_chunk(&self, coord: ChunkCoord, out: &mut DenseChunk);
    pub fn seed(&self) -> u64;
}
```

Public surface is unchanged in spirit from v1; callers in `app.rs`, persistence, and tests need only the rename-aware imports.

### Build pipeline (per fine region)

```
1. plates.rs           : evaluate plate_at over coarse samples in region + 2-region halo
2. climate.rs          : sample temperature / humidity (direct noise, not cached)
3. heightmap.rs::h_pre : plate base + plate-edge ridges + warped FBM relief over halo
4. macro_cache lookup  : trunk river accumulation from macro pass (recursive macro build if cold)
5. hydrology.rs        : fine D8 + sink fill + flow accumulation, injecting trunk accumulations
6. caves.rs            : roll cave systems for the region, build chamber graphs and tunnel splines
                         (the systems are stored; carving happens at chunk fill, not region build)
```

### Per-chunk fill

For each chunk `coord`:

1. Fetch the fine region containing the chunk.
2. For each column in the chunk:
    - Recompute `h_pre` from noise (cheap; not cached at column resolution).
    - Compute final height = `h_pre − valley_carve(world_xz, region.river_network)`.
    - Compute slope (`|∇h_pre|`) for cliff detection.
    - Look up biome from climate at column.
    - Pick surface block (cliffs > beach > snow line > biome cold cap > biome material).
    - Fill sub-surface (dirt/stone with the cliff exception).
3. For each cave system whose bounding box intersects the chunk:
    - Carve chambers (ellipsoid SDF), tunnels (capsule-along-spline SDF), respecting the surface buffer.
    - Carve entrance shafts/mouths (which *do* punch through the buffer).
4. In the deep band (`y < WORMHOLE_BAND_Y`), carve sparse 3D-noise wormholes.
5. Sea-level flood: any air cell at or below `SEA_LEVEL` becomes water.
6. Add trees on top via the existing per-cell deterministic pass.

## Plate model (`plates.rs`)

Each world cell of `PLATE_CELL_SIZE` (default 1024 blocks) rolls one Voronoi seed at a hashed jitter offset inside the cell. To find the plate at `world_xz`, scan the 3×3 surrounding cells (current + 8 neighbours), pick the nearest seed → `plate_id_a`, then second-nearest → `plate_id_b`.

**Per-plate properties** (deterministic from `hash(seed, plate_id)`):

- `kind: Continental | Oceanic`. `CONTINENTAL_RATIO = 0.45`.
- `base_elevation: f32`. Continental: `+18..+28` (sea-level frame). Oceanic: `−50..−20`.
- `roughness: f32`. `0.7..1.4`. Per-plate amplitude multiplier for base FBM relief.

**Boundary intensity** at `world_xz`: `t = (d_b − d_a) / (d_b + d_a)`. Smooth field, 0 on a plate boundary, 1 deep inside a plate.

**Plate-edge ridges** apply when `t < BOUNDARY_RIDGE_WIDTH` (default `0.12`) AND at least one of `{plate_a, plate_b}` is continental:

```
ridge_height = (1 − t / BOUNDARY_RIDGE_WIDTH) * peak
```

`peak` is modulated along the boundary curve by a 1D noise (parameterise the boundary by arc length, sample) so the chain has peaks and saddles. Maximum ridge contribution depends on the boundary type:

- Continental–continental: `RIDGE_PEAK_CC` (default `90` blocks). Tallest ranges.
- Continental–oceanic: `RIDGE_PEAK_CO` (default `54` blocks). Coastal ranges (Andes-style).
- Oceanic–oceanic: `RIDGE_PEAK_OO` (default `28` blocks). Island arc archipelagos.

**Continental shelf taper.** Base elevation contribution itself smoothly blends across the boundary by lerping `plate_a.base_elevation` and `plate_b.base_elevation` using `t`. Continental shelves emerge naturally as the elevation transitions from oceanic basin → coastal slope → continent.

## Heightmap (`heightmap.rs`)

### `h_pre` — pre-river heightmap

```
h_pre(wx, wz) = SEA_LEVEL
              + lerp(plate_a.base, plate_b.base, weight_from_t)
              + ridge_lift(t, plate_pair)
              + warped_fbm(wx, wz) * plate_a.roughness
```

`warped_fbm` is the trick for "continuous valleys" independent of rivers. A low-frequency vector field at `(wx, wz)` (sampled via two 2-octave noises) offsets the input coordinates before the height FBM eval. Amplitude `WARP_AMPLITUDE ≈ 40` blocks; period `~400` blocks. Breaks rounded-blob FBM signature; produces finger ridges and curved depressions.

### `h_final` — post-river-carve heightmap

```
h_final(wx, wz) = h_pre(wx, wz) − valley_carve(wx, wz, region.river_network)
```

`valley_carve` finds the nearest river segment in the region (via a small kd-tree built per region), gets the perpendicular distance to the centerline, gets the segment's width, returns a U-profile depth:

```
let d = distance_to_centerline_with_meander_warp(wx, wz, segment)
let half_width  = segment.width * 0.5
let half_valley = segment.width * VALLEY_HALF_WIDTH_MULT   // default 3.0
let depth = if d <= half_width {
    RIVER_BED_DEPTH                                         // flat bottom carved fully
} else if d < half_valley {
    let t = (d - half_width) / (half_valley - half_width)
    RIVER_BED_DEPTH * smoothstep(1.0, 0.0, t)
} else { 0 }
```

### Cliffs

Per-column: sample `h_pre` at `(wx ± 2, wz)` and `(wx, wz ± 2)`, compute `|∇h|` as the max delta. If `|∇h| > CLIFF_SLOPE_THRESH` (default `1.5` blocks/block), the surface is `Stone` (replaces v1's `MOUNTAIN_ROCK_LINE` rule). Below the surface, no dirt — `Stone` continues immediately.

### Hard cap

`h_final` is clamped to `[CAVE_FLOOR_Y + 8, MAX_TERRAIN_Y]`. `MAX_TERRAIN_Y` default `140` (was `136` in v1; +4 to give plate-edge ridges some headroom).

## Hydrology (`hydrology.rs`)

### Fine resolution

- **Cell size:** 8 blocks.
- **Region size:** 512 blocks → 64 × 64 fine samples per region.
- **Halo:** 2 regions. Build window 5 × 512 = 2560 blocks. Sees ~6.5 km² of drainage.
- **Sink-fill horizon:** ~2.5 km. Inland lakes up to this size fill correctly.

Algorithm per region:

1. Evaluate `h_pre` at every fine cell across the 5 × 5 region-window.
2. D8 flow direction at every cell (steepest downhill of 8 neighbours; sinks tagged).
3. Planchon–Darboux sink fill bounded to the window — sinks fill until they spill, but only locally; basins larger than the window may terrace at boundaries (handled by macro pass for large basins).
4. Inject trunk accumulations from the macro cache (see below).
5. Flow accumulation: topological sort by elevation descending; each cell donates its area (1 + injected trunk amount) to its downstream neighbour. O(N).
6. Tag river cells (`acc ≥ RIVER_THRESH`, default 50).
7. Compute width: `width = clamp(sqrt(acc) * RIVER_WIDTH_SCALE, MIN_WIDTH, MAX_WIDTH)`.

### Macro resolution (trunk pass)

- **Cell size:** 64 blocks (8× fine).
- **Region size:** 8192 blocks → 128 × 128 macro samples per macro region.
- **Halo:** 1 macro region. Build window 3 × 8192 = 24.5 km. Sees ~600 km² of drainage.

Same five steps at coarse resolution; trunk cells flagged where `macro_acc ≥ MACRO_RIVER_THRESH` (default ~500 macro cells ≈ 2000 km² drainage). Macro sink-fill handles continental-scale endorheic basins; fine regions inherit lake rim elevations from macro.

### Composition: macro trunk into fine accumulation

When the fine region build runs flow accumulation, before tagging river cells:

1. For each macro cell inside region + halo flagged as trunk, find the fine cell at its center.
2. Inject `macro_acc * 64` as starting accumulation at that fine cell (rescaling units).
3. Proceed with normal fine D8 accumulation.

Result: trunk flow propagates through the fine region via the local heightmap, so trunk rivers meander at fine resolution while preserving massive drainage. Maximum credible river width climbs from ~16 blocks (fine-only) to ~30 blocks (with trunk injection).

### Centerline reconstruction

The D8 path between cell centers is piecewise-linear (angular). When `valley_carve` queries a column's distance to the nearest river, it uses a **perturbed centerline**: the straight cell-center-to-cell-center segment plus a domain-warped offset sampled at `world_xz`. Meander amplitude scales with width:

```
meander_amp = clamp(width * MEANDER_AMP_PER_WIDTH, 0, MAX_MEANDER_AMP)
```

Default `MEANDER_AMP_PER_WIDTH = 0.4` → trunk rivers meander hard (12+ block lateral offsets), streams stay nearly straight.

### Mouths

When flow reaches a cell with `h_pre ≤ SEA_LEVEL`, accumulation stops. The cell is flagged `mouth = true` and gets `width × MOUTH_FLARE_MULT` (default `1.6`) so deltas read as flared.

### What the fine region caches

```rust
struct FineRegion {
    flow_acc:    Box<[u32; 64*64]>,
    flow_dir:    Box<[u8;  64*64]>,    // 0..8 (8 = sink/lake)
    is_river:    BitSet,                // 64*64
    is_lake:     BitSet,                // 64*64
    width:       Box<[f32; 64*64]>,     // 0 if not river
    lake_rim:    Box<[i16; 64*64]>,     // only meaningful if is_lake
    segments:    Vec<RiverSegment>,
    segment_kd:  KdTree<RiverSegmentId>, // built lazily on first valley_carve query
    cave_systems: Vec<CaveSystem>,
}
```

Approx size, breakdown: flow_acc 16 KB + flow_dir 4 KB + 2×bitset 2 KB + width 16 KB + lake_rim 8 KB ≈ **46 KB of arrays** per region. Add `segments` (typically a few KB) and `cave_systems` (1–4 systems × ~1 KB each) → **~55–70 KB per fine region**. Cap: 256 fine regions → **~16–18 MB**.

### What the macro region caches

```rust
struct MacroRegion {
    flow_dir:    Box<[u8;  128*128]>,
    flow_acc:    Box<[u32; 128*128]>,
    is_trunk:    BitSet,
    is_lake:     BitSet,
    lake_rim:    Box<[i16; 128*128]>,
}
```

Approx size, breakdown: flow_dir 16 KB + flow_acc 64 KB + 2×bitset 4 KB + lake_rim 32 KB ≈ **~116 KB per macro region**. Cap: 32 entries → **~4 MB**. Player view radius (~1 km) → typically 1–4 macro regions resident.

**Combined cache footprint** at default caps: fine ~17 MB + macro ~4 MB ≈ **~21 MB**.

## Caves (`caves.rs`)

### System rolls per region

`hash(seed, region_coord, salt)` deterministically rolls `1..=4` cave systems per region. Each system gets:

- A depth band: `Shallow (y ∈ [10, 50])`, `Middle (y ∈ [−40, 30])`, or `Deep (y ∈ [−110, −30])`. Bands overlap.
- A bounding box (~200 × 200 × 60 blocks) placed inside the region (with allowance to spill into neighbours).

### Chamber placement

Within the bounding box, 3D Poisson-disk samples produce 4–8 chamber centers (`POISSON_MIN_SPACING = chamber_radius × 3`). Each chamber's ellipsoid radii are sampled independently in `[6, 14]` per axis — chambers are oblong.

### Tunnel topology

1. Compute MST on chamber centers (Euclidean 3D distance) → connected component.
2. Add 1–2 random extra edges past the MST so some systems have loops (cycles → "inner chamber" landmarks, multi-path navigation).
3. Each edge becomes a cubic spline: 2–4 control points placed along the chamber-to-chamber line, pushed off by domain-warped 3D noise → meandering tunnels.
4. Tunnel radius: 2–3 blocks.

### Surface entrances

Each chamber independently rolls (against `hash(seed, system_id, chamber_idx, ENTRANCE_SALT)`) whether to seek the surface. Probability by depth band:

- Shallow: `0.40`
- Middle: `0.15`
- Deep: `0.05`

If the roll succeeds, evaluate the three entrance types in order; first geometrically possible wins:

1. **Sinkhole** — if `chamber.top_y` is within `SINKHOLE_DEPTH_MAX = 8` blocks of `h_pre` directly above. Carve a vertical shaft from chamber top to surface, radius 2–3, slightly funnel-shaped at the top. The surface block ring around the rim is broken — exposed dirt/stone, no grass cap.
2. **Cliff mouth** — if a steep-gradient cell (`|∇h_pre| > CLIFF_SLOPE_THRESH`) lies within `CLIFF_ENTRANCE_DIST = 30` blocks horizontal of the chamber. Carve a horizontal tunnel exiting the chamber toward the cliff, ending in a 4–6 block opening cut into the cliff face.
3. **Skylight** — if `chamber.top_y` is 30–60 blocks below `h_pre` directly above. Carve a narrow 1–2 block shaft straight up. Aesthetic; lets light into the chamber.

If none are geometrically possible, the chamber stays buried.

### Sparse wormhole fill (deep band only)

In `y < WORMHOLE_BAND_Y` (default `−40`), a single-octave 3D noise carves scattered passages independent of any system. Narrow band: ~2-block tunnels. Default `WORMHOLE_BAND = 0.05`. Above this Y, no wormholes — shallow caves are fully driven by the graph systems for coherence.

### Carve precedence (per chunk)

1. Find cave systems whose bounding boxes intersect the chunk (small spatial check against the chunk's fine region + halo).
2. Carve chambers (ellipsoid SDF) and tunnels (capsule-along-spline SDF), **skipping cells within `CAVE_SURFACE_BUFFER` (default 4) blocks of `h_pre`** — surface stays intact except at explicit entrances.
3. Carve entrance features (shafts, cliff mouths, skylights) — these *do* punch through the surface buffer.
4. In the deep band, carve sparse wormhole noise.
5. Sea-level flood (existing behavior): cave voids at/below `SEA_LEVEL` → water.

### Cross-region systems

Systems "own" their region (the one whose seed roll produced them) but bounding boxes can spill into neighbours. At chunk-fill time, the cave layer queries: "which systems intersect this chunk?" — checks the chunk's fine region cache plus 8 neighbour caches via `get_fine`. Standard lazy fetch.

## Climate, biomes, surface materials

### Climate noise (`climate.rs`)

Two 2D Simplex FBM fields — `temperature_map`, `humidity_map` — with `~512`-block period (unchanged from v1). Sampled directly per column; not cached.

### Biome classifier (`climate.rs`)

Adds `Tropical` to the existing five-biome set. With threshold perturbation (next section):

```
cold        → Tundra | SnowyForest   (split on humidity)
hot + dry   → Desert
hot + wet   → Tropical                                  (new)
temp + dry  → Plains
temp + wet  → Forest
```

### Biome blending (Section 6.5)

**Threshold perturbation (primary).** Each biome threshold adds a small high-frequency perturbation noise at the column:

```rust
let jitter = biome_jitter_noise(wx, wz) * BIOME_JITTER_AMPL;     // ±0.05
let is_desert = (desertness + jitter)  > 0.30;
let is_cold   = (temperature - jitter) < COLD_THRESHOLD;
let is_forest = (humidity + rotated_jitter) > FOREST_HUMIDITY;
```

Same shared 2-octave Simplex with ~24-block period; different sign / axis-rotation per threshold so deserts and cold zones don't co-jitter. Costs +1 noise eval per column.

**Sand/grass transition band.** Measured in *noise-value units*, not blocks: on the grass side of the desert boundary, when `0 < (0.30 − desertness) < SAND_TRANSITION_BAND` (default `SAND_TRANSITION_BAND = 0.05`, an opaque magic number whose spatial width depends on the desert noise's local gradient — roughly 4–8 blocks in practice), the surface block is rolled stochastically. Probability of `Sand` ramps linearly from 0 at the band's outer edge to ~50% at the boundary itself. Per-column deterministic via `hash(seed, wx, wz)`. Frayed, sand-spotted grass edge instead of a clean line.

**Tree-rate interpolation.** `tree_in_cell` blends the column's biome tree rate with the nearest adjacent biome's rate using distance-to-boundary within `TREE_BLEND_WIDTH = 12` blocks. Density ramps instead of stepping.

### Surface block selection (`surface.rs`)

Priority order, per column:

1. **Cliff** (`|∇h_pre| > CLIFF_SLOPE_THRESH`) → `Stone`. Dirt subsurface skipped.
2. **Beach** (`height ∈ [SEA_LEVEL − 1, SEA_LEVEL + 2]`, biome not cold, not a cliff) → `Sand`. 4-block-tall band.
3. **Snow line** (`height ≥ SNOW_LINE`, default 110) → `Snow`.
4. **Biome cold cap** (`Tundra`, `SnowyForest`) → `Snow`.
5. **Biome material** (`Desert` → `Sand`; `Tropical` → `Grass`; `Plains` → `Grass`; `Forest` → `Grass`).

Subsurface: 3 blocks `Dirt`, then `Stone`. Cliffs: `Stone` immediately.

### Trees (`trees.rs`)

Existing per-cell deterministic placement carries over. Per-biome rate table extended:

```
Tundra | Desert        → None
Plains                 → 12
Forest                 → 55
SnowyForest            → 55
Tropical               → 65       (new)
```

**`TreeKind`** enum selects shape:

- `Oak` (existing): trunk 4–6 blocks, spherical canopy radius 2–3 (current code).
- `Palm` (new, for `Tropical`): trunk 7–9 blocks straight, no canopy until top, then a "spreading fronds" silhouette — 4–6 horizontal arms one block thick emanating from the trunk top. Block reuse: existing `Wood` + `Leaves` for v1 (palm-specific blocks deferred to a follow-up PR).

## Region cache mechanics (`region.rs`)

### Layout

```rust
pub struct Generator {
    seed: u64,
    fine_cache:  Arc<Mutex<LruCache<RegionCoord,      Arc<FineRegion>>>>,
    macro_cache: Arc<Mutex<LruCache<MacroRegionCoord, Arc<MacroRegion>>>>,
}
```

`Arc<FineRegion>` allows cheap multi-thread sharing without holding the cache lock.

### Lookup-or-build

```rust
fn get_fine(&self, coord: RegionCoord) -> Arc<FineRegion> {
    if let Some(r) = self.fine_cache.lock().get(&coord) { return r.clone(); }
    let region = Arc::new(build_fine_region(self.seed, coord, &self.macro_cache));
    self.fine_cache.lock().put(coord, region.clone());
    region
}
```

Build happens outside the lock. Duplicate-build race is accepted (two threads on the same cold key both build; second insert overwrites with identical data because the build is pure). Single-flight deduplication via `OnceCell` is an optional optimization if profiling shows it matters.

### Cross-cache dependency

Fine build calls into macro cache (read-or-build). Macro build calls only noise functions (no cache recursion). **One-way dependency, no recursion risk.**

### Halo recomputation

Fine builds need neighbouring fine regions' `h_pre` samples for the 2-region halo. Same for macro. **Halo samples are recomputed from noise**, not pulled from the cache, to avoid a 9-region (fine) / 9-macro-region cascade per build. Cheap: an extra ~10k noise evals per build.

### Eviction

Pure LRU by access time. No pinning. Evicted entries rebuild on next access — the function is pure, so output is byte-identical.

### Capacity

```
FINE_CACHE_CAP   = 256 regions  → ~17 MB
MACRO_CACHE_CAP  = 32  regions  → ~4 MB
```

Total ≈ 21 MB at defaults. Tunable in `worldgen::tuning`.

### Thread safety

Single `Mutex` per cache. Concurrency at our chunk-job count (≤ 16) does not warrant sharding. If contention shows in profiling, shard by `hash(coord) % N` later.

## Save format additions

A new tiny manifest is written at world creation and read at world load:

```
saves/<name>/world.toml
```

```toml
seed             = 12345678901234567890
worldgen_version = 2
created_at       = 1715968800
```

- `seed`: `u64`. Generated randomly at world creation (replaces the hard-coded `seed = 42` in `app.rs`).
- `worldgen_version`: `u32`. Always `2` for now. Lets future "v3" detect old worlds and either re-generate (with player consent) or refuse-to-load.
- `created_at`: `i64` Unix timestamp. Informational.

Region files remain unchanged (`bincode(PalettedChunk)` + zstd). `Block` discriminants are append-only — adding new variants in future PRs is safe.

**Behaviour with pre-existing saves.** Saves without `world.toml` get a synthetic manifest written on first open (seed = 42 for compatibility, version = 1). v1 → v2 boundary terrain seam is accepted, as discussed.

## Tuning constants (`tuning.rs`)

All magic numbers consolidate into one file. This is the table for review and quick tweaking.

```rust
// Plates
pub const PLATE_CELL_SIZE: i32         = 1024;
pub const CONTINENTAL_RATIO: f32       = 0.45;
pub const CONTINENTAL_BASE_RANGE: (f32, f32) = (18.0, 28.0);
pub const OCEANIC_BASE_RANGE:     (f32, f32) = (-50.0, -20.0);
pub const ROUGHNESS_RANGE:        (f32, f32) = (0.7, 1.4);
pub const BOUNDARY_RIDGE_WIDTH: f32    = 0.12;
pub const RIDGE_PEAK_CC: f32           = 90.0;
pub const RIDGE_PEAK_CO: f32           = 54.0;
pub const RIDGE_PEAK_OO: f32           = 28.0;

// Heightmap
pub const SEA_LEVEL: i32               = 62;
pub const WARP_AMPLITUDE: f32          = 40.0;
pub const WARP_PERIOD: f32             = 400.0;
pub const CLIFF_SLOPE_THRESH: f32      = 1.5;
pub const MAX_TERRAIN_Y: i32           = 140;

// Rivers
pub const FINE_CELL: i32               = 8;
pub const FINE_REGION_SIZE: i32        = 512;
pub const FINE_HALO_REGIONS: i32       = 2;
pub const RIVER_THRESH: u32            = 50;
pub const RIVER_WIDTH_SCALE: f32       = 0.30;
pub const MIN_RIVER_WIDTH: f32         = 1.5;
pub const MAX_RIVER_WIDTH: f32         = 32.0;
pub const RIVER_BED_DEPTH: i32         = 4;
pub const VALLEY_HALF_WIDTH_MULT: f32  = 3.0;
pub const MEANDER_AMP_PER_WIDTH: f32   = 0.4;
pub const MAX_MEANDER_AMP: f32         = 16.0;
pub const MOUTH_FLARE_MULT: f32        = 1.6;

// Trunk pass
pub const MACRO_CELL: i32              = 64;
pub const MACRO_REGION_SIZE: i32       = 8192;
pub const MACRO_HALO_REGIONS: i32      = 1;
pub const MACRO_RIVER_THRESH: u32      = 500;

// Caves
pub const CAVE_SYSTEMS_PER_REGION: (u32, u32) = (1, 4);
pub const CAVE_BAND_SHALLOW: (i32, i32)   = (10, 50);
pub const CAVE_BAND_MIDDLE:  (i32, i32)   = (-40, 30);
pub const CAVE_BAND_DEEP:    (i32, i32)   = (-110, -30);
pub const CHAMBERS_PER_SYSTEM: (u32, u32) = (4, 8);
pub const CHAMBER_RADIUS_RANGE: (f32, f32) = (6.0, 14.0);
pub const POISSON_MIN_SPACING_MULT: f32   = 3.0;
pub const MST_EXTRA_LOOPS: (u32, u32)     = (1, 2);
pub const TUNNEL_RADIUS: (f32, f32)       = (2.0, 3.0);
pub const ENTRANCE_PROB_SHALLOW: f32      = 0.40;
pub const ENTRANCE_PROB_MIDDLE:  f32      = 0.15;
pub const ENTRANCE_PROB_DEEP:    f32      = 0.05;
pub const SINKHOLE_DEPTH_MAX: i32         = 8;
pub const CLIFF_ENTRANCE_DIST: i32        = 30;
pub const CAVE_SURFACE_BUFFER: i32        = 4;
pub const CAVE_FLOOR_Y: i32               = -120;
pub const WORMHOLE_BAND_Y: i32            = -40;
pub const WORMHOLE_BAND: f64              = 0.05;

// Biomes & surface
pub const COLD_THRESHOLD: f32         = -0.10;
pub const FOREST_HUMIDITY: f32        = 0.05;
pub const BIOME_JITTER_AMPL: f32      = 0.05;
pub const BIOME_JITTER_PERIOD: f32    = 24.0;
pub const SAND_TRANSITION_BAND: f32   = 0.05;   // noise-value units, not blocks
pub const TREE_BLEND_WIDTH: f32       = 12.0;
pub const SNOW_LINE: i32              = 110;

// Trees
pub const TREE_RATE_PLAINS:       u32 = 12;
pub const TREE_RATE_FOREST:       u32 = 55;
pub const TREE_RATE_SNOWY_FOREST: u32 = 55;
pub const TREE_RATE_TROPICAL:     u32 = 65;
pub const TREE_CELL_SIZE: i32         = 8;
pub const TREE_MARGIN: i32            = 5;

// Cache caps
pub const FINE_CACHE_CAP:  usize      = 256;
pub const MACRO_CACHE_CAP: usize      = 32;
```

## Testing strategy

### Tests that survive intact

- `fill_is_deterministic` — `(seed, coord) → chunk` is pure.
- `different_seeds_differ` — distinct seeds produce distinct chunks.
- `chunk_at_sea_level_has_water_or_solid` — basic sanity.

### Tests updated for new generator

- `golden_seed42_chunk_0_2_0` — re-baselined in each PR that changes the per-chunk output.
- `ridged_mountains_respect_height_cap` — cap moved to `MAX_TERRAIN_Y`.
- `underground_chunk_has_both_caves_and_solid` — numerical floors adjusted.
- `all_biomes_appear_in_a_large_scan` — adds `Tropical` to the expected set.
- `rivers_or_lakes_carve_inland_water` — still wants inland water; easier to satisfy with the new generator.
- `cold_biome_caps_with_snow` — unchanged.

### New tests by subsystem

**Plates (`plates.rs`):**

- `plate_at_is_pure` — same `(seed, xz)` → same plate ID.
- `continental_oceanic_ratio_within_5pct` — over a 4096² scan, continental fraction = `CONTINENTAL_RATIO ± 0.05`.
- `archipelago_islands_present` — in an oceanic-plate scan, ≥1 column above sea level due to ocean-ocean ridge contribution.

**Heightmap (`heightmap.rs`):**

- `warped_fbm_breaks_axis_symmetry` — `h_pre(x, z) ≠ h_pre(z, x)` somewhere. Catches accidental drop of domain warp.
- `cliff_threshold_exposes_stone` — synthesise a steep-gradient column with a fixed seed; assert surface is `Stone`.

**Hydrology (`hydrology.rs`):**

- `flow_path_terminates` — pick a river cell; follow `flow_dir`; reaches lake / ocean mouth / leaves region in `< 32` steps.
- `river_width_non_decreasing_downstream` — along any river path, width never decreases.
- `trunk_river_appears_in_continental_scan` — at least one trunk river cell with `macro_acc > MACRO_RIVER_THRESH` in a continental scan.
- `lake_surface_continuous_across_regions` — lake straddling a region boundary: rim elevation is identical sampled from both regions.

**Caves (`caves.rs`):**

- `cave_system_is_connected` — generated system's MST graph is one connected component.
- `surface_buffer_preserved_for_non_entrance` — over chambers without entrance flags, surface blocks above stay solid.
- `sinkhole_punches_through` — a chamber seeded to roll a sinkhole produces an air column from chamber top to surface.

**Region cache (`region.rs`):**

- `build_fine_region_is_pure` — two builds with same `(seed, coord)` produce byte-identical `FineRegion`.
- `chunk_is_cache_transparent` — same chunk filled with cold cache vs warm cache → identical blocks.
- `eviction_correctness` — fill past capacity; assert oldest entries dropped, freshly-inserted entries kept.

**Cross-region continuity:**

- `chunk_straddles_region_boundary_smooth` — adjacent columns across a region boundary: `|h_a − h_b| ≤ 2`.
- `river_path_agrees_at_region_boundary` — a river crossing a region boundary has the same fine-cell path from both regions' computations.

### Bench targets (not CI tests)

In `tests/bench/worldgen_bench.rs` as Criterion benchmarks (or equivalent):

- Cold-start single-chunk fill (cache empty): target `< 50ms`.
- Warm chunk fill: target `< 5ms`.
- 64-chunk region warmup: target `< 1s`.

Documented as alarm bells; not pass/fail gates.

### Map fingerprint test

Single PNG-hash fingerprint at a fixed seed: render a 256×256 top-down view of `h_final` at world coordinates `[−1024, 1024]² → 8-block stride`, encode as PNG, hash the bytes, pin the hash. PNG is regenerated locally for visual review when the hash mismatches. Catches macro-scale drift that single-chunk hashes miss.

## PR sequence

Even at "full in-place replacement", this is too big for one PR. Six PRs, each landing a coherent piece that compiles and passes tests on its own. Intermediate worldgen states between PRs may look weird (the rewrite is half-done) — accepted given there's no v2-flag fallback.

### PR 1 — Module split + region cache scaffolding + save manifest

- Split `worldgen/mod.rs` into the file layout above.
- Add `region.rs` with empty `FineRegion`/`MacroRegion` structs, the LRU caches on `Generator`, the `get_fine`/`get_macro` API. Caches start unused.
- Plate module added with `plate_at(seed, xz)` as a pure function. Not consumed yet.
- All existing tests still pass; existing heightmap/river/cave logic moved verbatim into intermediate `legacy_*` functions in the new files.
- **Save manifest:** `world.toml` written at world creation; read at world load; `seed` plumbed through `app.rs`.

### PR 2 — New heightmap (plates + warped FBM + cliffs)

- `heightmap.rs` builds `h_pre` from plate base + plate-edge ridges + warped FBM.
- Cliff detection in `surface.rs`; surface block selection switches to slope-driven rule.
- Old `mountain_noise` / `mountainness_map` / `MOUNTAIN_ROCK_LINE` removed.
- Rivers / lakes / biomes still on legacy paths — no carved valleys yet.
- `golden_seed42_chunk_0_2_0` re-baselined; bench timings recorded.

### PR 3 — Hydrology (macro + fine river network, valley carve)

- `hydrology.rs` builds D8 flow accumulation at fine and macro levels.
- `MacroRegion` populated with trunk pass.
- `FineRegion` populated with river segments, lake flags, rim elevations.
- `heightmap::h_final` subtracts valley carve.
- Old river / lake noise removed.
- New tests added: flow path termination, width monotonicity, trunk presence, lake continuity across regions.

### PR 4 — Caves (graph-based systems + surface entrances + deep wormhole fill)

- `caves.rs` rolls cave systems per region, generates chamber MSTs + 1–2 loops, spline tunnels, entrance features.
- Carve precedence rules wired into chunk fill.
- Surface buffer enforced for non-entrance carving.
- Old `tunnel_a` / `tunnel_b` / `cavern_noise` removed.
- Wormhole fill applied only to deep band.
- New tests added: connectivity, surface buffer preservation, sinkhole punch-through.

### PR 5 — Biome refresh (Tropical + blending + palm-shape trees)

- `climate.rs` and `surface.rs` add `Tropical`, threshold perturbation, sand/grass transition band, tree-rate interpolation.
- `trees.rs` adds `TreeKind::Palm` with the spreading-fronds silhouette (block-reused from `Wood` + `Leaves`).
- `all_biomes_appear_in_a_large_scan` updated for `Tropical`.

### PR 6 — Cleanup, tuning pass, map fingerprint test

- Remove all `legacy_*` shims. `mod.rs` becomes the thin public-API file.
- All constants consolidated into `worldgen::tuning`.
- Final tuning pass for visual quality.
- Add map fingerprint test (PNG hash) for macro-scale drift.
- Final `golden_seed42_chunk_0_2_0` re-baseline.

### Follow-up PR (post-merge of PR 6): palm-tree blocks

- Add `PalmLog` and `PalmFronds` variants to `Block` (append-only).
- Add atlas tile entries: palm bark, palm log top, palm fronds.
- Update `BlockRegistry` with the new entries.
- `trees.rs::stamp_palm` switches from `Wood`/`Leaves` to the dedicated blocks.

## Risks and open questions

### Risks

1. **Bench regression.** Cold-start chunk fill is currently `<10ms` (estimated). New cold-start touches a macro region build (~10k macro samples + flow accumulation) plus a fine region build (~25k fine samples + flow accumulation + cave system rolls). Target `<50ms`. If we miss, the player will see brief stalls when crossing into virgin macro regions. Mitigations available: smaller macro regions, parallelize macro/fine builds, warm-ahead-of-player.

2. **Continental-scale endorheic basins terrace.** Basins > ~25 km across the macro horizon will still produce a soft terrace at macro-region boundaries. Rare in practice (plate-driven geography produces continental tilt toward coasts), but not impossible. Documented limitation.

3. **MST loop edges occasionally cross tunnels.** Two tunnels at the same depth in the same system can occupy overlapping volumes. Visually fine — appears as a junction — but the carve logic will produce slightly larger voids at intersections. Acceptable; no fix planned.

4. **First-time-load latency.** Loading an existing save built with v1 will write a synthetic `world.toml` on first open (`seed = 42`, `version = 1`). Player won't notice; we should log it once.

### Open questions deferred to implementation

- **Macro cache: shared across `Generator` instances vs per-instance?** Default: per-instance (the simplest invariant — `Generator::new(seed)` is the only state). Revisit if tests show value in pooling.
- **`segment_kd` lazy vs eager.** Eager build per region adds milliseconds; lazy adds latency to the first `valley_carve` query. Default: lazy. Re-evaluate after profiling.
- **Domain warp shape for tunnel splines.** Default: 2-octave Simplex with ~24-block period, amplitude scaling with tunnel radius × 1.5. Likely to need tuning during PR 4 by visual review.

## Future work (not in scope)

- Palm-specific blocks and texture assets (follow-up PR above).
- Underwater caves with distinct biome treatments (reefs, kelp, etc.).
- Structures: villages, dungeons, ruins. Best built atop the new biome / surface system.
- Erosion simulation as a refinement pass — would mostly affect mountain silhouettes.
- Saved-region cache to disk for cold-start latency improvements.
- Procedural ore distribution conditioned on plate type and depth.
- Climate-driven ambient creatures (e.g., dolphins near tropical coasts).
