# Worldgen 3D Density — Design Spec

**Date:** 2026-05-19
**Status:** Drafted, awaiting approval
**Builds on:** `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`

## Summary

Replace the per-column "fill solid up to `height`" rule with a **3D density function** evaluated per voxel. Each voxel `(wx, wy, wz)` is solid iff `density(wx, wy, wz) > 0`. The 2D heightmap (`h_pre` + valley carve = `h_target`) becomes the *y-bias* inside that density rather than the literal surface height. Result: the terrain surface emerges block-by-block from a smooth 3D field instead of being a column-quantised staircase, so moderate slopes stop reading as clean chevron stripes. As a side effect, overhangs / cliffs / floating islands become possible.

## Goals

1. **Eliminate the voxel-staircase artifact** on moderate slopes (the chevron stripes the user flagged).
2. **Allow natural overhangs** where the noise field digs sideways into the terrain.
3. **Preserve everything that already works**: plate-driven heightmap, hierarchical river flow, graph caves, biome surface materials, region cache, save format.
4. **Keep performance acceptable** — chunk fill stays under ~10ms in release on the M4 Max baseline.

## Non-goals

- Floating-island archipelagos in the sky (the heightmap bias keeps terrain anchored).
- Erosion-style features (still no iterative process).
- New biomes or surface materials.
- River network rework — flow accumulation continues to produce a 2D `h_target` modification.
- Cave-system rework — caves stay as graph-based SDFs, just subtracted from the density field instead of overriding blocks post-fill.

## The density function

```
density(wx, wy, wz) = bias(wy, h_target(wx, wz))
                   + relief_noise(wx, wy, wz) * RELIEF_AMP
                   - cave_sdf_subtract(wx, wy, wz)
```

Where:

- **`h_target(wx, wz)`** = `h_pre(wx, wz) - valley_carve(wx, wz)`. Unchanged from current code.
- **`bias(wy, h_target)`** = `(h_target - wy) / FALLOFF_SCALE`. Positive below the target height, negative above, linear through `wy = h_target`. `FALLOFF_SCALE` controls how dramatic the noise-driven surface fuzz is:
  - Small (e.g. 1.0): noise has to overcome ±1 per block. Surface barely fuzzy. Almost like a heightmap.
  - Medium (4.0): noise can push the surface around by ~±4 blocks. Rougher, more organic.
  - Large (e.g. 8.0): substantial overhangs and floating spurs.
- **`relief_noise`** = 3D FBM (Simplex). 3 octaves, period ~32 blocks at the base octave, persistence 0.5. Amplitude `RELIEF_AMP ≈ 1.0` (it's compared against the bias, which is also unit-scale).
- **`cave_sdf_subtract`** = positive inside cave chambers/tunnels, zero outside. The existing cave code returns a boolean "is this cell inside a cave?" — for 3D we replace it with a soft SDF that returns `~5.0` deep inside a chamber, `~0.5` at the chamber wall, `0.0` outside the chamber's influence. The density field then naturally carves smooth-edged voids.

**Solid iff `density > 0`**. Air iff `density ≤ 0`. Water iff air AND `wy ≤ SEA_LEVEL` (or below a lake rim).

## Architectural changes

### `worldgen::heightmap` — split into 2D + 3D

- `HeightmapNoise` stays. Renamed conceptually: it's now the **bias** generator, not the heightmap.
- Add `DensityNoise`: owns the 3D relief FBM.
- `DensityNoise::evaluate(seed, h_target, wx, wy, wz) -> f32` returns the full density (bias + relief).

### `Generator`

- New field: `density: DensityNoise`.
- No removals — `heightmap` (the 2D bias source) stays.

### `worldgen/mod.rs::fill_chunk` — per-voxel evaluation

Per chunk, the existing top-of-fill setup (pre-fetch regions, gather cave systems) stays. The inner loop changes from "fill solid blocks up to height" to **per-voxel evaluation with top-down column scan for surface block selection**:

```rust
for each column (x, z) in chunk {
    let h_target = column_data_with(wx, wz, &regions).height as f32;
    let lake_rim = col.lake_rim;
    let biome = col.biome;
    let mut depth_below_surface: Option<i32> = None;
    let mut last_was_solid = false;
    for y in (0..CHUNK_DIM_U).rev() {  // top → bottom
        let wy = origin.y + y as i32;
        let d = self.density.evaluate(self.seed, h_target, wx, wy, wz);
        // Cave SDF subtraction (existing cave_systems list)
        let d = d - cave_sdf_at(wx, wy, wz, &cave_systems);
        let solid = d > 0.0;
        let block = if !solid {
            // Air or water.
            depth_below_surface = None;
            water_or_air(wy, lake_rim)
        } else {
            let new_depth = depth_below_surface.map(|d| d + 1).unwrap_or(0);
            depth_below_surface = Some(new_depth);
            select_solid_block(new_depth, biome, wy, ...)
        };
        out.set(local, block);
    }
}
```

`select_solid_block` is the existing depth → surface/dirt/stone logic, but it now uses the *actual* depth below the topmost solid voxel in this column (which can differ from `h_target` because the noise can push it up or down). For dirt depth: 3 blocks (the v1 default — no need for the coastal-extended cap since the 3D noise naturally hides the staircase).

### Sea-level / lake-rim water flood

Unchanged in spirit. When a voxel is air AND `wy ≤ effective_water_level`, fill with water. `effective_water_level = lake_rim.unwrap_or(SEA_LEVEL)`.

### Tree placement

`tree_in_cell` currently consults `column_data.height` to know where the surface is. In 3D the surface y differs from `h_target` by up to ±FALLOFF_SCALE blocks. Two options:

1. **Cheap:** during tree-cell selection, do a small top-down density walk (32 evals worst case) to find the actual topmost solid block in the cell's chosen column. Use that as `base_y`.
2. **Cached:** during chunk fill, store per-column "topmost solid Y" in a temporary array; trees read it after the main fill loop.

Option 2 is faster (the chunk fill already iterates the column top-down, just pass the topmost Y back to `add_trees`). Option 1 is simpler. We'll start with option 1 and switch to option 2 if profiling shows it matters.

### Cave SDF

The existing `caves::cave_air(wx, wy, wz, &systems)` and `caves::entrance_air(...)` return booleans. For 3D integration they become a single `cave_sdf(wx, wy, wz, &systems) -> f32` that returns:

- `0.0` outside any cave feature
- Increasing positive values toward chamber centers and along tunnel centerlines

We subtract this from density. Where the SDF is large positive, density drops below 0 → air. Where 0, density unchanged.

This is a chunkier refactor of `caves.rs` but the chamber/tunnel/entrance geometry stays the same — only the "is this cell inside" predicate becomes a "how far inside" measurement.

For PR A (hybrid) we can keep the boolean cave check and apply it post-density-test (override solid→air where cave_air). PR B is where we switch to the smooth SDF for soft-edged caves.

### Sub-band optimization

Full 3D evaluation costs one density call per voxel = `32³ = 32768` calls per chunk. Each density call = 1 noise + arithmetic. Compared to the current ~1024 height calls + ~1000 cave checks, this is ~30× more noise evaluations.

**Optimization:** skip density evaluation outside a narrow band around `h_target`:

```rust
if wy < h_target as i32 - SURFACE_BAND  → assume solid (deep underground)
if wy > h_target as i32 + SURFACE_BAND  → assume air
else                                      → evaluate density()
```

With `SURFACE_BAND = 8`, only ~16 voxels per column see a 3D density eval. Per-chunk noise budget: ~16k → only 1.5× the current pipeline. **No visible difference** for non-dramatic terrain (the noise inside the band still drives the natural roughness we're chasing).

For PR A we ship with `SURFACE_BAND = 8` — solves the chevron-stairs problem at minimal performance cost. For PR B we either widen the band (e.g. `24`) or remove it entirely to allow true overhangs and floating spurs. The user can choose at runtime based on a tuning constant.

## Phased rollout

### PR A — Surface-band 3D density (the de-stairs fix)

- Add `DensityNoise` to `Generator`.
- Replace `fill_chunk`'s solid/air decision with the density check inside `SURFACE_BAND = 8`.
- Surface block selection uses the actual topmost-solid Y from the top-down scan.
- Cave check stays boolean (post-density override).
- Tree placement uses the on-demand top-down density walk (option 1 above).
- Re-baseline golden + fingerprint hashes.
- Visual verification: chevron stripes on slopes should be replaced by organic roughness.

Risk: performance hit (~1.5× chunk gen). Bench before merge.

Behavioural changes from PR A:
- Surface terrain looks fuzzier / more natural on slopes.
- No overhangs yet (band too narrow).
- Caves still cookie-cuttered (boolean, sharp edges).

### PR B — Unbounded 3D + smooth cave SDF

- Widen / remove `SURFACE_BAND` so density evaluates everywhere (or in a much wider band). Allows real overhangs.
- Convert caves to soft SDF: `cave_sdf(wx, wy, wz)` returns smooth values, density subtracts. Caves end up with chamfered, organic walls instead of pixel-sharp ellipsoids.
- Optional: add a low-frequency "secondary relief" density layer that lets the noise carve dramatic cliffs and floating spurs in dramatic terrain.

Risk: substantial behavioural shift. Existing players see a new world (in unexplored chunks). Performance hit (3× chunk gen worst case).

Behavioural changes from PR B:
- Overhangs and floating spurs appear where 3D noise digs into the terrain.
- Caves have soft, organic walls.
- Players accustomed to clean Minecraft-like terrain may find this jarring.

PR B is optional. The de-stairs fix is fully delivered by PR A; PR B is for "true 3D" with its dramatic visual benefits and risks.

## Tuning constants (additions to `worldgen::tuning`)

```rust
// 3D density
pub const DENSITY_FALLOFF: f32 = 4.0;       // bias scale
pub const RELIEF_AMP: f32 = 1.0;            // 3D noise amplitude
pub const RELIEF_PERIOD: f32 = 32.0;        // 3D noise base period
pub const SURFACE_BAND: i32 = 8;            // density eval window around h_target
// PR B: widen this or use i32::MAX for unbounded
```

## Tests

### Tests that survive

- All existing tests: plate determinism, hydrology flow paths, caves connectivity, golden chunk hash (re-baselined), map fingerprint (re-baselined).
- `chunk_at_sea_level_has_water_or_solid` — still holds, water fills air below SEA_LEVEL.
- `all_biomes_appear_in_a_large_scan` — climate untouched.

### New tests

- `density_is_pure_in_seed_coord` — `density(seed, h, x, y, z)` is byte-stable.
- `density_above_target_is_mostly_air` — sample voxels at `wy = h_target + 4`; assert ≥95% are air (the noise sometimes pushes solid above the target, but rarely so far up).
- `density_below_target_is_mostly_solid` — same logic, opposite direction.
- `topmost_solid_within_band` — in a chunk far from caves, the topmost solid Y per column lies in `[h_target - SURFACE_BAND, h_target + SURFACE_BAND]`.
- `no_voxel_staircase` (visual fingerprint extension): render a side-on slice at a known steep slope; the per-column delta should now have variance > 0 (some columns +0, some +2, some +1) instead of perfectly +1 each step.

### Bench

Add `tests/bench/worldgen_3d_bench.rs` (or just measure in CI): chunk fill cold-start, target `<10ms` per chunk in release on the M4 Max.

## Risks

1. **Performance regression.** With `SURFACE_BAND = 8` the hit should be small, but worth measuring. Mitigation: tune `SURFACE_BAND` smaller; or short-circuit density when bias is overwhelmingly large.
2. **Visual style change.** Even PR A makes terrain look noticeably different — slopes are organic instead of clean. Mostly a positive change; mention in commit.
3. **Tree placement edge cases.** A tree picked on a column where the topmost solid is buried under an overhang might trunk through the overhang. Mitigation: have tree placement reject columns with overhangs (multiple air/solid transitions).
4. **Cave SDF re-engineering (PR B).** Soft SDFs for chamber + capsule-along-spline are doable but new code. Defer to PR B.

## Open questions deferred to implementation

- Whether to use 3 or 4 octaves on `relief_noise`. 3 octaves is faster; 4 looks slightly more detailed. Pick during PR A tuning.
- Whether `RELIEF_PERIOD = 32` is right. Smaller → bumpier; larger → smoother. Pick during PR A tuning.
- Whether to cache "topmost solid Y per column" in `FineRegion`. If profiling shows tree placement is slow, add it as a `Box<[i16; 64*64]>` next to existing flow fields.
- Cave SDF shape (Gaussian / linear / smoothstep). Decide during PR B.

## Decision points before implementation

Before writing code, confirm:

1. **Ship as PR A only, or A+B?** Recommended: A only, evaluate visually, then decide on B.
2. **`SURFACE_BAND` initial value?** Recommended: `8`.
3. **`DENSITY_FALLOFF` initial value?** Recommended: `4.0`. Smaller (e.g. `2.0`) = heightmap-like, larger (`8.0`) = dramatic. `4.0` is a good middle ground for the de-stairs fix.
4. **`RELIEF_AMP` initial value?** Recommended: `1.0` (matches bias scale).
