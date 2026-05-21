# Cave System Overhaul — Design

**Status:** design ready for review
**Builds on:** [2026-05-19-minecraft-worldgen-research.md](2026-05-19-minecraft-worldgen-research.md) (specifically Q3 Option C — keep graph caves + add noise carvers)
**Replaces:** the disabled graph-cave system + the deprecated spaghetti / wormhole / surface-entrance noise carvers

## Goal

Make caves engaging, cohesive, navigable, surface-accessible, and biased toward "deeper exploration is rewarded with vaster spaces."

The current engine has noise-driven local caves (cheese + spaghetti + pillars + wormholes) but the structured graph-cave system that was meant to give them a backbone — chambers connected by tunnels, identifiable cave systems with entrances — is disabled (`CAVE_SYSTEMS_PER_REGION = (0, 0)`). Result: caves feel like swiss-cheese pockets in every direction, no landmarks, no "I'm in a cave system", no incentive to descend.

This spec re-enables graph caves with structural enhancements (styles, vertical inter-band connectors, cross-region trunk lines), replaces the noise carvers with a single depth-driven Terasology-style ambient layer that produces more caves the deeper you go, retires the spaghetti / wormhole / surface-entrance code, adds smooth-min layer composition so close-but-not-touching pockets merge, and adds a Terasology-borrowed surface-block-fixer post-pass so cave breakthroughs to the heightmap produce correct surface materials.

## Decision summary

| Item | Choice | Why |
|---|---|---|
| Macro structure | Graph caves (existing types: `CaveSystem`, `Chamber`, `Tunnel`, `Entrance`) | Already in repo; Q3 decision was to keep |
| Per-system variety | 5-style table (Cathedral / Warren / Slot / Sump / Karst) | Variety as identity — players recognise district types |
| Inter-band navigation | Vertical connectors between systems in adjacent bands of same region | Guarantees deep systems are reachable from shallower ones |
| Cross-region cohesion | Trunk lines connecting nearest-system pairs in neighbour regions | Gives "the cave network goes on forever" feel |
| Ambient layer | **Terasology-style depth-driven 2-noise carver, anisotropic Y** | Coherent meandering tunnels, scarce near surface, vaster at depth — bonus criterion satisfied by math |
| Cheese layer | Kept (with existing `cave_layer²` Y-band gating) | Pocket detail inside larger structures |
| Pillars | Kept | Stone columns in chambers for naturalism |
| Spaghetti, wormhole, surface_entrance | **Retired** | Subsumed by Terasology ambient |
| Composition | `smin()` instead of `min()` between layers | Merges pockets within ~1 block of each other |
| Vertical run clamp | Cap any continuous vertical air column at 6 blocks | Eliminates rare-but-deadly drop hazards |
| Surface block fix | Terasology-style post-pass: move grass/dirt to actual cave floor where caves breach the heightmap | No more grass on cave ceilings |

## Layer composition

Convention: signed density, negative = carve. Final density value composed via smooth-min joins. Pipeline:

```rust
// Per voxel inside fill_chunk:
let cheese  = cheese_contribution(...);        // existing math, kept
let tera    = terasology_ambient(...);         // NEW — primary cave layer
let ambient = smin(cheese, tera, k);

let graph   = -graph_cave_sdf(...) * INTENSITY;   // existing graph SDFs, sign-flipped to signed-density
let cave    = smin(ambient, graph, k);

let pillar  = pillar_contribution(...);        // existing math, kept; ADDS to density
let density = base_density - cave + pillar;
```

In `min` terms before the smin substitution: a voxel carves whenever any of `cheese`, `tera`, `graph_sdf*4` go negative. With smin and a small `k`, two layers near zero both contributing slightly pull the result more negative — close-but-not-touching pockets merge.

## Layer details

### Graph caves (existing types, re-enabled with enhancements)

Build-time: `build_systems_for_region` rolls 0–3 cave systems per fine region. Each system rolls a depth band (Shallow / Middle / Deep), a style, a bounding box inside the region, then Poisson-sampled chambers connected by an MST + 1–2 loop edges.

Two new pieces of state added to `CaveSystem`:
- `style: CaveStyle` — one of the 5 named styles, hashed from cell id, band-biased.
- `trunk: Option<Trunk>` — optional cross-region connection to the closest system in any of the 8 neighbour regions, measured by Euclidean distance between the two systems' chamber-index-0 centers.

One new pass added at region-build time, after `build_systems_for_region`:
- `build_vertical_connectors(region)` — for each pair of systems in the same region whose bands are adjacent (Shallow↔Middle or Middle↔Deep), roll `vertical_connector_prob`. If pass, emit a `Tunnel` from the upper system's lowest chamber to the lower system's highest chamber.

Carve-time uses the existing `cave_sdf` math (chamber ellipsoid + tunnel capsule SDFs). Trunks and vertical connectors carve via the same `capsule_sdf`.

### Style table

```rust
pub enum CaveStyle {
    Cathedral,   // Few large chambers, wide tunnels. Deep-band-biased.
    Warren,      // Many small chambers, narrow tunnels. Shallow-band-biased.
    Slot,        // XZ-stretched chambers, narrow vertical sheets. Mid-band-biased.
    Sump,        // Low-clustered chambers (flooded look). Deep-band-biased.
    Karst,       // Default — medium chambers, medium tunnels.
}
```

Each style's parameters live in a `CaveStyleTable` in `CaveConfig`:

| Style | Chambers/sys | Chamber R_xz | Chamber R_y | Tunnel R |
|---|---|---|---|---|
| Cathedral | 3–5 | 22–32 | 16–26 | 4–6 |
| Warren | 9–13 | 6–12 | 5–10 | 3–4 |
| Slot | 5–8 | 10–24 (×1.4 X, ×0.6 Z stretch) | 4–8 | 3–5 |
| Sump | 4–7 | 14–24 | 8–14 | 4–6 |
| Karst | 6–10 | 10–18 | 8–14 | 3–5 |

Style is rolled from `hash(seed, region_coord, system_idx)` against per-band weights:
- Shallow: 5% Cathedral / 45% Warren / 20% Slot / 5% Sump / 25% Karst
- Middle: uniform 20% each
- Deep: 10% Cathedral / 5% Warren / 5% Slot / 45% Sump / 35% Karst

### Depth-scaled chamber sizes

Each chamber's `R_xz` and `R_y` are multiplied by `depth_mult(cy) = 1.0 + depth_scale * max(0, (40 - cy) / 80)`. With `depth_scale = 0.81` (chosen default), a chamber at `cy = -100` has radii ~2.4× a same-style chamber at `cy = 40`. Cathedrals at the world floor become visibly cavernous; Warrens at the surface stay tight.

### Terasology ambient (new — primary noise carver)

Borrowed verbatim from the Terasology `Caves` module's `CaveFacetProvider`, generalised to use anisotropic Y sampling for our shorter world.

```rust
pub fn terasology_ambient(wx: i32, wy: i32, wz: i32, cfg: &CaveConfig, seed: u64,
                          surface_y: f32) -> f32 {
    let depth = ((surface_y - wy as f32).max(0.0));
    let freq_reduction = (cfg.tera_supp - depth / cfg.tera_supp_depth).max(0.0);
    let freq_depth     = cfg.tera_thresh_base + depth / cfg.tera_thresh_depth;
    let wy_scaled      = wy as f32 * cfg.tera_y_factor;
    let n0 = fbm_simplex(wx as f32, wy_scaled, wz as f32,
                         1.0 / cfg.tera_wave, 4, seed.wrapping_add(2000));
    let n1 = fbm_simplex(wx as f32, wy_scaled, wz as f32,
                         1.0 / cfg.tera_wave, 4, seed.wrapping_add(2001))
             + freq_reduction;
    ((n0 * n0 + n1 * n1).sqrt() - freq_depth) * 5.0
    // returns signed value: <0 = carve, >0 = solid
}
```

Why this works:

1. **Two independent noises near zero → topologically 1D curves** (the comment in the Terasology source explicitly notes this gives continuous tunnels, not dead ends). Inherent coherence.
2. **`freq_depth` grows with depth** — the cave region in 2D noise space gets larger → more caves carved at depth → bonus criterion satisfied by math, no tuning needed.
3. **`freq_reduction` shifts the cave region off-zero near surface** — caves are rare in the surface band, with a smooth fade to "normal density" by `tera_supp_depth` blocks down.
4. **`wy * tera_y_factor` anisotropy** — sampling Y at higher rate than XZ forces tube iso-surfaces to be more horizontal. Critical for our shorter world; isotropic Y produces 30+-block vertical drops we don't want.

`fbm_simplex` is 4-octave FBM of Simplex (`noise::Fbm<Simplex>` already in repo).

### Cheese carver (kept, unchanged)

Existing `cheese_contribution` + `cave_layer²` gating from PR 8, no math changes. Composes with `tera` via `smin`. Provides swiss-cheese pocket detail inside larger structures and within Tera tunnels.

### Pillars (kept, unchanged)

Existing `pillar_contribution` — refills carved regions with stone columns. Added to density at the end of the composition (positive contribution).

### `smin` composition

Polynomial smooth-min:

```rust
fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 { return a.min(b); }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}
```

Applied at every layer join (cheese↔tera, ambient↔graph). With `k = 1.2`, two SDF values within ~1.2 in their respective scales pull the result below their individual min by up to `k/4 = 0.3`, effectively carving the "almost-air" zone between them. No more 1-block stone walls separating pockets that should connect.

### Vertical-run clamp (post-voxelization)

After density is composed and the voxel grid is filled, a single linear pass per XZ column:

```rust
for each (x, z) column:
    let mut run = 0;
    for y in chunk.y_range() {
        if voxel(x, y, z).is_air() {
            run += 1;
            if run > MAX_VERTICAL_AIR_RUN {
                voxel.set(x, y, z, Block::Stone);  // insert ledge
                run = 0;
            }
        } else {
            run = 0;
        }
    }
```

`MAX_VERTICAL_AIR_RUN = 6` (so the deepest unbroken fall is 6 blocks). This eliminates the rare-but-real fall-to-death scenarios that show up in the metric as `longest_run` outliers. Cheap O(voxels) pass.

### Surface block fixer (Terasology-borrowed)

Adapted from Terasology's `CaveToSurfaceProvider`. Runs after density and surface-block placement. For each voxel where the cave breaches the heightmap, move the surface block (grass / dirt / sand / snow) down to the first solid voxel in the column. Also disables cave voxels at or above `SEA_LEVEL + 1` whose column is below the heightmap (i.e. the seabed) — prevents ocean drain into caves below sea level. Spreads laterally by 3 blocks (Terasology's `SURFACE_SPREAD`) so adjacent cave-floor voxels near an opening also get correct surface treatment.

This replaces the current explicit-entrance-rolling approach (`EntranceKind::{Sinkhole, CliffMouth, Skylight}`) for getting natural-looking cave openings at the surface. Cave openings now emerge wherever the noise carves through, with correct surface materials at the floor.

## Default tunings (chosen via parameter sweep)

These are pick #3 from the v2 sweep (composite score 0.923; 0.99 verticality / 0.93 vastness / 1.00 depth reach / 0.92 surface access / 0.89 coherence / 0.71 navigability).

```ron
// Graph caves
systems_per_region_max: 3,
chamber_count_mult: 0.70,
depth_scale: 0.81,
deep_band_bias: 0.30,            // share of systems rolled into Deep band
vertical_connector_prob: 0.87,
vertical_connector_r: 3.4,
trunk_prob: 0.38,
trunk_r: 3.4,
entrance_boost: 0.20,            // bumps native entrance roll probabilities

// Terasology ambient
tera_wave: 200,                  // noise wavelength in blocks
tera_supp: 0.17,                 // surface freq_reduction at depth 0
tera_supp_depth: 123,            // blocks over which suppression fades to 0
tera_thresh_base: 0.073,         // cave region radius at surface
tera_thresh_depth: 2229,         // threshold growth: radius += depth / this
tera_y_factor: 3.56,             // Y anisotropy — higher = more horizontal tubes

// Cheese (retuned for new composition)
cheese_offset: 0.18,
cheese_scale: 24,
cave_layer_intensity: 2.0,

// Composition
smin_k: 1.2,                     // merge tolerance at every layer join
max_vertical_air_run: 6,         // ledge-insertion clamp
```

Style band weights live in `CaveStyleTable` (defaults in code, hot-reloadable).

## Engine integration

Touchpoints in the existing source tree:

| File | Change |
|---|---|
| `src/worldgen/caves.rs` | Re-enable graph-cave generation; add style table + per-style chamber/tunnel parameters; add `build_vertical_connectors` + `build_trunks`; add `terasology_ambient` carver; delete `spaghetti_*` / `WormholeNoise` / `surface_entrance_*` functions and tests; replace `min(cheese, …)` joins with `smin` |
| `src/worldgen/tuning.rs` | `CAVE_SYSTEMS_PER_REGION = (0, 0)` → `(0, 3)`; delete spaghetti / wormhole / surface_entrance constants; add `MAX_VERTICAL_AIR_RUN` and `SMIN_K_DEFAULT` |
| `src/worldgen/config.rs` | Delete spaghetti / wormhole / surface_entrance fields from `CaveConfig`; add the new fields above; add `CaveStyleTable` struct |
| `assets/worldgen/default.ron` | Update accordingly |
| `src/worldgen/mod.rs` | Update `fill_chunk` composition to use `smin`; integrate vertical-run clamp post-pass; integrate surface-block fixer post-pass |
| `src/worldgen/probe.rs`, `bin/worldgen_viz` | Remove references to retired carvers; add new layer toggles to the visualizer (Tera ambient, graph caves, cheese — match the brainstorm preview) |
| `src/worldgen/density_graph.rs` | If carver evaluator caches noise samples for retired layers, drop those channels |

The `CarverEvaluator` (corner-lattice trilerp) shrinks — the retired carvers had ~10 noise channels between them; the new design has 4 (cheese, cave_layer, 2× tera). Net win on per-chunk noise samples.

The new `terasology_ambient` is the heaviest single layer (4-octave FBM × 2 channels per voxel). Should fit within the existing corner-lattice trilerp pattern; sample at 4-block corners, lerp per voxel, exact at corners.

## Cleanup of failed experiments

Several iterations of this design explored dead ends before landing on the Terasology hybrid. **None of those experimental directions reached the source tree** — all the visualization work lived under `/tmp/*.js` and `.superpowers/brainstorm/`. The cleanup list for the *existing* source tree is the retirement of the noise carvers that PR 8 added but the design supersedes:

**Code to delete from `src/worldgen/caves.rs`:**
- `pub fn spaghetti_contribution`
- `pub fn spaghetti_roughness`
- `struct WormholeNoise` and `impl WormholeNoise`
- `pub fn surface_entrance_contribution`
- `surface_entrance_y_fade` helper
- All test functions under `#[cfg(test)]` referencing these (`spaghetti_*`, `wormhole_*`, `surface_entrance_*` tests)
- The `spag_*`, `wormhole_*`, `surface_entrance_*` fields on `NoiseCarvers` and `CarverEvaluator` + corresponding `CarverCorner` fields

**Code to delete from `src/worldgen/config.rs`:**
- `spaghetti_2d`, `spaghetti_2d_modulator`, `spaghetti_2d_elevation`, `spaghetti_2d_thickness`, `spaghetti_roughness` channel params
- All `spaghetti_*` scalar fields (elevation_min/max, gradient_*, thickness_*, clamp_*, cave_noise_offset)
- All `surface_entrance_*` fields
- `pillar_rareness`, `pillar_thickness` survive (still used)

**Code to delete from `src/worldgen/tuning.rs`:**
- `WORMHOLE_BAND_Y`, `WORMHOLE_BAND` constants
- Any `SPAG_*` or `SURFACE_ENTRANCE_*` constants

**Code to delete from `assets/worldgen/default.ron`:**
- All `spaghetti_*`, `wormhole_*`, `surface_entrance_*` keys

**Code to delete from `src/worldgen/noise_channel.rs`:**
- `weird_scaled_sample` (only used by spaghetti)
- `y_clamped_gradient` (only used by spaghetti)
- `map_from_unit_to` (only used by spaghetti)
- Keep `build_channel` (used by cheese, pillar, and tera)

**Bench / probe references:**
- `src/bin/doc_render/parity_dump.rs`, `src/bin/worldgen_viz/widgets/probe_table.rs`, `examples/probe_cliff.rs` — remove any rows / columns referencing retired carvers

**Test cleanup:**
- Drop the `#[ignore = "diagnostic only"]` `probe_surface_entrance_noise_distribution` test
- Drop `noise_carvers_*` tests that specifically reference spaghetti / wormhole
- Add new tests (see Testing section)

The net delta is roughly **−700 LOC** (deleted carvers + tests) **+500 LOC** (new style table, vertical connectors, trunks, terasology_ambient, vertical-run clamp, surface fixer) for a net **~−200 LOC** in the cave subsystem.

## Testing

Unit tests:
- `terasology_ambient_depth_monotonicity`: at fixed (wx, wz), the cave-fraction (sampled over wide XZ grid) at successive Y bands must be monotonically non-decreasing toward depth.
- `terasology_ambient_surface_suppression`: at depth 0, cave fraction must be < 1%; by `tera_supp_depth`, fade is complete.
- `terasology_ambient_anisotropy_horizontal`: the mean horizontal run / mean vertical run ratio of carved voxels must be > 1 (more horizontal than vertical extent) with `tera_y_factor >= 3`.
- `smin_extremes`: `smin(a, b, 0) == min(a, b)`; `smin(0, 0, k) == -k/4`; `smin(a, b, k)` ≤ `min(a, b)` for all `k >= 0`.
- `vertical_run_clamp`: synthetic column with 12 consecutive air voxels → after clamp, longest run ≤ 6.
- `style_band_distribution`: roll 1000 styles in each band and assert the distributions match the configured weights within ±5%.
- `vertical_connector_connects_systems`: build a region with one shallow + one middle system, force `vertical_connector_prob = 1.0`, assert the connector carves a continuous path between them.
- `trunk_links_neighbour_region_systems`: similar.

Integration tests (existing patterns):
- `deep_underground_has_no_surface_blocks` — keep, must still pass with surface fixer changes.
- `deep_caves_under_land_are_dry` — keep.
- `underground_chunk_has_both_caves_and_solid` — keep.
- New: `surface_breakthrough_has_correct_surface_block` — column where cave breaches heightmap: assert grass moved to cave floor, none on cave ceiling.
- New: `chunk_has_no_vertical_drops_over_six_blocks` — scan voxel columns, assert max air run ≤ `MAX_VERTICAL_AIR_RUN`.

Visual:
- New golden screenshot at `tests/screenshots/cave_overhaul_baseline.png` taken from a known seed/position. Use existing `tests/screenshots/diff.py` (per memory: ~6% noise floor is intrinsic to chunk streaming; pass at <10% diff for cave overhauls).

## Migration

- **Persistent chunk cache must be invalidated.** Per memory, `saves/default/regions/*.bin` caches chunks across runs and must be deleted after any worldgen change. The spec implementation should bump a `WORLD_GEN_VERSION` constant and the persistence layer should refuse to load caches from older versions (or just delete on mismatch with a log line).
- **Golden hashes must be re-baselined.** Per the worldgen-research doc, this is expected on any density-graph composition change. Run the fingerprint integration tests, observe the new hashes, commit them.

## What this design does NOT do

Out of scope for this spec, deliberately:

- **No new biome integration.** Biome-flavored caves (icy caverns in tundra, lava tubes in volcanic biomes, etc.) are a follow-up. The style table is biome-agnostic for now.
- **No mob spawn rules** keyed to style. Each `CaveSystem` knowing its style is the substrate for this; the spawn rules themselves are not in scope.
- **No biome-driven aquifer changes.** The existing primitive aquifer rule ("water only under lakes / ocean columns") survives unchanged; the proper MC-style aquifer migration remains the next big worldgen item per the research doc.
- **No data-driven `CaveStyle` enum.** Styles are Rust-side. Their *parameters* are RON-tunable; adding new style names requires a recompile.
- **No editable DAG integration.** Per memory, the editable DAG follow-up is deferred until the worldgen overhaul completes — this spec is a step toward that, not a part of it.

## Open follow-ups (after this lands)

- Decide whether to extract `CaveLocation` records (Terasology's `CaveLocationProvider` pattern) per chunk for downstream features (mob ecology, loot placement, lighting).
- Decide whether to add biome-driven style biasing (e.g., desert biome rolls more Sump for oasis-like cave-pools).
- Consider whether the surface-block fixer pass should also handle "ceiling stalactites" and "wet block transitions" near aquifers — both natural extensions but deferred.

## Verification

After implementation, verify by:

1. Re-run all worldgen tests (49 lib + 3 fingerprint integration). All pass; both golden hashes rebaselined.
2. Run `cargo run --bin oxium`, fly around an underground area for 60 seconds, screenshot. Confirm:
   - Caves are visibly rare near the surface and densely present at deep Y.
   - Cave systems are distinguishable as Cathedral / Warren / Slot / Sump / Karst by chamber-size signature.
   - No fall-to-death drops (the deepest unbroken vertical air column is ≤ 6 blocks).
   - Cave openings at the heightmap have grass/dirt on the actual cave floor, not on the ceiling.
3. Run the existing screenshot diff tool against the new baseline screenshot. Expect noise-floor-level differences (<10%) on subsequent runs.
