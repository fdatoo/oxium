# Minecraft 1.18+ Worldgen — Research Findings & Oxium Applicability

Source: decompiled MC source at `~/Downloads/out` (Java + data JSON).
Six parallel research agents covered (1) density-function graph, (2) splines, (3) climate/biomes, (4) chunkgen pipeline + interpolation, (5) surface/carvers/aquifer, (6) Oxium's current state.

This document is research output, not a plan. It exists to ground a design discussion.

---

## TL;DR — the four ideas that actually solve "interestingness vs chaos"

Oxium is fighting the same battle Minecraft fought from 1.7 → 1.18. Mojang's four breakthroughs:

1. **Splines, not noise sums, control macro shape.** Climate noise (continentalness, erosion, peaks-and-valleys) feeds *cubic Hermite splines* that produce per-column `offset(x,z)`, `factor(x,z)`, `jaggedness(x,z)`. Plateaus are flat because two adjacent knots have the same value and zero derivative — you *write down* that mountains peak at +0.5 and plains sit at 0, instead of hoping a noise product lands there.
2. **Density = 2D shape × Y-gradient + 3D noise, with asymmetric softening.** The literal formula is `4 * quarter_negative((depth + jaggedness * noise) * factor) + base_3d_noise`. The `quarter_negative` (scale negatives by 0.25) is the secret sauce — it lets 3D noise carve detail below the surface while keeping a hard ceiling above it, which is *exactly* the thing PR B's "soft cave SDFs with a `min(2.0)` cap" was hacking around.
3. **Biomes = nearest-hyperbox query in 6D climate space.** No nested if/else, no per-axis blending. Each biome owns a box (or several) in `(T, H, C, E, depth, weirdness)`. Lookup is an R-tree with squared-L2 distance and a `ThreadLocal` last-leaf cache. Per-block Voronoi jitter (no biome interpolation) gives organic borders for free.
4. **Sparse cell-grid evaluation: 4×8×4 corners + trilerp inside.** Density runs on a 5×49×5 corner lattice (~1225 evals) per chunk instead of per-voxel (~98k). Marker wrappers (`Interpolated`, `FlatCache`, `Cache2D`, `CacheOnce`, `CacheAllInCell`) annotate the function graph so the evaluator knows where to cache. This is the single biggest perf lever and Oxium currently has none of it.

Plus a few smaller-but-load-bearing tricks: **slides** at top/bottom of world, **aquifer barriers** with per-region water tables, and a **surface-rules DSL** that's just `Sequence | If(Condition, Then) | Block`.

---

## Part 1 — Mental model of MC 1.18+ worldgen

```
                                NOISE STAGE                                  POST-DENSITY
  ┌──────────────────┐    ┌────────────────────────────────────────┐    ┌────────────────┐
  │ Climate noises   │    │  Splines (Hermite, nested)             │    │ SurfaceSystem  │
  │ C, E, ridges,    ├───►│  → offset(x,z), factor(x,z),          ├───►│ Carvers (cave, │
  │ T, H, weirdness  │    │     jaggedness(x,z)                    │    │  canyon)       │
  └──────────────────┘    │                                        │    │ Aquifer        │
           │              │  depth = yGrad(-64→1.5..320→-1.5) +    │    │ Features       │
           │              │          offset                        │    └────────────────┘
           ▼              │                                        │
  ┌──────────────────┐    │  sloped_cheese =                       │
  │ 3D BlendedNoise  ├───►│    4 * quarter_negative(                │
  │ (anisotropic:    │    │      (depth + jaggedness * J_noise)    │
  │  Y wavelength    │    │      * factor)                         │
  │  = 2× XZ)        │    │    + base_3d_noise                     │
  └──────────────────┘    │                                        │
                          │  caves: subtract cheese/spaghetti/...   │
                          │  slide top→air, bottom→solid           │
                          │  squeeze(0.64 * interp(blend(...)))    │
                          │  min(noodle)                           │
                          └────────────────────────────────────────┘
                                          │
                       Evaluated only on a 4×8×4 cell grid → trilerp per voxel
                                          │
                          ┌───────────────▼───────────────┐
                          │ Biome assignment              │
                          │ 6D point (T,H,C,E,depth,W)    │
                          │ → R-tree nearest-hyperbox     │
                          │ → ~250 biome claims, sharp    │
                          └───────────────────────────────┘
```

Stages (`ChunkStatus`):  EMPTY → STRUCTURE_STARTS/REFS → BIOMES → NOISE → SURFACE → CARVERS → FEATURES → LIGHT → SPAWN → FULL. Each stage declares a per-radius dependency, enabling parallel chunkgen across non-conflicting cones.

---

## Part 2 — Numbers that matter (cheat sheet)

### Noise channels (overworld)
| Channel | firstOctave | amplitudes | XZ scale | ≈ wavelength | Role |
|---|---|---|---|---|---|
| continentalness | -9 | `[1,1,2,2,2,1,1,1,1]` | 0.25 | ~2000 blocks | ocean→inland; **drives height via spline** |
| erosion | -9 | `[1,1,0,1,1]` | 0.25 | ~2000 blocks | flatness / mountain factor |
| ridge (weirdness) | -7 | `[1,2,1,0,0,0]` | 0.25 | ~500 blocks | peaks-vs-valleys; biome variants |
| temperature | -10 | `[1.5,0,1,0,0,0]` | 0.25 | ~4000 blocks | climate band |
| vegetation (humidity) | -8 | `[1,1,0,0,0,0]` | 0.25 | ~1000 blocks | climate band |
| base_3d_noise | (blended -15..0/-7..0) | n/a | xz=0.25, **y=0.125** | XZ wavelength ~2× Y | per-voxel detail |
| jagged | -16 | sixteen 1.0s | xz=1500, y=0 | ~3 blocks | high-freq peak roughness |

**Anisotropy** — `base_3d_noise` y-scale is *half* the xz-scale, so vertical features are taller than they are wide. This is one of the cheapest visual wins available (one number).

**PV fold** — `peaksAndValleys(w) = -(||w| - 2/3| - 1/3) * 3`. Triangle wave on |w|. Raw weirdness still feeds biome lookup; folded PV feeds height splines.

### World shape
- Overworld: `min_y = -64`, `height = 384`, **cell size 4×8×4 blocks**
- Sea level: 63
- Top slide: starts y=240, target density −0.078125 (pull to air)
- Bottom slide: starts y=-64, target density +0.1171875 (pull to solid)
- "Underground" threshold: sloped_cheese < 1.5625

### Spline knot tables (offset, keyed on continentalness)
```
C        offset
-1.10    +0.044            (mushroom cap above deep ocean wall)
-1.02   -0.2222
-0.51   -0.2222            (DEEP_OCEAN floor — flat, repeated for plateau)
-0.44   -0.12
-0.18   -0.12              (ocean — still flat)
-0.16    beachSpline       (hard step into beach)
-0.15    beachSpline       (duplicated knot = visible coastline)
-0.10    lowSpline
 0.25    midSpline
 1.00    highSpline        (mountains)
```
Each "Spline" entry is itself a spline over erosion E, whose values are splines over ridges R. **Trilinear cascade of cubic Hermite, sparse: 5–10 knots per axis.** All derivatives are 0 (horizontal tangents) unless designed otherwise. Mountain factor ramps `0.10 → 0.70 → 1.00` walking inland — "mountains only happen far from coast" is data, not code.

### Biome table (samples)
- **MUSHROOM_FIELDS:** continentalness `[-1.2, -1.05]`, everything else full. Pure C gate.
- **DESERT:** T4 hot, any humidity, any inland C, E0..E4. The T4 row of `MIDDLE_BIOMES` is *all* desert across humidities.
- **DRIPSTONE_CAVES (3D!):** any T/H/E/W, C `[0.8, 1.0]`, depth `[0.2, 0.9]`. Underground biomes live in the same 6D table.
- **DEEP_DARK:** depth point `1.1` (world bottom) gate.

---

## Part 3 — Oxium today vs Minecraft equivalent

| Concern | Oxium today | Minecraft analogue | Take |
|---|---|---|---|
| Continents | Voronoi plates (1024-block cells, kind ∈ {continental, oceanic}, ridge_lift on boundaries) | Single 2D continentalness noise → spline → offset | MC's is smoother and seamless; Oxium has fought boundary cliff artifacts repeatedly (commits 870d35c, 515d9f0, b7e4b9f) |
| Mountains | `ridge_lift` along plate boundaries, peak height 24–60, smoothstep falloff width 0.30 | Erosion-driven spline; low erosion = mountainous | Replace ridge_lift with erosion noise + spline; no boundary geometry |
| 2D vs 3D | Hybrid: 2D h_pre is bias for 3D density `(h_target - wy)/4 + relief*1.0` | `(yGrad + offset) * factor + 3D_noise`, soft-floored by `quarter_negative` | MC's quarter_negative is the cleaner answer than Oxium's `min(2.0)` density cap |
| Climate | T, H, desertness — three independent 2D noises, classified by nested if (`Biome::classify` in mod.rs:710) | 6D (T, H, C, E, depth, W) hyperbox lookup with R-tree | Drop-in replacement; works with Oxium's current 6 biomes; adds depth axis for free |
| Biome borders | Per-classifier noise jitter (`BIOME_JITTER_AMPL=0.05`) | Sharp at quart grid + per-block hash-Voronoi jitter | Equivalent in spirit; MC's is cheaper (200 instructions/block) |
| Caves | Graph (Poisson chambers + MST tunnels) + wormhole noise sheets, subtracted as SDFs from density | Two-tier: noise carvers in density graph (cheese, spaghetti, noodle, pillars) + classic flood-fill walkers (cave, canyon) post-density | Keep Oxium's graph caves — they're a feature MC lacks. Add a noise-carver layer for ambient "everywhere" cave density. |
| Surface | Inline `if cliff/sand/snow/grass` in `fill_chunk` (mod.rs:420–474) | Surface Rules DSL: `Sequence | If(Cond, Then) | Block` with `ON_FLOOR`, `UNDER_FLOOR`, `stoneDepthAbove`, etc. | Pull surface logic out into a small DSL — fixes the `surface.rs` doc-rot stub too |
| Aquifers | Below sea = water, full stop (mod.rs:404–414) | Per-region 3D water table noise + barrier pressure between aquifers | Solves "cave chambers shouldn't all be flooded just because they're below sea" |
| Cell interpolation | None — 32768 density evals per chunk | 5×49×5 = 1225 corner lattice + trilerp + marker-driven cache | Biggest perf lever; PR B's commit message anticipated ~3× cost |
| Top/bottom slides | None | Lerp density to −0.078125 at y≥240, to +0.117 at y≤-40 | Prevents infinite caves at y=MAX_TERRAIN_Y; cleaner than CAVE_FLOOR_Y clamp |

---

## Part 4 — Specific things Oxium could borrow, in priority order

These are research-grade ideas — not commitments. Each line is a paragraph the team can evaluate, accept, or reject.

### 1. Replace Voronoi plates with (continentalness, erosion, PV) noise + splines

**Today:** `plates.rs` (366 lines) produces continents via a jittered Voronoi mosaic. Plate boundaries get `ridge_lift` for mountains. Repeatedly fought boundary artifacts.

**MC's approach:** Three independent 2D noises sampled at low frequency (firstOctave -9 ≈ 2000-block wavelength). Each maps through a `Spline<f32 → f32>` (Hermite, 5–10 knots). Splines compose: `offset = outerSpline_C(continents, innerSpline_E(erosion, innerSpline_R(ridges)))`. The innermost result is a scalar height.

**Why it helps:** No boundary geometry to fight. Continents fade in/out smoothly. Mountains are gated by `mountainFactor ramps 0.10 → 0.70 → 1.00` as you walk inland — *coastal mountains are physically impossible* in the data, no `CLIFF_MIN_HEIGHT` band-aid needed. The spline structure makes "interestingness" controllable per-knot.

**Cost:** Write a `CubicSpline<T>` type (~80 lines), three noise channels (already have `HeightmapNoise` infrastructure), and tune knots. The Mojang knot tables in `TerrainProvider.java` are a known-good starting point.

### 2. Adopt the `(depth + jaggedness*noise) * factor + 3D_noise` density composition

**Today:** `density = (h_target - wy)/DENSITY_FALLOFF + relief_noise*RELIEF_AMP`. Linear bias + isotropic noise. Soft cave SDFs subtracted with a `min(2.0)` cap to keep caves carvable when bias is huge.

**MC's approach:** `sloped_cheese = 4 * quarter_negative((depth + jagged * J_noise) * factor) + base_3d_noise`, where `quarter_negative(x) = x > 0 ? x : x * 0.25`. The `factor` from spline #1 controls vertical sharpness; `jaggedness` controls high-freq peak detail; `quarter_negative` softens below-surface so 3D noise can carve without breaking the surface.

**Why it helps:** No `min(2.0)` hack. Caves carve naturally underground because the bias is asymmetrically soft. Mountains stay sharp because factor is high there. Plains stay flat because factor is low there.

**Cost:** ~10 lines in `DensityNoise::evaluate`. The expensive part is computing `factor` and `jaggedness` — which falls out of (1).

### 3. Multi-noise biome lookup

**Today:** `Biome::classify(temp, humidity, desertness)` is a nested if-elif (mod.rs:710-735). Six biomes hard-coded.

**MC's approach:** Each biome owns one or more `Climate.ParameterPoint` boxes in 6D. Lookup is `RTree::search(quantized_target_point)` returning the nearest box by squared-L2 over axis gaps. Fanout 6, `ThreadLocal<Leaf>` cache for adjacency. Build once per world; query is ~3-4 hops.

**Why it helps:** Adds depth as a free axis → underground biomes work. Adds a "weirdness" axis decoupled from base climate → variant biomes (ice spikes, sunflower plains) without new noises. Decision logic is *data* — easy to tune, hot-reloadable, no nested code.

**Cost:** `Climate.Parameter`, `ParameterList`, `RTree`, `Sampler` ≈ 500 lines. Six biome table entries. Per-block Voronoi jitter (200 instructions/block) for organic borders.

### 4. Sparse cell-grid density evaluation

**Today:** Density is per-voxel. 32 KB chunks × 1 eval/voxel × (climate noises + relief + caves) is ~3× the cost it could be.

**MC's approach:** Cell size 4×8×4. Evaluate density on a `(cellCount+1)` corner lattice — overworld is 5×49×5 = 1225 corners. Per voxel does one trilerp (7 lerps for the nested Y/X/Z reductions). Marker types in the density graph (`Interpolated`, `CacheAllInCell`, `FlatCache`, `CacheOnce`) tell the per-chunk evaluator where to insert caches.

**Why it helps:** ~80× fewer expensive noise evals. Free side benefit: a "density graph" data model — composable, introspectable, and visually debuggable.

**Cost:** Non-trivial. Requires turning the density expression into a tree of `DensityFn` nodes (probably an enum) and writing the per-chunk evaluator with sliding YZ wall buffers and trilerp accumulators. ~400 lines. Best done after (1) and (2) so the graph shape is settled.

### 5. Surface rules as a small DSL

**Today:** Surface block selection is inline in `fill_chunk` (mod.rs:420–474), interleaving cliff / beach / snow / desert / grass logic. `surface.rs` is a 5-line "PR 1 stub" but its work lives in mod.rs (doc rot).

**MC's approach:** `RuleSource = Block(b) | Sequence([R]) | If(Condition, R) | Bandlands`. Conditions are predicates like `OnFloor`, `UnderFloor`, `StoneDepth(0, addSurfaceDepth)`, `Biome([...])`, `YAbove(y)`, `NoiseThreshold(noise, min, max)`. The runtime walks columns top-down, maintains `stoneDepthAbove` as state, evaluates rules lazily-cached per-column and per-row.

**Why it helps:** Surface is data, not code. Adding "wet biomes get podzol" is one line. Decouples surface from terrain. Easy to test in isolation. Restores `surface.rs` to non-stub.

**Cost:** ~150 lines for the enum, evaluator, and lazy-caching context. Migrate the existing inline logic into a `Vec<SurfaceRule>` constructed in `surface.rs`.

### 6. Per-region aquifers instead of "sea level fills all"

**Today:** `mod.rs:404–414` floods any air voxel below sea level with water. Cave chambers under the ocean are bathtubs.

**MC's approach:** Place "aquifer centers" on a jittered 16×12×16 grid. Each center samples (floodedness, spread, lava) noises to derive a local `(fluidLevel, fluidType)`. Per voxel: find 3 nearest centers, soft-blend by `similarity(d²₁, d²₂) = 1 - (d²₂ - d²₁)/25`. A `pressure(s₁, s₂)` term re-adds rock between aquifers with mismatched levels (asymmetric: thicker walls between water and lava, thicker barrier going down than up). Water-vs-lava pressure = constant 2.0 (guaranteed seal).

**Why it helps:** Caves deep below sea level can be *dry*. Water tables can vary by region. Lava pools can exist without flood-filling. The pressure function is the elegant bit — it prevents lava-water contact mathematically, not by special cases.

**Cost:** ~250 lines. Worth doing after the density graph (4) since the aquifer is naturally the last node before block-state decision.

### 7. Top/bottom slides

**Today:** Hard clamp `height = clamp(h_pre - carve, CAVE_FLOOR_Y+8, MAX_TERRAIN_Y)`. Caves can't open at y=MAX_TERRAIN_Y but the clamp is visible as flat ceilings.

**MC's approach:** Lerp density toward a target near the top (-0.078125 = mild air pull) and bottom (+0.1171875 = mild solid pull). 4 lines:
```rust
fn slide(d: f64, y: i32, min_y: i32, height: i32) -> f64 {
    let top_f = inv_lerp(min_y + height - 80, min_y + height - 64, y as f64).clamp(0.0, 1.0);
    let d = lerp(top_f, d, -0.078125);
    let bottom_f = inv_lerp(min_y as f64, min_y + 24 as f64, y as f64).clamp(0.0, 1.0);
    lerp(bottom_f, d, 0.1171875)
}
```
**Why it helps:** Even without other changes, this eliminates the most jarring failure modes of unbounded 3D density. Bedrock floor, sky ceiling, smooth approach.

**Cost:** ~10 lines. Probably the cheapest immediate win.

### 8. Anisotropic 3D noise (free win)

**Today:** Relief noise sampled at `RELIEF_PERIOD=32` uniformly in xyz.

**MC's approach:** `xz_scale=0.25`, `y_scale=0.125` — the y wavelength is *half* the xz wavelength. Vertical features look taller than wide. This is one number.

**Why it helps:** Cliffs and overhangs become character features rather than incidental noise wiggle. Costs nothing.

**Cost:** One line.

---

## Part 5 — A possible incremental migration path

Each step is independently shippable; each makes the next step easier.

**Step 1 — anisotropic noise + slides.** Tiny diffs in `heightmap.rs`/`DensityNoise`. Should noticeably change the look. Validates that density-only changes can move the needle.

**Step 2 — surface rules DSL.** Pull existing inline logic into `surface.rs::SurfaceRule`. No visual change. Tests verify equivalence. Fixes the doc-rot in `surface.rs`.

**Step 3 — `CubicSpline<f32→f32>` + erosion noise + new offset/factor formula.** Replace `h_pre = SEA + shelf + ridge + warped_fbm` with a single height computed via spline lookup. Keep plate Voronoi for now as the source of "continentalness" (i.e. use plate_t as continentalness input). Use `quarter_negative` + factor + 3D noise composition. This step removes the need for `CLIFF_MIN_HEIGHT` and the coastal dirt cap.

**Step 4 — replace Voronoi plates with continentalness noise.** Once splines are doing the work, the plate mosaic becomes redundant. Replace `plates.rs::plate_at` with `noise::continentalness(x, z)`. Boundary artifacts disappear by construction.

**Step 5 — multi-noise biome table + R-tree lookup.** Big code drop but isolated. Add depth as a 6th axis even if no underground biomes yet — future-proofs the structure.

**Step 6 — sparse cell-grid density.** Refactor density expression into a graph; build per-chunk interpolator. This is the perf step; do it after the model is settled.

**Step 7 — aquifers.** Last, because it depends on the post-density-graph hook point.

Steps 1, 2, 7 (aquifers) are independent of the rest and can be done out of order if desired.

---

## Part 6 — Open questions for discussion

1. **Plates as a *feature*, not a fault.** Oxium's Voronoi plates produce *distinguishable* continents — a player can recognise "this is the same landmass as last week". MC's pure-noise continentalness is more organic but less narrative. Keep plates as a coarse classifier (continental kind, base height) that *modulates* the spline inputs, rather than replacing them outright?

2. **Hydrology — does MC's lack of D8/Planchon-Darboux matter?** Oxium's `hydrology.rs` is 803 lines of real watershed analysis. MC has no such thing — rivers are noise zero-crossings, lakes are explicit features. Oxium's approach is *better* for realism but expensive. Do we keep it? If yes, splines/density work feeds the heightfield that hydrology operates on; the hydrology module survives untouched.

3. **Graph caves as identity.** Oxium's chambered cave systems with entrances are more "designed" than MC's noise caves. Worth keeping. Should they sit alongside a noise-carver layer (cheese caves for ambient density) like MC has both, or stay alone?

4. **Cell size choice.** MC uses 4×8×4. Oxium's chunks are 32³ vs MC's 16×384×16 — cell size that divides 32 evenly: 4×4×4 (512 evals/chunk) or 4×8×4 (256 evals) or 8×8×8 (64 evals). 4×8×4 matches MC and gives an 8-block vertical resolution.

5. **JSON or Rust for the density graph?** MC is fully data-driven (JSON via Codecs). Oxium's `tuning.rs` is Rust constants. Going JSON is a big commitment but enables hot-reload, mod support, and the same kind of tweakability that made MC's worldgen iteration fast. Cheaper alternative: keep Rust but use the graph type so we *could* JSON-ify later.

6. **PR sequencing vs the in-flight `worldgen-3d-design`.** PR A and PR B already landed parts of the 3D migration. The `quarter_negative` + factor composition above is the natural PR C that replaces PR B's `min(2.0)` hack. Discuss before writing.

---

## Appendix — Where to find the key MC source

All paths absolute under `/Users/fdatoo/Downloads/out`:

| Concept | File |
|---|---|
| Density graph nodes | `net/minecraft/world/level/levelgen/DensityFunctions.java` |
| Named density router | `net/minecraft/world/level/levelgen/NoiseRouterData.java` |
| Spline knots (offset/factor/jaggedness) | `net/minecraft/data/worldgen/TerrainProvider.java` |
| Spline data structure | `net/minecraft/util/CubicSpline.java` |
| Biome table | `net/minecraft/world/level/biome/OverworldBiomeBuilder.java` |
| Climate / R-tree | `net/minecraft/world/level/biome/Climate.java` |
| Chunk gen pipeline | `net/minecraft/world/level/levelgen/NoiseBasedChunkGenerator.java` |
| Cell interpolator | `net/minecraft/world/level/levelgen/NoiseChunk.java` |
| Surface DSL | `net/minecraft/world/level/levelgen/SurfaceRules.java` |
| Overworld surface tree | `net/minecraft/data/worldgen/SurfaceRuleData.java` |
| Aquifer + pressure | `net/minecraft/world/level/levelgen/Aquifer.java` |
| Carvers | `net/minecraft/world/level/levelgen/carver/*.java` |
| Cave noise data | `data/minecraft/worldgen/density_function/overworld/caves/*.json` |
| Top-level overworld config | `data/minecraft/worldgen/noise_settings/overworld.json` |

---

## Decisions Log

Outcomes of the design discussion, recorded in Q&A order. The questions are from Part 6; this section captures what we picked and what was deferred.

### Q1: Plates — decision

**Chosen: Option C + C2** (plates become the continentalness input; add an independent erosion noise with per-plate bias).

- Plate Voronoi distance field (signed: positive inland, negative offshore, blended via `plate_t`) becomes the `continentalness` input to the spline pipeline.
- Plate `kind` (Continental/Oceanic) becomes a threshold check on continentalness, not a separate field.
- Add a new independent 2D *terrain-shape* noise channel (low frequency, ~2000m wavelength like MC's erosion).
- Per-plate `roughness` becomes an *additive bias* on it: `effective_shape = shape_noise(x,z) + plate.roughness_bias`. Preserves "the rocky continent" identity.
- `ridge_lift` is removed — mountains fall out of the spline at low-shape-noise bands.
- `BOUNDARY_RIDGE_WIDTH`, `RIDGE_PEAK_CC/CO/OO`, `CLIFF_MIN_HEIGHT`, and the coastal-gated dirt cap all become obsolete.
- Rename "erosion" to avoid collision with `hydrology.rs`'s actual erosion. Candidates: `terrain_shape`, `mountainness`, `roughness`. Decide at implementation time.

### Q2: Hydrology — decision

**Keep `hydrology.rs` untouched in the migration.** Watershed analysis is Oxium's distinctive strength; MC has nothing comparable.

Two integration adjustments at the interface:

1. **`valley_carve` applies to spline `offset(x,z)`**, not to raw `h_pre`. One-line interface change; module untouched.
2. **Above-water 3D-noise clamp.** In a 4-block Y-band above any column with `is_river || lake_rim.is_some()`, clamp positive 3D-noise contribution to zero. Prevents overhang ceilings from turning rivers/lakes into tunnels. ~5 lines in `mod.rs`. Apply at the point density is composed.

**Cross-region terracing approach: A now, B if needed.**

- **Now (Option A): Boundary stitching.** At fine-region build, sample edge `flow_dir` / `flow_acc` from the four neighbor regions and force agreement at the boundary line. ~80 lines, no new caches. Touch points: `hydrology.rs::build_fine_hydro` (read neighbor edges before flow-direction solve), `region.rs::get_fine` (already has neighbor lookup machinery for the 3×3 halo).
- **Deferred (Option B): Mega-macro tier** — 64km regions, 512m cells. Same sink-fill / D8 / trunk-inject pattern as the existing macro→fine path. ~250 lines, ~1MB extra RAM, catches basins up to 64km.
- **Rejected (Option C):** bake global heightmap at world creation — incompatible with infinite world.

**Trigger for escalating to B:** any of (a) playtest reports of river kinks at region boundaries, (b) visible disagreement of flow direction across a seam, (c) endorheic lakes whose drainage doesn't connect across region edges. None of these are observable until the new worldgen lands, so re-evaluate then.

**Open follow-up:** if A doesn't suffice, implement B. Sketch:
- `MegaMacroRegionCoord` — 64km cells (`MEGA_MACRO_REGION_SIZE = 65536`).
- `MegaMacroRegion` struct holding a 128×128 grid of 512m cells (height, flow_dir, flow_acc, trunk graph).
- Macro pass changes: read mega-macro trunk at region center, inject as macro-scale trunk source the same way macro currently injects into fine.
- New cache `mega_macro_cache: LruCache<MegaMacroRegionCoord, Arc<MegaMacroRegion>>`, cap ~4 entries.
- Constants: `MEGA_MACRO_RIVER_THRESH ≈ 5000` (10× MACRO_RIVER_THRESH).

### Q3: Cave layers — decision

**Chosen: Option C** (keep graph caves + wormholes + add MC-style noise carver layers).

**P0 surface-block-placement bug identified (diagnosed in discussion):** user reported being unable to find caves; on digging straight down, they observed grass-dirt-stone cycles **with no air or water voxels in between** (continuous solid).

Root cause at **`mod.rs:351-352`**: `depth_below_surface: Option<i32> = None` is declared inside the `(x,z)` column loop but the `y` loop only iterates `0..CHUNK_DIM_U` (32 blocks — one chunk's vertical range). When the chunk below this one is generated, `depth_below_surface` resets to `None` again. The top solid voxel of every chunk below the true surface gets `depth=0` → surface block (grass).

Expected cycle period: exactly **32 blocks vertical** (one chunk). Easily verifiable in-game.

There is also a *separate* latent bug (which this discussion originally chased before re-diagnosing): when caves DO carve, the cave floor gets the same treatment because `depth_below_surface = None` resets on the air voxel at `mod.rs:403`. The fix below addresses both at once.

**Important consequence:** the chunk-cycle bug means we currently **cannot conclude from in-game observation whether caves are working**. The cycles are not evidence of caves — they happen even with no carving. After the fix below lands, looking for air voxels underground is the actual test.

**Fix landed in-session (two changes around `mod.rs:351` and `mod.rs:415`):**

1. **Seed `depth_below_surface` from the chunk above.** Before the y-loop, evaluate density one voxel above the chunk's top; if solid, start with `Some(4)` so the first solid voxel in this chunk skips the grass/dirt branches. Fixes the visible 32-block cycle.
2. **Add `near_surface` gate inside the solid branch.** `near_surface = (h_target - wy).abs() <= SURFACE_BAND`. Replace the depth-only conditional with `if col.is_cliff || !near_surface → Stone; else if depth == 0 → surface_selector; else if depth <= 3 → Dirt; else Stone`. Catches the latent cave-floor-grass bug *and* overhangs above the surface band.

Verification: new test `deep_underground_has_no_surface_blocks` (chunk-Y=-2, 16×16 scan, asserts grass=dirt=sand=snow=0). Test failed at **250,484 grass blocks** before fix, passes at 0 after. All other worldgen tests (49 lib + 3 fingerprint integration) unchanged including both golden hashes — confirming the fix is precisely scoped to the bug zone (chunks with above-surface chunk-tops or far-from-surface voxels).

**Caves can now be confirmed visible in-game.** The pre-existing `underground_chunk_has_both_caves_and_solid` test was already passing (caves were carving, just visually hidden by underground "grass meadows"). With the surface bug gone, cave air voxels and water-filled chambers below sea level should be directly observable.

**This bug remains strong evidence for adopting the surface-rules DSL (Q5):** the gate that prevents this class of bug is exactly what an `above_preliminary_surface` condition provides as a first-class primitive. Without a DSL, the same gate has to be remembered in every future surface change.

**Cave-quality tunings also landed in-session** (after the user observed caves were jagged/narrow/flooded):

1. **Lower density cap** (`mod.rs:392-393`): `raw_density.min(2.0)` → `raw_density.min(1.0)`. The carve threshold becomes `cap - cave_contribution > 0 → cave_contribution > 1`, putting the cave wall at SDF=1 instead of SDF=2. With `CAVE_SDF_INTENSITY=4`, that's ratio≈0.75 of chamber radius and 75% of tunnel radius (vs 50% before). Cave volume ~2× bigger; walls smoother (less voxel granularity per surface block).
2. **Primitive aquifer rule** (`mod.rs:399-415`): replaced "any air below sea level → water" with "water only under lake_rim or in ocean columns (`height <= SEA_LEVEL`)". Caves under land columns now stay dry. Caves under the seabed still flood (correct). Caves under lakes flood from the lake (correct). This is a stand-in until the real MC aquifer lands.
3. **Bump tunnel radius** (`tuning.rs:181`): `(2.0, 3.0)` → `(3.0, 4.5)`. With 75% effective carve, navigable tunnel widths become 4.5..6.75 blocks.

Verification: new test `deep_caves_under_land_are_dry` (finds a lake-free all-land chunk, asserts no Water in deep underground). Failed at 453 water blocks (after restricting to lake-free chunks: still 263), passes at 0 after the aquifer fix. All other tests preserved, including both golden hashes (golden chunk at Y=64..95 is above cave bands; fingerprint test samples the 2D heightmap which is untouched).

**All three tunings are temporary.** They will be replaced by the MC `quarter_negative * factor` composition + real aquifer in the larger migration. Don't get attached to the exact values.

### Q4: Cell size for sparse density evaluation — decision

**Chosen: 4×4×4 cells, with 2D fields flat-cached at per-chunk resolution.**

- Cells/chunk: 8×8×8 = 512. Corner lattice: 9×9×9 = 729. Speedup vs per-voxel for the heavy density function: ~45×.
- Why 4×4×4 not MC's 4×8×4: Oxium chunks are 32 tall (not MC's 384). 4-block vertical cells give 8 cells per chunk vs 4 — comfortable margin against visible interpolation banding. Cliffs and overhangs need vertical fidelity to match horizontal. Also keeps the `SURFACE_BAND=16` near-surface gate spread across 4 cells (vs 2 with 4×8×4).
- Caves stay per-voxel. Cave SDFs / noise carvers (cheese, spaghetti) need 1-voxel resolution; interpolating at 4-block scale would blur tunnel walls. Density gets the interpolation; caves are subtracted from the trilerped density.
- 2D fields flat-cached per chunk: climate (T, H, C, E, PV), per-column heights, river masks. Quart resolution (8×8 = 64 entries × ~6 fields) ≈ 1.5KB per chunk. ~16× fewer 2D noise samples per chunk. Negligible memory cost, big speedup, bundle with cell interpolation.

**Implementation pattern:** MC's "sliding YZ wall" interpolator from `NoiseChunk.java`. Outer loop over cell-X, fill a `(cellCount_Z+1) × (cellCount_Y+1)` wall of corner samples per X advance, then `swap_slices`. Hierarchical Y→X→Z lerp accumulators inside, keeping each voxel at 7 lerp ops total. Marker wrappers (`Interpolated`, `FlatCache`, `CacheOnce`, `CacheAllInCell`) annotate the density graph; a per-chunk pre-pass walks the tree and substitutes runtime cache instances.

**Sequencing:** **defer implementation until the spline migration is in flight.** The graph structure depends on the spline output shape (offset/factor/jaggedness 2D fields, 3D base noise, cave subtractions). Building the interpolator before the graph is settled would mean retrofitting it twice. Estimated effort when we get to it: ~400 LOC.

### Q5: Data-driven worldgen — decision

**Chosen: Hybrid C + B1 — graph in Rust, tunable values in RON, file watcher with hot reload (always on, including release builds).**

What lives where:

| Rust (recompile to change) | RON (hot-reload) |
|---|---|
| `DensityFn` enum + graph topology | Noise frequencies, amplitudes, octaves |
| Algorithm choices (D8, Voronoi, MST) | Spline knot tables (loc/val/slope) |
| Marker placement (Interpolated, FlatCache) | Biome climate boxes (T/H/C/E ranges) |
| Surface-rule DSL nodes | Cave probabilities, depth bands, radii |
| Aquifer / carver structure | All `tuning.rs` constants |

Implementation:
- `assets/worldgen/default.ron` bundled with binary (ground truth for tests).
- `WorldgenConfig` struct, Serde-derived, owns all tunable values.
- `Generator::new(seed, &WorldgenConfig)` — config injected at construction.
- `notify-debouncer-mini` crate for file watching (handles editor save-bursts).
- On change: parse → validate → rebuild Generator → clear chunk cache → trigger regen of loaded chunks.
- Validation errors: log, keep old config, never crash on typo.
- Tests always load bundled default — determinism preserved.
- **Always on in release builds.** Enables data packs / mods if the project ever goes there. Watcher thread cost is negligible.

Format: **RON, not JSON.** Comments allowed, Rust-syntax-like (`Knot(loc: -0.5, val: 0.3, slope: 0.0)`), Serde-native.

**Sequencing:** bundle with PR 1 of the spline migration. PR 1 introduces the first spline knot tables — natural insertion point for `WorldgenConfig`. Migrate existing `tuning.rs` constants opportunistically as later PRs touch them.

**`WorldgenConfig` schema design note:** acts as public surface for tuning. Hard to remove fields once shipped (compatibility). Design with the migration's full vision in mind, not just PR 1.

### Q6: PR sequencing — decision

**Chosen: Path A — write plan docs for all 8 PRs upfront, then execute in order.**

Rationale: each PR's design depends on the structural shape set by earlier PRs (especially PR 2 foundation and PR 5 graph). Writing them all upfront catches inter-PR shape mismatches before any code lands. Heavier upfront cost, but the team avoids "we built X expecting Y, but actually Y needed Z" mid-stream.

**Locked-in PR sequence:**

| # | Title | LOC est. | Depends on |
|---|---|---|---|
| 1 | Hydrology boundary stitching | ~80 | landed work |
| 2 | Foundation: density composition + hot reload + flat cache + splines | ~800 | landed work |
| 3 | Spline-driven heightmap with continentalness + erosion | ~400 | PR 2 |
| 4 | Multi-noise biome lookup (6D R-tree) | ~600 | PR 3 |
| 5 | Cell interpolation + density graph as enum | ~500 | PR 4 |
| 6 | Surface rules DSL | ~400 | PR 5 |
| 7 | Real aquifer (replacing ocean-column rule) | ~400 | PR 5 |
| 8 | Noise carver layers (cheese, spaghetti) | ~300 | PR 5 |

PR 1 is independent; PRs 2→3→4→5 are sequential; PR 5 unblocks PR 6/7/8 which can be done in any order.

Total: ~3500 LOC. Both golden hashes need re-baselining in PR 2, 3, 4, 5, 7, 8.

**Each PR gets its own plan doc** under `docs/superpowers/plans/`, written upfront before any PR begins implementation. The plan docs are the authoritative blueprint for each PR; this research doc is the design rationale they refer back to.

---

## Implementation status (live)

Landed in-session, uncommitted (a single small standalone fix):
- Per-chunk depth-reset bug fix (`mod.rs:351-360`)
- `near_surface` gate inside surface block selector (`mod.rs:415-485`)
- Density cap lowered: `min(2.0)` → `min(1.0)` (`mod.rs:392-405`)
- Primitive aquifer rule: "water only in ocean columns or under lakes" (`mod.rs:415-432`)
- Tunnel radius bumped: `(2.0, 3.0)` → `(3.0, 4.5)` (`tuning.rs:181`)
- Two new TDD-grown tests: `deep_underground_has_no_surface_blocks`, `deep_caves_under_land_are_dry`

This work is **not** considered PR 0 — it's a separate small fix that can ship independently or fold into PR 1.

To plan (in order): PR 1 → PR 2 → PR 3 → PR 4 → PR 5 → PR 6 → PR 7 → PR 8.

**After the P0 investigation:**

Noise layers to add (in priority order):

1. **Cheese caves** — single 3D noise threshold, tuned mild, active in the Y-window between Shallow band (10..50) and Deep band (-110..-30) — roughly `[-30, 10]`. ~20 lines. Provides the "honeycombed underground" feeling.
2. **Spaghetti tubes** — two noises gated by a rarity field, Y-clamped gradient for slow drift. ~40 lines. Inter-region tendrils that complement local graph chambers.

**Composition under the new MC-style density:**

```
density = 4 * quarter_negative((depth + jagged) * factor) + base_3d_noise
cave_subtraction = max(
    cheese_noise_contribution,
    spaghetti_contribution,
    graph_cave_sdf,
    entrance_sdf,
    wormhole_contribution,
)
density -= cave_subtraction
```

The `quarter_negative` softening above the surface eliminates the need for PR B's `min(2.0)` cap — the bias above the iso-surface is small enough that any positive cave contribution carves cleanly. PR B's cap can be removed when the new composition lands.

**Non-negotiable:** keep graph caves, wormholes, entrance rolls. MC has nothing equivalent and these are Oxium's identity.
