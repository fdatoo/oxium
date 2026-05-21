# Part III — Per-Region Build Implementation Plan (Phase 1, Plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the six chapters of *Part III — Per-Region Build* (3.1 Plates, 3.2 Climate, 3.3 Heightmap, 3.4 Hydrology, 3.5 Rivers/Valleys/Lakes, 3.6 Cave Systems), each grounded in the actual Oxium source. These chapters expand the stages that 2.1 introduced; they're the first place a reader gets the *full* story of what each per-region-cached subsystem does.

**Architecture:** One task per chapter. Each task: (a) implementer reads the relevant source file(s); (b) generates any chapter-specific images via `doc_render` and `gen-images.sh` (most chapters reuse 2.1's images, but a few need closer crops or different zooms); (c) writes the chapter; (d) builds; (e) commits. **One widget** in this plan: `D8Stepper` in chapter 3.4, which the design spec called out as the most-useful interactive demo for hydrology.

**Tech Stack:** Same as prior plans — Docusaurus 3 + TypeScript MDX; existing `doc_render` Rust binary; existing widget infrastructure.

**Out of plan scope:** Parts IV, V, appendices.

---

## Pre-flight: enter a worktree

```
EnterWorktree(name: "docs-part3")
```

Then:

```bash
git merge main --ff-only
cd docs/book && npm ci && cd ../..
cargo build --release --bin doc_render
```

Verify Part II content is present:

```bash
ls docs/book/content/part-2-overview/
ls docs/book/static/img/generated/  # should show ~13 images from Plans 1-3
```

---

## Shared patterns for every task

Read these once at the start of the plan; they're not repeated per task.

- **Chapter skeleton template:** `docs/book/content/part-1-foundations/1.5-voronoi.mdx` and `1.7-splines.mdx` are the closest references — they each combine engine grounding, math intuition, and forward links to other chapters. Use the standard 6 sections (opening hook → intuition → build-it → in-the-engine → what-you-can-now-do → next).
- **Image-embedding pattern:** `![Caption](/oxium/img/generated/<filename>.png)` followed by an italic caption paragraph. Path prefix `/oxium/` is the Docusaurus baseUrl; matches what chapter 2.1 does.
- **Forward-link pattern:** Relative paths to siblings (`./3.2-climate.mdx`) and to other parts (`../part-4-chunk-fill/4.3-composing-caves.mdx`).
- **Code-block convention:** `// src/worldgen/<file>.rs` header line on every Rust snippet that's from the engine, plus `// simplified for exposition` if simplified.
- **Honest about engine reality:** Prior implementers have repeatedly found that the design spec diverges from current code. **Every task starts with reading the source.** If a planned claim doesn't match reality, the implementer fixes the chapter rather than the code.
- **No widgets unless the task explicitly says so.** Only Task 4 (Hydrology) has a widget.

---

## Task 1: Chapter 3.1 — Plates

The full Voronoi-plate decomposition story: how plates are rolled per cell, how they get their kind/elevation/roughness, how the boundary-intensity `t` shapes geography, how `signed_continentalness` is derived.

### Files

- Modify: `docs/book/content/part-3-region-build/3.1-plates.mdx`
- Possibly modify: `docs/book/tools/gen-images.sh` (add a closer-zoom plate-id image at zoom 8)
- Possibly modify: `docs/book/static/img/generated/` (new image)

### Steps

- [ ] **Step 1: Read the plate module**

```bash
cat src/worldgen/plates.rs
grep -n "PLATE_CELL_SIZE\|CONTINENTAL_RATIO\|signed_continentalness" src/worldgen/tuning.rs src/worldgen/plates.rs src/worldgen/heightmap.rs | head -20
```

Key things to surface in the chapter:
- The `PlateId { cell_x, cell_z }` struct (cell coords ARE the ID).
- `PlateKind { Continental, Oceanic }`.
- The 3×3 nearest-seed scan returning a "look" with `a`, `b`, and boundary `t`.
- `PLATE_CELL_SIZE` value (likely 1024 blocks per side — verify).
- `signed_continentalness` (in `heightmap.rs` or `plates.rs`) — the climate axis derived from plates.
- The `CONTINENTAL_RATIO` constant (default ~0.45) — what fraction of plates are continental.

- [ ] **Step 2: Generate a closer-zoom plate-id image (optional)**

Chapter 2.1's plate-id image is at zoom 16 — wide enough to see continents. For this chapter, a closer zoom (z=4, ~1024 blocks across) lets the reader see individual cells more clearly. Add to `gen-images.sh`:

```bash
# Chapter 3.1 — closer plate view, ~1024 blocks across.
render plate-id 0,0 4 256
```

Run `bash docs/book/tools/gen-images.sh` and verify the new PNG `plate-id-s42-0_0-z4-w256.png` appears.

(Optional — you can skip this and reuse the z=16 image from chapter 2.1 if the z=16 view shows enough detail. Use your judgment after looking at it.)

- [ ] **Step 3: Write the chapter**

Replace `docs/book/content/part-3-region-build/3.1-plates.mdx`. Target ~1500–2000 words.

Structure:
- **Hook:** Continents and oceans need clean shapes that hold together across thousands of blocks. Pure noise gives you a "Swiss-cheese" world where land and sea interleave at every scale. Plates give you the macro structure that noise can then decorate.
- **The picture (with widget reference to 1.5 and one or two images).** Image: plate-id (z=4 or reuse z=16). Caption explains the cells are tectonic plates, each ~1024 blocks per side.
- **Build it.** Step through the algorithm:
  1. Per cell, hash the cell coords + a salt to get jitter offset → seed position.
  2. Per cell, hash the cell coords + a different salt to get `kind: Continental | Oceanic`, with `CONTINENTAL_RATIO` controlling the fraction.
  3. Per cell, hash to get a `roughness` multiplier (e.g., `0.8`–`1.4`) — what plays into the FBM amplitude for that plate's interior.
  4. For a query column, scan the 3×3 neighborhood, find `a` (nearest) and `b` (second-nearest); compute `d_a`, `d_b`, and `t = (d_b - d_a) / (d_b + d_a)`.
- **In the engine.** Show the actual `plate_at` function signature and how it returns a `PlateLookup` (or whatever the real struct name is — verify). Show `signed_continentalness` — a function that derives a smooth `[-1, 1]` field from the plate kinds and boundary `t`:
  - Inside continental plate: positive (e.g., +0.8).
  - Inside oceanic plate: negative (e.g., -0.5).
  - At boundaries: smoothly interpolated.
- **What you can now do.** Read `src/worldgen/plates.rs` end to end. Recognize `signed_continentalness` calls in `heightmap.rs`.
- **Next:** → `3.2 Climate`.

Reference `[1.5 Voronoi](../part-1-foundations/1.5-voronoi.mdx)` for the underlying Voronoi pattern and `[1.2 Determinism](../part-1-foundations/1.2-determinism.mdx)` for the hash mixer.

- [ ] **Step 4: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 5: Commit**

```bash
git add docs/book/content/part-3-region-build/3.1-plates.mdx docs/book/tools/gen-images.sh docs/book/static/img/generated/
git commit -m "content(book): write 3.1 Plates

Chapter on the Voronoi plate decomposition. Continental vs oceanic
rolls, per-plate properties, the 3x3 nearest-seed scan, the boundary
intensity t and signed_continentalness derived from it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Chapter 3.2 — Climate

The full 6-axis climate vector story: temperature, humidity, continentalness, terrain_shape, ridges_pv, depth, plus weirdness. The biome R-tree lookup. The per-block Voronoi jitter that makes biome edges organic.

### Files

- Modify: `docs/book/content/part-3-region-build/3.2-climate.mdx`

### Steps

- [ ] **Step 1: Read the climate module**

```bash
cat src/worldgen/climate.rs | head -200
grep -n "TargetPoint\|ParameterList\|lookup\|Biome::" src/worldgen/climate.rs | head -30
grep -n "voronoi_jitter" src/worldgen/climate.rs | head -5
```

Key things to surface:
- The `Biome` enum (Tundra, SnowyForest, Plains, Forest, Desert, Tropical — verify the variant list).
- The `TargetPoint` struct — what climate axes does it carry?
- The `ParameterList` (the R-tree). What does an entry look like — a hyper-rectangle in climate space mapping to a Biome?
- The per-block hash jitter perturbing `(wx, wz)` before the lookup. Where is `voronoi_jitter_offset` and how is it computed?
- The role of each axis: temperature (cold↔warm), humidity (dry↔wet), continentalness (ocean↔inland), terrain_shape (?), ridges_pv (?), depth (high-altitude↔buried), weirdness (variant unlock).

Climate has a lot in it. Don't try to explain every detail — focus on the *shape* (6+1 axes, R-tree lookup, jitter for organic edges) and what each axis means.

- [ ] **Step 2: Write the chapter**

Replace `docs/book/content/part-3-region-build/3.2-climate.mdx`. Target ~1800–2200 words (this chapter is heavier — there's more to explain).

Structure:
- **Hook:** Biomes look discrete (desert vs forest vs tundra) but the engine doesn't store a biome per column — it stores six continuous climate axes and runs a *lookup* against an authored set of biome rectangles. This decouples "where is climate" from "what biome name lives there" and lets the same climate produce different biomes depending on which biome table is loaded.
- **The picture.** Reuse 2.1's temperature, humidity, weirdness, biome-id images. Briefly walk through what each axis is and how it varies spatially.
- **Build it.** The 6 (or 7, with weirdness) climate axes:
  1. **Temperature, humidity** — large-period FBM noise, simple.
  2. **Continentalness** — from plates (carry-over from 3.1).
  3. **Terrain_shape** — another FBM that subtly varies terrain character (peaks vs plateaus).
  4. **Ridges_pv** (peaks and valleys) — yet another FBM signal that gates whether a "high" terrain_shape produces sharp ridges or rounded hills.
  5. **Depth** — derived from world-Y for the biome lookup; lets underground regions get cave biomes.
  6. **Weirdness** — mid-frequency FBM. The "rare variant" axis (analog of Minecraft's weirdness): high positive values unlock unusual biomes like ice spikes or sunflower plains (in future expansions).
- **Build it (continued).** The R-tree lookup: each biome entry is a hyper-rectangle in 6D space; the lookup finds the entry whose rectangle contains the query point. R-trees are spatial-index data structures that can answer "which boxes contain this point?" in O(log n) for small n; the engine has ~20–50 biome entries so the practical cost is negligible.
- **Build it (continued).** The per-block jitter: before looking up, the engine perturbs `(wx, wz)` by a small hash-derived offset. This makes biome edges *look* organic rather than perfectly aligned with FBM contours. The jitter amplitude (`BIOME_JITTER_RADIUS` — verify in `tuning.rs`) is small (e.g., 4 blocks).
- **In the engine.** Show:
  - `Generator::column_data` excerpt where it samples all axes and builds a `TargetPoint`.
  - `ParameterList::lookup(target)` returning a `Biome`.
- **What you can now do.** Read `climate.rs` end to end; understand every `cfg.biomes.entries` entry; tweak the biome table in `assets/worldgen/default.ron` (the runtime config).
- **Next:** → `3.3 Heightmap`.

- [ ] **Step 3: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 4: Commit**

```bash
git add docs/book/content/part-3-region-build/3.2-climate.mdx
git commit -m "content(book): write 3.2 Climate

Chapter on the 6+1 axis climate vector and R-tree biome lookup.
Covers temperature/humidity/continentalness/terrain_shape/ridges_pv/
depth/weirdness, the biome rectangle table, and the per-block Voronoi
jitter that makes biome edges look organic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Chapter 3.3 — Heightmap (`h_pre`)

How the spline pipeline turns climate axes into a target surface height per column. The offset/factor/jaggedness splines. The role of the 3D base FBM. Cliff detection.

### Files

- Modify: `docs/book/content/part-3-region-build/3.3-heightmap.mdx`

### Steps

- [ ] **Step 1: Read the heightmap module**

```bash
cat src/worldgen/heightmap.rs | head -100
grep -n "h_pre\|offset_spline\|factor_spline\|jaggedness_spline\|slope_at\|is_cliff" src/worldgen/heightmap.rs | head -20
grep -n "ClimateConfig\|offset_spline\|factor_spline\|jaggedness_spline" src/worldgen/config.rs | head -20
```

Key things to surface:
- The mapping from `(continentalness, terrain_shape, ridges_pv)` triple to a height value via a **nested** spline pipeline (each "outer" spline maps continentalness to either a scalar or to another spline keyed on the next axis).
- The three spline kinds: `offset_spline` (the base height curve), `factor_spline` (a multiplier), `jaggedness_spline` (controls peak sharpness). They compose into the final density bias.
- The 3D FBM that adds finer-grained roughness (in `DensityNoise::evaluate_base_3d`).
- `slope_at` and `is_cliff` — how the engine detects steep gradients for cliff-surface rules (links to 4.6).

- [ ] **Step 2: Write the chapter**

Replace `docs/book/content/part-3-region-build/3.3-heightmap.mdx`. Target ~1600–2000 words.

Structure:
- **Hook:** The heightmap is the engine's most carefully tuned subsystem. Climate axes go in; a target height per column comes out. The trick is that the mapping is *authored*, not computed — three spline curves (offset, factor, jaggedness) shape every continent's altitude profile.
- **The picture.** Reuse 2.1's `h-pre` image. Show what "target surface height per column" looks like — terrain ramp from teal (sea level) to warm tones (peaks).
- **Build it.** The spline pipeline:
  1. Sample the climate axes at the column.
  2. `offset_spline(continentalness, terrain_shape, ridges_pv)` → a base elevation contribution (signed).
  3. `factor_spline(...)` → a multiplier that scales the FBM roughness regionally (continental plates get more relief; oceanic less).
  4. `jaggedness_spline(...)` → a sharpness control that determines whether high terrain looks like rounded hills or knife-edge ridges.
  5. The 3D FBM `base_3d(wx, wy, wz)` provides roughness.
  6. Combine: `density(wx, wy, wz) = offset + base_3d × factor × jaggedness_modifier - bias_from_y(wy)`.

  The result is a signed density field, not directly a height. Where density goes from positive (solid) to negative (air) is the surface — found later in the per-chunk fill via a top-down scan.

- **Build it (continued).** Slopes and cliffs:
  - `slope_at(wx, wz)` = magnitude of the gradient of `h_pre` at the column, computed by sampling `h_pre` at small `(wx ± δ, wz)` offsets.
  - `is_cliff` thresholds the slope. Cliffs become bare stone with no soil.

- **In the engine.** Show `ClimateConfig::offset_spline_at` (note: implementers in prior plans verified `ClimateConfig` is the actual struct, not `HeightmapNoise`). Show how `Generator::column_data_with` calls `heightmap::h_pre` and `heightmap::is_cliff`.

- **What you can now do.** Read `heightmap.rs` end to end. Understand the spline configs in `assets/worldgen/default.ron`. Tweak the curves to change terrain character globally.

- **Next:** → `3.4 Hydrology`.

- [ ] **Step 3: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 4: Commit**

```bash
git add docs/book/content/part-3-region-build/3.3-heightmap.mdx
git commit -m "content(book): write 3.3 Heightmap (h_pre)

Chapter on the climate-driven spline pipeline that produces target
height per column. Covers offset/factor/jaggedness splines, the 3D
base FBM that adds roughness, slope/cliff detection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Chapter 3.4 — Hydrology + `D8Stepper` widget

Flow accumulation on the heightmap: D8 direction, Planchon–Darboux sink fill, accumulation by topological sort. Fine + macro hierarchical pass. The trunk-injection trick. Includes the headline `D8Stepper` widget.

### Files

- Modify: `docs/book/content/part-3-region-build/3.4-hydrology.mdx`
- Create: `docs/book/src/widgets/D8Stepper.tsx`

### Steps

- [ ] **Step 1: Read the hydrology module**

```bash
head -100 src/worldgen/hydrology.rs
grep -n "fn d8_dir\|fn flow_acc\|fn sink_fill\|fn build_fine\|fn build_macro\|trunk_inject" src/worldgen/hydrology.rs | head -20
grep -n "FINE_CELL\|FINE_REGION_SIZE\|MACRO_CELL\|MACRO_REGION_SIZE\|RIVER_THRESH\|MACRO_RIVER_THRESH" src/worldgen/tuning.rs | head -10
```

Hydrology is 39 KB of code — don't try to read all of it. Focus on the *shape*:
- D8: each cell picks 1 of 8 neighbours as its downhill direction (or marks itself a sink).
- Planchon–Darboux: iterative sink-fill that raises sink cells just enough that water can flow out, bounded to a local window.
- Flow accumulation: topological-sort cells by elevation descending; each cell donates its area (1 + injected upstream) to its downhill neighbour. O(N) per region.
- Fine vs macro: same algorithm at two cell sizes (8 m fine, 64 m macro). Macro provides trunk-river starting accumulation that fine builds inject before their local accumulation pass.
- Threshold (`RIVER_THRESH` ≈ 50 in fine cells) determines which cells become rivers.

- [ ] **Step 2: Build the `D8Stepper` widget**

The widget shows D8 flow accumulation running step-by-step on a small synthetic heightmap. The reader presses "Step" to advance one cell at a time, watching cells light up as their accumulation arrives at downstream neighbours.

Create `docs/book/src/widgets/D8Stepper.tsx`. Spec:
- 16×16 grid of cells (256×256 canvas, 16 px per cell).
- Default heightmap: a downward-sloping ridge with some local variation. Could be `h(x, y) = (15 - y) * 1.0 + sin(x * 0.7) * 0.5` plus a couple of synthetic bumps to create interesting flow patterns.
- Compute D8 direction per cell (8 neighbours; pick steepest downhill; if no downhill, mark sink).
- Sort cells by elevation descending.
- Animate the accumulation: maintain a `currentStep: number`, advance it by 1 each "Step" button click. Render cells colored by their accumulated area: 1 = pale; logarithmic ramp up to bright red for high-accumulation cells.
- Cells beyond `currentStep` show their elevation (grayscale).
- Cells up to `currentStep` show their flow_acc (color ramp).
- Draw small arrows from each processed cell to its downhill neighbour.
- Toggle: "show D8 directions only" — overlays an arrow per cell without coloring.
- Auto-step button (advances every 200ms until done).

Model after `SplineEditor.tsx` for mouse/state management complexity. Aim for ~250 lines. If it gets bigger, simplify by dropping the auto-step or the toggle.

The math is short:

```typescript
type Cell = { x: number; y: number; elev: number; acc: number; downhill: number | null };
const N = 16;

function buildGrid(): Cell[][] {
  const g: Cell[][] = [];
  for (let y = 0; y < N; y++) {
    const row: Cell[] = [];
    for (let x = 0; x < N; x++) {
      const elev = (N - 1 - y) * 1.0 + Math.sin(x * 0.7) * 0.5 + (x === 4 && y === 5 ? 2.5 : 0);
      row.push({ x, y, elev, acc: 1, downhill: null });
    }
    g.push(row);
  }
  return g;
}

function d8(grid: Cell[][], cx: number, cy: number): number | null {
  const here = grid[cy][cx];
  let bestSlope = 0;
  let best: number | null = null;
  const dirs = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
  for (let i = 0; i < 8; i++) {
    const [dx, dy] = dirs[i];
    const nx = cx + dx, ny = cy + dy;
    if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
    const drop = here.elev - grid[ny][nx].elev;
    if (drop <= 0) continue;
    const dist = Math.hypot(dx, dy);
    const slope = drop / dist;
    if (slope > bestSlope) { bestSlope = slope; best = i; }
  }
  return best;
}
```

Process by topological order (sort cells by elev descending; each cell donates its `acc` to its downhill neighbour).

- [ ] **Step 3: Write the chapter**

Replace `docs/book/content/part-3-region-build/3.4-hydrology.mdx`. Target ~1700–2100 words.

Structure:
- **Hook:** Real worlds have rivers that flow downhill from sources to oceans. Random noise can't do that — every column would be a local minimum. The engine simulates flow accumulation: D8 picks each cell's downhill neighbour; topological sort lets each cell donate its area to its donator. The result is a tree of rivers.
- **The picture (widget).** Embed `<D8Stepper>` — let the reader step through the algorithm.
- **Build it.** Walk the algorithm:
  1. D8 flow direction per cell.
  2. Planchon–Darboux sink fill (bounded — only fills sinks within a window).
  3. Topological sort by elevation descending.
  4. Flow accumulation: each cell starts at 1, donates to downhill, accumulates.
- **Build it (continued).** Hierarchical pass:
  1. Macro grid (64-block cells, 8192-block regions): same algorithm at coarser resolution. Identifies trunk rivers that span many fine regions.
  2. Fine grid (8-block cells, 512-block regions): runs after macro, *injects* macro trunks as starting accumulation before running its own accumulation pass.
  3. Why hierarchical? A purely fine grid has a 512-block horizon — only sees drainage from ~6 km² of upstream area. Macro's 8 km horizon sees ~60 km² of drainage, identifying continental-scale rivers that fine grids can't.
- **Build it (continued).** River threshold (`RIVER_THRESH` from `tuning.rs`) — cells with accumulation above this become rivers. Width: `sqrt(acc) × WIDTH_SCALE`.
- **In the engine.** Show the `build_fine_hydro` flow and how it injects macro trunk data. Cite where `flow_acc` and `is_river` live in `FineRegion`.
- **What you can now do.** Read `hydrology.rs`. Understand the macro/fine cache caps. Tune `RIVER_THRESH` to change what counts as a river.
- **Next:** → `3.5 Rivers, Valleys, Lakes`.

- [ ] **Step 4: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 5: Commit**

```bash
git add docs/book/src/widgets/D8Stepper.tsx docs/book/content/part-3-region-build/3.4-hydrology.mdx
git commit -m "content(book): write 3.4 Hydrology + D8Stepper widget

Chapter on flow accumulation. Covers D8 direction, Planchon–Darboux
sink fill, accumulation by topological sort, and the macro/fine
hierarchical pass with trunk injection. Widget steps through D8 +
accumulation cell-by-cell on a 16x16 grid.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Chapter 3.5 — Rivers, Valleys, Lakes

How river cells become river *segments* with width and a meandering centerline. The U-profile valley carve. Lake rims. Mouth flare.

### Files

- Modify: `docs/book/content/part-3-region-build/3.5-rivers-lakes.mdx`

### Steps

- [ ] **Step 1: Read the relevant hydrology code**

```bash
grep -n "fn perpendicular_distance\|fn valley_carve\|RiverSegment\|kd-tree\|MEANDER\|MOUTH" src/worldgen/hydrology.rs | head -20
grep -n "VALLEY_HALF_WIDTH\|RIVER_BED_DEPTH\|MOUTH_FLARE\|MEANDER" src/worldgen/tuning.rs
```

Key things:
- `RiverSegment` struct: a piecewise-linear path between cell centers.
- The kd-tree built per fine region for fast "nearest segment" lookups.
- `perpendicular_distance` — distance from query column to the centerline of the nearest segment, perturbed by hash-derived meander offset (this is the domain-warping from 1.6 still alive in the engine).
- `valley_carve` — the U-profile depth formula: flat bottom within `half_width`, smoothstep falloff out to `half_valley = width × VALLEY_HALF_WIDTH_MULT`.
- Lake rims — for sink-fill basins that became lakes, the rim elevation is stored and used by `fill_chunk` to flood any air voxel at or below `rim_y`.
- Mouth flare — river-mouth cells get `width × MOUTH_FLARE_MULT` (default 1.6) so deltas read as flared.

- [ ] **Step 2: Write the chapter**

Replace `docs/book/content/part-3-region-build/3.5-rivers-lakes.mdx`. Target ~1500–1800 words.

Structure:
- **Hook:** D8 gave us flow cells; this chapter is about turning cells into shapes a player sees. Rivers carve U-profile valleys; lakes flood basins; deltas flare at the mouth.
- **The picture.** Reuse 2.1's `valley-carve` and `h-target` images. Caption: valley-carve as the sparse depth field, h-target as the result after subtraction.
- **Build it.**
  - **Segments.** A river cell points to its downhill neighbour; concatenating those pointers gives a piecewise-linear centerline.
  - **Width.** `width = sqrt(acc) × RIVER_WIDTH_SCALE`, clamped to `[MIN_WIDTH, MAX_WIDTH]`.
  - **Meander warp.** The straight-line centerline gets perturbed perpendicular by a hash-derived offset whose amplitude scales with width: `meander_amp = clamp(width × MEANDER_AMP_PER_WIDTH, 0, MAX_MEANDER_AMP)`. Big rivers meander; streams stay straight.
  - **Perpendicular distance.** For a query column, find the nearest segment (kd-tree), compute distance to the warped centerline, return.
  - **U-profile carve.** Given that distance `d` and the segment's width:

```rust
let half_width  = segment.width * 0.5;
let half_valley = segment.width * VALLEY_HALF_WIDTH_MULT; // default 3.0
let depth = if d <= half_width {
    RIVER_BED_DEPTH                                       // flat bottom carved fully
} else if d < half_valley {
    let t = (d - half_width) / (half_valley - half_width);
    RIVER_BED_DEPTH * smoothstep(1.0, 0.0, t)
} else { 0.0 };
```

  - **Lakes.** Sink-fill cells with basin elevation. `fill_chunk` reads `lake_rim` per column and floods any air voxel `wy <= lake_rim` to water.
  - **Mouths.** A mouth is a river cell whose downstream is at or below sea level. Get `width × MOUTH_FLARE_MULT`.
- **In the engine.** Show `valley_carve` function signature and how `column_data` calls it. Mention the kd-tree is built lazily on first query per region.
- **What you can now do.** Read the rivers section of `hydrology.rs`. Tune `RIVER_WIDTH_SCALE` and `MEANDER_AMP_PER_WIDTH` to change river character globally.
- **Next:** → `3.6 Cave Systems`.

- [ ] **Step 3: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 4: Commit**

```bash
git add docs/book/content/part-3-region-build/3.5-rivers-lakes.mdx
git commit -m "content(book): write 3.5 Rivers, Valleys, Lakes

Chapter on segment-based rivers: width = sqrt(acc) * SCALE, meander
warp scaled with width, U-profile carve depth, kd-tree lookup, lake
rims from sink-fill basins, mouth flare.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Chapter 3.6 — Cave Systems

Per-region cave system rolls: 1–4 systems per region, each in a depth band. Chamber Poisson-disk sampling. MST + 1–2 extra loops. Spline tunnels with control-point offsets (1.6 callback). The three surface entrance types.

### Files

- Modify: `docs/book/content/part-3-region-build/3.6-cave-systems.mdx`

### Steps

- [ ] **Step 1: Read the caves module (just the system-rolling part)**

```bash
grep -n "build_systems_for_region\|CaveSystem\|chamber\|tunnel\|sinkhole\|cliff_mouth\|skylight\|MST\|poisson" src/worldgen/caves.rs | head -40
sed -n '1,120p' src/worldgen/caves.rs
```

Caves is the biggest file (55 KB) — don't try to read it all. Focus on system-building (vs the SDF carving math which is in 4.3 / 4.4 / 4.5).

Key things:
- Per region, hash to decide 1–4 systems.
- Each system gets a depth band: Shallow (y 10–50), Middle (y -40–30), Deep (y -110– -30).
- Within a system bounding box, 3D Poisson-disk samples produce 4–8 chamber centers.
- Chambers get oblong ellipsoid radii (independent per axis, ~6–14 blocks).
- MST connects chambers + 1–2 random extra edges (loops).
- Each edge is a cubic spline of 2–4 control points pushed off by domain-warped 3D noise — domain warping from 1.6 alive and well.
- Per chamber: roll for surface entrance. Three types: Sinkhole (chamber top near surface), Cliff Mouth (steep gradient nearby), Skylight (chamber 30–60 blocks below surface).

- [ ] **Step 2: Write the chapter**

Replace `docs/book/content/part-3-region-build/3.6-cave-systems.mdx`. Target ~1800–2200 words.

Structure:
- **Hook:** Caves can't be a uniform noise function — players need to recognize chambers, follow tunnels, find exits. The engine generates each cave system as an explicit graph: chambers connected by tunnels, with named surface entrances. The carving math (how chambers become voxels) lives in 4.3; this chapter is about *which* chambers, *where*, and *how they connect*.
- **The picture.** Hard to image-render caves in 2D. Either embed a Mermaid diagram showing chambers + edges + entrance types, OR reuse 2.1 verbiage and lean on prose. Pick the cleanest path.
- **Build it.** System rolling per region:
  1. `hash::mix(seed, &[region.x, region.z, SALT]) % 4 + 1` → 1–4 systems.
  2. Per system, roll a depth band (Shallow / Middle / Deep) — different bands have different entrance probabilities.
  3. Within the bounding box, run 3D Poisson-disk sampling: pick chamber centers such that no two are closer than `chamber_radius × 3` (default `POISSON_MIN_SPACING_MULT = 3.0`). Iterate ~30 candidate attempts per accepted point until convergence.
- **Build it (continued).** Chamber → MST → tunnels:
  1. Each chamber's ellipsoid radii: 3 independent rolls (X, Y, Z) in `[6, 14]` blocks.
  2. Build an MST on the chamber centers (Euclidean 3D distance) → tree.
  3. Add 1–2 random extra edges past the MST → loops. Why loops? Multi-path navigation; "inner chamber" landmarks.
  4. Per edge: 2–4 control points along the chamber-to-chamber line, each pushed perpendicular by domain-warped 3D noise. The path becomes a cubic spline; the tunnel SDF is a capsule along that spline.
- **Build it (continued).** Surface entrances:
  - Per chamber, roll against `ENTRANCE_PROB_{SHALLOW|MIDDLE|DEEP}` whether to seek surface.
  - If yes, evaluate three entrance types in order; first geometrically possible wins:
    1. **Sinkhole** — `chamber.top_y` is within `SINKHOLE_DEPTH_MAX` of `h_pre` above. Vertical shaft punches through.
    2. **Cliff mouth** — A steep `|∇h_pre|` cell is within `CLIFF_ENTRANCE_DIST` of the chamber. Horizontal tunnel exits at the cliff face.
    3. **Skylight** — Chamber is 30–60 blocks below surface. Narrow 1–2 block shaft straight up.
  - If none possible, chamber stays buried.
- **In the engine.** Show `build_systems_for_region` function structure. Show the `CaveSystem` struct holding chambers + edges + entrances.
- **What you can now do.** Read the system-rolling section of `caves.rs`. Understand `CHAMBER_RADIUS_RANGE`, `POISSON_MIN_SPACING_MULT`, `ENTRANCE_PROB_*` in `tuning.rs`. The carving math (how SDFs from chambers + tunnels + entrances compose into the density field) is in [§ 4.3](../part-4-chunk-fill/4.3-composing-caves.mdx).
- **Next:** → [Part IV — Per-Chunk Fill](../part-4-chunk-fill/4.1-density-graph.mdx).

- [ ] **Step 3: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 4: Commit**

```bash
git add docs/book/content/part-3-region-build/3.6-cave-systems.mdx
git commit -m "content(book): write 3.6 Cave Systems

Chapter on per-region cave system rolling. 1-4 systems per region in
depth bands; chamber Poisson-disk sampling; MST + 1-2 loops; spline
tunnels with domain-warped control points; three surface entrance
types (Sinkhole, Cliff Mouth, Skylight).

Closes Part III — Per-Region Build.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

### Spec coverage

| Spec chapter | Plan task |
|---|---|
| 3.1 Plates | Task 1 |
| 3.2 Climate | Task 2 |
| 3.3 Heightmap (h_pre) | Task 3 |
| 3.4 Hydrology + D8Stepper | Task 4 |
| 3.5 Rivers, Valleys, Lakes | Task 5 |
| 3.6 Cave Systems | Task 6 |

All six chapters and the one designated widget have a task. ✅

### Placeholder scan

No "TBD", "TODO", or "fill in details" patterns. Each task gives the implementer concrete source files to read and a chapter skeleton to fill in. ✅

### Type consistency

The chapters are independent — no shared types or APIs that need to match. Cross-references between chapters use the existing file IDs from the sidebar (which were locked in Plan 1). ✅

### Risks worth flagging

1. **Engine reality may diverge from chapter claims.** Same risk that Plans 2 and 3 hit and corrected for. Every task starts with reading source; implementers have proven they fix the chapter rather than ship a fake claim.

2. **Hydrology and caves are the biggest source files (39 KB, 55 KB).** Implementers will need to skim, not read line-by-line. The plan emphasizes *shape* over completeness for those chapters — that's the right framing.

3. **3.6 has no obvious image.** Caves are 3D and the existing 2D snapshot stages don't capture them well. The plan suggests a Mermaid diagram as the visual anchor; if the implementer finds a better option (e.g., a stylized cave-system graph), great.

4. **D8Stepper is the only new widget; aim ~250 lines.** The auto-step + toggles can be cut if implementation balloons. The minimum viable demo is "press Step to see one cell process per click".
