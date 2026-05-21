# Appendices Implementation Plan (Phase 1, Plan 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the four appendices that close Phase 1 of *How Oxium Builds a World*: Module Index, Technique Index, Constants Catalog, Glossary. These are reference pages — the chapters are the prose, the appendices are the lookups.

**Architecture:** Four parallel tasks (all writes are to disjoint files). Each agent reads enough source/chapters to build its lookup; writes its MDX file; controller batches build + commit.

**Tech Stack:** Same as prior plans.

**Out of plan scope:** Phase 2 (renderer, mesher, lighting) and Phase 3 (ECS, jobs, persistence, UI). After this plan, Phase 1 is complete.

---

## Pre-flight

```
EnterWorktree(name: "docs-appendices")
```

Then:

```bash
git merge main --ff-only
cd docs/book && npm ci && cd ../..
```

---

## Chapter ID catalog (used by ALL tasks)

Every appendix cross-references chapters. The full list of chapter IDs from the sidebar:

**Part I — Foundations**
- `1.1-voxel-world` — The Voxel World
- `1.2-determinism` — Deterministic Randomness
- `1.3-coherent-noise` — Coherent Noise
- `1.4-fbm` — Fractional Brownian Motion
- `1.5-voronoi` — Voronoi Diagrams
- `1.6-domain-warping` — Vector Fields & Domain Warping
- `1.7-splines` — Splines (cubic Hermite)
- `1.8-sdfs` — Signed Distance Functions
- `1.9-trilerp` — Trilinear Interpolation

**Part II — Pipeline Overview**
- `2.1-big-picture` — The Big Picture

**Part III — Per-Region Build**
- `3.1-plates` — Plates
- `3.2-climate` — Climate
- `3.3-heightmap` — Heightmap (h_pre)
- `3.4-hydrology` — Hydrology
- `3.5-rivers-lakes` — Rivers, Valleys, Lakes
- `3.6-cave-systems` — Cave Systems

**Part IV — Per-Chunk Fill**
- `4.1-density-graph` — The Density Graph
- `4.2-cell-evaluator` — The Cell Evaluator
- `4.3-composing-caves` — Composing Caves
- `4.4-noise-carvers` — Noise Carvers
- `4.5-procedural-carvers` — Procedural Carvers
- `4.6-surface-rules` — Surface Rules
- `4.7-aquifers` — Aquifers
- `4.8-fluid-settle` — Fluid Settle
- `4.9-trees` — Trees

**Part V — Engineering**
- `5.1-determinism` — Determinism Everywhere
- `5.2-region-cache` — The Region Cache
- `5.3-config-hot-reload` — Config & Hot-Reload
- `5.4-visualizer` — The Visualizer

Relative link paths from `appendices/` to chapters: `../part-N-name/X.Y-slug.mdx`. Verify file existence before linking.

---

## Task 1 (parallel): Appendix A — Module Index

**File:** `docs/book/content/appendices/a-module-index.mdx` (currently `<Stub />`).

**Frame:** For a reader who is in the engine code and wants the chapter that explains a particular module. Sorted alphabetically by module file.

### Steps

- [ ] **Step 1: List the engine modules**

```bash
ls src/worldgen/ src/bin/worldgen_viz/ 2>&1 | head -40
ls src/voxel/ 2>&1
```

Cover everything under `src/worldgen/` and (for completeness) the most-relevant supporting modules: `src/voxel/{block,chunk,coords}.rs`, `src/bin/worldgen_viz/`.

- [ ] **Step 2: Write the appendix**

Replace `docs/book/content/appendices/a-module-index.mdx`. Use a table layout:

```mdx
---
id: a-module-index
title: "Appendix A: Module Index"
---

A cross-reference from engine source files to the chapters that explain them. Sorted alphabetically by path.

## src/voxel/

| File | Chapter | Brief |
|---|---|---|
| `block.rs` | [1.1 The Voxel World](../part-1-foundations/1.1-voxel-world.mdx) | `Block` enum, palette |
| `chunk.rs` | [1.1 The Voxel World](../part-1-foundations/1.1-voxel-world.mdx) | `DenseChunk` and `PalettedChunk` |
| `coords.rs` | [1.1 The Voxel World](../part-1-foundations/1.1-voxel-world.mdx) | `ChunkCoord`, `LocalPos`, conversions |

## src/worldgen/

| File | Chapters | Brief |
|---|---|---|
| `aquifer.rs` | [4.7 Aquifers](../part-4-chunk-fill/4.7-aquifers.mdx) | Jittered 16×12×16 cell grid, pressure model |
| `carver.rs` | [4.5 Procedural Carvers](../part-4-chunk-fill/4.5-procedural-carvers.mdx) | MC-style tunnel system (the active cave generator) |
| `caves.rs` | [3.6 Cave Systems](../part-3-region-build/3.6-cave-systems.mdx), [4.3 Composing Caves](../part-4-chunk-fill/4.3-composing-caves.mdx), [4.4 Noise Carvers](../part-4-chunk-fill/4.4-noise-carvers.mdx) | Graph-cave system (currently disabled), noise carvers |
| `climate.rs` | [3.2 Climate](../part-3-region-build/3.2-climate.mdx) | TargetPoint, ParameterList, Voronoi jitter |
| `config.rs` | [5.3 Config & Hot-Reload](../part-5-engineering/5.3-config-hot-reload.mdx) | WorldgenConfig, ConfigHolder, RON schema |
| `density_graph.rs` | [4.1 The Density Graph](../part-4-chunk-fill/4.1-density-graph.mdx), [4.2 The Cell Evaluator](../part-4-chunk-fill/4.2-cell-evaluator.mdx) | DensityFn enum + CellEvaluator |
| `fluid.rs` | [4.8 Fluid Settle](../part-4-chunk-fill/4.8-fluid-settle.mdx) | settle_fluid post-pass |
| `hash.rs` | [1.2 Deterministic Randomness](../part-1-foundations/1.2-determinism.mdx), [5.1 Determinism Everywhere](../part-5-engineering/5.1-determinism.mdx) | The mix function and salt convention |
| `heightmap.rs` | [3.3 Heightmap](../part-3-region-build/3.3-heightmap.mdx) | h_pre, spline pipeline, slope/cliff |
| `hydrology.rs` | [3.4 Hydrology](../part-3-region-build/3.4-hydrology.mdx), [3.5 Rivers, Valleys, Lakes](../part-3-region-build/3.5-rivers-lakes.mdx) | D8 flow, segments, valley carve |
| `mod.rs` | [2.1 The Big Picture](../part-2-overview/2.1-big-picture.mdx), [4.9 Trees](../part-4-chunk-fill/4.9-trees.mdx) | Generator + fill_chunk + tree_in_cell |
| `noise_channel.rs` | [4.4 Noise Carvers](../part-4-chunk-fill/4.4-noise-carvers.mdx) | NoiseCarvers (cheese/spaghetti/pillar FBM channels) |
| `plates.rs` | [3.1 Plates](../part-3-region-build/3.1-plates.mdx) | Voronoi plate decomposition |
| `probe.rs` | [5.4 The Visualizer](../part-5-engineering/5.4-visualizer.mdx) | Stage enum + ColumnProbe (used by viz) |
| `region.rs` | [5.2 The Region Cache](../part-5-engineering/5.2-region-cache.mdx) | FineRegion, MacroRegion, LRU caches |
| `spline.rs` | [1.7 Splines](../part-1-foundations/1.7-splines.mdx) | CubicSpline (Hermite formula) |
| `surface.rs` | [4.6 Surface Rules](../part-4-chunk-fill/4.6-surface-rules.mdx) | RuleSource (priority tree for surface block) |
| `trees.rs` | [4.9 Trees](../part-4-chunk-fill/4.9-trees.mdx) | (empty stub — actual tree code is in mod.rs around line 1325) |
| `tuning.rs` | [Appendix C: Constants Catalog](./c-constants-catalog.mdx) | All compile-time constants |

## src/bin/

| File | Chapter | Brief |
|---|---|---|
| `worldgen_viz/` | [5.4 The Visualizer](../part-5-engineering/5.4-visualizer.mdx) | egui-on-wgpu debug visualizer |
| `doc_render/` | (engineering — not a worldgen module; produces the book's images via the same viz_render code) | Snapshot/parity tooling |
```

VERIFY every file actually exists with `ls`. If a file doesn't exist (e.g., engine state has changed), drop the row. If a file exists but isn't in this list, add a row.

- [ ] **Step 3: After writing, verify with `head -5`**

After writing the file, run `head -5 docs/book/content/appendices/a-module-index.mdx` to confirm the write took effect.

- [ ] **Step 4: Report**

DO NOT commit. DO NOT run `npm run build`. Report:
- Status
- Output of `head -5`
- Any modules you found that weren't in the spec
- Any modules from the spec that don't actually exist

---

## Task 2 (parallel): Appendix B — Technique Index

**File:** `docs/book/content/appendices/b-technique-index.mdx` (currently `<Stub />`).

**Frame:** For a reader who knows a technique and wants every chapter where it appears. Sorted alphabetically by technique.

### Steps

- [ ] **Step 1: Identify techniques covered in Phase 1**

This is mostly a recall exercise from reading the chapters. The major techniques:

- Catmull-Rom spline (3.6 — cave tunnels)
- Cubic Hermite spline (1.7, 3.3 climate spline pipeline)
- D8 flow direction (3.4)
- Domain warping (1.6, 3.5 meander, 3.6 tunnel control points)
- FBM / fractal Brownian motion (1.4, 3.2 climate axes, 3.3 base_3d, 4.4 noise carvers)
- Hash mixer (1.2, 5.1)
- LRU cache (5.2)
- Mermaid (used throughout for diagrams; not a runtime technique)
- MST (3.6 — chamber graph; engine uses Kruskal)
- Planchon–Darboux sink fill (3.4)
- Poisson-disk sampling (3.6 chamber placement)
- R-tree (mentioned in early Plan 4 drafts but Plan 6's 5.3 implementer found the biome lookup is actually a linear-scan `Vec<ParameterPoint>` — note this)
- Signed distance functions / SDFs (1.8, 3.6, 4.3, 4.4)
- Smoothstep (3.5 U-profile)
- Trilinear interpolation (1.9, 4.2)
- Voronoi (1.5, 3.1, 4.7 aquifer cells)
- xor-shift / golden-ratio multiply (the mix function family; 1.2 detail)

- [ ] **Step 2: Write the appendix**

Use a table. Each row: technique + 1–2 sentence intuition + chapters that use it.

```mdx
---
id: b-technique-index
title: "Appendix B: Technique Index"
---

A cross-reference from reusable techniques (each defined in Part I) to every place the engine uses them. Sorted alphabetically.

| Technique | Where it's defined | Where the engine uses it |
|---|---|---|
| Catmull-Rom spline | [1.7 Splines](../part-1-foundations/1.7-splines.mdx) (Hermite primer covers the math) | [3.6 Cave Systems](../part-3-region-build/3.6-cave-systems.mdx) — tunnel paths (verified by 3.6 implementer: tunnels are Catmull-Rom, not Hermite) |
| Cubic Hermite spline | [1.7 Splines](../part-1-foundations/1.7-splines.mdx) | [3.3 Heightmap](../part-3-region-build/3.3-heightmap.mdx) — the offset/factor/jaggedness pipeline |
| D8 flow direction | [3.4 Hydrology](../part-3-region-build/3.4-hydrology.mdx) | [3.4 Hydrology](../part-3-region-build/3.4-hydrology.mdx) (introduces it) |
| Domain warping | [1.6 Vector Fields & Domain Warping](../part-1-foundations/1.6-domain-warping.mdx) | [3.5 Rivers, Valleys, Lakes](../part-3-region-build/3.5-rivers-lakes.mdx) — meander warp on river centerlines; [3.6 Cave Systems](../part-3-region-build/3.6-cave-systems.mdx) — tunnel control-point offsets |
| FBM (Fractional Brownian Motion) | [1.4 FBM](../part-1-foundations/1.4-fbm.mdx) | Everywhere noise is sampled — [3.2 Climate](../part-3-region-build/3.2-climate.mdx), [3.3 Heightmap](../part-3-region-build/3.3-heightmap.mdx) base_3d, [4.4 Noise Carvers](../part-4-chunk-fill/4.4-noise-carvers.mdx) |
| Hash mixer | [1.2 Deterministic Randomness](../part-1-foundations/1.2-determinism.mdx), audit in [5.1 Determinism Everywhere](../part-5-engineering/5.1-determinism.mdx) | Every random decision — Voronoi jitter in [3.1 Plates](../part-3-region-build/3.1-plates.mdx) and [3.2 Climate](../part-3-region-build/3.2-climate.mdx), tree placement in [4.9 Trees](../part-4-chunk-fill/4.9-trees.mdx), per-cell rolls everywhere |
| LRU cache | [5.2 The Region Cache](../part-5-engineering/5.2-region-cache.mdx) | [5.2 The Region Cache](../part-5-engineering/5.2-region-cache.mdx) (FineCache + MacroCache); [4.5 Procedural Carvers](../part-4-chunk-fill/4.5-procedural-carvers.mdx) (carver tunnel cache) |
| MST (Kruskal) | [3.6 Cave Systems](../part-3-region-build/3.6-cave-systems.mdx) | [3.6 Cave Systems](../part-3-region-build/3.6-cave-systems.mdx) — chamber connectivity graph |
| Planchon–Darboux sink fill | [3.4 Hydrology](../part-3-region-build/3.4-hydrology.mdx) | [3.4 Hydrology](../part-3-region-build/3.4-hydrology.mdx) — bounds-window iteration to raise basins |
| Poisson-disk sampling | [3.6 Cave Systems](../part-3-region-build/3.6-cave-systems.mdx) | [3.6 Cave Systems](../part-3-region-build/3.6-cave-systems.mdx) — chamber center placement |
| SDFs (Signed Distance Functions) | [1.8 Signed Distance Functions](../part-1-foundations/1.8-sdfs.mdx) | [3.6 Cave Systems](../part-3-region-build/3.6-cave-systems.mdx) (chamber ellipsoids, tunnel capsules), [4.3 Composing Caves](../part-4-chunk-fill/4.3-composing-caves.mdx) (composition via min/max into density) |
| Simplex noise | [1.3 Coherent Noise](../part-1-foundations/1.3-coherent-noise.mdx) | Foundation for every FBM — engine uses `noise::Simplex` exclusively |
| Smoothstep | [3.5 Rivers, Valleys, Lakes](../part-3-region-build/3.5-rivers-lakes.mdx) (U-profile falloff) | [3.5 Rivers, Valleys, Lakes](../part-3-region-build/3.5-rivers-lakes.mdx) (valley walls); commonly throughout for soft thresholds |
| Trilinear interpolation | [1.9 Trilinear Interpolation](../part-1-foundations/1.9-trilerp.mdx) | [4.2 The Cell Evaluator](../part-4-chunk-fill/4.2-cell-evaluator.mdx) — 9×9×9 corner lattice + trilerp |
| Voronoi diagram | [1.5 Voronoi Diagrams](../part-1-foundations/1.5-voronoi.mdx) | [3.1 Plates](../part-3-region-build/3.1-plates.mdx) (tectonic plates), [3.2 Climate](../part-3-region-build/3.2-climate.mdx) (biome-edge jitter), [4.7 Aquifers](../part-4-chunk-fill/4.7-aquifers.mdx) (3D aquifer cells) |
| xor-shift / multiply | [1.2 Deterministic Randomness](../part-1-foundations/1.2-determinism.mdx) (mix function internals) | [1.2 Deterministic Randomness](../part-1-foundations/1.2-determinism.mdx); see also [5.1 Determinism Everywhere](../part-5-engineering/5.1-determinism.mdx) for salt conventions |
```

Add 1–2 sentence intuitions where helpful — "what is FBM" gets answered in a couple of sentences before pointing at 1.4.

- [ ] **Step 3: Verify with `head -5`**

- [ ] **Step 4: Report.** DO NOT commit. DO NOT build.

---

## Task 3 (parallel): Appendix C — Constants Catalog

**File:** `docs/book/content/appendices/c-constants-catalog.mdx` (currently `<Stub />`).

**Frame:** `src/worldgen/tuning.rs` walked group by group with brief prose for each constant. Reader can scan to find the knob that controls a given engine behavior.

### Steps

- [ ] **Step 1: Read `tuning.rs`**

```bash
cat src/worldgen/tuning.rs
```

(~11 KB.) Note the natural groupings (plates, heightmap, rivers, caves, aquifer, surface, trees, caches). The file already has section comments — use them as section headers.

- [ ] **Step 2: Write the appendix**

Replace `docs/book/content/appendices/c-constants-catalog.mdx`. Structure:

```mdx
---
id: c-constants-catalog
title: "Appendix C: Constants Catalog"
---

Every compile-time constant in `src/worldgen/tuning.rs` with a one-line description and a link to the chapter that explains why it has that value. Runtime-tunable parameters (spline curves, biome rates, surface rules) live in `assets/worldgen/default.ron` — see [5.3 Config & Hot-Reload](../part-5-engineering/5.3-config-hot-reload.mdx).

## Plates

| Constant | Value | Description | Chapter |
|---|---|---|---|
| `PLATE_CELL_SIZE` | 1024 | World cell size for the Voronoi plate grid (blocks) | [3.1 Plates](../part-3-region-build/3.1-plates.mdx) |
| `CONTINENTAL_RATIO` | 0.45 | Fraction of plates rolled as Continental | [3.1 Plates](../part-3-region-build/3.1-plates.mdx) |
| `ROUGHNESS_RANGE` | (0.7, 1.4) | Per-plate FBM amplitude multiplier | [3.1 Plates](../part-3-region-build/3.1-plates.mdx) |

... (continue for every group: heightmap, rivers, macro, caves, aquifer, surface, trees, caches)
```

For each constant: pull the **actual value** from `tuning.rs`, write a one-line description, link to the most-relevant chapter. If a constant is mentioned in multiple chapters, link to the primary one.

Don't fabricate values — read `tuning.rs` line by line.

- [ ] **Step 3: Verify with `head -5`.**

- [ ] **Step 4: Report.** DO NOT commit. DO NOT build.

---

## Task 4 (parallel): Appendix D — Glossary

**File:** `docs/book/content/appendices/d-glossary.mdx` (currently `<Stub />`).

**Frame:** Every domain term defined in Phase 1 with a short definition. Sorted alphabetically. Each entry links to the chapter that introduces the term.

### Steps

- [ ] **Step 1: Collect terms from the chapters**

This is mostly a memory exercise. Roughly the terms to define:

- Aquifer
- ArcSwap
- Avalanche (in hash-mixer context)
- Base 3D noise
- Bilinear interpolation
- Biome
- Block (the voxel kind)
- Carver tunnel
- Cell evaluator
- Chamber
- ChunkCoord
- Coherent noise
- Composition scale
- Config holder
- Continentalness
- Cubic Hermite
- D8 flow direction
- Density field / DensityFn
- Domain warping
- Drainage / flow accumulation
- DenseChunk vs PalettedChunk
- Entrance (cave: sinkhole, cliff mouth, skylight)
- FBM
- Fine region / macro region
- Hash mixer
- h_pre / h_target
- Jaggedness spline
- Jittered grid
- Kruskal's algorithm (MST)
- Lacunarity
- Lake rim
- LRU cache
- Meander warp
- MST
- Octave
- Persistence
- Planchon–Darboux
- Plate / PlateKind / PlateLookup
- Poisson-disk sampling
- Pressure model (aquifer)
- Procedural carver
- R-tree
- Region cache
- River segment / river width
- RON
- RuleSource
- Salt (in hash context)
- SDF
- Sea level
- Settle (fluid)
- Simplex noise
- Sink fill
- Slide
- Smoothstep
- Spline (Hermite / Catmull-Rom / Bezier)
- Stage (visualizer)
- Surface buffer
- TargetPoint
- TreeKind
- Trilerp
- Tuning constants
- Underground density threshold
- Voronoi diagram
- Weirdness
- Wormhole noise
- Xor-shift / golden-ratio multiply
- y_gradient

- [ ] **Step 2: Write the appendix**

```mdx
---
id: d-glossary
title: "Appendix D: Glossary"
---

Every domain term used in Phase 1 of *How Oxium Builds a World*, alphabetical. Links point to the chapter that introduces the term.

### A

**Aquifer.** A region of subterranean fluid (water or lava) defined by a jittered 3D cell grid. Each cell has a `y_top` (fluid surface) and a fluid kind. See [4.7 Aquifers](../part-4-chunk-fill/4.7-aquifers.mdx).

**ArcSwap.** The atomic-swap pointer type from the `arc_swap` crate. Used by [`ConfigHolder`](../part-5-engineering/5.3-config-hot-reload.mdx) so config snapshots can be loaded lock-free.

**Avalanche.** Property of a hash function: flipping one bit of input should flip roughly half the output bits. Achieved by the xor-shift + multiply finaliser in [`hash::mix`](../part-1-foundations/1.2-determinism.mdx).

...
```

For each term: 1–2 sentence definition, link to the chapter that introduces or canonically defines it. Don't repeat the chapter content — the glossary entry is a *pointer*, not a re-explanation.

Use H3 (`### A`, `### B`, ...) section headers for alphabetical sectioning. This is the standard appendix-D format.

- [ ] **Step 3: Verify with `head -5`.**

- [ ] **Step 4: Report.** DO NOT commit. DO NOT build.

---

## Controller post-dispatch (after all 4 return)

1. Verify all 4 files were written. Use `head -5` on each:
   ```bash
   for f in a-module-index b-technique-index c-constants-catalog d-glossary; do
     echo "=== $f ==="
     head -5 docs/book/content/appendices/$f.mdx
   done
   ```
2. Run `npm run build`. Fix any broken-link errors via Edit inline (links to non-existent chapter IDs are the most likely failure).
3. Commit 4 separate commits, one per appendix.
4. Final review subagent over the four.
5. `finishing-a-development-branch` to merge.

---

## Self-review

### Spec coverage

| Spec appendix | Plan task |
|---|---|
| A. Module Index | Task 1 |
| B. Technique Index | Task 2 |
| C. Constants Catalog | Task 3 |
| D. Glossary | Task 4 |

All four covered. ✅

### Placeholder scan

No "TBD" or "fill in details". Every task explicitly references the source files and the chapter ID list above. ✅

### Risks

1. **Hallucinated chapter IDs or paths.** All four appendices link heavily to chapters. If an agent uses a wrong slug or wrong relative path, the build will catch it via `onBrokenLinks: 'throw'`. Mitigation already wired into the controller post-dispatch step.
2. **R-tree mention.** The original design spec said the biome lookup uses an R-tree. Plan 6's 5.3 implementer found it's actually a linear-scan `Vec<ParameterPoint>`. Appendix B should NOT call it an R-tree — list the actual implementation if mentioning it.
3. **Empty trees.rs.** Plan 5's 4.9 implementer found `trees.rs` is just a docstring stub; actual code is in `mod.rs`. Appendix A should reflect this honestly.
