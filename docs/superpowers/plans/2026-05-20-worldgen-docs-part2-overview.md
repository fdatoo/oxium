# Part II — The Big Picture Implementation Plan (Phase 1, Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write chapter 2.1 "The Big Picture" — the no-math, image-heavy spine of the book that walks through the whole worldgen pipeline from `(seed, ChunkCoord)` to `DenseChunk`, with annotated images at every stage. Every later chapter (Parts III–V) refers back to it.

**Architecture:** One task. The chapter is built around ~12 generated images, one per pipeline stage, all rendered at the book seed via `doc_render snapshot` (already wired up in Plan 1). The implementer extends `tools/gen-images.sh` with the new entries, regenerates the PNGs, and writes the chapter prose so each image is the visual anchor for a paragraph explaining what the engine did to produce it.

**Tech Stack:** Same as prior plans — Docusaurus 3 + TypeScript MDX; existing `doc_render` Rust binary with `Stage` enum and per-stage colormaps.

**Out of plan scope:** Parts III–V chapters (those *explain* each stage in detail; this chapter just *introduces* them with one image + one paragraph each). Appendices.

---

## Pre-flight: enter a worktree

```
EnterWorktree(name: "docs-part2")
```

Then fast-forward to local main (since `origin/main` may be behind):

```bash
git merge main --ff-only
```

Verify Part I content is present:

```bash
ls docs/book/content/part-1-foundations/
# Should show all 9 chapters: 1.1 through 1.9
```

Run `cd docs/book && npm ci && cd ../..` to install deps (npm cache survives across worktrees but a fresh worktree needs the symlinks).

---

## Task 1: Write chapter 2.1 "The Big Picture" with full image suite

The chapter is `docs/book/content/part-2-overview/2.1-big-picture.mdx` (currently a `<Stub />`). It exists as a single MDX file.

### Files

- Modify: `docs/book/tools/gen-images.sh` — add ~12 new render entries.
- Run: `bash docs/book/tools/gen-images.sh` — produces new PNGs under `docs/book/static/img/generated/`.
- Modify: `docs/book/content/part-2-overview/2.1-big-picture.mdx` — full chapter.

### Reference patterns

- Existing demonstration chapters: `docs/book/content/part-1-foundations/1.1-voxel-world.mdx` (no widgets) and `1.4-fbm.mdx` (with widget).
- Existing chapters that already cite the engine accurately (use these as the style anchor): 1.5, 1.6, 1.7.
- `doc_render snapshot` stage list (from `src/worldgen/probe.rs` `Stage` enum): continentalness, plate-id, temperature, humidity, desertness, weirdness, h-pre, valley-carve, h-target, flow-accum, biome-id, aquifer-y, aquifer-substance.
- The per-chunk fill pipeline lives in `src/worldgen/mod.rs::Generator::fill_chunk` — read it to understand the actual sequence.

### Steps

- [ ] **Step 1: Read the per-chunk fill pipeline**

Run:

```bash
cat src/worldgen/mod.rs | head -100
```

The first 100 lines have the module-level doc comment that walks the pipeline. Read it carefully — that's the canonical sequence from `(seed, ChunkCoord)` to `DenseChunk`. Note what each stage does.

Then peek at `fill_chunk` itself:

```bash
grep -n "fn fill_chunk\|fn column_data\|fn build_fine_region\|fn gather_chunk_regions" src/worldgen/mod.rs
```

You don't need to memorize the code — you need the *order of operations*. The chapter follows that order.

- [ ] **Step 2: Plan the image suite**

List the images you'll render, in the order they appear in the chapter. The recommended set (12 images):

1. **`plate-id`** at zoom 16 (256 px = 4096 blocks across). Shows the Voronoi plate decomposition over a wide area — continents and oceans are visible at this zoom.
2. **`continentalness`** at zoom 16. Smooth gradient from -1 (deep ocean) to +1 (deep continent).
3. **`temperature`** at zoom 16. Low-frequency climate map.
4. **`humidity`** at zoom 16. Companion to temperature.
5. **`weirdness`** at zoom 16. Mid-frequency variant axis.
6. **`h-pre`** at zoom 8 (256 px = 2048 blocks). Pre-river heightmap. Shows mountain ranges, plains.
7. **`valley-carve`** at zoom 8. River-carved valley depth field (mostly black except along rivers).
8. **`h-target`** at zoom 8. Post-carve final height. Rivers should be visible cuts in the terrain.
9. **`flow-accum`** at zoom 8. Log-scaled drainage. Tree-of-rivers structure.
10. **`biome-id`** at zoom 8. Categorical biome map.
11. **`aquifer-y`** at zoom 8. Aquifer water-level surface.
12. **`aquifer-substance`** at zoom 8. Where the aquifer holds lava vs water.

If you'd rather skip `desertness` and `weirdness` to keep the chapter tight (they're auxiliary signals, not main pipeline stages), do so — the chapter prioritizes clarity over completeness. Document which ones you dropped in your final report.

For all images, use `--seed 42 --center 0,0`. Zoom and width as above.

- [ ] **Step 3: Add image entries to `gen-images.sh`**

Open `docs/book/tools/gen-images.sh`. There's one existing `render humidity 0,0 4 256` line — that's the FBM fallback used by 1.4. Add a new section below it for chapter 2.1.

Append entries like:

```bash
# Chapter 2.1 "The Big Picture" — one image per pipeline stage.
# All at the book seed (42), centered at world (0, 0), 256 px wide.
# Zoom 16 = each pixel is 16 blocks (256 px = 4096 blocks across)
# — wide enough to see continents.
render plate-id        0,0 16 256
render continentalness 0,0 16 256
render temperature     0,0 16 256
render humidity        0,0 16 256
render weirdness       0,0 16 256

# Zoom 8 = each pixel is 8 blocks (256 px = 2048 blocks across)
# — closer-in views where per-chunk detail starts to show.
render h-pre           0,0 8 256
render valley-carve    0,0 8 256
render h-target        0,0 8 256
render flow-accum      0,0 8 256
render biome-id        0,0 8 256
render aquifer-y       0,0 8 256
render aquifer-substance 0,0 8 256
```

(The existing `render humidity 0,0 4 256` line stays — it's the fallback for chapter 1.4. The new humidity entry at zoom 16 is a different file.)

- [ ] **Step 4: Generate the images**

Run from the worktree root:

```bash
bash docs/book/tools/gen-images.sh
```

Expected: rebuilds `doc_render` if needed, then renders each PNG. Should take ~30 seconds depending on the chunk fill cost at zoom 16 (which evaluates a wide area).

Verify output:

```bash
ls docs/book/static/img/generated/
```

Expected files (filenames derived from gen-images.sh's `render` helper: `<stage>-s42-0_0-z<zoom>-w<width>.png`):

```
.gitkeep
aquifer-substance-s42-0_0-z8-w256.png
aquifer-y-s42-0_0-z8-w256.png
biome-id-s42-0_0-z8-w256.png
continentalness-s42-0_0-z16-w256.png
flow-accum-s42-0_0-z8-w256.png
h-pre-s42-0_0-z8-w256.png
h-target-s42-0_0-z8-w256.png
humidity-s42-0_0-z16-w256.png
humidity-s42-0_0-z4-w256.png         # existing FBM fallback
plate-id-s42-0_0-z16-w256.png
temperature-s42-0_0-z16-w256.png
valley-carve-s42-0_0-z8-w256.png
weirdness-s42-0_0-z16-w256.png
```

If any image looks degenerate (all-black, all-one-color), that's worth flagging — note it in your report and either re-render with adjusted zoom OR document the cause in the chapter (e.g., "valley-carve is mostly black because most pixels aren't near a river").

- [ ] **Step 5: Write the chapter**

Replace `docs/book/content/part-2-overview/2.1-big-picture.mdx`. This is a substantial chapter (~1500 words plus images). Use the standard 6-section skeleton but adapt — the "build it" section becomes the pipeline walkthrough.

**Chapter purpose** (frame it like this in the opening hook): a reader has worked through Part I and learned each primitive (hash, noise, Voronoi, warp, splines, SDFs, trilerp). Now they need the *map* — the sequence of stages that those primitives compose into. After reading 2.1, the reader should know what happens between calling `Generator::fill_chunk(coord)` and getting blocks out, in the right order.

**The chapter's body is the pipeline walkthrough.** For each pipeline stage:
- Image with a one-line caption (the seed = 42, center = (0, 0)).
- 2–4 paragraphs explaining: what this stage produces, what the previous stage(s) feed into it, what the next stage uses it for.
- Forward link to the Part III/IV/V chapter that explains the stage in detail (Parts III–V are stubs but the file paths exist — link to them; the reader will follow when those chapters are written).

Pipeline stages in chapter order — match the order in `src/worldgen/mod.rs` module doc comment:

1. **Pre-fetch regions.** No image; one paragraph: the engine pulls a 3×3 grid of fine regions around the chunk so per-column queries don't lock the cache 1024 times. Link → `5.2 Region Cache`.
2. **Plates.** `plate-id` image. Voronoi plate decomposition. Each cell is one plate, continental or oceanic. Link → `3.1 Plates`.
3. **Continentalness.** `continentalness` image. Smooth signed field derived from plate boundaries — the *first* climate axis. Link → `3.1 Plates` (same chapter as plates because they're directly derived).
4. **Climate axes.** `temperature`, `humidity`, `weirdness` images, side by side or stacked. Three more independent FBM fields. Together with continentalness (and depth + terrain-shape + ridges-pv which we don't render directly because they're cheaper to discuss in 3.2), these form the 6-axis climate vector. Link → `3.2 Climate`.
5. **Heightmap `h_pre`.** `h-pre` image. Climate axes drive spline pipelines to produce a target height per column. The result is what terrain would look like *without* any river or cave carving. Link → `3.3 Heightmap`.
6. **Hydrology.** `flow-accum` image. D8 flow direction + Planchon–Darboux sink fill + accumulation, run at fine + macro resolutions. Trees of rivers visible. Link → `3.4 Hydrology`.
7. **Valley carve.** `valley-carve` image. River cells drive a perpendicular-distance U-profile carve. Link → `3.5 Rivers, Valleys, Lakes`.
8. **`h_target` = `h_pre - valley_carve`.** `h-target` image. The final surface height per column. Link → `3.5 Rivers, Valleys, Lakes`.
9. **Biome.** `biome-id` image. R-tree lookup over the 6-axis climate vector + jittered query produces a biome enum per column. Link → `3.2 Climate` (the biome classifier lives there).
10. **Density graph + cell evaluator** (no image — it's a per-voxel 3D thing; the heightmap captures the 2D projection). Brief paragraph: density at every voxel via the spline pipeline + 3D noise + slide; sampled at a 9×9×9 lattice and trilerped per-voxel. Link → `4.1 Density Graph` and `4.2 Cell Evaluator`.
11. **Caves.** No image (caves are 3D). Brief paragraph: graph-based cave systems + noise carvers + procedural carvers + wormhole noise all compose into the density field via signed-density min/max operations. Link → `4.3 Composing Caves`, `4.4 Noise Carvers`, `4.5 Procedural Carvers`, `3.6 Cave Systems`.
12. **Aquifers.** `aquifer-y` and `aquifer-substance` images side by side. Per-column aquifer cell determines water/lava and `y_top`. Pressure model decides whether each voxel is "claimed" by the aquifer or yields to the surface flood. Link → `4.7 Aquifers` and `4.8 Fluid Settle`.
13. **Surface rules.** No image. Brief paragraph: the surface rule system (cliffs, beaches, snow, biome materials, sand-transition band) picks the visible block for the top few voxels of each column. Link → `4.6 Surface Rules`.
14. **Trees.** No image. Brief paragraph: per-cell deterministic placement using the hash mixer from 1.2. Link → `4.9 Trees`.

After the walkthrough, a closing "What you can now do" section: the reader can now read `fill_chunk` in `src/worldgen/mod.rs` and recognize what every block does.

**Image embedding pattern in MDX:**

```mdx
![Plate decomposition at the book seed](/oxium/img/generated/plate-id-s42-0_0-z16-w256.png)
*Plate decomposition. Each cell is one plate; cool tones are oceanic, warm tones are continental. 4096 blocks across (zoom 16). The boundaries are where the engine grows mountain ranges.*
```

The path prefix is `/oxium/img/generated/` (the Docusaurus `baseUrl` is `/oxium/`). Hardcode it for now — the parity-test page does similarly; if we ever change baseUrl we'll have a `useBaseUrl()` cleanup pass.

The italic caption sits just below the image and is parsed as a paragraph by MDX.

**Forward-link pattern:**

```mdx
The Voronoi plate decomposition itself is covered in detail in [§ 3.1 Plates](../part-3-region-build/3.1-plates.mdx).
```

The path is relative from `part-2-overview/2.1-big-picture.mdx` to `part-3-region-build/3.1-plates.mdx`. Verify with `ls docs/book/content/part-3-region-build/` that the target file exists (it should — Part III stubs landed in Plan 1).

**Next link at the end:** Part III's first chapter:

```mdx
→ [3.1 Plates](../part-3-region-build/3.1-plates.mdx)
```

**Honest disclosure:** several of the forward-linked chapters are still stubs (`<Stub />`). That's fine for the chapter to forward-link to — when those chapters are written in Plans 4–6, the links resolve to content. For now the reader hits a stub and knows where the topic *will* land. Don't apologize in the prose; just link as normal.

- [ ] **Step 6: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -15 && cd ../..
```

Expect: `[SUCCESS]`. The build should complete with no broken-link warnings (Docusaurus tolerates broken markdown links but warns by default — adjust if you see warnings about your forward links).

- [ ] **Step 7: Commit**

```bash
git add docs/book/tools/gen-images.sh docs/book/static/img/generated/ docs/book/content/part-2-overview/2.1-big-picture.mdx
git commit -m "content(book): write 2.1 The Big Picture + generate pipeline images

Spine chapter that walks the worldgen pipeline from (seed, ChunkCoord)
to DenseChunk, with ~12 generated images (one per stage at the book
seed). Forward-links to every chapter in Parts III–V that explains
each stage in detail. Closes Part II.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

## Self-review

### Spec coverage

The design spec (`docs/superpowers/specs/2026-05-20-worldgen-docs-design.md`) describes Part II as:
> 2.1 The Big Picture | One chapter, no math. "From `(seed, chunk_coord)` to a `DenseChunk` of blocks — every step, in order, with annotated images from `worldgen_viz` at each stage." This chapter is the spine; later chapters refer back to it.

Single task implements exactly this. ✅

### Placeholder scan

No "TBD", "TODO", "fill in details" in the task. Several places explicitly invite the implementer to make judgment calls (which images to drop if any, how degenerate images get described) — those are scoped questions, not placeholders. ✅

### Type consistency

The plan only modifies markdown + shell scripts + checking-in PNGs. No type signatures or method names to keep consistent. ✅

### Risks worth flagging

1. **Images may take time at zoom 16.** Rendering 256×256 at zoom 16 means evaluating worldgen at 4096²÷256² = 256 columns per pixel, with `mod.rs::column_data` calls happening at high frequency. The release build is fast, but 12 images at this scale may take a couple minutes the first time. Acceptable.
2. **Some Stage values may produce mostly-uniform images at this seed.** Especially `weirdness` (mid-frequency FBM) and `aquifer-substance` (categorical water-vs-lava). If a render is essentially a single color, the implementer is invited to either re-render with different `--center` to find more interesting territory or skip that image and document it.
3. **Forward links land on stubs.** This is intentional — stubs already exist in Part III/IV/V — but if the implementer encounters issues with Docusaurus warning about stubs, the warnings are not errors and shouldn't block the merge.
4. **`useBaseUrl` is not used for the image paths.** Hardcoded `/oxium/` prefix. Same trade-off as 1.4-fbm.mdx made (the build verifier flagged this as Important but Plan 1 deferred). Defer again — consistent with the surrounding chapters; a cleanup pass can fix all image paths together.
