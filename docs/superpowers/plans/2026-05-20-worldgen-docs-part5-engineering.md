# Part V — Engineering Scaffolding Implementation Plan (Phase 1, Plan 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the four chapters of *Part V — Engineering Scaffolding*: 5.1 Determinism, 5.2 Region Cache, 5.3 Config & Hot-Reload, 5.4 The Visualizer. These chapters wrap up Phase 1 (only appendices remain after).

**Architecture:** All four chapters are independent (different source files, no shared widgets). Dispatch all 4 in parallel via `dispatching-parallel-agents`. Each agent writes its MDX file, does NOT commit, does NOT run npm build. The controller batches build + commits after all 4 return.

**Tech Stack:** Same as prior plans.

**Out of plan scope:** Appendices A-D (Plan 7).

---

## Pre-flight

```
EnterWorktree(name: "docs-part5")
```

Then:

```bash
git merge main --ff-only
cd docs/book && npm ci && cd ../..
```

---

## All four chapters in parallel

### Task 1 (parallel): Chapter 5.1 — Determinism Everywhere

**Source:** `src/worldgen/hash.rs` + the seed-offset patterns in `src/worldgen/mod.rs::new_internal` + a survey of `hash::mix*` call sites.

**File:** `docs/book/content/part-5-engineering/5.1-determinism.mdx`.

**Frame:** Chapter 1.2 introduced the hash mixer as a tool. This chapter is the *audit* — the entire engine's purity claim ("byte-identical output from `(seed, ChunkCoord)`") depends on every random decision routing through `hash::mix_*` and every noise field getting a distinct salt. Walk through the salting conventions and the consequences if purity slipped.

Cover:
- Quick recap of `hash::mix` (link back to 1.2).
- The axis-salted seed pattern in `Generator::new_internal`: each `Fbm::<Simplex>::new(seed.wrapping_add(N))` uses a different `N` so the noise fields are uncorrelated. List the salts the engine actually uses (grep for `wrapping_add` in `mod.rs`).
- Per-call-site salt convention: `hash::mix_*(seed, &[coord1, coord2, SALT])` — the inline integer salts that give independence between rolls at the same coordinate.
- The pure-on-(seed, coord) guarantee: `Generator::fill_chunk(coord)` produces byte-identical chunks regardless of which chunks were generated first.
- The negative case: where purity would slip if you accidentally used `Instant::now()` or `thread_rng()` — none of these exist in worldgen.
- Test infrastructure: the golden-chunk tests, the determinism unit tests.

### Task 2 (parallel): Chapter 5.2 — The Region Cache

**Source:** `src/worldgen/region.rs` (15 KB — read it fully) + the lookup-or-build call sites in `mod.rs`.

**File:** `docs/book/content/part-5-engineering/5.2-region-cache.mdx`.

**Frame:** Chapter 2.1 mentioned "pre-fetch regions" as the first step of `fill_chunk`. This chapter explains why and how the region cache exists, the lookup-or-build mechanic, and why purity isn't compromised by having a cache.

Cover:
- The two cache levels: `FineCache` and `MacroCache` (LRU + sizes).
- `RegionCoord` and `MacroRegionCoord` keying.
- `get_fine` / `get_macro` lookup-or-build patterns. Show the actual call signatures.
- The halo: regions are built with a halo of neighbouring data sampled from noise (NOT pulled from the cache, to avoid recursive build).
- Cross-cache dependency: fine build → macro cache. Macro build only calls noise. No recursion.
- Thread safety: `Arc<Mutex<LruCache<...>>>`. Build happens outside the lock; duplicate-build race accepted.
- Why purity isn't compromised: caches are *memoisation*, not state. Evicted entries rebuild byte-identically.
- Cache footprint at default caps (FINE_CACHE_CAP, MACRO_CACHE_CAP) — verify in `tuning.rs`.

### Task 3 (parallel): Chapter 5.3 — Config & Hot-Reload

**Source:** `src/worldgen/config.rs` (19 KB) + `assets/worldgen/default.ron`.

**File:** `docs/book/content/part-5-engineering/5.3-config-hot-reload.mdx`.

**Frame:** Worldgen has many tunable parameters. Some live in `src/worldgen/tuning.rs` (compile-time constants) and some live in `assets/worldgen/default.ron` (runtime, hot-reloadable). This chapter explains the split, the `ConfigHolder` snapshot machinery, and what's safe to swap at runtime.

Cover:
- The two config sources: compile-time `tuning.rs` constants vs runtime RON.
- `WorldgenConfig` and its sub-configs (`ClimateConfig`, `DensityConfig`, `CaveConfig`, `AquiferConfig`, `SurfaceConfig`, `BiomesConfig`). Sketch the structure.
- The RON format. Show a snippet of `assets/worldgen/default.ron`.
- The `ConfigHolder`: an `Arc<ArcSwap<WorldgenConfig>>` (or whatever the actual type is — verify). Atomic-swap.
- The file watcher (`notify-debouncer-mini` crate). Debounced reload on file save.
- `Generator::config_snapshot()` — called once per chunk so the chunk uses a consistent config even if a swap races mid-fill.
- What's safe to hot-swap vs what requires a `Generator` restart. Examples:
  - Safe: spline knot values, biome rate percentiles, surface rule tweaks.
  - Not safe: noise frequencies (baked into FBM objects at `Generator::new`), aquifer cell sizes (baked into the aquifer system).
- The trade-off: hot-reloadable knobs help authoring loop; baked frequencies guarantee determinism for noise channels.

### Task 4 (parallel): Chapter 5.4 — The Visualizer

**Source:** `src/bin/worldgen_viz/` (multiple files, mostly UI). Read `main.rs`, `app.rs`, `session.rs`, `overlays/mod.rs`, plus a peek at the layout/paint files.

**File:** `docs/book/content/part-5-engineering/5.4-visualizer.mdx`.

**Frame:** A reference chapter for `worldgen_viz` — the debug tool that's been visible throughout the book as the source of every "engine-authoritative image". Now we close the loop: how the visualizer's architecture works, what its panels show, and why it's useful for tuning.

Cover:
- Architecture: egui-on-wgpu UI bound to a live `Generator`.
- The map view (`overlays/`): per-stage sampler + colormap, pan/zoom. The same code path used by `doc_render` (link to chapter 2.1 / Plan 1 infrastructure).
- The column probe (`probe.rs` in viz): pin a column, view all 30+ pipeline values for that column.
- The density breakdown panel: per-voxel signed-density decomposition (each `DensityFn` leaf's contribution separately).
- The cross-section view (`crosssection.rs`): vertical slice through the world showing density / blocks / aquifer.
- Live config reload: changes to `assets/worldgen/default.ron` flow into the viz automatically (via the file watcher from 5.3).
- The screenshot / preset infrastructure (`preset.rs`).
- Honest note: `worldgen_viz` is a developer tool, not user-facing. Future runtime visualization for end-users is a Phase 3 topic.

End the chapter with:

```mdx
This is the last chapter of Part V — Phase 1 of *How Oxium Builds a World* is now complete.
The Appendices that follow are reference material: a module-by-module index, a technique
cross-reference, a constants catalog, and a glossary.

→ [Appendix A — Module Index](../appendices/a-module-index.mdx)
```

---

## What each parallel agent does

1. Read the relevant source.
2. Write ONLY their chapter file (one MDX file each, no other changes).
3. DO NOT commit, DO NOT run `npm run build`.
4. Return a brief report.

## What the controller does after all 4 return

1. Verify all 4 files were written (`git status` should show all 4 modified).
2. **Use `Read` on each MDX file head to confirm content** (defends against the misrouted-write bug from Plan 5 — sample the first ~5 lines of each file).
3. Run `npm run build`. Fix any broken links or other issues inline (Edit tool).
4. Commit the 4 chapters as 4 separate commits to preserve per-chapter authorship granularity.
5. Run the final-review subagent.
6. Use `finishing-a-development-branch` to merge.

---

## Self-review

### Spec coverage

| Spec chapter | Plan task |
|---|---|
| 5.1 Determinism Everywhere | Task 1 |
| 5.2 The Region Cache | Task 2 |
| 5.3 Config & Hot-Reload | Task 3 |
| 5.4 The Visualizer | Task 4 |

All four chapters have a task. No widgets in Part V per the design spec. ✅

### Placeholder scan

No "TBD" or "fill in details" patterns. Each task gives concrete source to read and a chapter skeleton. ✅

### Risks worth flagging

1. **Same risks as Plan 5 parallel dispatch** — silent write failures (misrouted Write paths), hallucinated link paths. Mitigation: explicit verification step in the controller's post-dispatch flow (read each file's head, run build, fix inline).
2. **5.4 The Visualizer covers a lot of code** — at least 8 files in `src/bin/worldgen_viz/`. Agent should focus on architecture/UX rather than line-by-line.
3. **Chapter 5.1's salt-survey** depends on grep accuracy. If the engine has many `wrapping_add` calls, the chapter should list the salient ones (Generator::new_internal seed offsets) rather than every call site.
