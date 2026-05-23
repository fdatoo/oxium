# Worldgen Module — Agent & Contributor Guide

## Orientation

The worldgen package is a **pure `(seed, ChunkCoord) → DenseChunk` map**. Every chunk can be regenerated from the seed alone; caches are memoisation, not ground truth.

**Layer order (plates → chunk):**
1. `plates.rs` — Voronoi tectonic plates → continental mask + roughness
2. `heightmap.rs` — plate-driven FBM relief + domain warp → `h_pre`
3. `density_graph.rs` — 9³ corner-lattice trilerp → per-voxel 3D density
4. `climate.rs` — 6D R-tree biome lookup, Voronoi jitter
5. `region.rs` — LRU caches for fine/macro region builds
6. `hydrology.rs` — D8 flow accumulation, sink-fill, valley carve, rivers
7. `caves.rs` — graph cave systems, noise carvers (cheese, Terasology)
8. `surface.rs` — rule-tree surface block selection
9. `fluid.rs` — fluid placement (sea, lakes, cave pools)
10. `mod.rs` / `pipeline.rs` — orchestrates everything per chunk

**Start reading:** `mod.rs` `//!` header → `tuning.rs` → `pipeline.rs`.

---

## `tuning.rs` vs `default.ron` — Where Does a New Knob Go?

This is the most important convention in the package. There are two places worldgen reads numbers from:

| File | What goes here | Hot-reload? |
|---|---|---|
| `src/worldgen/tuning.rs` | Architectural invariants: cell/region sizes, cache caps, derived constants, anything another module imports | No — recompile required |
| `assets/worldgen/default.ron` | World-character knobs: terrain splines, biome hyperboxes, noise params, cave style tables, surface rules | Yes — file-watcher swaps atomically |

**Litmus test for a new constant:**

| Question | Where it goes |
|---|---|
| Could a non-coder tune this live to reshape the world? | `default.ron` |
| Does another module's `const` or array size depend on it? | `tuning.rs` |
| Is it derived from other constants? | `tuning.rs` (under `// ── Derived ──`) |
| Is it imported by `voxel/`, `render/`, `persistence/`, or tests? | `tuning.rs` |
| Is it a noise channel, spline knot, biome claim, or surface rule? | `default.ron` |

When in doubt: if deleting the value from the file would break a `use` import somewhere, it belongs in `tuning.rs`.

> **Note on "future RON candidates":** Hydrology knobs (`RIVER_WIDTH_SCALE`, meander params) and cave entrance probabilities live in `tuning.rs` for now but are annotated `// future: RON candidate`. Moving them to `default.ron` is a separate follow-up that requires a schema migration.

---

## The Fingerprint Test Is Sacred

`tests/worldgen_fingerprint.rs` pins a `DefaultHasher` over a 256×256 PNG of the heightmap at seed 42. **This is the only automated correctness signal for procedural terrain — ground truth is otherwise impossible to verify automatically.**

Rules:
1. Run `cargo test --test worldgen_fingerprint --release` after every commit that touches anything upstream of `HeightmapNoise::h_pre` (plates, climate splines, `DensityNoise`, hydrology valley carve).
2. **A refactor that flips the hash is a bug.** Do not rebaseline — revert the float-op reorder.
3. Bump `EXPECTED_HASH` **only** when the terrain change is the intent of the PR, with a dated comment: `// Rebaselined YYYY-MM-DD: <reason>`.
4. Companion tests: `cargo test --test smoke --release && cargo test --test gen_with_neighbors --release`.

---

## Documentation Style

Every file in this package must have:

- **`//!` module header**: one sentence on the module's role, bullet-list of key concepts, back-reference to a design spec or book chapter.
- **`///` on every `pub` item**: one-line summary + non-obvious invariants.
- **Tuning constant docstrings**: "Smaller → X. Larger → Y." + units.
- **Math prose**: a paragraph *above* each dense formula naming the algorithm (Planchon-Darboux, D8, Catmull-Rom, etc.) and explaining *why* this formula is correct — not just what it computes.
- **"Why" inline comments** on workarounds, surprising shapes, hot-path constraints.

Comments explain **why**. The code shows **what**.

---

## Hot-Path Constraints

- Any per-voxel trait (like `CornerLatticeEvaluator`) must use **generics** (`<E: Trait>`), never `dyn`. A vtable indirection in the per-voxel inner loop costs ~10% throughput.
- The inner loops in `fill_chunk` and `cave_sdf` must not allocate.
- Region cache access happens **once per chunk** in `gather_chunk_regions`; per-column queries read from the prefetched 3×3 grid without re-locking.

---

## Naming Conventions

- **Hash salts**: `const SALT_<PURPOSE>: i32 = N;` — never bare integers in `mix_u32`/`mix_unit` calls. See the `// ── Hash domain separators ──` block in `caves.rs` for the template.
- **Tuning constants**: `SCREAMING_SNAKE` in `tuning.rs`, organized under `// ── Section ──` banners.
- **Newtypes** where a value carries clamp semantics (`RiverWidth`, `ChamberRadius`); leave plain `i32`/`f32` for purely arithmetic locals.

---

## Submodule Layout Is Intentional

Do not merge submodule directories back into flat files. The directories are:

| Directory | Contents |
|---|---|
| `density/` | `HeightmapNoise`, `DensityNoise`, cell evaluator, density math helpers |
| `region/` | `FineRegion`, `MacroRegion`, cache, concurrency (BuildCache/Condvar) |
| `hydrology/` | D8 flow, sink-fill (Grid), fine/macro passes, valley carve, rivers |
| `caves/` | Styles, system builder, chambers, tunnels, entrances, SDFs, noise carvers |

New additions go into the correct submodule. New subdirectory-worthy topics may be created; merge only for clear degeneracy.

---

## Persistence Gotcha

Delete `saves/default/regions/` after **any worldgen change** before manual play-testing. Chunks are cached by `(seed, coord)` — the cache does not invalidate on code changes. Stale chunks from old code will appear at the boundary of newly-generated chunks.

Always note "delete saves/default/regions/ before testing" in the PR description.

---

## Common Edit Patterns

### Add a new biome
1. Edit `assets/worldgen/default.ron` → `biomes.entries`: add a `ParameterPoint` claim across 6 climate axes.
2. Add the `Biome` variant in `src/worldgen/biome.rs`; update its `match` arms (color, snow rule, tree rate).
3. Update the `surface` rule tree in `default.ron` if the biome needs a special surface block.
4. Verify: `cargo test --release`; eyeball the visualizer (`cargo run --release --bin worldgen_viz -- --seed 42`).
5. **No fingerprint bump** (biome tags don't feed the heightmap PNG).

### Tune river width / depth / meander
1. Edit `tuning.rs`: `RIVER_WIDTH_SCALE`, `MIN_RIVER_WIDTH`, `MAX_RIVER_WIDTH`, `RIVER_BED_DEPTH`, `VALLEY_HALF_WIDTH_MULT`, `MEANDER_AMP_PER_WIDTH`. Recompile required.
2. **Fingerprint will flip** — intentional terrain change. Rebaseline `EXPECTED_HASH` with a dated comment.
3. Verify: visualizer + screenshot baseline regen.

### Add a cave style
1. Edit `assets/worldgen/default.ron` → `cave.style_table`: add chamber count / radius / tunnel ranges, update each `style_weights_*` array (must sum to 1.0).
2. Add the variant to `CaveStyle` enum in `src/worldgen/caves/style.rs`; update `pick_style` + `style_params`.
3. New variants require recompile; pure value changes are hot-reload.
4. Verify: smoke test + visualizer cave probe.

### Add a new tuning constant
1. Apply the litmus test above.
2. `default.ron` path: add field to `WorldgenConfig` (or sub-config) in `config.rs`, populate in `default.ron`.
3. `tuning.rs` path: add `pub const` under the appropriate banner with "Smaller → X. Larger → Y." doc.
4. Verify: `cargo test --release` + visualizer.

### Adjust cave depth bands
1. Edit `tuning.rs`: `CAVE_BAND_SHALLOW`, `CAVE_BAND_MIDDLE`, `CAVE_BAND_DEEP`, `CAVE_FLOOR_Y`, `CAVE_SURFACE_BUFFER`. Recompile required.
2. Verify: smoke test + `tests/cave_vertical_clamp.rs` + visualizer.
3. Check fingerprint — may or may not flip depending on whether bands affect the heightmap path.

### Add a new cave entrance type
1. Add variant to `EntranceKind` in `src/worldgen/caves/entrance.rs`; add a roller function.
2. Register it in `roll_entrances` in `caves/system.rs`.
3. Add entrance carve in `caves/sdf.rs::entrance_sdf`.
4. Add probability constants to `tuning.rs`.
5. Verify: smoke + `tests/cave_surface_breakthrough.rs`.

### Change a surface block rule
1. Edit `assets/worldgen/default.ron` → `surface` rule tree. No recompile needed.
2. Verify: smoke test + visualizer.

### Modify density composition
1. Edit `default.ron` → `density` section and/or `src/worldgen/density/density_3d.rs`.
2. **Fingerprint will almost certainly flip** — every voxel is affected. Confirm intent before rebaselining.
3. Companion: visual regression via `tests/screenshots/diff.py`.

### Add a tree variant
1. Edit `src/worldgen/trees.rs`; add `TREE_RATE_<BIOME>` to `tuning.rs`.
2. Verify: smoke + visualizer + screenshot baseline.
3. **No fingerprint bump** (trees don't feed the heightmap PNG).

---

## Reading Order for New Contributors

`mod.rs` → `tuning.rs` → `pipeline.rs` → `density/mod.rs` → `region/mod.rs` → `hydrology/mod.rs` → `caves/mod.rs`

The `//!` headers on each are designed to be readable end-to-end without reading bodies.
