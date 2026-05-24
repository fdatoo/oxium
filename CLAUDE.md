# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Oxium** is a single-player voxel sandbox engine in Rust (edition 2024, rust-version 1.95). It targets real-time procedural terrain, streaming chunk I/O, and chunk relighting. Current phase: post-M3 streaming foundation, with worldgen, voxel, persistence, and mesher organized around documented domain modules.

- Published book: https://fdatoo.github.io/oxium/
- Design specs: `docs/superpowers/specs/`
- Implementation plans: `docs/superpowers/plans/`

---

## Commands

```bash
# Build & run (release is default via .cargo/config.toml)
cargo run --release --bin oxium

# Dev build (opt-level=1) — must use --profile dev; `cargo run` alias defaults to release
cargo run --profile dev -- [FLAGS]

# Tests
cargo test
cargo test <test_name>           # single test
RUST_LOG=debug cargo test -- --nocapture

# Lint / format
cargo clippy
cargo fmt

# Profiling build (release + full debug info for samply/Instruments)
cargo build --profile profiling
samply record cargo run --profile profiling -- [ARGS]
```

### CLI flags (main binary)

```
--spawn x,y,z          Override spawn position
--look yaw,pitch       Camera orientation (degrees)
--find-water           Auto-locate water column for spawn
--time 0..=1           Time of day (0=midnight, 0.5=noon)
--uncapped             Disable vsync
--profile <path>       Per-frame CSV profiling
--seed <u64>           Override world seed
--screenshot-and-exit  Render one frame, save PNG, exit
```

`OXIUM_SCREENSHOT_WARMUP_FRAMES` – frames before screenshot capture (default 60).

### Other binaries

```bash
# Interactive terrain tuning
cargo run --release --bin worldgen_viz -- --seed 42 --radius-xz 3

# Generate documentation images
cargo run --release --bin doc_render -- --help

# LLM testing harness — inspect world state as JSON (no renderer)
cargo run --release --bin oxium-probe -- inspect --column 0,0
cargo run --release --bin oxium-probe -- inspect --find cave --max-radius 2048 --count 3
cargo run --release --bin oxium-probe -- inspect --at 100,64,-200

# oxium-probe inspect modes:
#   --column WX,WZ          ColumnData JSON (biome, height, water_surface_y, …)
#   --at WX,WY,WZ           DensityBreakdown JSON (per-voxel density decomposition)
#   --find <kind>           Nearest feature JSON; kinds: water, lava, cave, river,
#                           forest, tropical, desert, tundra, plains, snowy_forest
#   --max-radius N          Search radius in blocks (default 8192)
#   --count N               Number of results (default 1)
```

---

## Code Style

Write beautiful, idiomatic Rust. Prefer:

- **Iterators over index loops** — `chunks.iter().filter_map(...)` over `for i in 0..n`
- **`?` everywhere** — no `.unwrap()` in library code; reserve `.expect("reason")` for invariants that are genuinely impossible to violate
- **Expressive types** — newtype wrappers (`ChunkCoord`, `BlockPos`, `LocalPos`, `RegionCoord`, `RegionSlot`, `LightLevel`, `PackedRgbLight`) over raw integers; `impl Trait` return types where concrete types are unimportant to callers
- **Traits with purpose** — add traits only for a real shared contract or hot-path specialization point; prefer newtypes for single-value invariants
- **`match` / `if let`** over chained `.is_some()` / `.unwrap()`
- **`From`/`Into`** for conversions between domain types
- **`derive` first** — `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]` before writing manual impls
- **Flat over nested** — early returns and `?` to reduce nesting; one level of indentation per logical step

`cargo clippy` is the style arbiter — code must be warning-free. `cargo fmt` is non-negotiable.

---

## Architecture

### Frame loop (`app.rs::AppState::step`)

1. `input_system` – keyboard/mouse → `InputBuf`
2. `movement_system` – player physics/position
3. `world_stream_system` – determine chunks to load/unload, spawn rayon jobs
4. `drain_jobs_system` – install completed gen/mesh/lighting results into `World`
5. `world_unload_system` – despawn out-of-radius chunks, save to disk
6. `render_system` – GPU draw call
7. `clear_input_buf`

### Library crates (in `src/lib.rs` — no windowing dependency)

| Module | Responsibility |
|--------|---------------|
| `voxel/` | `Block`, coordinate types, `DenseChunk`/`PalettedChunk`, light packing metadata, `World`, raycasting |
| `worldgen/` | Seed-deterministic terrain pipeline: plates → heightmap → caves → hydrology → climate → surface → trees |
| `lighting/` | Chunk relight worker: sky light plus RGB block light BFS |
| `mesher/` | Public mesh types, greedy mesh (production), naive (debug), LOD1/2, ambient occlusion |
| `jobs/` | Two rayon pools (`gen_pool`, `mesh_pool`) + crossbeam result channels |
| `persistence/` | World manifest, 16^3 region file format, `SaveIndex` occupancy cache, dedicated I/O thread |
| `physics/` | AABB collision helpers |

### Binary-only modules (depend on winit/wgpu)

- `app.rs` – `AppState` owns renderer, world, ECS, jobs, persistence
- `ecs/` – `hecs`-based components + per-frame systems
- `render/` – wgpu pipelines, camera, atlas, HDR, bloom, HUD, font, screenshot
- `ui/` – egui pause menu, chat, command dispatcher
- `src/bin/worldgen_viz/`, `src/bin/doc_render/`

---

## Key Concepts

### Coordinate spaces

```rust
ChunkCoord(IVec3)      // 32³-block chunk positions (infinite grid, negatives valid)
BlockPos               // global voxel position
LocalPos               // [0..32)³ within a chunk

let local = block.to_local();
let chunk = block.to_chunk();
```

### Chunk lifecycle

```
Pending -> Stored(Generated/Meshing/Ready) -> (unload)
```

Each stored chunk carries `ChunkMeta::dirty.mesh` and `ChunkMeta::dirty.light`; systems re-run the relevant job when set.

### Lighting

The checked-in lighting path is recompute-on-dirty BFS. Sky light drops from the top face, RGB block light spreads from emissive blocks, and neighbour boundary values seed cross-chunk continuity. Graph-engine designs may exist in `docs/`, but they are not the current runtime authority unless implemented in code.

### Chunk compression

- `DenseChunk` - hot, unpacked 32^3 block and light arrays used by generation, lighting, meshing, and edits
- `PalettedChunk` - compressed in-memory/on-disk palette plus bit-packed indices
- Compression happens post-worldgen/lighting, before disk write.

### Worldgen config hot-reload

Edit `assets/worldgen/default.ron` while the engine runs — `notify-debouncer-mini` detects the change and swaps the `Arc` immediately. Only affects *newly generated* chunks.

### Persistence

16 x 16 x 16 chunk region files live at `saves/default/regions/*.bin`. Delete this directory after any worldgen change or after seeing chunk-aligned artifacts at the stale/fresh boundary.

---

## Visual Regression

PNG baselines live in `tests/screenshots/`. To diff after a code change:

```bash
python3 tests/screenshots/diff.py tests/screenshots/baseline_<scene>.png /tmp/new.png
# Exit 0 = within noise floor; exit 1 = regression
```

~6% of pixels naturally differ ±1–2 between identical runs (physics drift → LOD/vertex jitter). `diff.py` uses calibrated thresholds to filter this; `cmp -l` will always show noise — don't use it.

To regenerate a baseline after an intentional visual change, use `OXIUM_SCREENSHOT_WARMUP_FRAMES=1800` (default 60 frames is insufficient at streaming radius 16):

```bash
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release -- \
  --screenshot-and-exit tests/screenshots/baseline_noon_outdoor.png \
  --look 45,-15 --time 0.5
```

Full regeneration commands for all six baselines are in `tests/screenshots/README.md`.

---

## Test Organization

```
tests/
  smoke.rs                 Full pipeline: gen → light → mesh → round-trip
  lighting_colored.rs      RGB propagation correctness
  worldgen_fingerprint.rs  Worldgen output stability — update expected values when terrain gen changes intentionally
  gen_with_neighbors.rs    Chunk gen with neighbor context
```

Unit tests are inline (`#[cfg(test)]` blocks) or in sibling `tests.rs` / `*_tests.rs` modules.

---

## Gotchas

- **Negative chunk coords are valid** — the world grid is infinite in all directions.
- **`Pending` chunks can't be read** — wait for `Stored` (next frame).
- **Persistence is async** — the I/O thread saves in the background; on-disk state lags.
- **Mesh must be re-uploaded after lighting changes** — handled by system ordering, but matters if you bypass the normal pipeline.
- **Config hot-reload is not retroactive** — existing chunks don't refresh when `default.ron` changes.
- **Docs can lead code** - design plans may describe future systems. Inspect the runtime module before treating a plan as implemented.

For package-specific conventions, see the local `CLAUDE.md` files under `src/worldgen/`, `src/voxel/`, `src/mesher/`, and `src/persistence/`.

---

## Performance Metrics

All per-frame performance data flows through a single struct pipeline so the HUD overlay, the `--profile <path>` CSV output, and (future) `oxium-probe capture` metrics CSV all see the same numbers.

### How metrics flow

| Layer | File | Role |
|---|---|---|
| `FrameCounters` | `src/profiler.rs:35` | Named integer counters gathered once per frame |
| `PerfSnapshot` | `src/app.rs:111` | Runtime copy read by HUD and perf CSV |
| `Profiler::finish_frame` | `src/profiler.rs` | Writes one CSV row (base counters + per-span µs columns) |
| HUD overlay | `src/render/hud.rs:355` | Renders on-screen line from `PerfSnapshot` |
| `oxium-probe` sidecar JSON | `src/bin/oxium-probe/` | Embeds `perf` object in each `<frame>.png.json` |

### Adding a new counter

1. **Add a field** to `FrameCounters` (`src/profiler.rs`) and `PerfSnapshot` (`src/app.rs:~111`).
2. **Populate it** in `AppState::step` where the existing counters are gathered — chunk counts at `src/app.rs:~497–506`, draw calls at `src/render/mod.rs:~1567`.
3. **Add a CSV column** in `Profiler::open` (header) and `Profiler::finish_frame` (value).
4. **Append to the HUD line** in `src/render/hud.rs:355–451` if it should appear on screen.
5. Re-run `cargo test --release` and the six visual baselines (`tests/screenshots/diff.py`) before merging — the CSV header change is observable in any test that diffs profiler output.

### Adding a new span timer

Wrap the work with `profiler::time(prof, "name", || ...)` in `AppState::step`. New span columns are auto-discovered the first frame they fire and appended to the CSV. No CSV header or struct changes needed — spans are dynamic.

### Counters not yet tracked (follow-up work)

The following are not currently instrumented and would need new `FrameCounters` fields:

- GPU work time — requires `wgpu::QuerySet` timestamps.
- Per-frame vertex / triangle counts — only `chunk_mesh_count` exists today (`src/render/mod.rs:877`).
- Jobs queue depth (gen / mesh / relight in-flight) — currently inferred via `ChunkSlot::Pending` walk.
- Persistence bytes written / read, queue depth.
- Lighting BFS nodes visited per relight.
