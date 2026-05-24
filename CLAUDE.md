# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Oxium** is a single-player voxel sandbox engine in Rust (edition 2024, rust-version 1.95). It targets real-time procedural terrain, streaming chunk I/O, and incremental lighting. Current phase: post-M3 streaming foundation, with ongoing work on the graph-based lighting engine and worldgen documentation.

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
```

---

## Code Style

Write beautiful, idiomatic Rust. Prefer:

- **Iterators over index loops** — `chunks.iter().filter_map(...)` over `for i in 0..n`
- **`?` everywhere** — no `.unwrap()` in library code; reserve `.expect("reason")` for invariants that are genuinely impossible to violate
- **Expressive types** — newtype wrappers (`ChunkCoord`, `BlockPos`, `LocalPos`) over raw integers; `impl Trait` return types where concrete types are unimportant to callers
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
| `voxel/` | `Block`, `DenseChunk`/`PalettedChunk`, `World`, `BlockPos`/`ChunkCoord`/`LocalPos`, raycasting |
| `worldgen/` | Seed-deterministic terrain pipeline: plates → heightmap → caves → hydrology → climate → surface → trees |
| `lighting/` | Graph-engine `LightEngine` (default); legacy BFS behind `--features legacy-lighting` |
| `mesher/` | Greedy mesh (production), naive (debug), LOD1/2, ambient occlusion |
| `jobs/` | Two rayon pools (`gen_pool`, `mesh_pool`) + crossbeam result channels |
| `persistence/` | Region file format, dedicated I/O thread, `SaveIndex` occupancy cache, world manifest |
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
Pending → Generated → Stored → (unload)
```

Each chunk carries `mesh_dirty` / `light_dirty` flags; systems re-run the relevant job when set.

### Lighting

**Graph engine (default):** Directed DAG per-voxel RGB light. Increase phase on block removal, decrease phase on block placement. Incremental per-frame ticks — no full recompute. Uses `TickCache` to amortize chunk decompress in the hot path.

**Legacy BFS** (opt-in via `--features legacy-lighting`): Full BFS from sky ceiling + emissive blocks on every edit. Simpler but slower.

### Chunk compression

- `DenseChunk` – flat 32³ `Block` array (~50 KB)
- `PalettedChunk` – palette + bit-packed indices (1–4 bits/voxel, ~1 KB typical)
- Compression happens post-worldgen/lighting, before disk write.

### Worldgen config hot-reload

Edit `assets/worldgen/default.ron` while the engine runs — `notify-debouncer-mini` detects the change and swaps the `Arc` immediately. Only affects *newly generated* chunks.

### Persistence

Region files at `saves/default/regions/*.bin`. Delete this directory after any worldgen change or after seeing chunk-aligned artifacts at the stale/fresh boundary.

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

Unit tests are inline (`#[cfg(test)]` blocks) throughout the library modules.

---

## Gotchas

- **Negative chunk coords are valid** — the world grid is infinite in all directions.
- **`Pending` chunks can't be read** — wait for `Stored` (next frame).
- **Persistence is async** — the I/O thread saves in the background; on-disk state lags.
- **Mesh must be re-uploaded after lighting changes** — handled by system ordering, but matters if you bypass the normal pipeline.
- **Config hot-reload is not retroactive** — existing chunks don't refresh when `default.ron` changes.
- **Legacy lighting requires a feature flag** — `cargo build --features legacy-lighting`; the graph engine is the default authority.

For worldgen-specific conventions (tuning.rs vs default.ron boundary, fingerprint test rules, common edit patterns, submodule layout), see `src/worldgen/CLAUDE.md`.
