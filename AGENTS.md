# AGENTS.md

Repository guidance for Codex and other coding agents working in this project.

## Project

Oxium is a single-player voxel sandbox engine in Rust 2024. It targets real-time procedural terrain, streaming chunk I/O, and chunk relighting. Current work is post-M3 streaming foundation, with the worldgen, voxel, persistence, and mesher packages shaped around documented domain modules.

- Published book: https://fdatoo.github.io/oxium/
- Design specs: `docs/superpowers/specs/`
- Implementation plans: `docs/superpowers/plans/`

## Commands

```bash
# Build and run; release is the default via .cargo/config.toml
cargo run --release --bin oxium

# Dev build; use --profile dev because cargo run defaults to release
cargo run --profile dev -- [FLAGS]

# Tests
cargo test
cargo test <test_name>
RUST_LOG=debug cargo test -- --nocapture

# Lint and format
cargo clippy
cargo fmt

# Profiling build
cargo build --profile profiling
samply record cargo run --profile profiling -- [ARGS]
```

Main binary flags:

```text
--spawn x,y,z          Override spawn position
--look yaw,pitch       Camera orientation, in degrees
--find-water           Auto-locate water column for spawn
--time 0..=1           Time of day; 0=midnight, 0.5=noon
--uncapped             Disable vsync
--profile <path>       Per-frame CSV profiling
--seed <u64>           Override world seed
--screenshot-and-exit  Render one frame, save PNG, exit
```

`OXIUM_SCREENSHOT_WARMUP_FRAMES` controls frames before screenshot capture. Default is `60`.

Other binaries:

```bash
cargo run --release --bin worldgen_viz -- --seed 42 --radius-xz 3
cargo run --release --bin doc_render -- --help
```

## Coding Standards

- Keep patches minimal and surgical. Do not refactor unrelated code.
- Prefer performance-conscious, low-latency paths. Avoid hidden state and magic behavior.
- Never silence errors. Use explicit error handling with meaningful types.
- Do not guess APIs, data layouts, or schemas. Inspect the code or ask for missing context.
- Write idiomatic Rust using current stable Rust 2024 patterns.
- Prefer iterators over index loops where that remains clear and efficient.
- Use `?` for error propagation. Avoid `.unwrap()` in library code.
- Use `.expect("reason")` only for invariants that are genuinely impossible to violate.
- Prefer expressive domain types such as `ChunkCoord`, `BlockPos`, `LocalPos`, `RegionCoord`, `RegionSlot`, `LightLevel`, and `PackedRgbLight` over raw integers.
- Add traits only when they encode a real shared contract or hot-path specialization point; newtypes are preferred for single-value invariants.
- Prefer `match` and `if let` over chained `.is_some()` plus `.unwrap()`.
- Use `From` and `Into` for conversions between domain types.
- Derive standard traits before writing manual implementations.
- Keep control flow flat with early returns and `?`.
- `cargo fmt` is required. `cargo clippy` should be warning-free.

## Architecture

Frame loop in `app.rs::AppState::step`:

1. `input_system` maps keyboard and mouse input into `InputBuf`.
2. `movement_system` updates player physics and position.
3. `world_stream_system` determines chunks to load or unload and spawns rayon jobs.
4. `drain_jobs_system` installs completed gen, mesh, and lighting results into `World`.
5. `world_unload_system` despawns out-of-radius chunks and saves them to disk.
6. `render_system` issues GPU draw calls.
7. `clear_input_buf` resets per-frame input state.

Library modules in `src/lib.rs` must not depend on windowing:

| Module | Responsibility |
| --- | --- |
| `voxel/` | Blocks, coordinate types, chunk storage, light packing metadata, world storage, raycasting |
| `worldgen/` | Seed-deterministic terrain pipeline |
| `lighting/` | Chunk relight worker: sky light plus RGB block light BFS |
| `mesher/` | Public mesh types, greedy mesh, debug naive mesh, LOD, ambient occlusion |
| `jobs/` | Rayon pools and crossbeam result channels |
| `persistence/` | World manifest, 16^3 region files, `SaveIndex`, I/O thread |
| `physics/` | AABB collision helpers |

Binary-only modules may depend on `winit` and `wgpu`:

- `app.rs`
- `ecs/`
- `render/`
- `ui/`
- `src/bin/worldgen_viz/`
- `src/bin/doc_render/`

## Domain Notes

Coordinate spaces:

```rust
ChunkCoord(IVec3) // 32^3-block chunk positions; negative coordinates are valid
BlockPos          // Global voxel position
LocalPos          // [0..32)^3 within a chunk

let local = block.to_local();
let chunk = block.to_chunk();
```

Chunk lifecycle:

```text
Pending -> Stored(Generated/Meshing/Ready) -> unload
```

Each stored chunk carries `ChunkMeta::dirty.mesh` and `ChunkMeta::dirty.light` flags. Systems rerun the relevant job when these are set.

The checked-in lighting path is recompute-on-dirty BFS: sky light drops from the top face, RGB block light spreads from emissive blocks, and neighbour boundary values seed cross-chunk continuity. Graph-engine designs may exist in `docs/`, but they are not the current runtime authority unless implemented in code.

Chunk compression uses `DenseChunk` for a flat 32^3 `Block` array and `PalettedChunk` for palette plus bit-packed indices. Compression happens after worldgen and lighting, before disk writes.

Worldgen config hot-reloads from `assets/worldgen/default.ron`, but only affects newly generated chunks.

Persistence stores 16 x 16 x 16 chunk region files at `saves/default/regions/*.bin`. Delete that directory after worldgen changes or after seeing chunk-aligned stale/fresh artifacts.

Package-specific guidance lives in `src/worldgen/`, `src/voxel/`, `src/mesher/`, and `src/persistence/`.

## Visual Regression

PNG baselines live in `tests/screenshots/`.

```bash
python3 tests/screenshots/diff.py tests/screenshots/baseline_<scene>.png /tmp/new.png
```

Use the diff script instead of `cmp -l`; small pixel differences are expected between identical runs.

For intentional visual baseline updates, use a longer warmup:

```bash
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release -- \
  --screenshot-and-exit tests/screenshots/baseline_noon_outdoor.png \
  --look 45,-15 --time 0.5
```

Full regeneration commands are in `tests/screenshots/README.md`.

## Tests

Integration tests:

- `tests/smoke.rs`: full gen, light, mesh, round-trip pipeline
- `tests/lighting_colored.rs`: RGB propagation correctness
- `tests/worldgen_fingerprint.rs`: worldgen stability fingerprints
- `tests/gen_with_neighbors.rs`: chunk gen with neighbor context

Unit tests are inline or in sibling `tests.rs` / `*_tests.rs` modules.

## Gotchas

- Negative chunk coordinates are valid.
- `Pending` chunks cannot be read; wait for `Stored`.
- Persistence is async, so on-disk state lags.
- Meshes must be re-uploaded after lighting changes.
- Config hot-reload is not retroactive.
- Do not assume plans in `docs/` are implemented; inspect the runtime module before changing adjacent code.
