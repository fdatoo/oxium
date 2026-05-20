# Agent-Friendly Testing Harness — Design

**Date:** 2026-05-19
**Status:** Draft, pending implementation plan
**Audience:** Coding agents (Claude Code and successors) maintaining this codebase, and humans reviewing those changes.

---

## 1. Goals & non-goals

The harness exists so a coding agent can make a change, verify it works end-to-end, and surface a precise failure when it doesn't — without a human watching the game window.

**In scope:**

- **Worldgen at scale.** Biome distribution, cave connectivity, structure invariants — beyond the existing single-chunk hash and 2 km heightmap fingerprint.
- **Gameplay & physics behavior.** Walking, falling, collision, raycast targeting, place/break correctness, persistence round-trip.
- **Visual regressions.** Rendered frame diffs against PNG goldens for stable surfaces (sky, water, lighting, terrain mesh, HUD, paused/chat overlays).
- **Performance budgets.** Per-system tick budgets pinned so a 2× regression in meshing or world streaming fails CI, not production.
- **Feature-development workflow.** "Observe mode" capture for new visual features where there is no baseline to diff against (a pig is being added, not regressed).

**Out of scope:**

- In-game NPC/AI runtime. The harness is for the coding agent, not for gameplay agents.
- Replay format for full play sessions.
- Networking / multiplayer test infrastructure.
- Mutation testing or property-based fuzzing infrastructure.
- Asset hot-reload validation.
- VLM-as-judge automation (no test passes because a model thinks the pig looks stocky).

---

## 2. Architecture overview

Two layers, both invoked by `cargo test`. The agent never has to remember a separate runner.

```
                            cargo test
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
       Layer A: In-process            Layer B: Binary + capture
       (≈99% of scenarios)            (visual: observe or golden diff)
              │                                   │
   tests/scenario_*.rs              tests/visual_*.rs, observe_*.rs
   tests/common/harness.rs                  │ std::process::Command
              │                             ▼
              ▼                  cargo run --release -- \
        Harness {                  --screenshot-and-exit <path> \
          world, ecs, jobs,        --spawn / --look / --time /
          generator, registry,     --shader-time / --ui
          tick(), input(),                  │
          place(), break(),                 ▼
          snapshot(), ...             capture PNG + state.json
        }                                   │
              │                             ▼
              ▼                  observe: artifacts are deliverable
        assert! on engine        golden: 3-up diff + diff.json on fail
        state directly
```

**Why this shape.**

- **Layer A** builds directly on the library boundary already drawn in `src/lib.rs`. `oxium::voxel::world::World`, `oxium::worldgen::Generator`, `oxium::lighting`, `oxium::mesher`, `oxium::physics` are all importable today. The `Harness` is a thin façade that wires them together the same way `AppState` does — without `winit` or `wgpu` — and exposes a deterministic fixed-step `tick()` loop plus synthetic input and edit operations.
- **Layer B** reuses the `--screenshot-and-exit` path that already exists in `src/main.rs`. The only new pieces are a small comparator (`tests/common/visual.rs`) that diffs against `tests/goldens/`, and an `observe` mode that captures + dumps state without comparing.
- The shared `tests/common/` directory holds reusable helpers; per Rust convention, `tests/common/mod.rs` is shared across integration test files without being treated as its own integration test.

**Layer A doesn't render.** If a scenario needs to verify rendering, that is Layer B. This is the single rule that keeps Layer A fast and self-contained.

---

## 3. File layout

```
tests/
  common/
    mod.rs              — re-exports
    harness.rs          — Layer A: Harness struct + builder + tick loop
    snapshot.rs         — WorldSnapshot, PlayerSnapshot
    asserts.rs          — assert_pos!, assert_block!, assert_tick_budget! macros
    visual.rs           — Layer B: spawn binary, capture, observe + golden modes
  scenario_physics.rs       — falling, collision, jump arcs
  scenario_edit.rs          — raycast targeting, place, break
  scenario_worldgen.rs      — biome stats, cave connectivity, water placement
  scenario_persistence.rs   — round-trip save/load, torn-write recovery
  scenario_perf.rs          — per-system tick budgets
  visual_world.rs           — sky/water/terrain goldens
  visual_ui.rs              — HUD, paused menu, chat overlay goldens
  observe_world.rs          — observe-mode template scenario
  goldens/
    sky_noon.png
    sky_night.png
    water_surface.png
    terrain_default.png
    hud_default.png
    ui_paused.png
    ui_chat.png

  # Pre-existing, kept as-is:
  smoke.rs
  worldgen_fingerprint.rs
```

`tests/smoke.rs` and `tests/worldgen_fingerprint.rs` stay put. They are still useful, run fast, and renaming churns git history for no gain.

---

## 4. Layer A — in-process `Harness` API

```rust
// tests/common/harness.rs

pub struct Harness {
    pub world: World,
    pub ecs: GameEcs,
    pub generator: Arc<Generator>,
    pub registry: Arc<BlockRegistry>,
    pub jobs: Jobs,
    input_buf: InputBuf,
    input_state: InputState,
    tick_count: u64,
    last_timings: TickTimings,
    persistence: Option<Persistence>,   // Some in TempDir mode
    saves_dir: Option<PathBuf>,         // Some in TempDir mode
}

pub struct HarnessBuilder {
    seed: u64,                          // default 42
    spawn: Vec3,                        // default (0, 250, 0)
    preload_radius: i32,                // default 2 chunks
    persistence: PersistenceMode,       // default InMemory
    worldgen_config: Option<PathBuf>,   // default embedded baseline
}

pub enum PersistenceMode {
    InMemory,
    TempDir,                            // real region files in tempfile::TempDir
}

impl Harness {
    pub fn builder() -> HarnessBuilder;

    // Time ────────────────────────────────────────────
    pub fn tick(&mut self) -> &TickTimings;     // dt = 1/60s
    pub fn tick_n(&mut self, n: u32) -> &TickTimings;
    pub fn drain_jobs(&mut self);

    // Input (synthetic) ───────────────────────────────
    pub fn press(&mut self, key: KeyCode);
    pub fn release(&mut self, key: KeyCode);
    pub fn mouse_motion(&mut self, dx: f32, dy: f32);
    pub fn mouse_button(&mut self, button: MouseButton, state: ElementState);
    pub fn scroll(&mut self, lines: f32);

    // World edits ─────────────────────────────────────
    pub fn set_block(&mut self, pos: BlockPos, block: Block);  // god-mode
    pub fn block_at(&self, pos: BlockPos) -> Block;
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<RayHit>;
    pub fn place_block(&mut self, pos: BlockPos, block: Block); // real edit path
    pub fn break_block(&mut self, pos: BlockPos);               // real edit path

    // Player ──────────────────────────────────────────
    pub fn player_pos(&self) -> Vec3;
    pub fn player_velocity(&self) -> Vec3;
    pub fn teleport(&mut self, pos: Vec3);
    pub fn look(&mut self, yaw: f32, pitch: f32);

    // Snapshots ───────────────────────────────────────
    pub fn snapshot(&self) -> WorldSnapshot;
    pub fn last_timings(&self) -> &TickTimings;

    // Persistence (TempDir mode) ──────────────────────
    pub fn flush_persistence(&mut self);
}

pub struct TickTimings {
    pub input: Duration,
    pub movement: Duration,
    pub world_stream: Duration,
    pub drain_jobs: Duration,
    pub world_unload: Duration,
    pub total: Duration,
}

pub struct WorldSnapshot {
    pub tick: u64,
    pub player: PlayerSnapshot,
    pub loaded_chunks: Vec<(ChunkCoord, u64)>,   // sorted by coord, chunk hash
    pub edits: u32,
}
```

### 4.1 Design notes

- **Fixed step.** `tick()` always advances `dt = 1/60s`. No wall-clock anywhere — movement and physics already take `dt` as a parameter (Instant::now is only used in `app.rs` for the FPS meter, autosave, and shader time; none of those run under the harness).
- **Synthetic input goes through the real `InputBuf`** — the same path `winit` feeds in `main.rs`. So testing "WASD walks the player" exercises the actual input-buffering and movement code, not a parallel mock. Edge/level semantics are whatever the existing input system already uses; the harness inherits them.
- **Edit operations have two flavors.** `set_block` is a god-mode write — fast, ignores raycast/range. `place_block` and `break_block` go through the real edit pipeline (raycast targeting from camera, adjacency rules, lighting relight queue). Most physics scenarios use the first; place/break scenarios use the second.
- **`drain_jobs()` is explicit.** Worldgen, meshing, and lighting are off-thread. A scenario that needs chunks loaded calls `drain_jobs()` until the queue settles.
- **Persistence is opt-in.** `InMemory` is default. `TempDir` mode runs the real `Persistence` thread pointing at a `tempfile::TempDir`.
- **`WorldSnapshot` is the diffing currency.** When an assert fails, a scenario can print "expected chunk hash X at (0,0,0), got Y" with file/line.

### 4.2 What is deliberately NOT in v1

- A scripting language. Scenarios are plain Rust `#[test]` functions.
- A general world-edit DSL. Setting 100 blocks is a `for` loop.
- A mock GPU surface. Layer A doesn't render.

---

## 5. Layer B — visual capture + state dump

Reframed from the original "golden diff only" idea. The harness needs **observe** as well as **compare**, because new features have no baseline.

### 5.1 Two operating modes

**Observe mode** (default for new visual work). Capture a frame plus a structured state dump and exit successful. Artifacts are the deliverable, not the assertion. Workflow:

1. `tests/observe_pig.rs` boots the world, spawns a pig at known coords, ticks N times, takes a screenshot, dumps state.
2. `cargo test observe_pig` runs it. Always passes.
3. The agent reads `target/harness/observe_pig/frame.png` and `state.json`, judges the result, iterates.

No goldens to update. Every iteration during tuning is intentional.

**Golden mode** (opt-in, for stable surfaces). Same capture, but the scenario also diffs against `tests/goldens/<name>.png` using `PixelTolerance`. Used for: sky, water shader, terrain rendering, HUD layout, paused/chat overlays — things that should not move once they are right.

### 5.2 API

```rust
// tests/common/visual.rs

pub struct VisualScenario {
    pub name: &'static str,
    pub flags: Vec<String>,            // --spawn / --look / --time / --shader-time / --ui
    pub tolerance: PixelTolerance,     // golden mode only
}

pub struct PixelTolerance {
    pub avg_l1: f32,                   // default 2.0 (calibrate per scenario)
    pub max_bad_frac: f32,             // default 0.005
    pub per_pixel_l1: u32,             // default 16
}

// Macros: write artifact bundle on failure, panic with standard template.
observe!(VisualScenario { ... });
golden!(VisualScenario { ... });
```

### 5.3 Flow per scenario

1. Build the flag list (`--screenshot-and-exit target/harness/<name>/actual.png` plus scenario-specific spawn/look/time/shader-time/ui).
2. Shell out to `cargo run --release -- <flags>` via `std::process::Command`. Reuse the cached release binary if `target/release/oxium` exists. No daemon.
3. In observe mode: write `actual.png`, `state.json`, `cmd.txt`. Pass.
4. In golden mode: also decode `tests/goldens/<name>.png`, compute the diff, write `diff.png` (3-up composite) and `diff.json`. Pass if within tolerance, fail otherwise.

### 5.4 Determinism

The screenshot path already gives most of what is needed: deterministic seed, deterministic spawn (`--spawn`), deterministic camera (`--look`), deterministic time-of-day (`--time`). The water shader animates on wall-clock `start_time`; for goldens we need shader animation phase to be pinnable.

**New binary flag: `--shader-time <secs>`.** Locks shader animation phase. `--time` remains semantic sun-cycle control. Both pin together for every visual scenario.

### 5.5 Structured diff output

When a golden test fails, the agent needs to triage *where* the diff is and *how much*, not just look at three PNGs.

`diff.png` is one 3-up horizontal composite: `[actual | golden | overlay]`. The overlay is the actual frame at 30% opacity with bad pixels painted bright magenta. One Read call shows the regression in context.

`diff.json` is structured stats:

```json
{
  "name": "ui_paused",
  "avg_l1": 4.2,
  "bad_frac": 0.018,
  "bad_bbox": { "x": 240, "y": 380, "w": 180, "h": 120 },
  "bad_region_pct_of_frame": 4.2,
  "dominant_channel": "g",
  "mean_offset": { "r": -1, "g": 12, "b": 3 }
}
```

`state.json` is the world-side context behind the frame:

```json
{
  "tick": 600,
  "camera": { "pos": [10.0, 65.0, 12.0], "yaw": 0.0, "pitch": -0.2 },
  "time_of_day": 0.5,
  "shader_time": 0.0,
  "entities_visible": [
    { "id": 42, "kind": "pig", "pos": [12.0, 64.0, 8.0],
      "components": { "health": 20, "wander_state": "Idle", "facing": 1.57 } }
  ],
  "loaded_chunks": 47,
  "draw_calls": 312,
  "mesh_count": 47,
  "perf": { "movement_ms": 0.18, "world_stream_ms": 1.4 }
}
```

### 5.6 What is deliberately NOT in v1

- Cross-platform goldens. Goldens are pinned to the machine that generated them; CI either uses matching hardware or skips Layer B via `OXIUM_SKIP_VISUAL=1`.
- Perceptual color metrics (delta-E, SSIM). Per-channel L1 is enough and easier to debug.
- Video / multi-frame goldens. Single frame only.
- Automatic golden regeneration. Goldens are committed by humans (or by the agent running the printed `cp` accept-line) — never by the test runner.
- HTML or web reporters.

---

## 6. Determinism & input plumbing

### 6.1 Tick contract

```
harness.tick():
  apply queued synthetic input
  input         system   (consumes InputBuf)
  movement      system   (uses dt = 1/60s)
  world_stream  system
  drain_jobs    system
  world_unload  system
  (render skipped)
  clear_input_buf
  tick_count += 1
  TickTimings recorded
```

Same order as `AppState::step` minus `render`. The harness owns the same fields `AppState` owns, minus `window`, `renderer`, `persistence` (unless `TempDir` mode), `last_autosave`, `last_tick`, `start_time`, `fps_meter`, `profiler`.

### 6.2 Sources of nondeterminism, closed

| Source | How it is handled |
|---|---|
| Wall-clock time | Fixed `dt = 1/60s`. No `Instant::now()` calls in any system the harness runs. |
| Rayon job ordering | Jobs run on the real pool. `drain_jobs()` waits until the channel is empty — final state converges. Asserts run after drain. |
| Persistence I/O | Default `InMemory` means no I/O thread. `TempDir` mode exposes `flush_persistence()` for round-trip tests. |
| `rand` crate | Only used for `WorldManifest::seed` at world creation. Harness builder takes an explicit seed — never invokes `rand`. |
| HashMap iteration | `WorldSnapshot` sorts loaded chunks by `ChunkCoord` before serializing. |
| GPU driver noise | Layer A doesn't touch GPU. Layer B absorbs it via `PixelTolerance`. |
| Float determinism | Within a single binary on a single machine: fine. Cross-platform: not promised. |

### 6.3 What is deliberately NOT done

- Pinning rayon to a single thread. Would break the deterministic-by-design property of `drain_jobs` waiting to settle, AND would make tests artificially slow.
- Implementing a "fake clock" abstraction. The codebase's dt-as-parameter convention is already the cleanest design.
- Building a separate event queue for input. The harness piggybacks on the existing `InputBuf`.

---

## 7. Failure ergonomics & artifacts

The single most important property of this harness: when a test fails, the agent gets enough information **in one place** to fix it without re-running.

### 7.1 Per-scenario artifact directory

Every scenario writes to `target/harness/<scenario>/`. Predictable paths, no timestamps, overwritten on rerun.

**Layer A failures:**

```
target/harness/scenario_physics::gravity_lands_player_on_block/
  state_initial.json     — world + player at tick=0
  state_final.json       — same at the asserting tick
  tick_timings.json      — per-system durations for every tick
  trace.log              — block edits, raycast hits, input events (grep-friendly)
```

`trace.log` is plain text, one event per line:

```
t=0     spawn_player pos=(0,80,0)
t=0     set_block (0,79,0)=Stone
t=1     input press W
t=60    movement: pos=(0.0, 79.5, -3.2) vel=(0.0, 0.0, -8.0)
t=72    raycast origin=(0,80,0) dir=(0,-1,0) hit=(0,79,0)
```

**Layer B failures (golden mode):**

```
target/harness/visual_world::sky_noon/
  actual.png             — captured frame
  diff.png               — 3-up composite: actual | golden | overlay
  diff.json              — structured stats
  state.json             — entities, camera, time, draw stats
  cmd.txt                — the `cargo run` invocation
```

Observe mode: only `actual.png`, `state.json`, `cmd.txt`. No diff, no asserts.

### 7.2 Panic message template

Layer A failure:

```
SCENARIO FAILED: scenario_physics::gravity_lands_player_on_block
  expected: player.y ≈ 79.5 (tolerance 0.05)
  actual:   player.y = 77.2
  tick:     72
  artifacts: target/harness/scenario_physics::gravity_lands_player_on_block/
    state_final.json   — world/player state at assert
    trace.log          — event log
```

Layer B golden failure:

```
SCENARIO FAILED: visual_world::sky_noon
  avg_l1 = 4.2 (threshold 1.0)
  bad_frac = 1.8% (threshold 0.1%)
  bad_bbox: (240,380)–(420,500), green-dominant
  artifacts: target/harness/visual_world::sky_noon/
    diff.png    — 3-up: actual | golden | overlay
    diff.json   — structured stats
  accept: cp target/harness/visual_world::sky_noon/actual.png tests/goldens/sky_noon.png
```

The accept-line is a literal, copy-paste-runnable command.

### 7.3 Helper macros

`tests/common/asserts.rs` provides macros that wire artifact-writing into panic messages:

```rust
assert_pos!(harness, expected: Vec3::new(0.0, 79.5, 0.0), tol: 0.05);
assert_block!(harness, (0, 79, 0), Block::Stone);
assert_tick_budget!(harness, "movement", 0.5ms);
golden!(VisualScenario { ... });
observe!(VisualScenario { ... });
```

Each macro on failure: writes the artifact bundle, panics with the standardized template. No scenario needs to remember to dump state.

### 7.4 Environment knobs

| Var | Effect |
|---|---|
| `OXIUM_HARNESS_KEEP_ON_PASS=1` | Write the full artifact bundle even on pass. Useful when debugging "test passes but does the wrong thing". |
| `OXIUM_SKIP_VISUAL=1` | Skip Layer B entirely. CI uses this on mismatched hardware. |
| `OXIUM_HARNESS_VERBOSE=1` | Emit `trace.log` for input/edit events on every passing test. |

Three knobs total. No config file.

---

## 8. Initial scenario catalog

A starter set small enough to land alongside the harness and concrete enough that each test maps to a pain point.

### 8.1 Layer A

| File | Scenario | What it pins |
|---|---|---|
| `scenario_physics.rs` | `gravity_lands_player_on_block` | Drop player at y=80 over a stone column; tick to rest; assert final y is at column-top + half player height ± ε. |
| | `walking_forward_advances_position` | Press W, tick 60; assert player advanced ≥ X blocks along facing, no Y drift. |
| | `wall_stops_player` | Place wall, press W into it, tick 60; assert player position unchanged on the blocked axis. |
| | `jump_arc_peaks_then_falls` | Press jump, tick; assert peak Y above start, returns within ε. |
| `scenario_edit.rs` | `raycast_hits_nearest_block` | Place block at known coord; raycast from camera; assert hit coord matches. |
| | `place_block_in_air` | Aim at block face; `place_block`; tick; assert target voxel set, neighbor lighting updated. |
| | `break_block_clears_voxel_and_relights` | Place opaque block in lit area; break; tick; assert voxel is air, light propagated back. |
| `scenario_worldgen.rs` | `biome_distribution_at_seed_42` | Sample 16k columns over 4 km²; histogram biome ids; assert each biome frequency within ±5% of expected band. |
| | `caves_form_connected_network` | Flood-fill from a known cave cell in seed-42 world; assert connected cells > N. |
| | `no_water_above_sea_level_inland` | Sample 5k columns far from coast; assert no Water blocks above `SEA_LEVEL`. |
| | `surface_band_density_no_chevrons` | Assert no axis-aligned stair patterns by checking heightmap variance across a small block-scale window. |
| `scenario_persistence.rs` | `chunk_round_trip_bytewise` | Generate chunk → save to TempDir → load → assert byte-identical compressed payload. |
| | `save_index_resists_torn_writes` | Simulate mid-write crash; reopen; assert index recovers consistent state. |
| `scenario_perf.rs` | `meshing_one_chunk_under_budget` | Mesh a representative chunk; assert `< 5ms` on release build. |
| | `worldstream_tick_under_budget` | Steady-state tick with full load radius; assert tick total `< 8ms`. |

Budgets in `scenario_perf.rs` are pinned at observed value × 1.5 on first run, tightened later.

### 8.2 Layer B

| File | Scenario | Mode | What it pins |
|---|---|---|---|
| `visual_world.rs` | `sky_noon`, `sky_night` | golden | Sky gradient, sun/moon position. |
| | `water_surface` | golden | Water shader output at locked `--shader-time`. |
| | `terrain_default` | golden | Greedy mesh + AO + lighting at seed 42 default view. |
| `visual_ui.rs` | `hud_default` | golden | Crosshair, hotbar, font. |
| | `ui_paused` | golden | Pause menu layout. |
| | `ui_chat` | golden | Chat overlay. |
| `observe_world.rs` | `observe_default_spawn` | observe | Smoke check; template for new observe scenarios. |

### 8.3 Visual scenario flag set

| Name | Flags |
|---|---|
| `sky_noon` | `--spawn 0,250,0 --look 0,0 --time 0.5 --shader-time 0` |
| `sky_night` | `--spawn 0,250,0 --look 0,0 --time 0.85 --shader-time 0` |
| `water_surface` | `--find-water --look 0,-10 --time 0.5 --shader-time 0` |
| `terrain_default` | `--spawn 0,80,0 --look 90,-15 --time 0.5 --shader-time 0` |
| `hud_default` | `--spawn 0,80,0 --look 0,0 --time 0.5 --shader-time 0` |
| `ui_paused` | `--spawn 0,80,0 --ui paused --time 0.5 --shader-time 0` |
| `ui_chat` | `--spawn 0,80,0 --ui chat --time 0.5 --shader-time 0` |

### 8.4 Catalog growth

New scenarios are 95% of future work. Adding one is: create a `#[test]` fn that builds a `Harness`, ticks, asserts. No registration, no central catalog file. The catalog grows by accretion the same way `src/**/*.rs` unit tests do.

---

## 9. Binary changes required

Exactly one: add `--shader-time <secs>` to `src/main.rs` CLI and thread it through to the shader-time uniform that today reads `state.start_time.elapsed().as_secs_f32()`. When `--shader-time` is set (used in screenshot mode), use the override; otherwise behave as today.

Everything else in `src/main.rs` stays put. The existing `--screenshot-and-exit`, `--spawn`, `--look`, `--find-water`, `--time`, `--ui`, and `OXIUM_SCREENSHOT_WARMUP_FRAMES` cover the rest of Layer B's needs.

---

## 10. Risks to validate during implementation

- **Mesher / lighting determinism under rayon.** The harness story leans on `drain_jobs` making final state deterministic regardless of work-stealing order. Most likely true (mesh output is a function of chunk + neighbors, not order), but worth a fingerprint test that generates a region with N=1 thread vs N=8 threads and asserts byte-equal `WorldSnapshot`. If it diverges, either pin worker count for tests or fix the offending system.
- **Persistence byte-equality.** `bincode` is deterministic for a fixed schema; `zstd` is deterministic at a fixed level. The "chunk save → load → re-save matches" assertion needs the level pinned. A future `zstd` bump implies golden regeneration.
- **GPU driver noise on Layer B goldens.** Start tolerances at `avg_l1=2.0, bad_frac=0.5%`. Tighten case-by-case once first-batch goldens are captured. Calibration is part of the initial PR.

---

## 11. Rollout

Single PR. Lands the harness, all scenarios from Section 8, the `--shader-time` flag, and the initial golden PNG batch together. Justification: the pieces depend on each other (Layer A asserts use macros that depend on the snapshot serializer, Layer B depends on `--shader-time`, the catalog depends on both layers) and splitting would mean three weeks of half-merged tooling sitting in branches.

Review-aid: the PR description lists the file boundaries from Section 3 explicitly so a reviewer can sweep by area (`tests/common/` first, then `scenario_*.rs`, then `visual_*.rs` + `observe_*.rs` + `goldens/`).

---

## 12. Open questions for follow-up

- **Mob AI scenarios** — when mobs land, the observe-mode workflow gains scenarios. No prep work needed in this spec; the API surface already covers it.
- **Inventory / crafting** — same: scenarios will arrive with the feature.
- **CI hardware tier** — to decide once Layer B goldens exist and have been profiled across at least two machines. If consistent: enable on CI. If not: keep `OXIUM_SKIP_VISUAL=1` set in CI and treat goldens as a local pre-merge check.
- **VLM-as-judge** — explicitly deferred. May revisit when adjudication of subjective visuals becomes a real bottleneck.
