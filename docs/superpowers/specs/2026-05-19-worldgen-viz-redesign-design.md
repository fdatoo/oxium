# Worldgen visualizer redesign — Design Spec

**Date:** 2026-05-19
**Status:** Drafted, awaiting approval
**Branch:** `worktree-worldgen-viz-redesign` (worktree at `.claude/worktrees/worldgen-viz-redesign`)
**Approach:** Approach 1 — Studio, reusing the game's chunk/mesher/jobs subsystems. Editable DAG (Approach 3) is captured as a follow-up artifact, gated on the worldgen overhaul completing.

## Summary

Replace `src/bin/worldgen_viz/` with a streaming, multi-workspace **studio** that exposes every stage of the worldgen pipeline. A Dense-Dashboard layout (no tabs, no docking) shows the 3D world (camera-following chunk streaming), pannable 2D pipeline overlays, cross-sections, the parameter editor, a column probe, a read-only DAG view, and iteration tools (A/B compare, preset library, parameter sweep atlas, headless render). The viz reuses the game's `voxel`, `mesher`, and `jobs` modules; it adds a thin paint-mode layer on top of the existing mesher and a small set of additive, read-only introspection methods on `Generator`. No worldgen-generation-path changes.

The current binary's experience (fixed 64×128×64 region, orbit camera, side panels, top-down thumbnail) is replaced wholesale at PR 1, then expanded over six follow-on PRs. After PR 1 the new viz is already strictly better than today's; each later PR adds a workflow.

## Goals

1. **See every stage of the pipeline**, not just the final mesh. Plates, climate, hydrology, biomes, caves, density — all queryable as 2D overlays or via a column probe.
2. **Tuning iterates in &lt;1 s for the common case.** Editing a knob re-meshes a tight ring of chunks around the camera synchronously after debounce; outer chunks update async over the next ~1–2 s.
3. **Reason about WHY** any column looks the way it does: click it, see every intermediate value the pipeline produced for it.
4. **Compare** two configs (A/B) and **sweep** one parameter (atlas grid) without leaving the tool.
5. **Render headless** for PR comparison images and image-diff regression tests.

## Non-goals

- Game-engine features (entities, physics, lighting, players).
- Editable DAG / worldgen-as-runtime-graph — captured as the sibling follow-up artifact, not built here. Gated on the worldgen overhaul completing.
- Multi-monitor docking (egui_dock is rough; Dense Dashboard with fixed splitters is sufficient).
- Multiplayer / shared worlds, networking, undo/redo.
- Configurable worlds that don't share the game's `WorldgenConfig` schema.
- Persisting sweep-atlas thumbnail grids to disk (the grid is ephemeral in v1).

## Architecture

### Module layout

```
src/bin/worldgen_viz/
├── main.rs              # winit app, frame loop, surface management
├── app.rs               # AppState: sessions, cameras, ui state
├── session.rs           # one "view" of a world: Generator + caches + cameras
├── layout.rs            # dashboard egui panels (splitter positions)
├── world/
│   ├── mod.rs           # streaming chunk cache + meshing pipeline
│   ├── stream.rs        # camera-follow load/unload (radius)
│   ├── mesher.rs        # thin wrapper over oxium::mesher with paint hook
│   └── invalidate.rs    # config-change → wipe affected chunks
├── camera.rs            # fly cam (WASD + mouse-look) + orbit fallback
├── paint.rs             # paint modes: block, biome, height-Δ, density, …
├── overlays/
│   ├── mod.rs           # 2D MapView; pan/zoom; share-camera link
│   ├── stages.rs        # one sampler per Stage (plate, climate, hydro, …)
│   └── colormap.rs      # named gradient palettes
├── probe.rs             # column probe — click → ProbeResult
├── crosssection.rs      # cut-plane heatmap (XZ / XY / YZ, draggable)
├── compare.rs           # A/B: two sessions, linked cameras, split viewport
├── sweep.rs             # atlas: parameter × N → thumbnail grid
├── preset.rs            # named presets w/ notes; on-disk library
├── dag.rs               # read-only DAG view with live trace
├── widgets/
│   ├── spline.rs        # refined spline editor (today's widget evolved)
│   ├── colormap.rs      # gradient picker
│   └── probe_table.rs   # field-value list for probe panel
├── headless.rs          # offscreen-render subcommand (no winit, no egui)
└── render/
    ├── mod.rs
    ├── scene.rs
    └── shader.wgsl
```

Game subsystems reused as libraries (no fork): `oxium::voxel::DenseChunk`, `oxium::mesher::*`, `oxium::jobs::*`, `oxium::worldgen::*`.

### Sessions

```rust
pub struct Session {
    pub generator: Arc<Generator>,
    pub config: ConfigHolder,
    pub chunks: Arc<RwLock<ChunkCache>>,
    pub camera: FlyCamera,
    pub probe: Option<ColumnProbe>,
    pub paint: PaintMode,
}
```

The dashboard's "world" is normally a single `Session`. A/B compare spawns a second one. The Atlas mode spawns N short-lived sessions (each owns a small region; they don't stream). Sessions share the egui context and the wgpu device but own their own chunk caches and camera state.

## Worldgen API additions

All additive, read-only, non-mutating to the generation path. Should be safe to land alongside the other worldgen overhaul session because they touch only the public surface.

```rust
// in oxium::worldgen

pub struct ColumnProbe {
    pub wx: i32, pub wz: i32,
    pub plate: PlateLookup,                  // primary, secondary, boundary_t
    pub climate: ClimateProbe,               // temperature, humidity, desertness
    pub heightmap: HeightmapProbe,           // h_pre, valley_carve, h_target, slope, is_cliff
    pub hydrology: HydrologyProbe,           // flow_accum, lake_rim, river_width
    pub splines: SplineProbe,                // offset_corr, factor_corr, jagged
    pub biome: Biome,
    pub caves: Vec<CaveSystemRef>,           // intersecting bbox; (SystemId, bbox)
    pub aquifer_y: Option<i32>,
}

pub struct DensityBreakdown {
    pub bias: f32,           // (h_target - wy) / FALLOFF
    pub base_3d: f32,        // 3D noise contribution
    pub cave_sdf: f32,
    pub final_density: f32,  // bias + base_3d - cave_sdf, post-slide
}

impl Generator {
    pub fn probe_column(&self, wx: i32, wz: i32) -> ColumnProbe;
    pub fn evaluate_density_breakdown(&self, wx: i32, wy: i32, wz: i32) -> DensityBreakdown;
    pub fn sample_stage(&self, stage: Stage, wx: i32, wz: i32) -> f32;
    pub fn cave_systems_in(&self, region: Aabb2) -> Vec<CaveSystemRef>;
    pub fn density_slab(&self, chunk: ChunkCoord) -> DensitySlab; // 18^3 halo
}

pub enum Stage {
    Continentalness, PlateId,
    Temperature, Humidity, Desertness,
    HPre, ValleyCarve, HTarget, FlowAccum,
    BiomeId, CaveCoverage, AquiferY,
}
```

All four methods derive from values the pipeline already computes; we expose them instead of hiding them behind `column_data`. `CaveSystemRef` leaks no internals — just an opaque `SystemId` plus the system's bounding box, suitable for overlay rendering.

`DensitySlab` is an 18³ `Box<[f32; 5832]>` (chunk + 1-block halo) sampled once during chunk fill and stored beside the chunk. Costs ~3% of chunk-fill time. Consumed by the **density-gradient paint mode** so paint passes don't re-evaluate noise.

## Streaming world view

### Load radius and cache

- 8 chunks XZ, 4 chunks Y around the camera. ~256 chunks resident at any time.
- `LruCache<ChunkCoord, MeshedChunk>` for geometry: `(positions, normals, voxel_position_per_face)` — no colors baked in.
- Parallel `LruCache<(ChunkCoord, PaintMode), ColorBuffer>` for colors. Toggling paint mode rebuilds colors only (no remesh).

### Background pipeline

Two stages, both on `oxium::jobs`:

1. **Fill** — `Generator::fill_chunk(coord, &mut DenseChunk)`. Pure CPU. ~5–10 ms per chunk.
2. **Mesh** — `oxium::mesher`. Geometry only.

Priority queue: closer chunks first. The mesher upload buffers as it completes each chunk; the 3D viewport draws whatever is currently in the cache (no waiting for a full radius to be ready).

### Edit-time invalidation (conservative)

Any `WorldgenConfig` change after the 200 ms debounce wipes both caches. The 25 chunks closest to the camera run a **synchronous high-priority pass** (so the tuning loop feels responsive); the outer ~230 chunks fill in over ~1–2 s in the background.

Localising invalidation per-field is brittle (caves move when continentalness moves; small changes cascade). v1 picks the conservative path. If profiling shows the wipe-all approach hurts the tuning loop, PR 7 introduces a field-tagged invalidation map.

### Camera

Default **fly cam**: WASD + mouse-look + Shift-boost + Q/E up/down. `O` toggles to an orbit fallback (today's behaviour). Both implement `crate::camera::Camera` so paint modes and overlays don't care which is active. Teleport-to-coord and teleport-to-probe-column buttons in the toolbar.

## Paint modes

Paint modes are CPU-side functions `(wx, wy, wz, face_normal, &PaintCtx) -> Rgb`. The mesher gives us per-face voxel positions; the paint pass walks each face and writes the color buffer. Toggling paint mode = rerun paint pass only (cheap, ~ms per chunk, no remesh).

| Mode | What it shows | Useful for |
|---|---|---|
| **Block** | block-type color (today's behaviour) | default; recognising terrain |
| **Biome** | column-biome color, with desert-transition dither | tuning biome thresholds |
| **Height-Δ** | `(h_target - h_pre)` as divergent gradient | seeing valley carve in 3D |
| **Density gradient** | density at the just-air neighbour, hot gradient | surface-fuzz / 3D-density tuning |
| **Cave distance** | distance to nearest cave centerline / SDF | tuning cave geometry; spotting dead zones |
| **Plate ID** | hashed color per plate; blended near boundary | sanity-checking continent decomposition |
| **Slope** | `|∇h_pre|` magnitude, warm→cool | cliff threshold tuning |

A `PaintCtx` carries per-column data precomputed when a chunk is meshed (so the paint pass doesn't re-sample noise). Paint mode + colormap are chosen from a small toolbar at the top of the 3D viewport.

## Pipeline overlays (2D map panel)

A pan/zoom 2D map in the dashboard, separate from the 3D viewport. Linked-camera mode optional. The map renders one **stage** at a time:

| Stage | Sampler | Colormap |
|---|---|---|
| Continentalness | `plate_at(seed, wx, wz).signed_continentalness` | divergent (ocean blue ↔ continent green) |
| Plate ID | `plate_at(...).primary_id` | categorical (stable hash → hue) |
| Temperature | climate noise sampler | viridis |
| Humidity | climate noise sampler | viridis |
| Desertness | desert-mask noise | viridis |
| h_pre | heightmap (pre-carve) | terrain ramp |
| Valley carve | `valley_carve(wx, wz, &region)` | hot (intensity = depth) |
| h_target = height | `column_data.height` | terrain ramp |
| Flow accumulation | from `FineRegion` | log-scaled blues |
| Biome | column biome | categorical |
| Cave coverage | bool: "any cave system intersects this column" | binary |
| Aquifer Y | aquifer water table | divergent around sea level |

Each stage is a struct implementing:

```rust
trait StageSampler {
    fn sample(&self, generator: &Generator, wx: i32, wz: i32) -> f32;
    fn colormap(&self) -> Colormap;
}
```

Rendering: `MapView` walks pixels, calls the sampler, writes `ColorImage`, uploads to egui texture. 512×512 px at 2 blocks/px = 1024-block side; ~262k samples — most stages are cheap noise eval, ~30–100 ms cold. Render result cached per `(stage, world_x, world_z, zoom)`; invalidated on config change.

Pan = drag; zoom = scroll (changes blocks-per-pixel). Hover shows world-coords. Crosshair overlay = currently-probed column. Optional linked-camera mode pans the 3D viewport along.

## Column probe

Click a column on the 2D map or 3D viewport. (3D click = raycast into visible mesh → top solid voxel → its `(wx, wz)`.) The clicked column becomes the **pinned** column and turns into a vertical highlight in 3D + a crosshair on the map. A right-side Probe panel shows:

```
Column @ (wx, wz)
─── Geometry ──────────────────────────
  plate primary       Continental#7
  plate secondary     Oceanic#3      (t=0.18 boundary)
  h_pre               73.4
  valley_carve        −2.1
  height (h_target)   71
  is_cliff            false
  slope               0.12
─── Climate ──────────────────────────
  temperature         0.08
  humidity            0.32
  desertness          −0.22
  biome               Forest
─── Hydrology ────────────────────────
  flow_accum          417  (stream)
  lake_rim            —
  river width         3 blocks
─── Density (sliding y, slider in panel) ──
  bias(h, y=70)       +0.5     ← evaluate_density_breakdown.bias
  bias(h, y=80)       −1.2
  base_3d(y=75)       +0.15    ← evaluate_density_breakdown.base_3d
  cave_sdf(y=75)      0.00     ← evaluate_density_breakdown.cave_sdf
  final density(y=75) +0.65
─── Caves ────────────────────────────
  cave systems        2 (#5e1a, #5e1c)
  inside_chamber      no
─── Spline outputs ───────────────────
  offset_corr         +0.04
  factor_corr         +0.12
  jagged              +0.08
```

Every value shown corresponds to one node in the DAG view (PR 5); probing a column lights that DAG node up with the value.

Two probe-panel actions:
- **Teleport camera here** — fly cam centered on the column.
- **Copy as RON test fixture** — emits a snippet you can paste into a `#[test]` to lock in this behavior.

## Cross-sections (cut-plane heatmap)

A separate dashboard panel:

- **Orientation:** XZ (horizontal, fixed Y), XY (vertical, fixed Z), YZ (vertical, fixed X).
- **Position:** slider for the constant axis (e.g., XY slice, drag Z from −512 to +512).
- **Size:** 256 blocks × 256 blocks. 1 block per pixel = 256×256 image.

Selectable **layer**:

- **Block** — fill solid voxels with block color (most useful for vertical slices showing strata).
- **Density** — signed scalar, divergent gradient through 0.
- **Cave SDF** — distance to nearest chamber/tunnel.
- **Biome** — XZ slice; column biome.
- **Aquifer water table** — XZ slice colored by `aquifer.water_table_y - y_of_slice`.

**Annotations** overlaid:

- River segments where they cross the slice (cyan polylines).
- Cave-system bounding boxes intersecting the slice (dashed yellow).
- Pinned column from the probe (white crosshair).

The cut plane is also drawn in the 3D viewport as a translucent rectangle so the user sees where the slice is in space.

## A/B compare

Press `[ Compare ]` in the toolbar → dashboard splits into **two synced viewports** side-by-side. Each is backed by its own `Session` (own `Generator`, own caches), so they can run different `WorldgenConfig`s. Linked controls by default; unlinkable per-axis:

- **Camera link** — pan/zoom one, both move.
- **Probe link** — clicking a column probes both sessions; the Probe panel shows fields in two columns marked `A`/`B`, with deltas highlighted (e.g. `temp: 0.08 → 0.12 (+0.04)`).
- **Map link** — both 2D maps show the same stage at the same coords. Toggle "diff" mode → a third map shows `B − A` for the active scalar stage.
- **Cross-section link** — same cut plane, two heatmaps.

The param panel grows an `A | B` toggle at the top: edits land on whichever session is active. A `Swap` button exchanges them. A `Sync B from A` button copies one config to the other (so edits can be made from a shared baseline).

Exiting compare drops B's cache; A becomes the only active session.

## Preset library

`assets/worldgen/presets/` directory; each preset is `<name>.ron` + an optional `<name>.notes.md` for human commentary. A dashboard panel `Presets`:

- List of all presets with a short label.
- Click → load into the active session (with a confirmation prompt if there are unsaved edits).
- `Save current as…` prompts a name.
- `Rename`, `Duplicate`, `Delete`.
- Right-click → "Load into B side" for compare flows.

The default config (`assets/worldgen/default.ron`) is the immutable base preset; can be cloned, never overwritten by the viz. Naming convention `<short-handle>-vN.ron` (e.g., `alpine-v2.ron`) so iterations don't trample each other. The notes file is freeform.

## Sweep atlas

A separate dashboard mode (toolbar `[ Atlas ]`).

**Setup form:**

- **Parameter** — dropdown of every numeric `WorldgenConfig` field with associated range metadata. Today's UI panels carry implicit ranges (e.g., `cheese_threshold: -1.0..=1.0`); v1 maintains a small `crate::sweep::registry` mapping field paths to `(min, max, label, group)` rather than introducing a derive macro.
- **From / To / Steps** — numeric range and N (clamped to N ≤ 25).
- **View** — top-down 2D, or a fixed-angle 3D thumbnail.
- **Render** — kicks off N independent generations.

Each thumbnail is labelled with the parameter value. **Click** a thumbnail → loads that value into the active session and exits the atlas. **Hover** → see the probe values for the same seed/coords across thumbnails so the user can spot what changed.

Render budget: per-thumbnail, a fixed 64×64×96 voxel region; 128×128 px output. ~30 ms per thumbnail × 16 = ~500 ms total single-threaded; parallel across cores ~80 ms. Background task, non-blocking.

Sweep results are not persisted; the grid is ephemeral. Persistence can come later if needed.

## Read-only DAG view (with live trace)

Hardcoded topology — no graph engine. egui drawing of boxes and arrows. Each node:

```
┌────────────────────────────────┐
│ offset_spline           [PR 3] │   ← name + worldgen-PR tag
│ in:  c, s, r                   │
│ out: offset_corr               │
│ live:  +0.04                   │   ← lit when a column is probed
└────────────────────────────────┘
```

Nodes grouped into columns roughly matching pipeline stages:

```
[Plate]  →  [Climate]  →  [Spline corrections]  →  [Density]  →  [Cave SDF]  →  [Solid?]  →  [Block / Water]
```

- **Hover** a node → tooltip with formula in math notation + source file:line.
- **Click** a node → focuses the relevant panel (e.g., click `offset_spline` → spline editor opens with offset selected).

Layout: hardcoded node positions in PR 5 (a ~20-node graph fits comfortably). Auto-layout only if needed. Implementation budget ≈ 300–500 lines of egui code.

The read-only DAG is also the *canvas* for the future editable-DAG follow-up: when the worldgen-as-graph refactor lands, this panel becomes interactive.

## Headless render mode

The same binary, called with a subcommand instead of opening a window:

```
worldgen_viz render \
    --preset assets/worldgen/presets/alpine-v2.ron \
    --view topdown \            # topdown | crosssection-xy | crosssection-xz | crosssection-yz | 3d-fixed
    --stage h_target \          # for topdown / cross: which stage to render
    --paint biome \             # for 3d-fixed: paint mode
    --seed 42 \
    --size 2048x2048 \
    --out alpine-v2.png
```

No winit, no egui — direct wgpu offscreen render to an `image::DynamicImage`. The render functions used here are the **same ones** the windowed binary uses (factored into `crate::overlays` + `crate::paint` modules with no UI dependencies); the binary just feeds them a deterministic camera + size + output buffer instead of an egui texture handle.

Two uses:

1. **PR comparison images.** Generate before/after side-by-side PNGs from two presets, attach to a PR description.
2. **Regression-test fingerprints.** A test renders a known seed + preset → byte-compares (or image-diffs with tolerance) against a golden PNG. If worldgen output changes inadvertently, the test fails with the actual delta image. Image-diff via the `image` crate; no extra deps.

Per-render time budget: 2048² topdown ≈ 2–3 s; 3D-fixed view ≈ 5–10 s (chunks must fill). Acceptable for a CLI tool.

## Testing

The viz is a graphical app; integration-testing a window doesn't make sense. Strategy:

- **Unit tests** on logic with no GPU or window deps:
  - Cache invalidation (config change → cache wiped, expected chunks re-filled in priority order).
  - Paint-mode functions are pure: input voxel/column → known color.
  - Stage samplers are pure and byte-stable per `(seed, wx, wz)`.
  - Sweep enumeration: given parameter range + steps → correct value list.
- **Headless render tests** (the real regression suite):
  - `tests/viz_render_golden.rs`: for each `(preset, view, paint, seed)` tuple, render to PNG, compare to `tests/golden/<name>.png` with a small per-pixel tolerance.
  - Update goldens with `UPDATE_GOLDENS=1 cargo test viz_render_golden`.
- **CI smoke test**: `cargo run --bin worldgen_viz -- --check` exits after one frame; CI runs this on every PR.
- **Windowed self-test** (manual, not CI): `cargo run --bin worldgen_viz -- --self-test` opens, exercises every panel/tab via a scripted input sequence, exits clean. Catches "the dashboard layout broke on resize" bugs.

## Phasing (PR plan)

Seven PRs, each independently shippable. Each PR rebases on top of the worktree branch. The current `worldgen_viz` binary stays usable until **PR 1** lands — at which point the new one replaces it. The user has confirmed no one is actively depending on the current viz, so the cutover is safe.

| PR | Title | Scope |
|---|---|---|
| **1** ✅ | viz: streaming skeleton | New binary; fly cam; LRU chunk cache; reuse mesher; block paint mode only; dashboard layout with placeholder panels. **Replaces old binary.** Landed at branch `worktree-worldgen-viz-redesign`. |
| **2** | viz: pipeline overlays + column probe | 2D map panel with all stages; click-to-probe; probe panel with full field list; worldgen API additions land here. |
| **3** | viz: cross-sections + extra paint modes | Cut plane (XZ/XY/YZ); biome / height-Δ / density / cave-distance / plate / slope paint modes. |
| **4** | viz: A/B compare + preset library | Dual-session compare; preset directory + notes; toolbar Compare button. |
| **5** | viz: read-only DAG view | egui-drawn DAG with live trace; click-to-jump-to-panel. |
| **6** | viz: sweep atlas + headless render | Atlas mode; `worldgen_viz render` subcommand; golden-PNG regression tests. |
| **7** | viz: polish + docs | Status-bar refinements, keybinding map, README in `docs/`, CLAUDE.md note, field-tagged invalidation if needed. |

Each PR self-contained, mergeable independently. After PR 1, the viz is already strictly better than today (streaming, fly cam, dashboard). Each subsequent PR adds a workflow.

## Risks

1. **Worldgen overhaul collisions.** Another session is rewriting `worldgen`. Our API additions are purely additive, but if the overhaul renames `ColumnData` or splits `Generator`, we will need to rebase. *Mitigation:* minimise API surface added in PR 1; expand in PR 2 once the overhaul settles.
2. **Streaming cost on edit.** "Wipe all on edit" might feel sluggish for tiny edits. *Mitigation:* bench in PR 1; if painful, add a config-field-aware invalidation map in PR 7.
3. **DAG layout aesthetics.** Auto-layouting a 20-node graph in egui without overlapping is non-trivial. *Mitigation:* hardcode node positions in PR 5; auto-layout only if needed.
4. **Goldens-vs-driver-jitter.** GPU rendering can differ across driver versions; pure byte equality won't survive. *Mitigation:* use the `image` crate's perceptual diff with a small tolerance; document the tolerance in the test.

## Open questions deferred to implementation

- **Mesher color hook.** Whether `oxium::mesher` exposes a clean "produce geometry without colors" path today, or whether we need to add one. Decide in PR 1 — if invasive, fork a `viz_mesher` shim locally.
- **DensitySlab cost.** Whether the 18³ density cache per chunk really lands at ~3% fill cost. Measure in PR 3 when the density-gradient paint mode lands.
- **Atlas concurrency.** Whether N generators in parallel hit memory pressure on smaller machines. PR 6 starts with a 4-worker cap; expand if profiling allows.

## Decision points

Before implementation, confirm:

1. **Worktree branch name**: `worktree-worldgen-viz-redesign` is the current branch. PRs target `main` once each is ready, **after** the worldgen overhaul session has merged its branch — to minimise merge conflicts. If urgent, PRs can target `main` independently and accept the rebase cost.
2. **Default viz seed**: today's hardcoded `42` becomes the default. PR 1 adds a `--seed N` CLI flag that overrides it at launch. PR 4 (presets/compare) adds a runtime seed override in the toolbar so the user can change worlds without restarting.

## Sibling artifact

`docs/superpowers/specs/2026-05-19-worldgen-editable-dag-followup.md` — the deferred Approach 3 outline. Committed alongside this spec; promoted to a real implementation spec once the worldgen overhaul lands.
