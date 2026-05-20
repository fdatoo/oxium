# Render performance rework — design

**Date:** 2026-05-20
**Status:** Approved (brainstorming → writing-plans)

## Goal

Cut CPU per-frame render cost from the per-chunk wgpu encoding loop that dominates today's profile, and along the way replace the `render` module's structural mess (~1.5k-line `render/mod.rs`, three duplicated chunk-iteration loops, per-chunk `BindGroup` allocations every frame) with a small set of focused components.

The primary symptom is **CPU-bound rendering**. The secondary motivation is **code quality** — the two are the same problem: the structural mess *is* the CPU cost. GPU-side changes are deliberately conservative.

## Non-goals

- No MSAA change. MSAA was added at commit `d0f1b74` specifically for voxel-silhouette smoothing and remains a visual asset for this game. We make it configurable (default 4×, switchable to 2×/1×) but don't replace it with FXAA/SMAA.
- No reflection-quality change. Reflections already render at 1/3 resolution (`REFLECTION_SCALE = 3`); the rework makes the reflection pass nearly free on CPU automatically, so there's no reason to drop visual quality.
- No new visual features. This is a pure rework — the rendered image at every baselined screenshot scene must match (within SSIM tolerance for floating-point drift).
- No worldgen / mesher algorithm changes. The mesher's **output format** changes (split opaque/water streams, new `chunk_id` field) but the meshing algorithm itself is untouched.

## Current state — what's slow and why

Reading `src/render/mod.rs` end-to-end identifies the following per-frame CPU costs:

1. **Per-chunk `BindGroup` creation × 3 passes.** Lines 1183–1204 (reflection), 1321–1342 (opaque), 1440–1461 (water) each call `device.create_bind_group` inside the per-chunk loop. At ~500 visible chunks × 3 passes, that's ~1500 fresh `BindGroup` objects per frame. Each one triggers wgpu validation + Metal/Vulkan descriptor allocation. This is the single biggest CPU cost.
2. **Three independent chunk iteration loops** with duplicated cull + LOD logic. Each pass re-runs frustum cull, distance cull, LOD pick.
3. **`upload_chunk_mesh` allocates a fresh `wgpu::Buffer` pair per chunk per upload** (`render::mesh::upload_mesh`). Every remesh = 2 GPU allocations.
4. **Per-chunk `ChunkLightVolume` is its own `wgpu::Texture`** with its own bind-group entry. Per-chunk allocation, per-chunk binding.
5. **`ChunkUniform` is a 16-byte per-chunk uniform buffer** (chunk origin) with its own bind. Trivial data, treated as bulky GPU state.
6. **Mid-frame `copy_texture_to_texture` of an MSAA depth attachment** (`render/mod.rs:1362–1380`) between opaque and water passes. Full-screen 4×-sample blit every frame.
7. **HUD allocates fresh vertex + index buffers per frame** (lines 1051–1062). Two `device.create_buffer_init` per HUD batch.
8. **Water uses the opaque vertex stream** and discards non-water fragments in the shader. Vertex shader runs over millions of opaque verts during the water pass for no output.

## Approach — GPU-driven chunk renderer + pass restructuring (combined A + C)

The rework is built around one structural idea: **the renderer should not iterate chunks**. It builds a *visibility list* once per frame, then each pass walks the list and issues draws against a shared GPU arena.

### New components

#### `render::arena::ChunkArena` *(new file)*

- Owns: `vbuf: wgpu::Buffer`, `ibuf: wgpu::Buffer`, free-list of `(offset, len)` slabs per buffer.
- Public API: `alloc(vertices: &[Vertex], indices: &[u32]) -> ChunkAllocation`, `free(alloc)`, `growth_metrics()`.
- Implementation: bump-then-coalesce allocator. Buffers start at ~32 MB each, grow ×1.5 when an alloc doesn't fit. Free slots are reused; fragmentation is bounded because all allocations are similar-sized chunk meshes.
- Replaces: today's per-upload `device.create_buffer_init` pair in `render::mesh::upload_mesh`.

#### `render::chunk_table::ChunkTable` *(new file)*

- Owns: `slots: Vec<ChunkSlot>`, `gpu_buf: wgpu::Buffer` (storage), `dirty_set: BitSet`.
- `ChunkSlot { origin: [f32; 4], light_layer: u32, flags: u32, _pad: [u32; 2] }` — 32 bytes, vec4-aligned.
- Public API: `register(coord) -> ChunkId`, `unregister(coord)`, `set_light_layer(id, layer)`, `flush_dirty(queue)`.
- The slot index is the `ChunkId` baked into the vertex stream. Once allocated, it's stable for the chunk's lifetime — re-meshes don't change it.
- Replaces: today's per-chunk `ChunkUniform` + per-chunk `chunk_bg` `BindGroup` creation.

#### `render::light_array::LightVolumeArray` *(new file)*

- Owns: a single `wgpu::Texture` of dimension `D3` with `depth_or_array_layers = max_chunks`, **or** a 2D mega-atlas if the wgpu 23 backend doesn't expose 3D texture arrays — detected at startup, behavior selected once.
- `alloc_layer() -> u32`, `free_layer(idx)`, `upload(layer, blob)`.
- Replaces: today's `chunk_lights: HashMap<ChunkCoord, ChunkLightVolume>` of one `wgpu::Texture` per chunk.

#### `render::visibility::VisibilityList` *(new file)*

- Built per `render()` call. Iterates `ChunkTable` slots, runs frustum + distance culling per slot, writes a packed list of `(chunk_id, lod_level, opaque_range, water_range)` entries.
- **Two lists per frame:** one for the main camera (shared by the opaque + water passes — same frustum, same distance, same LOD) and one for the reflected camera (different frustum + below-water clip). Distance + LOD inputs derive from the *unmirrored* eye in both cases, so only the frustum input differs.
- Two outputs: a `Vec<DrawEntry>` for CPU-issued draws (always), and a populated `IndirectDrawBuffer` for `multi_draw_indexed_indirect` (when the wgpu adapter feature is available; transparent fallback otherwise).

#### `render::passes` *(refactored from `render::mod`)*

- New module: `passes/reflection.rs`, `passes/opaque.rs`, `passes/water.rs`, `passes/composite.rs`, `passes/hud.rs`.
- Each pass takes `(encoder, frame_resources, visibility_list)` and emits draws. No iteration over `chunk_meshes` inside the pass — the visibility list is the single source of truth.
- `render/mod.rs` shrinks from 1538 lines to ~200 lines of frame-orchestrator code.

### Changes to existing code

#### Mesher (`src/mesher/`)

- `ChunkMesh` gains a separation: `opaque_vertices`, `opaque_indices`, `water_vertices`, `water_indices`. Eliminates the "draw all chunks through the water pipeline and `discard` non-water" waste.
- `Vertex` gains a `chunk_id: u16` field, replacing the per-chunk uniform bind. The mesher stamps it once per chunk during emission. Vertex grows from 16 → 20 bytes (pad to vec4 alignment; the existing `_pad`/`_pad2` slots absorb most of it).

#### Shaders (`assets/shaders/`)

- `opaque.wgsl` / `water.wgsl`: replace `@group(1) @binding(0) chunk: ChunkUniform` with `@group(1) @binding(0) chunk_table: array<ChunkData>` (storage). Read `chunk_table[in.chunk_id]` for origin.
- Light sampling: `texture_3d<f32>` → `texture_3d_array<f32>` (or atlas math when falling back to the 2D mega-atlas variant). Same UVW math, plus a `layer` argument from `chunk_table[in.chunk_id].light_layer`.

#### Pass structure (`render/mod.rs` → `render/passes/`)

- The mid-frame `copy_texture_to_texture(depth)` is eliminated: the opaque pass keeps `StoreOp::Store` on the depth attachment; the water pass binds the *same* depth texture as a read-only depth view (legal in wgpu when the pipeline has `depth_write_enabled: false` and the attachment is bound `DepthOnly` + read-only).
- HUD vertex/index buffers move into a persistent ring buffer with per-frame sub-allocation. No `create_buffer_init` per frame.

### MSAA — kept, made configurable

- `MSAA_SAMPLES` becomes a runtime setting (CLI flag + config file). Default 4×. Valid values: 1, 2, 4.
- All MSAA-aware resources (`make_depth_texture`, `make_msaa_color_texture`, `make_reflection_*`) take the sample count as a parameter.
- Pipelines are rebuilt on change (the value is fixed for the renderer's lifetime; a settings change requires a restart).

### Reflections — unchanged structurally

- Already 1/3 resolution. The CPU cost of re-iterating chunks goes to ~zero automatically once the GPU-driven path lands (same visibility list, same indirect buffer, different camera UBO). No design changes to the reflection pass beyond reusing the new components.

## Data flow

### Per-frame (the hot path)

```
camera + sun update                              (unchanged)
  ↓
write CameraUniform + ReflectionCameraUniform    (unchanged; 2 small queue writes)
  ↓
ChunkTable::flush_dirty()                        (NEW: one queue write for changed slots)
  ↓
VisibilityList::build(frustum, eye, lod_thresholds)
  ↓ (single iteration over ChunkTable; zero BindGroup allocations)
  ↓
Reflection pass:
  set_pipeline(opaque_reflection)
  set_bind_group(0, reflection_camera_bg)
  set_bind_group(1, chunk_table_bg)              (ONE bind group, not N)
  set_bind_group(2, atlas_bg)
  set_bind_group(3, light_array_bg)
  for entry in visibility_list (reflection frustum):
      draw_indexed(entry.range)
  ↓
Opaque pass: same shape, main camera.
  ↓
(NO depth copy here — same depth texture is bound read-only to water pass)
  ↓
Water pass: same shape, water pipeline, separate water VB ranges.
  ↓
Cursor + composite + HUD                          (essentially unchanged)
```

### Per-chunk upload (cold path)

```
job thread emits ChunkMesh + light blob
  ↓
main thread: drain_jobs receives them
  ↓
renderer.upload_chunk_mesh(coord, lod, mesh):
    chunk_id = chunk_table.register_or_lookup(coord)
    arena.alloc(mesh.opaque_vertices, mesh.opaque_indices) → opaque_range
    arena.alloc(mesh.water_vertices,  mesh.water_indices)  → water_range
    chunk_lods[chunk_id][lod] = (opaque_range, water_range)
    set chunk_table.flags |= has_lod{n}
  ↓
renderer.upload_chunk_light_volume(coord, blob):
    chunk_id = chunk_table.lookup(coord)
    layer    = light_array.layer_for_chunk(chunk_id)
    light_array.upload(layer, blob)
    chunk_table.set_light_layer(chunk_id, layer)
```

### Per-chunk unload

```
arena.free(opaque_range); arena.free(water_range)
light_array.free_layer(layer)
chunk_table.unregister(coord)
```

## Failure modes

**Arena exhaustion** — `ChunkArena::alloc` returns `None` if the buffer can't grow. Behavior: log `warn!`, drop the upload, leave the previous LOD's allocation in place. The chunk stays visible at its last successful LOD. No crash, no flicker.

**Light array exhaustion** — same shape. New chunks fall back to the placeholder volume until a layer frees. Visible as a single dark frame, fixed on the next upload.

**Stale `chunk_id` in visibility list** — if a chunk unloads mid-frame between visibility build and pass encoding, the `flags` field reads "no LOD" and the entry is filtered. Slots are not reused until next frame, so no aliasing hazard.

**LOD fallback** — preserved from today: visibility builder picks `preferred_lod`, falls back to any populated LOD slot. Logic relocated from per-pass loops into the visibility builder.

**Wgpu feature gating** — `multi_draw_indexed_indirect` is behind `Features::MULTI_DRAW_INDIRECT` and not universally available in wgpu 23 (notably weaker on Metal). The design starts with a **CPU-iterated indirect path** (a `Vec<DrawEntry>` looped on the CPU, but with all per-chunk state work *already done* — no `BindGroup` creation, no `set_vertex_buffer` calls, just `draw_indexed`). This captures the bulk of the CPU win. True MDI is enabled where supported as a transparent follow-up.

**Screenshot path** — `render_to_view` reuses the same `(visibility_list, frame_resources)` shape, so existing screenshot tests keep working without per-pass duplication.

## Testing

The lighting work already established a screenshot-baseline harness (`test(screenshot): re-baseline …` commits). We piggy-back on that.

**Visual regression** — existing PNG goldens must match within SSIM tolerance for floating-point drift from the new shader binding layout. Run `--screenshot-and-exit` across every baselined scene before-and-after. If a scene drifts, either the new shader has a bug or the rework legitimately changed output and the baseline needs re-baking — both decisions are user-facing.

**Unit tests** for new components:

- `ChunkArena`: alloc/free/grow cycle; fragmentation under randomized alloc-free sequences; free-list correctness.
- `ChunkTable`: `ChunkId` stability across register/unregister; dirty-set coalescing produces one queue write per frame.
- `LightVolumeArray`: layer allocation stable across unload/reload cycles.
- `VisibilityList`: cull output bit-identical to today's per-pass test on randomized chunk inputs.

**Performance test** — the goal of the whole rework:

- New `bin/render_bench` that loads a fixed save, places the camera at a known position, runs N frames in `PresentMode::Immediate`, writes a CSV row per frame matching the existing `Profiler` format.
- Compare against a baseline CSV captured on `main` *before* the rework lands.
- **Target:** median CPU `render` span < 1 ms at the documented benchmark scene. Today's number gets captured as part of the baseline; the spec target updates if reality is materially different from the estimate.

**Hot-reload check** — shaders keep loading from `assets/shaders/`. A smoke test that boots the game and inspects the first frame's `last_draw_calls()` catches binding-layout mismatches early.

## Out of scope (explicit non-goals, restated)

- Bindless textures, descriptor indexing — would require newer wgpu features and changes throughout.
- GPU-driven culling (compute-shader frustum cull writing the indirect buffer) — possible follow-up after the CPU path lands.
- Temporal anti-aliasing, screen-space reflections — visual changes, not part of this rework.
- Async compute, multi-queue submission — wgpu 23 doesn't expose them cleanly.
- HDR tonemap changes — the composite pass is left as-is.

## Implementation order (informational; the writing-plans skill will produce the real plan)

1. `ChunkArena` standalone, with unit tests. No renderer integration yet.
2. `ChunkTable` standalone, with unit tests.
3. `LightVolumeArray` standalone, with backend detection.
4. Mesher: split opaque/water streams; add `chunk_id` to `Vertex`.
5. Shaders: storage-buffer `chunk_table` + light array sampling.
6. `VisibilityList` builder; replace per-pass loops one at a time (opaque first, then water, then reflection).
7. Eliminate mid-frame depth copy.
8. HUD persistent ring buffer.
9. MSAA configurable.
10. `bin/render_bench` + baseline capture.

Each step is independently testable and the screenshot harness catches regressions at every step.
