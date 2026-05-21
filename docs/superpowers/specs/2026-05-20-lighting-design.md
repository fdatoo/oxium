# Modern Lighting Overhaul — Design Spec

**Date:** 2026-05-20
**Status:** Partially superseded — PRs 1-4 merged; PRs 5-8 deferred until after the graph-engine rewrite ships.
**Supersedes:** lighting section of `2026-05-18-oxium-design.md`
**Partly superseded by:** [`2026-05-21-lighting-graph-engine-design.md`](2026-05-21-lighting-graph-engine-design.md) — the CPU-propagation portion (the "CPU light propagation" section and parts of "Module shape" / "Data model") is replaced by the graph-engine design. The GPU/visual portion (HDR, light_volume sampling, shadow map, bloom, ambient bounce, volumetrics, day/night colors) is unaffected and PRs 5-8 still apply once the engine lands.

## Summary

Replace oxium's per-vertex baked-light shading with a modern, stylized lighting pipeline targeting a Vintage Story / Hytale aesthetic: directional sun with single-cascade shadow mapping, smooth per-pixel light sampling from a 3D light volume, colored RGB block light, HDR + bloom, baked sky-channel ambient bounce, and volumetric atmospherics. The CPU side keeps the existing recompute-on-dirty BFS model but widens to four channels (sky + R + G + B). The GPU side gains an HDR pipeline, a shadow pass, a bloom chain, a volumetric pass, and a composite pass.

Primary hardware target: M4 Max-class GPU. Visual fidelity is the goal; quality scaling for weaker GPUs is a follow-on.

## Scope

**In scope:**

- 4-channel light volume (sky + R + G + B), 4 bits per channel, propagated by chunk-local BFS
- Colored emissive blocks (`BlockInfo::emission: [u8; 3]`)
- Per-pixel trilinear sampling of light volume in opaque fragment shader
- Wrap-diffuse directional sun with `sky_level`-gated occlusion
- Single-cascade sun shadow map (2048², heavy PCF) within an 80-block radius
- HDR offscreen pipeline (`Rgba16Float`) with composite pass
- 5-level Kawase-style bloom chain
- Volumetric atmospherics raymarch driven by the shadow map
- Sky-channel ambient-bounce term in BFS (caves and overhangs get a soft fill)
- Day/night sun + sky color curves (CPU)
- Save format extension for colored block light, backwards-compatible loading

**Out of scope (deferred to v2):**

- Block-light bounce / true GI (voxel cone tracing or SDF GI)
- Incremental BFS (recompute-on-dirty is retained)
- Auto-exposure / eye adaptation
- Screen-space reflections (water pass already does planar reflection)
- Multi-cascade shadow maps
- Specular highlights
- Per-block-light shadow casting
- Quality scaling enum / weaker-GPU configuration

## Architecture

### Module shape

```
src/lighting/
├── mod.rs              # public API: recompute_chunk, snapshot_face_boundaries
├── volume.rs           # ChannelVolume<u4>: sky + RGB per-voxel storage (NEW)
├── propagate.rs        # 4-channel BFS with ambient-bounce term (NEW; absorbs current mod.rs guts)
└── seam.rs             # cross-chunk boundary seeding (extracted)

src/render/
├── shadow.rs           # single-cascade sun shadow pass (NEW)
├── hdr.rs              # offscreen HDR target + bloom chain (NEW)
├── pipelines/
│   ├── shadow.wgsl     # depth-only sun pass (NEW)
│   ├── opaque.wgsl     # samples light volume + shadow map; outputs linear HDR (MODIFIED)
│   ├── bloom.wgsl      # downsample/upsample chain (NEW)
│   ├── volumetrics.wgsl # fullscreen raymarched fog/god-rays (NEW)
│   └── composite.wgsl  # HDR → fog → bloom → tonemap → underwater → SDR (NEW)
```

The threading model is unchanged. Chunk light recompute runs on the rayon pool. Results flow through the existing `Relit` channel. Shadow, bloom, volumetric, and composite are GPU-only and run every frame.

The public lighting API stays the same — callers in `voxel/world.rs` are not touched.

### Per-frame render schedule

```
1. shadow pass        → depth-only into shadow_map (one cascade)
2. opaque pass        → linear HDR into hdr_color, samples shadow_map + light_volume
3. water pass         → into hdr_color, as today
4. volumetrics pass   → fullscreen, samples shadow_map, adds scatter to hdr_color
5. bloom chain        → 5 downsample + 4 upsample levels on hdr_color
6. composite pass     → hdr_color + bloom → fog → tonemap → underwater → swapchain
```

## Data model

### Per-voxel storage

```rust
// src/voxel/chunk.rs (DenseChunk)
pub sky_light: Box<[u8; CHUNK_VOL]>,   // 4 bits used (unchanged)
pub block_rgb: Box<[u16; CHUNK_VOL]>,  // R<<8 | G<<4 | B, 4 bits each (NEW; replaces block_light)
```

Per-chunk memory grows from 96 KB (32 KB blocks + 32 KB sky + 32 KB block) to 128 KB (32 KB blocks + 32 KB sky + 64 KB block_rgb). At horizontal render radius 16, ~17k chunks loaded → in-RAM cost grows from ~1.6 GB to ~2.1 GB worst case. Acceptable on the M4 Max target.

**Why u16 not three separate u8 arrays:** BFS updates R/G/B in lockstep — one comparison cell per neighbor visit, three channels updated together. Avoids running three separate BFS passes.

### Block emission

```rust
// src/voxel/block.rs (BlockInfo)
pub emission: [u8; 3],   // R, G, B levels 0..15
```

Examples:
- Torch: `[13, 8, 4]` (warm white-orange)
- Glowstone: `[15, 14, 10]` (bright warm white)
- Future redstone torch: `[12, 0, 0]` (pure red)
- Default: `[0, 0, 0]`

### Propagation rules

| Channel | Per-step cost in air | Per-step cost in water | Blocked by opaque |
|---------|----------------------|------------------------|-------------------|
| Sky     | 1                    | 3                      | yes               |
| R/G/B   | 1                    | 3                      | yes               |

The column-drop for sky stays as today: each `(x, z)` column drops `15` from the world ceiling until it hits an opaque block. Non-opaque non-air cells (leaves, water) attenuate by 1 per cell during the drop.

### Ambient-bounce term

The BFS has one new parameter on the sky channel:

```rust
const AMBIENT_THRESHOLD: u8 = 8;   // source-cell sky level needed to leak
const AMBIENT_FLOOR: u8 = 2;       // minimum sky level imparted to neighbor

// In bfs_spread, when visiting a neighbor:
let prop = next.saturating_sub(cost.saturating_sub(1));
let ambient = if level >= AMBIENT_THRESHOLD { AMBIENT_FLOOR } else { 0 };
let final_val = prop.max(ambient);
```

Effect: cave mouths and the shadow side of cliffs get a soft fill instead of falling to pitch black one cell into shadow. Tunable; setting `AMBIENT_FLOOR = 0` disables the term entirely.

The bounce is **sky-channel only.** Block-light bounce is deferred to v2.

### GPU upload

Each chunk's light volume uploads as a 33³ `rgba8unorm` 3D texture:
- `R = block_red / 15`
- `G = block_green / 15`
- `B = block_blue / 15`
- `A = sky / 15`

The 33rd row in each axis borrows from the +X/+Y/+Z neighbor so trilinear sampling at the chunk's far edge sees valid neighbor values. Sampler is `linear` + `clamp_to_edge`.

Loaded chunks live in a `texture_3d_array<rgba8unorm>`, indexed by chunk coord through a small per-frame uniform table.

### Persistence

The on-disk paletted chunk format extends from sky + 1-channel block light to sky + 3-channel block light, two extra 4-bit fields per voxel. Old saves load with `G = B = 0`; the player sees today's monochrome warm-white torch light. World manifest version bumps. Forward-only — old binaries can't read new saves.

## CPU light propagation

### Pass structure (per chunk recompute)

```
1. Clear all channels (sky → 0, block_rgb → 0)
2. Sky pass:
   a. Column drop from chunk-above's bottom row
   b. Seed cross-chunk boundaries from ±X/±Z/-Y neighbors
   c. BFS spread with ambient-bounce term
3. RGB pass:
   a. Seed every emissive block with its (R, G, B) emission tuple
   b. Seed cross-chunk boundaries from all 6 neighbors
   c. BFS spread over u16-packed cells (three channels updated together)
```

### Cross-chunk seam handling

`seed_from_neighbors` keeps its shape — the per-channel max-merge generalizes from u8 to u16 cleanly. `snapshot_face_boundaries` grows from 2 bytes/cell (sky + block) to 5 bytes/cell (sky + R + G + B + reserved). The `Relit` cascade rule ("only re-mark neighbor as `light_dirty` if its facing boundary values actually changed") works unchanged with more bytes to compare.

### Cost

Per-chunk recompute (worst case, ~200k BFS visits over 32³ cells):
- Sky pass: ~80 µs
- RGB pass: ~150 µs
- Total: **~250 µs on the rayon pool** (vs ~150 µs today)

Negligible end-user impact; chunk relight already debounces one frame behind block edits.

## Surface lighting (opaque shader)

### Bindings

```wgsl
@group(0) @binding(0) var<uniform> camera: CameraUniform;  // extended

@group(1) @binding(0) var<uniform> chunk:         ChunkUniform;
@group(1) @binding(1) var          light_volume:  texture_3d<f32>;  // NEW
@group(1) @binding(2) var          light_sampler: sampler;          // NEW (linear, clamp)

@group(2) @binding(0) var atlas_tex:     texture_2d<f32>;  // unchanged
@group(2) @binding(1) var atlas_sampler: sampler;

@group(3) @binding(0) var shadow_map:     texture_depth_2d;          // NEW
@group(3) @binding(1) var shadow_sampler: sampler_comparison;        // NEW
```

### Camera uniform additions

```wgsl
struct CameraUniform {
    view_proj:         mat4x4<f32>,
    sun_view_proj:     mat4x4<f32>,   // NEW
    sun_dir:           vec4<f32>,
    sun_color:         vec4<f32>,     // NEW
    sky_color:         vec4<f32>,     // NEW
    sun_intensity:     f32,
    time:              f32,
    underwater_factor: f32,
    clip_y_min:        f32,
    eye:               vec4<f32>,
    inv_view_proj:     mat4x4<f32>,
};
```

### Vertex shader changes

The greedy mesher's vertex payload drops the baked light byte. The vertex shader still receives `normal_face` (face index, 0..5) and derives a `face_normal: vec3<f32>` to pass to the fragment shader. Bandwidth is unchanged (the byte slot was already padded); the meaning shifts from "baked light here" to nothing — light is sampled per pixel now.

### Fragment composition

```wgsl
// Sample the light volume air-side of the surface.
let sample_world = in.v_world + in.face_normal * 0.5;
let sample_local = sample_world - chunk.origin.xyz;
let uvw          = (sample_local + 0.5) / 33.0;
let l            = textureSampleLevel(light_volume, light_sampler, uvw, 0.0);
let block_rgb    = l.rgb;   // 0..1
let sky_level    = l.a;     // 0..1

// Wrap-diffuse directional sun with sky-channel occlusion.
let n_dot_l = max(dot(in.face_normal, -camera.sun_dir.xyz), 0.0);
let wrap    = (n_dot_l + 0.4) / 1.4;            // soft, ambient-dominant
let shadow  = sample_shadow(in.v_world);        // 0..1, see Shadow Map section
let sun_lit = wrap * shadow * sky_level;

// Sky ambient — tinted, scaled by sky exposure.
let sky_amb = camera.sky_color.rgb * sky_level * 0.35;

// Colored block light — direct from the volume.
let block_lit = block_rgb;

// Combine.
let direct = camera.sun_color.rgb * sun_lit * camera.sun_intensity;
let lit    = direct + sky_amb + block_lit;
let ao     = mix(0.45, 1.0, in.v_ao);
let shade  = ao * (lit + MIN_SHADE);            // small baseline, e.g. vec3(0.02)
let lit_rgb = base_rgb * shade;

// Distance fog and post effects move to composite.wgsl.
return vec4<f32>(lit_rgb, 1.0);   // linear HDR
```

### Design choices

**`sky_level` gates the sun.** A cell whose sky channel is 0 (deep underground) gets zero sun contribution even if it's geometrically sun-facing. BFS-propagated sky_light *is* the long-range shadow occlusion term; the shadow map handles near-range dynamic occlusion. The product is what makes one low-res cascade sufficient.

**Wrap diffuse `(N·L + 0.4) / 1.4`** instead of `max(N·L, 0)`. Lifts glancing angles into soft mid-tones, matching the Vintage Story / Hytale ambient-dominant look. Constant is tunable.

**`face_mul` is removed.** The current hardcoded top=1.0 / bottom=0.55 / sides=0.80 is superseded by the real `dot(N, sun_dir)` — which produces the same intuition automatically (tops bright at noon, sides medium, bottoms dim).

**`+ face_normal * 0.5` air-side sampling.** Sampling the light volume exactly at the surface would hit the opaque interior's (zero) cell. Half-block normal offset puts the sample point in the air-side cell, mirroring the mesher's "sample one cell outside the face" rule.

### What survives unchanged

- Atlas sampling + `fract(uv)` tile repetition
- Per-block `block_variation_hash` jitter
- `biome_tint_shift` low-frequency color shift
- Leaf alpha-test discard

### What moves to composite

- ACES tonemap
- Distance fog (`fog_start`, `fog_end`, sky/cave fog mix)
- Underwater tint
- Horizon color computation

## Shadow map

Single cascade, 2048², `Depth32Float`. Heavy PCF for soft Vintage Story-style penumbras.

### Specifics

```
Resolution:     2048 × 2048, Depth32Float
Fitting:        AABB around camera frustum, clamped to sphere radius 80 blocks
                centered on camera, snapped to texel boundaries (kills shimmer)
Projection:     orthographic, sun_dir from existing TimeOfDay
Update cadence: every frame
Bias:           slope-scaled depth bias + small constant (0.0005)
PCF kernel:     3×3 with linear filtering (9 taps, 5×5 effective area)
```

The 80-block radius is well inside the 16-chunk horizontal render distance. Beyond that, fragments fall back to `sky_level`-only occlusion. Atmospheric fog starts to dominate at the same distance anyway.

### Shadow pass

A depth-only pass that runs before opaque each frame. Reuses chunk vertex buffers via `shadow.wgsl` — a position-only vertex shader writing depth through `sun_view_proj`, no fragment shader.

Frustum cull for the shadow pass uses an inflated bound (`shadow_radius + 2`) to include casters behind the camera. ~150 chunks at radius 80.

### Sample function

```wgsl
fn sample_shadow(world_pos: vec3<f32>) -> f32 {
    let light_clip = camera.sun_view_proj * vec4<f32>(world_pos, 1.0);
    let light_ndc  = light_clip.xyz / light_clip.w;
    if (any(abs(light_ndc.xy) > vec2<f32>(1.0))) {
        return 1.0;   // outside shadow map → unshadowed
    }
    let uv             = light_ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    let receiver_depth = light_ndc.z - SHADOW_BIAS;

    var sum = 0.0;
    let texel = 1.0 / 2048.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
            let sd = textureSampleLevel(shadow_map, shadow_sampler,
                                         uv + offset, 0.0).r;
            sum = sum + select(0.0, 1.0, sd > receiver_depth);
        }
    }
    return sum / 9.0;
}
```

### How shadow combines with sky_level

`sun_lit = wrap × shadow × sky_level`. Both must agree for the sun to land:
- `sky_level` (BFS, slow-changing): "is this cell exposed to the sky?" Handles long-range geometric occlusion.
- `shadow` (shadow map, every frame): "is this fragment dynamically occluded right now?" Handles per-frame near-range shadows.

### Cost on M4 Max

- Shadow pass: ~0.5 ms (depth-only, ~150 chunks)
- 3×3 PCF in opaque: 9 taps per fragment, ~0.2-0.3 ms total at 1440p
- Total budget: under 1 ms

## HDR, bloom, atmospherics, composite

### HDR target

`hdr_color: Rgba16Float`, full-resolution. Alpha doubles as "this fragment is sky" mask (sky writes 0, opaque writes 1).

### Bloom chain (Kawase-style)

```
Pass 0:           hdr_color → bloom_0 (½ res), threshold + downsample
                  (smoothstep around threshold=1.0; only HDR-bright bloom)
Passes 1-4:       bloom_n → bloom_{n+1} (¼, ⅛, 1/16, 1/32), 13-tap box filter
Passes 5-8:       upsample with tent filter, accumulate
Output:           bloom_0 at ½ res, blended in composite
```

### Volumetric raymarch

Fullscreen pass between water and bloom:

```
Per pixel:
  Reconstruct world-space ray from depth + inv_view_proj
  16-step march from camera to fragment depth
  At each step:
    sample shadow_map at world position
    if lit: accumulate density × phase(L·V) × extinction
  Multiply by camera.sun_color, add to hdr_color
```

Uniform density, Henyey-Greenstein phase function with `g = 0.6`. ~0.4 ms at 1440p on M4 Max.

### Composite pass

```wgsl
@fragment
fn fs_composite(...) -> @location(0) vec4<f32> {
    var color = textureSample(hdr_color, ...).rgb;
    let bloom = textureSample(bloom_result, ...).rgb;

    // 1. Distance fog (pre-bloom so distant lights still bloom through fog).
    color = mix(color, fog_color, fog_t);

    // 2. Bloom (pre-tonemap so HDR drives bloom proportionally).
    color = color + bloom * BLOOM_STRENGTH;   // default 0.06

    // 3. ACES tonemap.
    color = aces_tonemap(color);

    // 4. Underwater tint (post-tonemap so the tint stays a pure pull).
    color = underwater_tint(color, world_pos, time, underwater_factor);

    return vec4<f32>(color, 1.0);
}
```

Pass order is deliberate: fog → bloom → tonemap → underwater.

### Day/night sun & sky colors

A CPU function `time_of_day_colors(time) -> (sun_color, sky_color, sun_intensity)` produces all three with smooth curves, fed to the camera uniform once per frame.

| Time | sun_color | sky_color | sun_intensity |
|------|-----------|-----------|---------------|
| Sunrise/sunset | warm peach (1.0, 0.75, 0.55) | matches sun (peach) | low |
| Noon | neutral-warm white (1.0, 0.96, 0.90) | cool blue (0.55, 0.70, 0.95) | full |
| Night | irrelevant | cool indigo (0.10, 0.12, 0.20) | 0 |

Same shape as the existing `sky.wgsl` `horizon_color` math, lifted to CPU so the values can drive surface lighting too.

### Cost budget on M4 Max (1440p)

| Pass | Cost |
|------|------|
| Shadow | 0.5 ms |
| Opaque + water + sky | ~2-3 ms (unchanged) |
| Volumetric | 0.4 ms |
| Bloom | 0.6 ms |
| Composite | 0.1 ms |
| **Post overhead** | **~1.6 ms** |

## PR sequence

| # | PR | Scope | Visual diff | Risk |
|---|----|-------|-------------|------|
| 1 | HDR pipeline + composite refactor | Move tonemap/fog/underwater from opaque.wgsl into composite.wgsl. Switch render target to `Rgba16Float`. | None (regression test only). | Low |
| 2 | Colored block light | Widen `BlockInfo::emission` to `[u8; 3]`. `block_rgb: Box<[u16]>`. BFS over u16. Save format bump. | Torches in v1 colors become warm; mixes compose. | Medium |
| 3 | Per-pixel light volume sampling | Upload light volume as 33³ `rgba8unorm` 3D texture array. Drop per-vertex light bake. Add wrap diffuse. Remove `face_mul`. | Smooth gradients across faces. Soft shaded sides. | Medium-high |
| 4 | Bloom + HDR | Bloom chain on `hdr_color`. Composite blends. | Emissives glow. Sunsets read bright. | Low |
| 5 | Sun shadow map | `shadow.rs` + `shadow.wgsl`. PCF in opaque. Texel snap, slope bias. | Trees cast shadows. Overhangs darken walls. | High |
| 6 | Sky ambient-bounce | Sky-channel ambient floor in BFS. Re-relight loaded chunks. | Caves stop pitch-black. Overhangs get soft fill. | Low |
| 7 | Volumetric atmospherics | Fullscreen raymarch sampling shadow map. | God rays. Atmospheric depth. | Medium |
| 8 | Day/night sun + sky color | `time_of_day_colors` CPU function. New uniform fields. | Warm sunsets, cool nights, full day cycle. | Low |

PRs 1-5 are the must-have v1 core. PRs 6-8 are the polish layer. Each PR is independently shippable and produces a visual diff worth posting.

## Testing strategy

| Module | Tests |
|--------|-------|
| `lighting/propagate.rs` | Today's tests pass unchanged. Adds: `red_torch_does_not_emit_green`, `mixed_torches_compose`, `ambient_bounce_softens_overhang`, `cross_chunk_colored_light_continues`. |
| `render/shadow.rs` | Unit test for texel-snap math (sub-½-texel camera moves produce identical projections). |
| Headless screenshot harness (existing `--screenshot-and-exit`) | New reference PNGs per PR: `shadow_under_overhang`, `shadow_moves_with_sun`, `noon_outdoor`, `sunset_outdoor`, `night_with_torches`, `underwater_at_noon`, `bloom_disabled_vs_default`, `god_rays_through_canopy`, `red_green_torch_room`. |
| CI | PRs touching `render/`, `lighting/`, `mesher/` run the screenshot harness; >1-2% pixel diff flags for manual review. |
| Manual smoke | Each high-risk PR (3, 5, 7) gets a 5-minute manual playthrough at noon, sunrise, sunset, and night, plus underground. |

## Risks and mitigations

**Tuning iterations on PRs 3, 5, 7.** Wrap-diffuse constant, shadow bias, volumetric density, bloom threshold, ambient floor — each has a "this looks wrong, fiddle" phase that's hard to estimate. Mitigation: 1-2 day buffer per high-risk PR. Document final constants with visual-effect rationale in code.

**Save format migration (PR 2).** Old saves load with `G = B = 0` (equivalent to today's monochrome). New saves can't be read by pre-PR-2 binaries. Bump world manifest version. No destructive migration; forward-only.

**M4 Max-specific tuning.** Design targets one machine. Quality-scaling enum is deferred until a second target exists. Constants that would scale: bloom chain depth, shadow map size + PCF kernel, volumetric march steps. All collected behind named constants for future extraction.

**Bulk edits → BFS hot path.** Recompute-on-dirty at ~250 µs/chunk is fine for per-click edits. If future work adds explosions / world-mods, incremental BFS becomes mandatory. The `propagate.rs` structure is designed so an incremental implementation can drop in without touching callers.

**Shadow + sky_level interaction.** The two combine multiplicatively; if either is wrong, sun looks wrong. Mitigation: each ships in its own PR with its own screenshot tests so failures are isolatable.

## Open questions

None. Every decision has a concrete answer.
