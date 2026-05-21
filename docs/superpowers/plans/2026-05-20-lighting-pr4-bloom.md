# Lighting PR 4 — Bloom + HDR Glow

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5-level downsample / 4-level upsample bloom chain on the HDR target. Pre-tonemap HDR values above a soft threshold bleed into a halo, additively blended into the composite. Emissive surfaces (torches, sun, glowstone) glow; the rest of the frame is untouched.

**Architecture:** A new `BloomChain` owns five `Rgba16Float` textures sized ½, ¼, ⅛, 1/16, 1/32 of the swapchain — one per mip level of the bloom pyramid. Three new pipelines target those textures:

1. **`bloom_threshold`** — samples the HDR target, applies a smoothstep cutoff around `BLOOM_THRESHOLD = 1.0`, writes into `bloom[0]` (½ res). This is the first pass and isolates HDR-bright pixels.
2. **`bloom_downsample`** — 13-tap box filter (the "dual-filter" pattern from the COD Siggraph 2014 bloom talk; reduces firefly artefacts vs a naive 4-tap). Runs four times, each reading mip `n` and writing mip `n+1`.
3. **`bloom_upsample`** — 3×3 tent filter. Runs four times reading mip `n+1` and additively blending (`SrcAlpha: One, DstAlpha: One`) into mip `n`. The blend means each upsample step accumulates onto the destination's existing downsample result, producing the "glow spread" look.

After the nine passes finish, `bloom[0]` holds the full bloom result at ½ res. The composite pass gains a fourth binding for that view and adds `bloom * BLOOM_STRENGTH` to the tonemap input.

The HDR target's bind group layout still uses `filterable: false` to stay compatible with the existing composite sampler. The bloom passes use a *new* linear sampler with `filterable: true` — `Rgba16Float` is filterable in wgpu without any device feature flag (it's `Rgba32Float` that needs `FLOAT32_FILTERABLE`).

This PR is **intentional visual change** for scenes with HDR-bright pixels (noon sun, sunset sky, emissive blocks). Non-bright scenes (cave, underwater interior, fog horizon at noon) should be pixel-near-identical because the threshold gates the entire chain.

**Tech Stack:** Rust, wgpu 23, WGSL. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-20-lighting-design.md` (HDR, bloom, atmospherics, composite section).
**Previous plans:** PR 1 (HDR refactor), PR 2 (RGB block light), PR 3 (per-pixel sampling) are all merged to main.

---

## Files

**Create:**
- `src/render/bloom.rs` — `BloomChain` struct that owns the 5 mip textures + per-mip views + linear sampler. Resize handling.
- `src/render/pipelines/bloom.rs` — three `RenderPipeline`s (`threshold`, `downsample`, `upsample`) + their shared bind-group layout.
- `assets/shaders/bloom.wgsl` — fullscreen vertex shader + three fragment entry points (`fs_threshold`, `fs_downsample`, `fs_upsample`).

**Modify:**
- `src/render/mod.rs` — own `bloom: BloomChain` and `bloom_pipes: BloomPipelines`. Recreate the chain in `resize`. Encode bloom between the water pass and the composite pass. Composite bind group gains a fourth entry pointing at `bloom.mip(0).view`.
- `src/render/pipelines/mod.rs` — `pub mod bloom;`.
- `src/render/pipelines/composite.rs` — extend the bind-group layout with binding 3 (bloom texture, filterable) and binding 4 (bloom sampler, filtering). Note: the existing HDR sampler at binding 1 stays non-filtering — only the bloom sampler is new.
- `assets/shaders/composite.wgsl` — read the bloom texture at the same UV, apply `color = color + bloom * BLOOM_STRENGTH` *before* tonemap so HDR values drive bloom proportionally (per spec composite-order section).
- `tests/screenshots/baseline_noon_outdoor.png`, `tests/screenshots/baseline_sunset.png` — re-captured at the end of the PR (bright-scene baselines move).

**Do not touch in this PR:**
- `src/lighting/`, `src/mesher/`, `src/voxel/` — bloom is pure post.
- Cursor, HUD, sky, opaque, water pipelines.
- The HDR target itself (`src/render/hdr.rs`) — its layout / format / bind group are reused as-is.
- Shadow / volumetrics / day-night colors (PRs 5–8).

---

## Constants

```rust
// src/render/bloom.rs
pub const BLOOM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const BLOOM_MIP_COUNT: u32 = 5;          // ½, ¼, ⅛, 1/16, 1/32
pub const BLOOM_MIN_DIM:   u32 = 4;          // floor for the smallest mip so we never request a 0×0 texture
```

```wgsl
// assets/shaders/bloom.wgsl
const BLOOM_THRESHOLD: f32 = 1.0;
const BLOOM_KNEE:      f32 = 0.5;            // smoothstep falloff half-width around the threshold
```

```wgsl
// assets/shaders/composite.wgsl
const BLOOM_STRENGTH: f32 = 0.06;
```

---

## Tasks

The plan moves in a "skeleton → connect → verify" arc so each merge boundary leaves the build green and the renderer working. Visual change lands in Task 6.

### Task 0: Snapshot the pre-PR4 screenshots

The five baselines committed under `tests/screenshots/` were captured at the end of PR 1 and may have drifted slightly through PR 2 / PR 3 (PR 3 was explicitly an intentional visual change for colored lighting and wrap diffuse). This task captures the *current main* output as the pre-bloom reference so we can detect any unintended drift during Tasks 1–5 before the deliberate change in Task 6.

**Files:**
- Create: `tests/screenshots/pre_pr4_noon_outdoor.png`
- Create: `tests/screenshots/pre_pr4_sunset.png`
- Create: `tests/screenshots/pre_pr4_cave.png`
- Create: `tests/screenshots/pre_pr4_underwater.png`
- Create: `tests/screenshots/pre_pr4_fog_horizon.png`

- [ ] **Step 1: Confirm the worktree is on a clean `main`-based branch.**

Run:
```bash
git status
```
Expected: working tree clean. If not, stash before starting.

- [ ] **Step 2: Release build.**

Run: `cargo build --release`
Expected: clean build.

- [ ] **Step 3: Capture noon outdoor.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/pre_pr4_noon_outdoor.png --spawn 0,80,0 --look 45,-15 --time 0.5
```
Expected: PNG written, daylit outdoor terrain visible.

- [ ] **Step 4: Capture sunset.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/pre_pr4_sunset.png --spawn 0,80,0 --look 90,-10 --time 0.78
```
Expected: PNG with warm peach sky tones.

- [ ] **Step 5: Capture cave.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/pre_pr4_cave.png --spawn 32,20,32 --look 0,-30 --time 0.5
```
Expected: dim cave interior.

- [ ] **Step 6: Capture underwater.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/pre_pr4_underwater.png --find-water --time 0.5
```
Expected: blue underwater tint dominates.

- [ ] **Step 7: Capture fog horizon.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/pre_pr4_fog_horizon.png --spawn 0,80,0 --look 0,0 --time 0.5
```
Expected: distant terrain blending into haze.

- [ ] **Step 8: Commit the snapshots.**

```bash
git add tests/screenshots/pre_pr4_*.png
git commit -m "test(render): pre-PR4 screenshot snapshots for bloom-drift detection"
```

These files are temporary — they are deleted in Task 7 once the new baselines are captured.

---

### Task 1: `BloomChain` skeleton (unused)

**Files:**
- Create: `src/render/bloom.rs`
- Modify: `src/render/mod.rs` — declare `pub mod bloom;` near `pub mod hdr;`

- [ ] **Step 1: Write `src/render/bloom.rs`.**

```rust
//! Bloom mip chain: five `Rgba16Float` textures sized ½, ¼, ⅛, 1/16, 1/32
//! of the swapchain. Used by the bloom post pass (PR 4 of the lighting
//! overhaul) to spread HDR-bright pixels into a soft halo.
//!
//! All mips share one linear-filtering sampler (`Rgba16Float` is
//! filterable in wgpu without a device feature). Each mip carries its
//! own `TextureView` because the down/upsample passes bind a specific
//! level as both read source and write target.

use wgpu::TextureFormat;

pub const BLOOM_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const BLOOM_MIP_COUNT: u32 = 5;
pub const BLOOM_MIN_DIM: u32 = 4;

/// One mip of the bloom chain — its texture, the view used to bind it
/// as a write target, and the cached dimensions. The view doubles as
/// the read-source view in down/upsample passes; mips are written as
/// whole-texture render targets, so no level-of-detail subview is
/// needed.
pub struct BloomMip {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

/// Owned five-mip bloom chain + linear sampler. Recreated on resize via
/// `recreate`.
pub struct BloomChain {
    pub mips: [BloomMip; BLOOM_MIP_COUNT as usize],
    pub sampler: wgpu::Sampler,
}

impl BloomChain {
    pub fn new(device: &wgpu::Device, swapchain_w: u32, swapchain_h: u32) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let mips = std::array::from_fn(|i| make_mip(device, swapchain_w, swapchain_h, i as u32));
        Self { mips, sampler }
    }

    /// Re-create every mip at the new swapchain size. Sampler is reused.
    pub fn recreate(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        for (i, mip) in self.mips.iter_mut().enumerate() {
            *mip = make_mip(device, w, h, i as u32);
        }
    }
}

fn make_mip(device: &wgpu::Device, w: u32, h: u32, level: u32) -> BloomMip {
    // Mip 0 is ½ res; mip 1 is ¼; etc. The +1 in the shift compensates.
    let divisor = 1u32 << (level + 1);
    let width = (w / divisor).max(BLOOM_MIN_DIM);
    let height = (h / divisor).max(BLOOM_MIN_DIM);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("bloom-mip-{level}")),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: BLOOM_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    BloomMip { texture, view, width, height }
}
```

- [ ] **Step 2: Declare the module.**

In `src/render/mod.rs`, the module declaration block near the top of the file currently reads:
```rust
pub mod atlas;
pub mod camera;
pub mod font;
pub mod gpu;
pub mod hdr;
pub mod hud;
pub mod light_volume;
pub mod mesh;
pub mod pipelines;
pub mod screenshot;
```
Insert `pub mod bloom;` between `atlas` and `camera` (alphabetical order):
```rust
pub mod atlas;
pub mod bloom;
pub mod camera;
```

- [ ] **Step 3: Compile.**

Run: `cargo build`
Expected: clean build. `BloomChain` and `BloomMip` will be reported "never constructed" — fine, Task 4 uses them.

- [ ] **Step 4: Commit.**

```bash
git add src/render/bloom.rs src/render/mod.rs
git commit -m "render: add BloomChain skeleton (unused)"
```

---

### Task 2: Bloom WGSL shader (three fragment entry points)

**Files:**
- Create: `assets/shaders/bloom.wgsl`

- [ ] **Step 1: Write the shader.**

Create `assets/shaders/bloom.wgsl`:
```wgsl
// Bloom pass — three fragment entry points fed by a shared fullscreen
// vertex shader. The pipelines in `src/render/pipelines/bloom.rs` pick
// which entry point to use.
//
//   fs_threshold  — pass 0; HDR → bloom[0]. Applies a smoothstep
//                   cutoff so only HDR-bright pixels feed the chain.
//   fs_downsample — passes 1..4; bloom[n] → bloom[n+1]. 13-tap box
//                   filter (the "dual-filter" pattern from the COD
//                   Siggraph 2014 talk).
//   fs_upsample   — passes 5..8; bloom[n+1] → bloom[n] with additive
//                   blend configured at the pipeline level. 3×3 tent.

@group(0) @binding(0) var src_tex:     texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv       = vec2<f32>(x, y);
    return out;
}

const BLOOM_THRESHOLD: f32 = 1.0;
const BLOOM_KNEE:      f32 = 0.5;

// Smoothstep around `threshold` — pixels far above the threshold pass
// through unchanged, pixels at the threshold attenuate to zero. The
// `knee` controls the soft-falloff width.
fn threshold_curve(color: vec3<f32>) -> vec3<f32> {
    let brightness = max(color.r, max(color.g, color.b));
    let soft       = clamp(brightness - BLOOM_THRESHOLD + BLOOM_KNEE, 0.0, 2.0 * BLOOM_KNEE);
    let soft_q     = (soft * soft) / (4.0 * BLOOM_KNEE + 0.0001);
    let mult       = max(soft_q, brightness - BLOOM_THRESHOLD) / max(brightness, 0.0001);
    return color * mult;
}

@fragment
fn fs_threshold(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, src_sampler, in.uv).rgb;
    return vec4<f32>(threshold_curve(c), 1.0);
}

// 13-tap downsample (Jorge Jimenez / Activision 2014). One center +
// four "inner box" 2×2 averages + four corner averages. Reduces
// fireflies vs a 5-tap or 9-tap.
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(src_tex));
    let t = vec2<f32>(1.0) / tex_size;
    let uv = in.uv;

    let a = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-2.0,  2.0)).rgb;
    let b = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 0.0,  2.0)).rgb;
    let c = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 2.0,  2.0)).rgb;
    let d = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-2.0,  0.0)).rgb;
    let e = textureSample(src_tex, src_sampler, uv                              ).rgb;
    let f = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 2.0,  0.0)).rgb;
    let g = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-2.0, -2.0)).rgb;
    let h = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 0.0, -2.0)).rgb;
    let i = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 2.0, -2.0)).rgb;
    let j = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0,  1.0)).rgb;
    let k = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0,  1.0)).rgb;
    let l = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0, -1.0)).rgb;
    let m = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0, -1.0)).rgb;

    // Weighted average: the inner 2×2 (j,k,l,m) contributes 0.5, the
    // four outer 2×2 box-averages contribute 0.125 each. Sums to 1.0.
    var color = (j + k + l + m) * 0.125;
    color = color + (a + b + d + e) * 0.03125;
    color = color + (b + c + e + f) * 0.03125;
    color = color + (d + e + g + h) * 0.03125;
    color = color + (e + f + h + i) * 0.03125;
    return vec4<f32>(color, 1.0);
}

// 3×3 tent upsample. Output blends additively with the destination via
// pipeline blend state, so the value here is the bloom contribution to
// add, not the final color.
@fragment
fn fs_upsample(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(src_tex));
    let t = vec2<f32>(1.0) / tex_size;
    let uv = in.uv;

    var color = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0,  1.0)).rgb * 1.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 0.0,  1.0)).rgb * 2.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0,  1.0)).rgb * 1.0;

    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0,  0.0)).rgb * 2.0;
    color = color + textureSample(src_tex, src_sampler, uv                              ).rgb * 4.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0,  0.0)).rgb * 2.0;

    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0, -1.0)).rgb * 1.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 0.0, -1.0)).rgb * 2.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0, -1.0)).rgb * 1.0;

    // 1+2+1+2+4+2+1+2+1 = 16
    color = color / 16.0;
    return vec4<f32>(color, 1.0);
}
```

- [ ] **Step 2: Confirm the shader is found at compile time.**

Nothing imports it yet; the pipeline module in Task 3 uses `include_str!`. For now just verify the file is readable:
```bash
ls -la assets/shaders/bloom.wgsl
```
Expected: file exists, non-empty.

- [ ] **Step 3: Commit.**

```bash
git add assets/shaders/bloom.wgsl
git commit -m "render: bloom WGSL (threshold + downsample + upsample, unused)"
```

---

### Task 3: Bloom pipelines module

**Files:**
- Create: `src/render/pipelines/bloom.rs`
- Modify: `src/render/pipelines/mod.rs` — add `pub mod bloom;`

- [ ] **Step 1: Write `src/render/pipelines/bloom.rs`.**

```rust
//! Bloom pipelines — one per fragment entry point in `bloom.wgsl`.
//!
//! Threshold + downsample use a single-target color attachment with no
//! blending. Upsample uses an additive blend so each upsample pass
//! accumulates onto whatever the downsample chain wrote to the
//! destination mip — the "blur spread" look.
//!
//! All three share one bind-group layout (a 2D texture + linear
//! sampler) and one pipeline layout, so callers can swap pipelines
//! without rebuilding bind groups.

const SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/shaders/bloom.wgsl"
));

pub struct BloomPipelines {
    pub bgl: wgpu::BindGroupLayout,
    pub threshold:  wgpu::RenderPipeline,
    pub downsample: wgpu::RenderPipeline,
    pub upsample:   wgpu::RenderPipeline,
}

pub fn build(device: &wgpu::Device, bloom_format: wgpu::TextureFormat) -> BloomPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bloom-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bloom-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bloom-layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    // Helper: build a pipeline targeting `bloom_format` with a given
    // fragment entry point and blend state.
    let make_pipeline = |label: &str, entry: &str, blend: Option<wgpu::BlendState>| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: bloom_format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    };

    let threshold  = make_pipeline("bloom-threshold",  "fs_threshold",  None);
    let downsample = make_pipeline("bloom-downsample", "fs_downsample", None);
    // Upsample: additive blend so we accumulate onto the destination
    // mip's existing content. `SrcAlpha` lets future work tweak per-
    // mip weights via alpha; we write alpha = 1.0 in the shader, so
    // the current behavior is one-to-one additive.
    let upsample_blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation:  wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation:  wgpu::BlendOperation::Add,
        },
    };
    let upsample = make_pipeline("bloom-upsample", "fs_upsample", Some(upsample_blend));

    BloomPipelines { bgl, threshold, downsample, upsample }
}
```

- [ ] **Step 2: Declare the module.**

Edit `src/render/pipelines/mod.rs`. Current contents:
```rust
pub mod composite;
pub mod cursor;
pub mod hud;
pub mod opaque;
pub mod sky;
pub mod water;
```
Add `pub mod bloom;` at the top (alphabetical):
```rust
pub mod bloom;
pub mod composite;
pub mod cursor;
pub mod hud;
pub mod opaque;
pub mod sky;
pub mod water;
```

- [ ] **Step 3: Compile.**

Run: `cargo build`
Expected: clean build with an unused-warning on `BloomPipelines` (Task 4 uses it).

- [ ] **Step 4: Commit.**

```bash
git add src/render/pipelines/bloom.rs src/render/pipelines/mod.rs
git commit -m "render: bloom pipelines (threshold + downsample + upsample, unused)"
```

---

### Task 4: Wire BloomChain into Renderer + encode the 9 passes

This is the structural change — `BloomChain` and `BloomPipelines` get owned by `Renderer`, and a new `encode_bloom_pass` method runs the threshold + four downsamples + four upsamples after the water pass and before composite. **The composite pass still ignores bloom in this task** (Task 5 changes that), so the visual output should remain unchanged.

**Files:**
- Modify: `src/render/mod.rs`

- [ ] **Step 1: Import the new modules and add fields.**

At the top of `src/render/mod.rs`, near the other `use crate::render::pipelines::...` imports, add:
```rust
use crate::render::bloom::BloomChain;
use crate::render::pipelines::bloom::{build as build_bloom, BloomPipelines};
```

In the `Renderer` struct (around line 109), add two fields. A reasonable spot is right after the existing `composite_pipe: CompositePipeline,` line:
```rust
/// Bloom mip chain (5 levels, ½..1/32 swapchain). Recreated on resize.
pub bloom: BloomChain,
/// Threshold / downsample / upsample bloom pipelines. Shared bind-
/// group layout; instances of `BloomChain::sampler` + per-mip view
/// fill the layout per draw.
bloom_pipes: BloomPipelines,
```

- [ ] **Step 2: Construct them in `new_with_present_mode`.**

In `Renderer::new_with_present_mode`, after the line that creates `composite_pipe`:
```rust
let composite_pipe = build_composite(&gpu.device, gpu.surface_cfg.format);
```
add:
```rust
let bloom = BloomChain::new(
    &gpu.device,
    gpu.surface_cfg.width,
    gpu.surface_cfg.height,
);
let bloom_pipes = build_bloom(&gpu.device, crate::render::bloom::BLOOM_FORMAT);
```

Then in the `Self { ... }` constructor literal that returns from `new_with_present_mode`, add `bloom,` and `bloom_pipes,` alongside `composite_pipe,`.

- [ ] **Step 3: Recreate the chain on resize.**

In `Renderer::resize` (around line 656), after the existing `self.hdr.recreate(&self.gpu.device, w, h);` line, add:
```rust
self.bloom.recreate(&self.gpu.device, w, h);
```

- [ ] **Step 4: Add `encode_bloom_pass`.**

Add the following method to the `impl Renderer` block, ideally placed right above the existing `fn encode_composite_pass` definition (around line 1125):

```rust
/// Build the bloom mip chain by sampling `self.hdr` through the
/// threshold + downsample + upsample passes. Mutates `self.bloom`'s
/// textures in place; the final result lives at `self.bloom.mips[0]`
/// for the composite pass to read.
///
/// Bind groups are created per pass because each pass reads a
/// different source view. wgpu requires the bind group's layout to
/// match the pipeline's layout, so we reuse `self.bloom_pipes.bgl`
/// for every binding.
fn encode_bloom_pass(&self, enc: &mut wgpu::CommandEncoder) {
    // Helper: a one-shot fullscreen pass with a single color target
    // and one bind group.
    let one_pass = |label: &str,
                    pipeline: &wgpu::RenderPipeline,
                    bg: &wgpu::BindGroup,
                    target: &wgpu::TextureView,
                    load: wgpu::LoadOp<wgpu::Color>| {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    };

    // Pass 0: HDR → bloom[0]. Threshold + downsample in one shader.
    let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bloom-threshold-bg"),
        layout: &self.bloom_pipes.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.hdr.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.bloom.sampler),
            },
        ],
    });
    one_pass(
        "bloom-threshold",
        &self.bloom_pipes.threshold,
        &bg,
        &self.bloom.mips[0].view,
        wgpu::LoadOp::Clear(wgpu::Color::BLACK),
    );

    // Passes 1..4: bloom[n] → bloom[n+1] via the 13-tap downsample.
    for n in 0..(crate::render::bloom::BLOOM_MIP_COUNT as usize - 1) {
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bloom-downsample-{n}-bg")),
            layout: &self.bloom_pipes.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.bloom.mips[n].view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.bloom.sampler),
                },
            ],
        });
        one_pass(
            "bloom-downsample",
            &self.bloom_pipes.downsample,
            &bg,
            &self.bloom.mips[n + 1].view,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
        );
    }

    // Passes 5..8: bloom[n+1] → bloom[n] additive (3×3 tent). Walks
    // from the smallest mip outward, accumulating onto the destination
    // mip's existing downsample content.
    for n in (0..(crate::render::bloom::BLOOM_MIP_COUNT as usize - 1)).rev() {
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bloom-upsample-{n}-bg")),
            layout: &self.bloom_pipes.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.bloom.mips[n + 1].view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.bloom.sampler),
                },
            ],
        });
        // Upsample uses `LoadOp::Load` — destination mip already
        // holds the downsample result; we accumulate via additive
        // blend (configured in the pipeline).
        one_pass(
            "bloom-upsample",
            &self.bloom_pipes.upsample,
            &bg,
            &self.bloom.mips[n].view,
            wgpu::LoadOp::Load,
        );
    }
}
```

- [ ] **Step 5: Call `encode_bloom_pass` from `render`.**

In the `render` method (around line 1048), the current sequence is:
```rust
self.encode_opaque_pass(
    &mut enc,
    &self.msaa_color_view,
    &self.hdr.view,
    &visible_main,
);
// Composite: read HDR, write to the swapchain. Passthrough
// today; Tasks 5/6 add tonemap + underwater tint.
self.encode_composite_pass(&mut enc, &view);
```

Insert the bloom call between them:
```rust
self.encode_opaque_pass(
    &mut enc,
    &self.msaa_color_view,
    &self.hdr.view,
    &visible_main,
);
// Bloom chain: builds 5 mips on the HDR target. Result lives in
// self.bloom.mips[0]; composite reads it in Task 5.
self.encode_bloom_pass(&mut enc);
self.encode_composite_pass(&mut enc, &view);
```

- [ ] **Step 6: Compile.**

Run: `cargo build --release`
Expected: clean build.

- [ ] **Step 7: Smoke run — confirm bloom passes run without breaking the frame.**

Run:
```bash
cargo run --release -- --screenshot-and-exit /tmp/t4_noon.png --spawn 0,80,0 --look 45,-15 --time 0.5
```
Expected: PNG produced. **It should be byte-identical to `tests/screenshots/pre_pr4_noon_outdoor.png`** — composite still ignores bloom, so the user-visible frame is unchanged.

- [ ] **Step 8: Verify pixel-identical.**

Run:
```bash
cmp -l /tmp/t4_noon.png tests/screenshots/pre_pr4_noon_outdoor.png | wc -l
```
Expected: `0`.

If non-zero: bloom is leaking into composite somehow. Check that `encode_bloom_pass` only writes to bloom mips and not to the HDR target. The HDR view is *read* during the threshold pass, never written.

- [ ] **Step 9: Sweep the other four scenes too.**

Run each pre-PR4 baseline command but writing to `/tmp/t4_<scene>.png`, then `cmp -l` against `pre_pr4_<scene>.png`. All should diff to 0.

- [ ] **Step 10: Commit.**

```bash
git add src/render/mod.rs
git commit -m "render: build bloom mip chain each frame (still unused by composite)"
```

---

### Task 5: Composite reads the bloom + blends additively

**Files:**
- Modify: `src/render/pipelines/composite.rs` — extend the bind-group layout with bindings 3 (bloom texture) and 4 (bloom sampler, filtering)
- Modify: `assets/shaders/composite.wgsl` — add the bloom bindings + the `color + bloom * BLOOM_STRENGTH` pre-tonemap blend
- Modify: `src/render/mod.rs` — extend the composite bind group with two more entries

- [ ] **Step 1: Extend the composite bind-group layout.**

In `src/render/pipelines/composite.rs`, in the `entries: &[...]` array of the `device.create_bind_group_layout` call, add two new entries after the existing `binding: 2` (camera uniform):
```rust
wgpu::BindGroupLayoutEntry {
    binding: 3,
    visibility: wgpu::ShaderStages::FRAGMENT,
    ty: wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    },
    count: None,
},
wgpu::BindGroupLayoutEntry {
    binding: 4,
    visibility: wgpu::ShaderStages::FRAGMENT,
    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    count: None,
},
```

- [ ] **Step 2: Extend the composite shader.**

In `assets/shaders/composite.wgsl`, after the existing `@group(0) @binding(2) var<uniform> camera: CameraUniform;` line, add:
```wgsl
@group(0) @binding(3) var          bloom_tex:     texture_2d<f32>;
@group(0) @binding(4) var          bloom_sampler: sampler;

const BLOOM_STRENGTH: f32 = 0.06;
```

Then change `fs_main` to:
```wgsl
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var color = textureSample(hdr_tex,   hdr_sampler,   in.uv).rgb;
    let bloom = textureSample(bloom_tex, bloom_sampler, in.uv).rgb;

    // Bloom is additive in linear HDR, applied *before* the tonemap so
    // HDR-bright pixels drive bloom proportionally (per spec composite-
    // order: fog → bloom → tonemap → underwater).
    color = color + bloom * BLOOM_STRENGTH;

    let mapped  = aces_tonemap(color);
    let out_rgb = underwater_tint(mapped, in.uv, camera.time, camera.underwater_factor);
    return vec4<f32>(out_rgb, 1.0);
}
```

(The previous `let hdr = ...; let mapped = aces_tonemap(hdr);` lines are replaced by the four lines above.)

- [ ] **Step 3: Extend the composite bind group construction.**

In `src/render/mod.rs`, in `encode_composite_pass` (around line 1130), the current bind group is:
```rust
let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("composite-bg"),
    layout: &self.composite_pipe.bgl,
    entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.hdr.view) },
        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.composite_pipe.sampler) },
        wgpu::BindGroupEntry { binding: 2, resource: self.camera_buf.as_entire_binding() },
    ],
});
```

Extend the `entries` array with two more entries:
```rust
wgpu::BindGroupEntry {
    binding: 3,
    resource: wgpu::BindingResource::TextureView(&self.bloom.mips[0].view),
},
wgpu::BindGroupEntry {
    binding: 4,
    resource: wgpu::BindingResource::Sampler(&self.bloom.sampler),
},
```

- [ ] **Step 4: Compile.**

Run: `cargo build --release`
Expected: clean build.

- [ ] **Step 5: Capture a bloom-on noon screenshot.**

Run:
```bash
cargo run --release -- --screenshot-and-exit /tmp/t5_noon.png --spawn 0,80,0 --look 45,-15 --time 0.5
```
Expected: PNG produced. **This should now look subtly different from `pre_pr4_noon_outdoor.png`** — bright sky pixels and any HDR-bright surface should have a soft halo.

- [ ] **Step 6: Eyeball the diff.**

Open `/tmp/t5_noon.png` and `tests/screenshots/pre_pr4_noon_outdoor.png` side by side. The bloom should:
- Visibly soften / brighten the noon sky's brightest band
- Have **no visible effect** in non-HDR regions (terrain near the camera, mid-tone foliage)
- Not introduce blocky or ringed artefacts

Bloom that's too strong: tune `BLOOM_STRENGTH` down (e.g. 0.04). Too weak / invisible: tune up (e.g. 0.10). Default 0.06 is the spec's starting value. If the threshold curve clips too aggressively (sky barely glows), lower `BLOOM_THRESHOLD` in `bloom.wgsl` (0.8 is a reasonable next step).

Tuning is **expected** here. Spend at most ~15 minutes; document the chosen values in code comments before committing.

- [ ] **Step 7: Underwater scene smoke.**

Run:
```bash
cargo run --release -- --screenshot-and-exit /tmp/t5_underwater.png --find-water --time 0.5
```
Expected: underwater scene; bloom should be minimal (most pixels are below threshold) — output should look very close to `pre_pr4_underwater.png`.

Visual check, not pixel-identity — the screen-space caustic pattern noise interacts with the bloom blur very slightly.

- [ ] **Step 8: Cave scene smoke.**

Run the cave command. The cave has no emissives and no sky; expect output near-identical to `pre_pr4_cave.png`. If a noticeable glow appears, something is wrong in the threshold curve (cave brightness shouldn't exceed 1.0 in any channel given PR3's lighting).

- [ ] **Step 9: Commit.**

```bash
git add assets/shaders/composite.wgsl src/render/pipelines/composite.rs src/render/mod.rs
git commit -m "render: composite blends bloom additively pre-tonemap"
```

---

### Task 6: Final tuning + capture new baselines

This task is the visual-quality gate. Run the live app, walk around, place torches, look at the sun and at sunset, and confirm bloom feels right. Then re-capture the bright-scene baselines.

**Files:**
- Modify: `tests/screenshots/baseline_noon_outdoor.png` — re-captured (bloom changes it)
- Modify: `tests/screenshots/baseline_sunset.png` — re-captured (bloom changes it)
- Modify: `tests/screenshots/README.md` — update the description to mention bloom

(`baseline_cave.png`, `baseline_underwater.png`, `baseline_fog_horizon.png` should remain unchanged — verify rather than rebake.)

- [ ] **Step 1: Live smoke session.**

Run: `cargo run --release`

In the live window:
1. Look at the sun (orient toward the noon sun direction) — confirm a soft halo, not a hard edge.
2. Place a torch in an enclosed space (default keybind is left-click with a torch in hand if torches are in the hotbar) — confirm the torch produces a halo whose color matches its emission tint.
3. Look at the horizon at `--time 0.78` (re-launch with this flag) — confirm sunset sky bloom feels warm and natural.
4. Walk into water — confirm bloom doesn't visibly fight the underwater grade.
5. Walk into a dark cave — confirm bloom contributes essentially nothing (no halos around dim torches or ambient light).

If anything feels off, return to Task 5 Step 6 and re-tune `BLOOM_THRESHOLD`, `BLOOM_KNEE`, and `BLOOM_STRENGTH`. Commit the tuned values before continuing.

- [ ] **Step 2: Re-capture the bright-scene baselines.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/baseline_noon_outdoor.png --spawn 0,80,0 --look 45,-15 --time 0.5
cargo run --release -- --screenshot-and-exit tests/screenshots/baseline_sunset.png       --spawn 0,80,0 --look 90,-10 --time 0.78
```

- [ ] **Step 3: Re-verify non-bright scenes against the existing baselines.**

```bash
cargo run --release -- --screenshot-and-exit /tmp/cave.png       --spawn 32,20,32 --look 0,-30 --time 0.5
cargo run --release -- --screenshot-and-exit /tmp/underwater.png --find-water --time 0.5
cargo run --release -- --screenshot-and-exit /tmp/fog.png        --spawn 0,80,0 --look 0,0 --time 0.5

cmp -l /tmp/cave.png       tests/screenshots/baseline_cave.png        | wc -l
cmp -l /tmp/underwater.png tests/screenshots/baseline_underwater.png  | wc -l
cmp -l /tmp/fog.png        tests/screenshots/baseline_fog_horizon.png | wc -l
```

Expected: all three diff to `0`. If one is non-zero, investigate — the threshold curve may be triggering on borderline-bright pixels and producing a visible halo where there shouldn't be one. Tune `BLOOM_THRESHOLD` up if so.

- [ ] **Step 4: Update the README.**

Append a section to `tests/screenshots/README.md` (don't replace existing content):

````markdown

## PR 4 (Bloom) baseline notes

The `baseline_noon_outdoor.png` and `baseline_sunset.png` were re-captured during
PR 4 to include the bloom glow on HDR-bright sky pixels. Cave / underwater /
fog-horizon baselines were verified pixel-identical (bloom threshold gates them).

Bloom tuning constants (in code):
- `BLOOM_THRESHOLD` and `BLOOM_KNEE` in `assets/shaders/bloom.wgsl`
- `BLOOM_STRENGTH` in `assets/shaders/composite.wgsl`
````

- [ ] **Step 5: Commit.**

```bash
git add tests/screenshots/baseline_noon_outdoor.png tests/screenshots/baseline_sunset.png tests/screenshots/README.md
git commit -m "test(render): re-baseline bright scenes after bloom"
```

---

### Task 7: Cleanup + final validation + PR

**Files:**
- Delete: `tests/screenshots/pre_pr4_*.png` — drift-detection snapshots; not needed once PR4 merges

- [ ] **Step 1: Remove the drift snapshots.**

Run:
```bash
git rm tests/screenshots/pre_pr4_*.png
```
Expected: five files deleted.

- [ ] **Step 2: Clean build + all unit tests.**

Run: `cargo build --release && cargo test`
Expected: all tests pass; no new warnings.

- [ ] **Step 3: Full screenshot regression sweep.**

```bash
cargo run --release -- --screenshot-and-exit /tmp/final_noon.png       --spawn 0,80,0 --look 45,-15 --time 0.5
cargo run --release -- --screenshot-and-exit /tmp/final_sunset.png     --spawn 0,80,0 --look 90,-10 --time 0.78
cargo run --release -- --screenshot-and-exit /tmp/final_cave.png       --spawn 32,20,32 --look 0,-30 --time 0.5
cargo run --release -- --screenshot-and-exit /tmp/final_underwater.png --find-water --time 0.5
cargo run --release -- --screenshot-and-exit /tmp/final_fog.png        --spawn 0,80,0 --look 0,0 --time 0.5

for s in noon sunset cave underwater fog; do
    base=baseline_${s}_outdoor.png
    case $s in
        cave) base=baseline_cave.png ;;
        underwater) base=baseline_underwater.png ;;
        fog) base=baseline_fog_horizon.png ;;
        sunset) base=baseline_sunset.png ;;
        noon) base=baseline_noon_outdoor.png ;;
    esac
    diff=$(cmp -l /tmp/final_${s}.png tests/screenshots/$base 2>/dev/null | wc -l)
    echo "$s: $diff differing bytes"
done
```

Expected: every line prints `0 differing bytes`. If any non-zero, regenerate that baseline (the chunk-load order is deterministic so a freshly-captured PNG should reproduce). If it still differs, stop and investigate — the bloom chain may be non-deterministic (it shouldn't be).

- [ ] **Step 4: Live smoke walkthrough.**

Run: `cargo run --release`
- Walk around a sunlit area for ~30s — visual is stable, no flicker, no perf cliff
- Toggle through `--time 0.0`, `0.25`, `0.5`, `0.75` by restart — sunrise/noon/sunset/night all look right
- Stand near a torch — halo matches expectation
- Enter water — bloom + underwater grade compose cleanly

Document the bloom tuning constants used in this PR's commit message body (Step 6).

- [ ] **Step 5: Verify git status is clean.**

Run: `git status`
Expected: clean working tree.

- [ ] **Step 6: Push branch and open PR.**

```bash
git push -u origin worktree-lighting-pr4
gh pr create --title "Lighting PR 4: bloom + HDR glow" --body "$(cat <<'EOF'
## Summary
- New `BloomChain` owning 5 `Rgba16Float` mip textures (½..1/32 swapchain)
- New `bloom.wgsl` with threshold + 13-tap downsample + 3×3 tent upsample
- Composite pass blends bloom additively pre-tonemap at `BLOOM_STRENGTH = 0.06`
- Cave / underwater / fog-horizon scenes verified pixel-identical (threshold gates them)
- Noon / sunset baselines re-captured to include the glow

Spec: `docs/superpowers/specs/2026-05-20-lighting-design.md`

Tuning constants (final):
- `BLOOM_THRESHOLD`, `BLOOM_KNEE` in `assets/shaders/bloom.wgsl`
- `BLOOM_STRENGTH` in `assets/shaders/composite.wgsl`

## Test plan
- [x] `cargo build --release` clean
- [x] `cargo test` passes
- [x] Five-scene screenshot sweep pixel-identical against committed baselines
- [x] Live smoke: sunlit area, torch glow, sunset, underwater, cave

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

**Spec coverage check** (PR 4 row of `2026-05-20-lighting-design.md` + the HDR/bloom/composite section):

| Spec requirement | Plan task |
|------------------|-----------|
| 5-level mip chain on `hdr_color` (½, ¼, ⅛, 1/16, 1/32) | Task 1 (`BloomChain` with `BLOOM_MIP_COUNT = 5`) |
| Pass 0: threshold + downsample | Task 2 (`fs_threshold`), Task 3 (pipeline), Task 4 (first pass in `encode_bloom_pass`) |
| Passes 1-4: 13-tap box filter downsample | Task 2 (`fs_downsample`), Task 4 (loop) |
| Passes 5-8: tent upsample with accumulation | Task 2 (`fs_upsample`), Task 3 (additive blend), Task 4 (reverse-order loop with `LoadOp::Load`) |
| Bloom blended in composite | Task 5 |
| Composite order: bloom → tonemap (HDR-bright drives bloom proportionally) | Task 5 (the new `fs_main` adds bloom before `aces_tonemap`) |
| Threshold ≈ 1.0 with smoothstep around it | Task 2 (`BLOOM_THRESHOLD = 1.0`, `BLOOM_KNEE = 0.5`, `threshold_curve` smoothstep) |
| Default strength 0.06 | Task 5 (`BLOOM_STRENGTH = 0.06`) |
| No visual change for non-HDR scenes | Tasks 0, 5 (Step 7-8), 6 (Step 3) — regression check against `pre_pr4_*` then `baseline_*` for cave / underwater / fog |
| Visual change for emissives + bright sky | Task 5 (Step 6), Task 6 (Step 1 + Step 2 re-baseline) |
| `Rgba16Float` precision throughout | Task 1 (`BLOOM_FORMAT`), Task 3 (pipeline target format) |
| Cost budget ~0.6 ms on M4 Max | Not measured in this plan — acceptable per spec "design targets one machine; quality scaling deferred." Smoke test in Task 7 catches a perf cliff (>5 ms regression would be visible as stutter at uncapped FPS). |

**Deviations / risks:**
- The bloom pipeline uses a `Filtering` sampler (linear) — the existing composite HDR sampler is `NonFiltering`. These are two separate samplers and don't conflict; `Rgba16Float` is filterable in wgpu without enabling any device feature. Verified by reading `src/render/gpu.rs` (only `required_features: Features::empty()`).
- The 13-tap downsample comes from the COD Siggraph 2014 bloom talk, not strictly "Kawase" (the spec's word). The two produce visually equivalent results for our use case; the 13-tap is firefly-resistant out of the box. Documented in `bloom.wgsl`.
- Tuning (Task 5 Step 6 + Task 6 Step 1) is an unbounded-time activity inherent to bloom. Plan accepts ~15 min of fiddling per session and documents the chosen values in code.
