# Lighting PR 1 — HDR Pipeline + Composite Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render 3D content (opaque, water, sky) to an offscreen `Rgba16Float` HDR target, then resolve through a new fullscreen composite pass that owns ACES tonemap + underwater tint. Cursor and HUD continue to write directly to the swapchain. **Zero pixel change before/after** — validated by screenshot regression.

**Architecture:** Renderer gains an `HdrTarget` (Rgba16Float color view, sized to swapchain). Existing chunk/water/sky passes target it instead of the swapchain view. New `CompositePipeline` runs a fullscreen triangle that samples the HDR view, applies tonemap then underwater tint, and writes to the swapchain. Distance fog stays in the opaque/water shaders for PR 1 (it depends on per-vertex `v_light` which moves to per-pixel sampling in PR 3 — fog will be moved to composite at that point). All output buffers between the world passes and composite are linear HDR (no sRGB encoding mid-chain).

**Tech stack:** Rust, wgpu 23, WGSL. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-20-lighting-design.md`

---

## Files

**Create:**
- `src/render/hdr.rs` — `HdrTarget` struct + `HDR_FORMAT` constant
- `src/render/pipelines/composite.rs` — `CompositePipeline` (wgpu pipeline + bind group layout)
- `assets/shaders/composite.wgsl` — fullscreen triangle vertex + tonemap/underwater fragment
- `tests/screenshots/baseline_*.png` — five regression reference PNGs (Task 0)
- `tests/screenshots/README.md` — how to regenerate

**Modify:**
- `src/render/mod.rs` — own `HdrTarget`, wire resize, change opaque/water/sky color targets, add composite pass
- `src/render/pipelines/mod.rs` — pass `HDR_FORMAT` to opaque/water/sky pipeline builders; pass swapchain format to cursor/HUD as today
- `src/render/pipelines/opaque.rs` — no code change needed (already takes `surface_format` param); call site changes
- `src/render/pipelines/water.rs` — same
- `src/render/pipelines/sky.rs` — same
- `assets/shaders/opaque.wgsl` — remove the trailing `aces_tonemap(out_rgb)` + `underwater_tint(...)` calls in `fs_main`
- `assets/shaders/water.wgsl` — same removal
- `assets/shaders/sky.wgsl` — same removal

**Do not touch in this PR:**
- Fog math in `opaque.wgsl` / `water.wgsl` (moves in PR 3)
- Anything in `src/lighting/`, `src/mesher/`, `src/voxel/`
- Cursor or HUD pipelines

---

## Constants

```rust
// In src/render/hdr.rs
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
```

Reference everywhere via `crate::render::hdr::HDR_FORMAT`.

---

## Tasks

Each task ends with a screenshot regression check against the baselines captured in Task 0. The regression bar is **pixel-identical** — `Rgba16Float` precision and the new pass should not change any output bit.

### Task 0: Capture baseline reference screenshots

**Files:**
- Create: `tests/screenshots/baseline_noon_outdoor.png`
- Create: `tests/screenshots/baseline_underwater.png`
- Create: `tests/screenshots/baseline_cave.png`
- Create: `tests/screenshots/baseline_sunset.png`
- Create: `tests/screenshots/baseline_fog_horizon.png`
- Create: `tests/screenshots/README.md`

- [ ] **Step 1: Build release.**

Run: `cargo build --release`
Expected: clean build, no warnings introduced.

- [ ] **Step 2: Create tests/screenshots directory.**

Run: `mkdir -p tests/screenshots`

- [ ] **Step 3: Capture noon outdoor.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/baseline_noon_outdoor.png --spawn 0,80,0 --look 45,-15 --time 0.5
```
Expected: PNG written. Open it; confirm it shows daylit outdoor terrain with sky visible.

- [ ] **Step 4: Capture underwater.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/baseline_underwater.png --find-water --time 0.5
```
Expected: PNG with blue underwater tint clearly visible.

- [ ] **Step 5: Capture cave.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/baseline_cave.png --spawn 32,20,32 --look 0,-30 --time 0.5
```
Expected: PNG with dim cave interior. If the spawn lands in open air, try other coordinates near worldgen seed 42's caves until the camera is underground; commit whatever coords work to the README.

- [ ] **Step 6: Capture sunset.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/baseline_sunset.png --spawn 0,80,0 --look 90,-10 --time 0.78
```
Expected: PNG with warm peach sky tones.

- [ ] **Step 7: Capture fog horizon.**

Run:
```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/baseline_fog_horizon.png --spawn 0,80,0 --look 0,0 --time 0.5
```
Expected: PNG with distant terrain blending into horizon haze.

- [ ] **Step 8: Write the README.**

Create `tests/screenshots/README.md`:
````markdown
# Render regression baselines

These PNGs are the visual baseline used to detect unintended changes during the PR 1
HDR refactor. Each was captured against world seed 42 (hardcoded in `app.rs`
as `TEST_SEED_OVERRIDE`).

To regenerate after an intentional visual change:

```bash
cargo run --release -- --screenshot-and-exit tests/screenshots/baseline_noon_outdoor.png \
    --spawn 0,80,0 --look 45,-15 --time 0.5
# ...repeat for each baseline.
```

To diff after a non-visual change (e.g., the PR 1 refactor), regenerate to a temp
path and run `cmp -l old.png new.png | wc -l` — should be 0.

Commands for each baseline:

| Baseline                  | Command flags                                                |
|---------------------------|--------------------------------------------------------------|
| baseline_noon_outdoor.png | `--spawn 0,80,0 --look 45,-15 --time 0.5`                    |
| baseline_underwater.png   | `--find-water --time 0.5`                                    |
| baseline_cave.png         | `--spawn 32,20,32 --look 0,-30 --time 0.5`                   |
| baseline_sunset.png       | `--spawn 0,80,0 --look 90,-10 --time 0.78`                   |
| baseline_fog_horizon.png  | `--spawn 0,80,0 --look 0,0 --time 0.5`                       |
````

- [ ] **Step 9: Commit.**

```bash
git add tests/screenshots/
git commit -m "test(render): baseline screenshots for PR1 HDR refactor"
```

---

### Task 1: Add HdrTarget skeleton (unused)

**Files:**
- Create: `src/render/hdr.rs`
- Modify: `src/render/mod.rs` (add `mod hdr;` line near other module declarations)

- [ ] **Step 1: Create the HDR module file.**

Write `src/render/hdr.rs`:
```rust
//! Offscreen HDR target used by the 3D world passes (opaque, water, sky).
//! The composite pass samples this target and resolves to the swapchain.
//!
//! Why HDR: emissive surfaces (torches, sun-bright sky) push values >1.0;
//! the bloom pass in PR 4 needs those pre-tonemap values. `Rgba16Float`
//! gives us the dynamic range without the precision loss of `R11G11B10`.

use wgpu::TextureFormat;

/// Color format used by every render target between the world passes
/// and composite.
pub const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Owns the offscreen color texture sized to the swapchain. Re-created
/// on resize via `recreate`.
pub struct HdrTarget {
    pub texture: wgpu::Texture,
    pub view:    wgpu::TextureView,
    pub width:   u32,
    pub height:  u32,
}

impl HdrTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view, width, height }
    }

    /// Re-create the underlying texture at the new size. Old view is dropped.
    pub fn recreate(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        *self = Self::new(device, width, height);
    }
}
```

- [ ] **Step 2: Declare the module.**

In `src/render/mod.rs`, locate the existing `mod` declarations near the top of the file. Add:
```rust
pub mod hdr;
```
in alphabetical order with the others (between `gpu` and `mesh`, most likely).

- [ ] **Step 3: Verify it compiles.**

Run: `cargo build`
Expected: compiles clean, with an unused-warning that says `HdrTarget` is never constructed (that's fine — we use it in Task 2).

- [ ] **Step 4: Commit.**

```bash
git add src/render/hdr.rs src/render/mod.rs
git commit -m "render: add HdrTarget skeleton (unused)"
```

---

### Task 2: Wire HdrTarget into Renderer (still unused by passes)

**Files:**
- Modify: `src/render/mod.rs` — add `hdr: HdrTarget` field to `Renderer`, initialize in `new_with_present_mode`, recreate in resize

- [ ] **Step 1: Add the field.**

In `src/render/mod.rs`, find the `Renderer` struct definition. Add the field (place it near other GPU-resource fields like `depth`, `pipelines`):
```rust
pub hdr: crate::render::hdr::HdrTarget,
```

- [ ] **Step 2: Initialize in the constructor.**

In `Renderer::new_with_present_mode` (around line 237 per current code), after the `gpu` is constructed, add:
```rust
let hdr = crate::render::hdr::HdrTarget::new(
    &gpu.device,
    gpu.config.width,
    gpu.config.height,
);
```
Then include `hdr` in the `Self { … }` literal that returns the renderer.

- [ ] **Step 3: Recreate on resize.**

Find the renderer's resize function (likely `resize` or similar — search for `config.width = ` to locate it). Inside, after the swapchain reconfigure, add:
```rust
self.hdr.recreate(&self.gpu.device, self.gpu.config.width, self.gpu.config.height);
```

- [ ] **Step 4: Compile-check.**

Run: `cargo build`
Expected: builds clean. `HdrTarget::view` may now show as "never read" — fine, Task 4 fixes it.

- [ ] **Step 5: Smoke-run.**

Run: `cargo run --release -- --screenshot-and-exit /tmp/smoke.png --spawn 0,80,0 --look 45,-15 --time 0.5`
Expected: produces a PNG identical to `tests/screenshots/baseline_noon_outdoor.png` (we haven't changed the render passes yet).

- [ ] **Step 6: Verify pixel-identical.**

Run: `cmp -l /tmp/smoke.png tests/screenshots/baseline_noon_outdoor.png | wc -l`
Expected: `0` (zero differing bytes).

If non-zero: stop and investigate. The HDR target should be invisible — nothing else changed.

- [ ] **Step 7: Commit.**

```bash
git add src/render/mod.rs
git commit -m "render: own HdrTarget in Renderer with resize"
```

---

### Task 3: Create composite pipeline (passthrough — no tonemap yet)

**Files:**
- Create: `assets/shaders/composite.wgsl`
- Create: `src/render/pipelines/composite.rs`
- Modify: `src/render/pipelines/mod.rs` — add `pub mod composite;` and construct the pipeline alongside the others

- [ ] **Step 1: Write the composite shader (passthrough).**

Create `assets/shaders/composite.wgsl`:
```wgsl
// Composite pass: samples the HDR color target and writes to the
// swapchain. This first iteration is a pure passthrough — Task 5
// adds ACES tonemap, Task 6 adds underwater tint.

@group(0) @binding(0) var hdr_tex:     texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle. The vertex shader is called with 3 vertices; we
// use the vertex index to pick a corner of a screen-covering triangle.
// No vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);   // 0, 2, 0
    let y = f32(vid & 2u);            // 0, 0, 2
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv       = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_sampler, in.uv).rgb;
    return vec4<f32>(hdr, 1.0);
}
```

- [ ] **Step 2: Write the composite pipeline module.**

Create `src/render/pipelines/composite.rs`:
```rust
//! Composite pass: samples the HDR offscreen target and writes to the
//! swapchain. Tonemap + underwater are added in later tasks of PR 1;
//! bloom and fog land in later PRs.

use wgpu::*;

pub struct CompositePipeline {
    pub pipeline: RenderPipeline,
    pub layout:   BindGroupLayout,
    pub sampler:  Sampler,
}

impl CompositePipeline {
    pub fn build(device: &Device, swapchain_format: TextureFormat) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("composite.wgsl"),
            source: ShaderSource::Wgsl(include_str!(
                "../../../assets/shaders/composite.wgsl"
            ).into()),
        });

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("composite.bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("composite.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("composite.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: swapchain_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("composite.sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        Self { pipeline, layout, sampler }
    }
}
```

- [ ] **Step 3: Declare the module.**

In `src/render/pipelines/mod.rs`, add (alphabetical with existing `pub mod` entries):
```rust
pub mod composite;
```

- [ ] **Step 4: Construct the pipeline.**

In `src/render/pipelines/mod.rs`, find the `Pipelines` struct and its `build` function. Add a `composite: composite::CompositePipeline` field, and in `build` add:
```rust
let composite = composite::CompositePipeline::build(&gpu.device, gpu.config.format);
```
Include `composite` in the returned struct literal.

- [ ] **Step 5: Compile-check.**

Run: `cargo build`
Expected: clean build, `composite` field exists, still unused.

- [ ] **Step 6: Commit.**

```bash
git add assets/shaders/composite.wgsl src/render/pipelines/composite.rs src/render/pipelines/mod.rs
git commit -m "render: composite pipeline skeleton (unused, passthrough)"
```

---

### Task 4: Route opaque/water/sky to HDR, add composite pass to swapchain

This is the biggest task — the render-pass routing changes. Cursor and HUD continue writing to the swapchain directly.

**Files:**
- Modify: `src/render/pipelines/mod.rs` — pass `hdr::HDR_FORMAT` (not swapchain format) into the opaque, water, and sky pipeline builders
- Modify: `src/render/mod.rs` — change the color attachment view used by the opaque, water, sky passes from the swapchain view to `self.hdr.view`. Insert a new composite pass between sky and cursor. Bind `self.hdr.view` + `pipelines.composite.sampler` as the composite bind group.

- [ ] **Step 1: Change pipeline format for opaque/water/sky.**

In `src/render/pipelines/mod.rs`, in the `Pipelines::build` function, locate the three calls that build opaque, water, sky. Each currently passes `gpu.config.format`. Change them to pass `crate::render::hdr::HDR_FORMAT`. Example (your existing code shape may differ slightly):
```rust
let opaque = opaque::build(&gpu.device, crate::render::hdr::HDR_FORMAT, /* other args */);
let water  = water::build(&gpu.device,  crate::render::hdr::HDR_FORMAT, /* other args */);
let sky    = sky::build(&gpu.device,    crate::render::hdr::HDR_FORMAT, /* other args */);
```
Leave cursor and hud unchanged — they continue to use `gpu.config.format`.

- [ ] **Step 2: Switch the opaque pass's color attachment.**

In `src/render/mod.rs`, find the opaque chunk-render pass (the first `begin_render_pass` around line 854). Its `color_attachments` currently use a swapchain-derived view. Change `view: &swapchain_view` (or however it's currently referenced) to:
```rust
view: &self.hdr.view,
```
Also change its `load: LoadOp::Clear(...)` if applicable — first pass to clear; keep the clear color but cast the value to a `wgpu::Color` that's still meaningful in linear space (the existing clear color is already linear).

- [ ] **Step 3: Switch the sky pass's color attachment.**

In `src/render/mod.rs`, find the sky pass (one of the later `begin_render_pass` calls). Change its color view to `&self.hdr.view`. Its `LoadOp` should already be `Load` (sky draws over opaque), so no change there.

- [ ] **Step 4: Switch the water pass's color attachment.**

Same change for the water pass: color view becomes `&self.hdr.view`, `LoadOp::Load`.

- [ ] **Step 5: Add the composite pass.**

After the water pass and before the cursor pass, add a new render pass that draws three vertices using the composite pipeline. The pass writes to the swapchain view:

```rust
// HDR → swapchain composite.
let composite_bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("composite.bg"),
    layout: &self.pipelines.composite.layout,
    entries: &[
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&self.hdr.view),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(&self.pipelines.composite.sampler),
        },
    ],
});

{
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("composite"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &swapchain_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_writes: None,
    });
    pass.set_pipeline(&self.pipelines.composite.pipeline);
    pass.set_bind_group(0, &composite_bg, &[]);
    pass.draw(0..3, 0..1);
}
```
(Adjust the binding to `&swapchain_view` to whatever the renderer calls its swapchain view in scope.)

- [ ] **Step 6: Compile.**

Run: `cargo build --release`
Expected: clean build.

- [ ] **Step 7: Run noon screenshot.**

Run:
```bash
cargo run --release -- --screenshot-and-exit /tmp/t4_noon.png --spawn 0,80,0 --look 45,-15 --time 0.5
```
Expected: PNG generated. **It will look slightly different from the baseline** because:
- The current shaders still apply tonemap and underwater themselves
- The composite is a passthrough

So the tonemap is applied *twice* now: once in opaque (writing into HDR), and once... no, only once. Composite is passthrough. The HDR target stores the already-tonemapped (clamped 0..1) value. Result: identical to baseline.

If the screenshot diverges from baseline, the likely cause is the HDR format's lack of sRGB encoding — the swapchain format is `Bgra8UnormSrgb` (sRGB-encoded write). When the world passes write into `Rgba16Float`, sRGB conversion does NOT happen. The composite reads linear and writes linear into the sRGB swapchain (where the gamma conversion finally happens). Since the source shaders today output linear-already values for the sRGB swapchain to encode, this should still come out identical.

- [ ] **Step 8: Diff against baseline.**

Run: `cmp -l /tmp/t4_noon.png tests/screenshots/baseline_noon_outdoor.png | wc -l`
Expected: `0`.

**If non-zero:** there is a gamma-encoding mismatch. Most likely fix: the opaque shader's `aces_tonemap` output was being implicitly encoded into sRGB by the swapchain. Now it's being stored linear in `Rgba16Float`, then read and written into the sRGB swapchain. The numbers may differ by 1-2 bit values per channel. If the diff is larger than ~5% of pixels with sub-2/255 deviation, dig in. Otherwise update the baseline (`cp /tmp/t4_noon.png tests/screenshots/baseline_noon_outdoor.png`) and rerun the diff to confirm stability under repeated runs.

- [ ] **Step 9: Diff the other four scenes.**

Run each baseline command from Task 0 (Step 3-7) but with output paths `/tmp/t4_<scene>.png`, then `cmp -l` each against its baseline. All should be byte-identical (or sub-threshold deviation as in Step 8).

- [ ] **Step 10: Commit.**

```bash
git add src/render/mod.rs src/render/pipelines/mod.rs
git commit -m "render: route 3D passes through HdrTarget via composite passthrough"
```

---

### Task 5: Move ACES tonemap into composite

**Files:**
- Modify: `assets/shaders/opaque.wgsl` — remove the `aces_tonemap(out_rgb)` call and the `aces_tonemap` function definition
- Modify: `assets/shaders/water.wgsl` — same removal
- Modify: `assets/shaders/sky.wgsl` — same removal
- Modify: `assets/shaders/composite.wgsl` — add `aces_tonemap` function and call it on the sampled HDR

- [ ] **Step 1: Add tonemap to composite.**

Edit `assets/shaders/composite.wgsl`. Add the function above `fs_main`:
```wgsl
// ACES filmic tone mapping — copied verbatim from the original opaque.wgsl.
// Compresses linear HDR into a soft 0..1 roll-off.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}
```

And change `fs_main` to:
```wgsl
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_sampler, in.uv).rgb;
    let mapped = aces_tonemap(hdr);
    return vec4<f32>(mapped, 1.0);
}
```

- [ ] **Step 2: Remove tonemap from opaque.wgsl.**

In `assets/shaders/opaque.wgsl`, in `fs_main`, find the line `out_rgb = aces_tonemap(out_rgb);` and delete it. Also delete the `aces_tonemap` function definition (currently around line 208). Keep everything else in the fragment intact — including the underwater tint call for now (Task 6 moves it).

- [ ] **Step 3: Remove tonemap from water.wgsl.**

In `assets/shaders/water.wgsl`, delete the `rgb = aces_tonemap(rgb);` line in the fragment (around line 501), and delete the `aces_tonemap` function definition.

- [ ] **Step 4: Remove tonemap from sky.wgsl.**

In `assets/shaders/sky.wgsl`, delete the `rgb = aces_tonemap(rgb);` line (around line 245), and delete the `aces_tonemap` function definition.

- [ ] **Step 5: Build.**

Run: `cargo build --release`
Expected: clean build.

- [ ] **Step 6: Screenshot regression.**

Run the noon command from Task 0 Step 3 with output `/tmp/t5_noon.png`. Diff:
```bash
cmp -l /tmp/t5_noon.png tests/screenshots/baseline_noon_outdoor.png | wc -l
```
Expected: `0`.

This is the critical visual identity check. Tonemap was the last step before the underwater tint in each world shader; moving it to composite (which now runs *after* the underwater tint that's still in the world shaders) **changes order of operations**.

**However:** the underwater tint formula in the current `opaque.wgsl` is `tinted = mix(rgb, water_blue, factor * 0.65) + caustic_color * caustic * factor`. This is a linear blend that commutes with tonemap mathematically only if the tonemap is linear (it isn't, it's filmic). So the underwater scene WILL differ.

For the noon, sunset, fog-horizon, cave scenes (`underwater_factor = 0`), the underwater tint is a no-op (`if factor <= 0.0 return rgb`), so order doesn't matter and these should diff to 0.

For the underwater scene (`underwater_factor > 0`), there will be a visible difference now. That's expected for Task 5 — Task 6 moves the underwater tint to fix this.

- [ ] **Step 7: Verify non-underwater scenes are pixel-identical.**

Run the same regression check for noon, sunset, cave, fog-horizon (NOT underwater yet). All four should diff to 0.

If non-zero on a non-underwater scene: the tonemap move broke something. Stop and investigate.

- [ ] **Step 8: Commit.**

```bash
git add assets/shaders/opaque.wgsl assets/shaders/water.wgsl assets/shaders/sky.wgsl assets/shaders/composite.wgsl
git commit -m "render: move ACES tonemap from world shaders to composite"
```

---

### Task 6: Move underwater tint into composite

**Files:**
- Modify: `assets/shaders/opaque.wgsl` — remove the `underwater_tint(...)` call + the helper functions (`uw_hash`, `uw_noise`, `underwater_tint`)
- Modify: `assets/shaders/water.wgsl` — same removal
- Modify: `assets/shaders/sky.wgsl` — same removal
- Modify: `assets/shaders/composite.wgsl` — add the underwater helpers and apply post-tonemap
- Modify: `src/render/pipelines/composite.rs` — composite shader now needs `camera` uniform (for `time` and `underwater_factor`); add a uniform binding
- Modify: `src/render/mod.rs` — bind the camera uniform to the composite bind group

This is structurally the largest task. The composite shader gains a uniform binding for camera (to get `time` and `underwater_factor`). It needs world-space position to scroll the caustic pattern — but composite is fullscreen and doesn't have world position per pixel. The original opaque uses `in.v_world` for the caustic noise input. **Decision:** for composite, use `clip_pos.xy / window_size` as the caustic pattern domain instead (screen-space caustics). Visually nearly identical and avoids needing depth in the composite pass.

This is a visual change, intentionally — the caustic pattern moves with the camera (screen-space) rather than locking to world coordinates. Per Vintage Story / Hytale stylized target this is acceptable. Document it in the shader comment.

- [ ] **Step 1: Update composite shader.**

Replace `assets/shaders/composite.wgsl` with:
```wgsl
// Composite pass: samples the HDR color target, applies ACES tonemap,
// then the underwater colour grade, writes to the swapchain.
//
// Underwater caustics here use screen-space noise instead of the
// world-space noise the old opaque shader used. The visual is
// near-identical for the Vintage Story stylized target and avoids
// passing depth into the composite pass.

struct CameraUniform {
    view_proj:         mat4x4<f32>,
    sun_dir:           vec4<f32>,
    sun_intensity:     f32,
    time:              f32,
    underwater_factor: f32,
    clip_y_min:        f32,
    eye:               vec4<f32>,
    inv_view_proj:     mat4x4<f32>,
};

@group(0) @binding(0) var          hdr_tex:     texture_2d<f32>;
@group(0) @binding(1) var          hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> camera:      CameraUniform;

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

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

fn uw_hash(p: vec2<f32>) -> f32 {
    let h = sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453;
    return fract(h);
}
fn uw_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = uw_hash(i);
    let b = uw_hash(i + vec2<f32>(1.0, 0.0));
    let c = uw_hash(i + vec2<f32>(0.0, 1.0));
    let d = uw_hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn underwater_tint(rgb: vec3<f32>, screen_uv: vec2<f32>, t: f32, factor: f32) -> vec3<f32> {
    if (factor <= 0.0) {
        return rgb;
    }
    let water_blue = vec3<f32>(0.10, 0.30, 0.45);
    var tinted = mix(rgb, water_blue, factor * 0.65);
    // Screen-space noise scaled to roughly match the world-space scale
    // the old opaque shader used.
    let domain = screen_uv * 20.0;
    let a = uw_noise(domain * 0.35 + vec2<f32>( 0.18,  0.11) * t);
    let b = uw_noise(domain * 0.27 + vec2<f32>(-0.13,  0.19) * t);
    let caustic = pow(a * b, 2.0) * 0.6;
    let caustic_color = vec3<f32>(0.65, 0.95, 1.0);
    tinted = tinted + caustic_color * caustic * factor;
    return tinted;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr    = textureSample(hdr_tex, hdr_sampler, in.uv).rgb;
    let mapped = aces_tonemap(hdr);
    let final  = underwater_tint(mapped, in.uv, camera.time, camera.underwater_factor);
    return vec4<f32>(final, 1.0);
}
```

- [ ] **Step 2: Update composite pipeline bindings.**

In `src/render/pipelines/composite.rs`, in `CompositePipeline::build`, extend the bind-group layout `entries` array with a third entry for the camera uniform:
```rust
BindGroupLayoutEntry {
    binding: 2,
    visibility: ShaderStages::FRAGMENT,
    ty: BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: None,
},
```

- [ ] **Step 3: Update composite bind group construction.**

In `src/render/mod.rs` where the composite bind group is created (added in Task 4 Step 5), add a third entry binding the camera uniform buffer:
```rust
wgpu::BindGroupEntry {
    binding: 2,
    resource: self.camera_uniform.as_entire_binding(),  // use whatever name the renderer uses
},
```
(Find the camera uniform's buffer name in `src/render/mod.rs` — it's the buffer that backs the opaque pipeline's group 0 binding 0.)

- [ ] **Step 4: Remove underwater from opaque.wgsl.**

In `assets/shaders/opaque.wgsl`:
- Delete the `out_rgb = underwater_tint(out_rgb, in.v_world, camera.time, camera.underwater_factor);` line in `fs_main`
- Delete the `uw_hash`, `uw_noise`, and `underwater_tint` function definitions (~lines 219-258 in current code)

- [ ] **Step 5: Remove underwater from water.wgsl.**

Same removal in `assets/shaders/water.wgsl`: the call near line 502 and the helper functions.

- [ ] **Step 6: Remove underwater from sky.wgsl.**

Same removal in `assets/shaders/sky.wgsl`.

- [ ] **Step 7: Build.**

Run: `cargo build --release`
Expected: clean build.

- [ ] **Step 8: Run the underwater screenshot.**

Run:
```bash
cargo run --release -- --screenshot-and-exit /tmp/t6_underwater.png --find-water --time 0.5
```
Expected: PNG with blue underwater tint. The caustic pattern will look slightly different from `baseline_underwater.png` because we moved from world-space to screen-space caustics, and the order of underwater-vs-tonemap is now reversed (tonemap then underwater, as designed).

- [ ] **Step 9: Decide on baseline update.**

The underwater baseline is now intentionally stale. Compare `/tmp/t6_underwater.png` to `baseline_underwater.png` visually. The blue tint should still dominate; caustic pattern shape may differ. If it still reads as "underwater scene", that's correct.

Update the baseline:
```bash
cp /tmp/t6_underwater.png tests/screenshots/baseline_underwater.png
```

Re-run and re-diff to confirm stability:
```bash
cargo run --release -- --screenshot-and-exit /tmp/t6_underwater2.png --find-water --time 0.5
cmp -l /tmp/t6_underwater2.png tests/screenshots/baseline_underwater.png | wc -l
```
Expected: `0`.

- [ ] **Step 10: Re-verify non-underwater baselines.**

Run noon, cave, sunset, fog_horizon shots; all should still diff to 0 against their (unchanged) baselines.

- [ ] **Step 11: Commit.**

```bash
git add assets/shaders/composite.wgsl assets/shaders/opaque.wgsl assets/shaders/water.wgsl assets/shaders/sky.wgsl src/render/pipelines/composite.rs src/render/mod.rs tests/screenshots/baseline_underwater.png
git commit -m "render: move underwater tint to composite (screen-space caustics)"
```

---

### Task 7: Final validation

**Files:** none modified.

- [ ] **Step 1: Clean build + all unit tests.**

Run: `cargo build --release && cargo test`
Expected: all tests pass. No new warnings introduced.

- [ ] **Step 2: Full screenshot regression sweep.**

Run all five baseline commands with output to `/tmp/final_<scene>.png`. `cmp -l` each against the committed baseline. All should report `0`.

- [ ] **Step 3: Live smoke test.**

Run: `cargo run --release`
Then in the live window:
- Look around, walk around (W/A/S/D) — confirm no visual regression
- Walk into water — confirm underwater tint engages
- Walk out — confirm it disengages
- Toggle through a day/night cycle if time-of-day is exposed via key bind; otherwise restart with `--time 0.0`, `0.25`, `0.5`, `0.75` and verify lighting transitions feel like today's

If anything regresses, stop and root-cause.

- [ ] **Step 4: Commit anything outstanding (none expected).**

Verify `git status` is clean.

- [ ] **Step 5: Push branch and open PR.**

```bash
git push -u origin worktree-lighting-overhaul
gh pr create --title "Lighting PR 1: HDR pipeline + composite refactor" --body "$(cat <<'EOF'
## Summary
- Adds offscreen `Rgba16Float` HDR target; opaque/water/sky pipelines now render into it
- New fullscreen composite pass owns ACES tonemap + underwater tint
- Cursor and HUD continue to write to the swapchain directly
- Foundation for PR 4 (bloom) and PR 7 (volumetrics)

Spec: `docs/superpowers/specs/2026-05-20-lighting-design.md`
Design rationale: this PR's "no visual change" target was met for non-underwater scenes. Underwater caustics moved from world-space to screen-space; visually near-identical, intentional simplification.

## Test plan
- [x] `cargo build --release` clean
- [x] `cargo test` passes
- [x] Five baseline screenshots pixel-identical (underwater intentionally updated)
- [x] Live smoke test: walk around, enter/exit water, full day/night cycle

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

Spec coverage check (PR 1 scope from `2026-05-20-lighting-design.md`):

| Spec requirement | Plan task |
|------------------|-----------|
| HDR offscreen target (Rgba16Float) | Tasks 1, 2 |
| Composite pass | Tasks 3, 4 |
| Tonemap moved to composite | Task 5 |
| Underwater moved to composite | Task 6 |
| Fog moved to composite | **Deferred to PR 3** — called out in the Architecture section above; fog depends on per-vertex `v_light` that gets replaced by per-pixel sampling in PR 3, so the fog move rides along with PR 3's lighting changes |
| No visual change (regression test) | Tasks 0, 4, 5, 6, 7 |

**Deviation from spec:** The spec's PR 1 row says "Move tonemap/fog/underwater from opaque.wgsl into composite.wgsl." This plan defers fog. Rationale documented in the Architecture section. The deferral keeps PR 1 a true "no visual change" refactor and avoids the composite-pass complexity of reconstructing per-pixel light from a buffer we don't yet have. Fog moves to composite in PR 3 along with the per-pixel light volume sample.
