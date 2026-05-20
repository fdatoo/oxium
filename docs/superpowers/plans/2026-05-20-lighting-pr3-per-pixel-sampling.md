# Lighting PR 3 — Per-Pixel Light Volume Sampling

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mesher's per-vertex baked light byte with per-pixel sampling from a 33³ `Rgba8Unorm` 3D light texture owned per chunk. The opaque + water shaders pick up smooth gradients across face interiors and start consuming colored block light directly (RGB) instead of the legacy scalar brightness. Wrap-diffuse directional sun replaces the hardcoded `face_mul` table.

**Architecture:** A new per-chunk `ChunkLightVolume` owns a 33³ Rgba8Unorm 3D texture (R/G/B = block_red/green/blue, A = sky_light, each scaled to [0,1]). The 33rd row in each axis is borrowed from the +X/+Y/+Z neighbor so trilinear sampling at the chunk's far face sees valid neighbor values. The chunk bind group layout gains two new bindings (texture + sampler). Shaders sample the volume at `world_pos + face_normal * 0.5` (the air-side neighbor cell) to avoid self-occlusion. The vertex `light` byte stays in the `Vertex` struct for layout stability but the shader ignores it.

This is the first PR with **intentional visual change** — smooth gradients across face interiors, soft wrap-diffuse on glancing angles, colored torches actually rendering as colored. Screenshot baselines are expected to shift; the regression check is "does it look like the spec's stylized target".

**Tech Stack:** Rust, wgpu 23, WGSL. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-20-lighting-design.md` (Surface lighting section).
**Previous plans:** PR 1 (HDR refactor) and PR 2 (RGB infrastructure) are both merged to main.

---

## Files

**Create:**
- `src/render/light_volume.rs` — `ChunkLightVolume` struct + per-chunk 3D texture management

**Modify:**
- `src/render/camera.rs` — extend `make_chunk_bind_group_layout` from 1 binding to 3 (chunk uniform + light_volume texture + light sampler)
- `src/render/mod.rs` — own `chunk_lights: HashMap<ChunkCoord, ChunkLightVolume>`; build bind group per-draw including light volume; resize / unload accordingly. Camera uniform write site picks up new `sun_color` / `sky_color` fields
- `src/render/camera.rs` — `CameraUniform` adds `sun_color: [f32; 4]` and `sky_color: [f32; 4]` (initially hardcoded; PR 8 makes them time-of-day driven)
- `src/render/mod.rs` — `light_sampler: Sampler` shared resource for the light bind group entry
- `src/voxel/chunk.rs` — add `pub fn build_light_volume_blob(dense: &DenseChunk, neighbors: &Neighbors) -> Box<[u8; 33*33*33*4]>` helper that produces the 33³ Rgba8 blob with 1-cell borrow from neighbors
- `src/jobs/mod.rs` — `JobResult::Meshed` gains optional `light_volume: Option<Box<[u8; 33*33*33*4]>>` field (populated by LOD-0 mesher only); mesher worker constructs it inline; relight worker emits it via a new `light_volume` field on `JobResult::Relit`
- `src/ecs/systems/mesh_upload.rs` — Meshed and Relit handlers push the light volume to the renderer
- `src/mesher/greedy.rs` — leaves the vertex `light` byte writes as-is (value becomes unused); no other change
- `assets/shaders/opaque.wgsl` — sample light volume, wrap diffuse, drop `face_mul`, compose `block_lit + sky_amb + sun_lit`
- `assets/shaders/water.wgsl` — same sampling for water surface lighting

**Do not touch in this PR:**
- Vertex layout / `Vertex` struct in `src/mesher/mod.rs` (keep 16-byte size for pipeline stability)
- `src/render/pipelines/opaque.rs` — pipeline layout flows through `chunk_bgl` so layout changes propagate automatically
- HUD, cursor, sky pipelines
- `src/persistence/` — light volume is a derived GPU artifact, not persisted
- Composite / HDR (PR 1 already in)

---

## Constants

```rust
// In src/render/light_volume.rs
pub const LIGHT_VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const LIGHT_VOLUME_DIM: u32 = 33;          // chunk dim + 1 border row
pub const LIGHT_VOLUME_BYTES: usize = 33 * 33 * 33 * 4;
```

```rust
// In src/render/camera.rs CameraUniform — defaults for PR 3:
sun_color: [1.00, 0.96, 0.90, 0.0],   // neutral-warm white
sky_color: [0.55, 0.70, 0.95, 0.0],   // cool noon blue
```

(PR 8 makes these time-of-day driven. PR 3 hardcodes them so the new shader has live values to consume.)

---

## Tasks

Each task ends with `cargo test` + screenshot capture. Tasks 1-3 are infrastructure (no visual change); Tasks 4-5 introduce the visual change; Task 6 re-baselines.

### Task 1: `ChunkLightVolume` + 3D texture upload

**Files:**
- Create: `src/render/light_volume.rs`
- Modify: `src/render/mod.rs` — declare `pub mod light_volume;`

- [ ] **Step 1: Add module declaration.**

In `src/render/mod.rs`, near the existing `pub mod hdr;` line, add `pub mod light_volume;` in alphabetical order.

- [ ] **Step 2: Write the module.**

Create `src/render/light_volume.rs`:
```rust
//! Per-chunk 3D light texture. Resolution is 33³ — one extra cell per
//! axis beyond the 32³ chunk so trilinear sampling at the chunk's
//! +X/+Y/+Z boundary picks up the neighbor cell rather than clamping.
//!
//! Layout: Rgba8Unorm. R/G/B encode block_red/green/blue / 15.0;
//! A encodes sky_light / 15.0. The shader multiplies back as needed.
//!
//! Lifecycle: created on first Meshed or Relit completion for a chunk;
//! re-uploaded on every subsequent Relit; dropped when the chunk
//! leaves the load radius.

use wgpu::TextureFormat;

pub const LIGHT_VOLUME_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
pub const LIGHT_VOLUME_DIM: u32 = 33;
pub const LIGHT_VOLUME_BYTES: usize = 33 * 33 * 33 * 4;

pub struct ChunkLightVolume {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl ChunkLightVolume {
    /// Create + upload from a freshly-built `[u8; LIGHT_VOLUME_BYTES]` blob.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, blob: &[u8]) -> Self {
        assert_eq!(blob.len(), LIGHT_VOLUME_BYTES,
            "light volume blob size mismatch ({} vs {})", blob.len(), LIGHT_VOLUME_BYTES);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("chunk-light-volume"),
            size: wgpu::Extent3d {
                width: LIGHT_VOLUME_DIM,
                height: LIGHT_VOLUME_DIM,
                depth_or_array_layers: LIGHT_VOLUME_DIM,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: LIGHT_VOLUME_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            blob,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(LIGHT_VOLUME_DIM * 4),
                rows_per_image: Some(LIGHT_VOLUME_DIM),
            },
            wgpu::Extent3d {
                width: LIGHT_VOLUME_DIM,
                height: LIGHT_VOLUME_DIM,
                depth_or_array_layers: LIGHT_VOLUME_DIM,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }

    /// Reupload contents into the existing texture (no realloc).
    pub fn update(&self, queue: &wgpu::Queue, blob: &[u8]) {
        assert_eq!(blob.len(), LIGHT_VOLUME_BYTES,
            "light volume blob size mismatch");
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            blob,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(LIGHT_VOLUME_DIM * 4),
                rows_per_image: Some(LIGHT_VOLUME_DIM),
            },
            wgpu::Extent3d {
                width: LIGHT_VOLUME_DIM,
                height: LIGHT_VOLUME_DIM,
                depth_or_array_layers: LIGHT_VOLUME_DIM,
            },
        );
    }
}
```

- [ ] **Step 3: Build.**

Run: `cargo build`
Expected: compiles clean, unused-warning on the type (we use it in Task 2).

- [ ] **Step 4: Commit.**

```bash
git add src/render/light_volume.rs src/render/mod.rs
git commit -m "render: ChunkLightVolume skeleton (unused)"
```

---

### Task 2: Build the 33³ blob from `DenseChunk` + neighbors

**Files:**
- Modify: `src/voxel/chunk.rs` — add `build_light_volume_blob`

- [ ] **Step 1: Add the blob builder.**

In `src/voxel/chunk.rs`, near the existing `unpack_rgb` helper, add:
```rust
/// Build the 33³ Rgba8Unorm blob for a chunk's GPU light volume. Each
/// axis spans 0..=32 — index 32 reads from the +X/+Y/+Z neighbor's
/// index 0 so trilinear sampling at the chunk's far face sees valid
/// neighbor values (no clamp artefact). Cells where the neighbor isn't
/// loaded clamp to the local boundary value.
///
/// Layout per voxel:
///   R = block_red   / 15
///   G = block_green / 15
///   B = block_blue  / 15
///   A = sky_light   / 15
pub fn build_light_volume_blob(
    dense: &DenseChunk,
    neighbors: &Neighbors,
) -> Box<[u8; 33 * 33 * 33 * 4]> {
    use crate::mesher::Face;
    let mut out = vec![0u8; 33 * 33 * 33 * 4].into_boxed_slice().try_into().unwrap();
    // SAFETY: vec is exactly 33*33*33*4 bytes; try_into to fixed array succeeds.
    let mut out: Box<[u8; 33 * 33 * 33 * 4]> = out;
    let scale = |v: u8| ((v as u32 * 255 + 7) / 15) as u8;   // round-up to nearest
    for z in 0..33 {
        for y in 0..33 {
            for x in 0..33 {
                let (r, g, b, a) = sample_for_blob(dense, neighbors, x, y, z);
                let idx = (z * 33 * 33 + y * 33 + x) * 4;
                out[idx]     = scale(r);
                out[idx + 1] = scale(g);
                out[idx + 2] = scale(b);
                out[idx + 3] = scale(a);
            }
        }
    }
    out
}

/// Sample (R, G, B, sky) at local index (x, y, z) where each axis is
/// 0..=32. Indices 0..=31 read from this chunk; index 32 reads from the
/// +X/+Y/+Z neighbor's index 0. If the neighbor isn't loaded, returns
/// the local boundary cell (clamped).
fn sample_for_blob(
    dense: &DenseChunk,
    neighbors: &Neighbors,
    x: usize,
    y: usize,
    z: usize,
) -> (u8, u8, u8, u8) {
    use crate::mesher::Face;
    // Decide which chunk + local coords this index maps to.
    let (chunk_src, lx, ly, lz): (&DenseChunk, usize, usize, usize) = if x == 32 {
        match neighbors.chunks[Face::PosX as usize] {
            Some(n) => (n, 0, y.min(31), z.min(31)),
            None    => (dense, 31, y.min(31), z.min(31)),
        }
    } else if y == 32 {
        match neighbors.chunks[Face::PosY as usize] {
            Some(n) => (n, x.min(31), 0, z.min(31)),
            None    => (dense, x.min(31), 31, z.min(31)),
        }
    } else if z == 32 {
        match neighbors.chunks[Face::PosZ as usize] {
            Some(n) => (n, x.min(31), y.min(31), 0),
            None    => (dense, x.min(31), y.min(31), 31),
        }
    } else {
        (dense, x, y, z)
    };
    let idx = crate::voxel::coords::LocalPos(glam::UVec3::new(lx as u32, ly as u32, lz as u32))
        .to_index();
    let (r, g, b) = unpack_rgb(chunk_src.block_rgb[idx]);
    let a = chunk_src.sky_light[idx] & 0x0F;
    (r, g, b, a)
}
```

The `let mut out = ...; let mut out: Box<[u8; ...]> = out;` dance is to convert the runtime-sized `Box<[u8]>` into the fixed-size `Box<[u8; N]>`. If the implementer finds a cleaner version (e.g. `Box::new([0u8; N])` directly), use that — the spec is the function signature, not the body.

- [ ] **Step 2: Add a test.**

In `src/voxel/chunk.rs::tests` add:
```rust
#[test]
fn light_volume_blob_size_and_layout() {
    use crate::voxel::chunk::{build_light_volume_blob, DenseChunk, Neighbors};
    let mut d = DenseChunk::empty();
    d.sky_light[0] = 15;
    d.block_rgb[0] = pack_rgb(15, 0, 0);
    let n = Neighbors { chunks: [None; 6] };
    let blob = build_light_volume_blob(&d, &n);
    assert_eq!(blob.len(), 33 * 33 * 33 * 4);
    // First voxel (0,0,0): R should be ~255 (from block_rgb's R=15), A also ~255.
    assert!(blob[0] >= 240, "R channel scaled wrong: {}", blob[0]);
    assert_eq!(blob[1], 0);
    assert_eq!(blob[2], 0);
    assert!(blob[3] >= 240, "A channel scaled wrong: {}", blob[3]);
}
```

- [ ] **Step 3: Run.**

```bash
cargo test --release --lib voxel::chunk::tests::light_volume_blob_size_and_layout
```
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add src/voxel/chunk.rs
git commit -m "voxel: build_light_volume_blob — 33³ Rgba8 blob from DenseChunk + neighbors"
```

---

### Task 3: Extend chunk bind group layout + plumb light volume into draw

**Files:**
- Modify: `src/render/camera.rs:148-164` — `make_chunk_bind_group_layout` adds two more entries
- Modify: `src/render/mod.rs` — `chunk_lights: HashMap<ChunkCoord, ChunkLightVolume>`, shared `light_sampler`, bind-group construction at draw time

- [ ] **Step 1: Extend the bind group layout.**

In `src/render/camera.rs`, replace `make_chunk_bind_group_layout`:
```rust
pub fn make_chunk_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chunk-bgl"),
        entries: &[
            // Binding 0 — per-chunk world-space origin uniform (unchanged).
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                },
                count: None,
            },
            // Binding 1 — 3D light volume texture.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            // Binding 2 — linear sampler for the light volume.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}
```

- [ ] **Step 2: Add `chunk_lights` map to `Renderer`.**

In `src/render/mod.rs`, near the `chunk_meshes` field declaration (around line 225), add:
```rust
/// Per-chunk 3D light volume textures, keyed by world chunk coord.
/// One volume per chunk regardless of how many LOD slots are filled;
/// all LODs of the same coord sample the same volume.
chunk_lights: std::collections::HashMap<ChunkCoord, light_volume::ChunkLightVolume>,
/// Linear sampler used at chunk bind group entry 2 (light volume).
/// Single shared instance — every chunk's bind group references it.
light_sampler: wgpu::Sampler,
```

- [ ] **Step 3: Initialize in the constructor.**

In `new_with_present_mode`, after `gpu` is created:
```rust
let light_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
    label: Some("chunk-light-sampler"),
    address_mode_u: wgpu::AddressMode::ClampToEdge,
    address_mode_v: wgpu::AddressMode::ClampToEdge,
    address_mode_w: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::FilterMode::Nearest,
    ..Default::default()
});
```
And add `chunk_lights: HashMap::new(), light_sampler,` to the Self literal at the end of the constructor.

- [ ] **Step 4: Add upload + remove API.**

In the same `impl Renderer` block as `upload_chunk_mesh`, add:
```rust
/// Upload (or replace) the 3D light volume for chunk `coord`.
/// The blob must be exactly `LIGHT_VOLUME_BYTES` long.
pub fn upload_chunk_light_volume(&mut self, coord: ChunkCoord, blob: &[u8]) {
    use std::collections::hash_map::Entry;
    match self.chunk_lights.entry(coord) {
        Entry::Occupied(e) => {
            e.get().update(&self.gpu.queue, blob);
        }
        Entry::Vacant(e) => {
            let vol = light_volume::ChunkLightVolume::new(
                &self.gpu.device,
                &self.gpu.queue,
                blob,
            );
            e.insert(vol);
        }
    }
}

/// Drop the per-chunk light volume (called when the chunk unloads).
pub fn remove_chunk_light_volume(&mut self, coord: ChunkCoord) {
    self.chunk_lights.remove(&coord);
}
```

Then update `remove_chunk_mesh` (around line 698) to also drop the light volume:
```rust
pub fn remove_chunk_mesh(&mut self, coord: ChunkCoord) {
    self.chunk_meshes.remove(&coord);
    self.chunk_lights.remove(&coord);
}
```

- [ ] **Step 5: Change `upload_chunk_mesh`'s bind group creation.**

The existing `upload_chunk_mesh` creates a bind group containing only the chunk uniform at binding 0. Since the chunk_bgl now requires three bindings, the bind group must include the light volume too.

**Decision:** build the chunk bind group at draw time (inside the per-chunk draw loop) rather than at upload time. The light volume might be uploaded *after* the mesh (different worker timings), so the bind group can't always be created at mesh-upload time. Draw-time construction is simple and gives every draw a fresh bind group with the current light volume.

Drop the `bg` field from `ChunkGpu` and the bind-group construction inside `upload_chunk_mesh`. Replace `bg` with just `_ubuf` retention:
```rust
struct ChunkGpu {
    mesh: GpuMesh,
    /// Per-LOD chunk uniform (world-space origin). The bind group
    /// referencing this buffer is built at draw time so the same call
    /// can include the chunk's current light_volume view.
    ubuf: wgpu::Buffer,
}
```
And in `upload_chunk_mesh`, remove the bind-group construction; store `ubuf` directly:
```rust
slots[lod] = Some(ChunkGpu {
    mesh: gpu_mesh,
    ubuf,
});
```

- [ ] **Step 6: Build bind group at draw time + handle missing light volume.**

In `encode_opaque_pass` (around line 1090-ish), inside the per-chunk loop, after picking the LOD slot, build the bind group:
```rust
// Per-chunk bind group: uniform + light volume + sampler. If the
// light volume hasn't arrived yet (race between Meshed and Relit),
// fall back to a 1×1×1 black placeholder — that path produces a
// black-lit chunk for a single frame which is much better than
// crashing or skipping the draw.
let light_view = self.chunk_lights.get(coord)
    .map(|v| &v.view)
    .unwrap_or(&self.placeholder_light_view);
let chunk_bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("chunk-bg"),
    layout: &self.chunk_bgl,
    entries: &[
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &cg.ubuf,
                offset: 0,
                size: std::num::NonZeroU64::new(16),
            }),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(light_view),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::Sampler(&self.light_sampler),
        },
    ],
});
pass.set_bind_group(1, &chunk_bg, &[0]);
```

(Replace the existing `pass.set_bind_group(1, &cg.bg, &[0])` call.)

- [ ] **Step 7: Create the placeholder light view.**

In the constructor, create a 1×1×1 Rgba8Unorm black texture:
```rust
let placeholder_light_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
    label: Some("chunk-light-placeholder"),
    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D3,
    format: light_volume::LIGHT_VOLUME_FORMAT,
    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    view_formats: &[],
});
gpu.queue.write_texture(
    wgpu::ImageCopyTexture {
        texture: &placeholder_light_tex,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    },
    &[0u8, 0, 0, 0],
    wgpu::ImageDataLayout {
        offset: 0,
        bytes_per_row: Some(4),
        rows_per_image: Some(1),
    },
    wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
);
let placeholder_light_view = placeholder_light_tex.create_view(&Default::default());
```

Add to `Renderer`:
```rust
_placeholder_light_tex: wgpu::Texture,
placeholder_light_view: wgpu::TextureView,
```

And to the Self literal.

- [ ] **Step 8: Apply same draw-time bind-group construction to water + reflection passes.**

The water pass (line ~1185) and reflection opaque pass (inside `encode_reflection_pass`) also iterate chunk_meshes and set bind group 1. Apply the same per-draw bind-group construction. The reflection pass uses the placeholder for the light view (mirroring sun-lit reflection is fine without per-pixel light variance; the LDR reflection target doesn't need fine lighting).

- [ ] **Step 9: Build + verify no visual change (yet).**

```bash
cargo build --release
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/t3.png --look 45,-15 --time 0.5
python3 tests/screenshots/diff.py tests/screenshots/baseline_noon_outdoor.png /tmp/t3.png
```

Since the shader still ignores the light volume (Task 4 wires it in), output should remain within noise floor.

If it's not: the placeholder light view is bound at every draw (because chunk_lights is empty until Task 4 wires the upload). That's correct for this step but means the shader is getting a black light volume. The shader DOESN'T sample it yet — opaque.wgsl in this state still does the old `max(sky_l * sun_intensity, block_l)` math from the vertex byte. So no visual change.

- [ ] **Step 10: Commit.**

```bash
git add src/render/camera.rs src/render/mod.rs
git commit -m "render: extend chunk bind group with 3D light volume + sampler"
```

---

### Task 4: Wire light volume into the opaque shader (THE VISUAL CHANGE)

This task makes the shader actually consume the new light volume. It introduces wrap diffuse, drops `face_mul`, and changes `block_lit` from a scalar to RGB. Visual output WILL change.

**Files:**
- Modify: `assets/shaders/opaque.wgsl` — full fragment shader rewrite (lighting section)

- [ ] **Step 1: Update the camera uniform struct.**

In `src/render/camera.rs`, extend `CameraUniform`:
```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],   // NEW
    pub sky_color: [f32; 4],   // NEW
    pub sun_intensity: f32,
    pub time: f32,
    pub underwater_factor: f32,
    pub clip_y_min: f32,
    pub eye: [f32; 4],
    pub inv_view_proj: [[f32; 4]; 4],
}
```

Update `identity()`:
```rust
sun_color: [1.00, 0.96, 0.90, 0.0],
sky_color: [0.55, 0.70, 0.95, 0.0],
```

- [ ] **Step 2: Update render-side camera buffer writes.**

In `src/render/mod.rs`, find every site that writes `CameraUniform` (search for `CameraUniform {`). Add:
```rust
sun_color: [1.00, 0.96, 0.90, 0.0],
sky_color: [0.55, 0.70, 0.95, 0.0],
```

at each write. There are ~3 sites: main world camera write, reflection camera write, possibly a third in `render_to_view`.

- [ ] **Step 3: Update the opaque shader.**

Edit `assets/shaders/opaque.wgsl`. Replace the `CameraUniform` struct:
```wgsl
struct CameraUniform {
    view_proj:         mat4x4<f32>,
    sun_dir:           vec4<f32>,
    sun_color:         vec4<f32>,
    sky_color:         vec4<f32>,
    sun_intensity:     f32,
    time:              f32,
    underwater_factor: f32,
    clip_y_min:        f32,
    eye:               vec4<f32>,
    inv_view_proj:     mat4x4<f32>,
};
```

Add the light-volume bindings to group 1:
```wgsl
@group(1) @binding(0) var<uniform> chunk: ChunkUniform;
@group(1) @binding(1) var light_volume:  texture_3d<f32>;
@group(1) @binding(2) var light_sampler: sampler;
```

In `VsOut`, add a `face_normal: vec3<f32>` location (replace one of the unused slots or pick a new location number; the comments in opaque.wgsl mention which are in use):
```wgsl
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) v_color: vec4<f32>,
    @location(1) v_ao:    f32,
    @location(2) v_light: f32,            // legacy — still computed for fog
    @location(3) v_world: vec3<f32>,
    @location(4) v_uv:    vec2<f32>,
    @location(5) @interpolate(flat) v_tile_index: u32,
    @location(6) @interpolate(flat) v_face_normal: vec3<f32>,
};
```

In `vs_main`, derive `face_normal` from `face`:
```wgsl
var face_normal: vec3<f32>;
switch (face) {
    case 0u: { face_normal = vec3<f32>( 1.0,  0.0,  0.0); }  // PosX
    case 1u: { face_normal = vec3<f32>(-1.0,  0.0,  0.0); }  // NegX
    case 2u: { face_normal = vec3<f32>( 0.0,  1.0,  0.0); }  // PosY
    case 3u: { face_normal = vec3<f32>( 0.0, -1.0,  0.0); }  // NegY
    case 4u: { face_normal = vec3<f32>( 0.0,  0.0,  1.0); }  // PosZ
    case 5u: { face_normal = vec3<f32>( 0.0,  0.0, -1.0); }  // NegZ
    default: { face_normal = vec3<f32>( 0.0,  1.0,  0.0); }
}
out.v_face_normal = face_normal;
```

Keep computing `v_light` from `face_light.y` so the fog path still works during PR 3 (fog moves to per-pixel sky_level in a follow-up; for now we keep the vertex byte fed to fog).

In `fs_main`, replace the lighting section (the part that computes `lit_rgb`). Find the existing block that does:
```wgsl
let ao = mix(0.45, 1.0, in.v_ao);
let lit = max(0.05, in.v_light);
let shade = ao * lit;
...
var lit_rgb = base_rgb * shade * variation;
```

Replace with:
```wgsl
// ── Per-pixel light volume sample, air-side of the surface.
let sample_world = in.v_world + in.v_face_normal * 0.5;
let chunk_local  = sample_world - chunk.origin.xyz;
let uvw          = (chunk_local + vec3<f32>(0.5, 0.5, 0.5)) / 33.0;
let lvol         = textureSampleLevel(light_volume, light_sampler, uvw, 0.0);
let block_rgb    = lvol.rgb;        // already 0..1 from Rgba8Unorm
let sky_level    = lvol.a;          // 0..1

// ── Wrap-diffuse directional sun with sky-channel occlusion.
let n_dot_l = max(dot(in.v_face_normal, -camera.sun_dir.xyz), 0.0);
let wrap    = (n_dot_l + 0.4) / 1.4;
// PR 5 introduces real cast shadows; until then `shadow = 1` everywhere.
let shadow  = 1.0;
let sun_lit = wrap * shadow * sky_level;

// ── Sky ambient — tinted, scaled by sky exposure.
let sky_amb = camera.sky_color.rgb * sky_level * 0.35;

// ── Colored block light — direct from the volume.
let direct  = camera.sun_color.rgb * sun_lit * camera.sun_intensity;
let lit     = direct + sky_amb + block_rgb;

// AO stays a per-vertex bake.
let ao_term = mix(0.45, 1.0, in.v_ao);
let MIN_SHADE = vec3<f32>(0.02, 0.02, 0.02);

// Per-block + biome variation as before (don't touch — these still apply).
let variation = 1.0 + block_variation_hash(in.v_world) * 0.06;
var lit_rgb = base_rgb * (lit + MIN_SHADE) * ao_term * variation;
let is_blendable = in.v_tile_index == 2u || in.v_tile_index == 4u;
if (is_blendable) {
    lit_rgb = lit_rgb + biome_tint_shift(in.v_world.xz) * (lit + MIN_SHADE) * ao_term;
}
```

Remove the `face_mul` computation entirely (lines around 90-99 in current opaque.wgsl). Note: `face_mul` was applied to `in.color.rgb` — without it the per-face hardcoded brightness goes away; the directional sun now does the work.

Adjust the vertex output: keep `v_color = vec4<f32>(in.color.rgb, in.color.a)` (no face_mul multiplication) but leave the existing alpha-based "is water" discard logic unchanged.

The fog block (around line 333) still references `in.v_light` for cave/sky fog tinting. Keep it as-is for PR 3.

- [ ] **Step 4: Apply same sampling to water shader.**

In `assets/shaders/water.wgsl`, mirror the changes:
- Add the same `CameraUniform` fields
- Add `@group(1) @binding(1) var light_volume: texture_3d<f32>; @group(1) @binding(2) var light_sampler: sampler;`
- In water's fragment, sample the volume at the water surface position and use it to tint the water color. Water gets the same `sky_level`-gated sun and the same `block_rgb` ambient.

Specifically, find the line in water.wgsl where water mixes its surface color with the deep tint. Right before that mix, multiply by the sampled lighting:
```wgsl
let sample_world = in.v_world + vec3<f32>(0.0, 0.5, 0.0);  // water surface faces up
let chunk_local  = sample_world - chunk.origin.xyz;
let uvw          = (chunk_local + vec3<f32>(0.5, 0.5, 0.5)) / 33.0;
let lvol         = textureSampleLevel(light_volume, light_sampler, uvw, 0.0);
let sky_level    = lvol.a;
let block_rgb    = lvol.rgb;
rgb = rgb * (camera.sky_color.rgb * sky_level * 0.35 + block_rgb + vec3<f32>(0.2));
```

(Exact formula will need tuning; pick a starting point and iterate. The above keeps water reasonably bright at noon while letting cave-water go dim.)

- [ ] **Step 5: Compile.**

```bash
cargo build --release
```
Expected: clean build.

- [ ] **Step 6: Single-shot smoke screenshot.**

```bash
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/t4_noon.png --look 45,-15 --time 0.5
```

Open the PNG. **Expect a visible change** from the baseline:
- Smooth gradients across face interiors (no faceted per-vertex shading)
- Sides of blocks darker than tops (from dot(N, sun))
- Wrap-diffuse softens the shadowed sides (not pitch black)
- Torches' light should bleed into surrounding blocks smoothly

If the scene is mostly BLACK, the light volume isn't being uploaded yet (Task 5 wires the upload pipeline). Proceed to Task 5.

If the scene looks roughly normal but with smooth gradients, great — visual change is showing.

- [ ] **Step 7: Commit.**

```bash
git add assets/shaders/opaque.wgsl assets/shaders/water.wgsl src/render/camera.rs src/render/mod.rs
git commit -m "render: per-pixel light volume sampling + wrap diffuse in opaque/water"
```

---

### Task 5: Wire the light volume upload pipeline

**Files:**
- Modify: `src/jobs/mod.rs` — extend `JobResult::Meshed` (light volume option) + `JobResult::Relit` (light volume always); workers build the blob
- Modify: `src/ecs/systems/mesh_upload.rs` — handlers call `renderer.upload_chunk_light_volume`

- [ ] **Step 1: Extend `JobResult::Meshed` + `JobResult::Relit`.**

```rust
Meshed {
    coord: ChunkCoord,
    lod: u8,
    mesh: ChunkMesh,
    version: u64,
    /// Light volume blob (33³ × 4 bytes), populated by LOD-0 mesher
    /// only — LODs 1 and 2 share the LOD-0 volume so they always set
    /// `None`. `None` on LOD-0 means the chunk's light wasn't ready
    /// yet (rare race; the next Relit will catch up).
    light_volume: Option<Box<[u8; 33 * 33 * 33 * 4]>>,
},
...
Relit {
    coord: ChunkCoord,
    data: PalettedChunk,
    changed_faces: [bool; 6],
    /// Always present — the relight worker built it from the same
    /// DenseChunk it just relit.
    light_volume: Box<[u8; 33 * 33 * 33 * 4]>,
},
```

- [ ] **Step 2: Populate the light volume in mesher worker (LOD-0).**

In `src/jobs/mod.rs`, find the mesher worker (around line 253 — the first `JobResult::Meshed` send). It builds a `ChunkMesh` from a `DenseChunk` + `Neighbors`. Add:
```rust
let light_volume = if lod == 0 {
    Some(crate::voxel::chunk::build_light_volume_blob(&dense, &neighbors))
} else {
    None
};
let _ = tx.send(JobResult::Meshed { coord, lod, mesh, version, light_volume });
```

Same change on the second `JobResult::Meshed` send (around line 301) — that's the relight-driven re-mesh path.

- [ ] **Step 3: Populate the light volume in relight worker.**

In the same file, find the `JobResult::Relit` send (line 218). The worker has a `DenseChunk` + `Neighbors` in scope. Add:
```rust
let light_volume = crate::voxel::chunk::build_light_volume_blob(&dense, &ns);
let _ = tx.send(JobResult::Relit { coord, data, changed_faces, light_volume });
```

- [ ] **Step 4: Wire upload into the handlers.**

In `src/ecs/systems/mesh_upload.rs`:
```rust
JobResult::Meshed { coord, lod, mesh, version, light_volume } => {
    ...existing version check + upload_chunk_mesh...
    if version >= current && let Some(blob) = light_volume {
        renderer.upload_chunk_light_volume(coord, blob.as_ref());
    }
}
JobResult::Relit { coord, data, changed_faces, light_volume } => {
    renderer.upload_chunk_light_volume(coord, light_volume.as_ref());
    ...existing relight handling...
}
```

- [ ] **Step 5: Plumb world unload to drop the light volume.**

The `world_unload` system already calls `renderer.remove_chunk_mesh(coord)`. We extended that to also drop the light volume in Task 3 step 4, so this should be automatic — verify by searching for `remove_chunk_mesh` in `src/ecs/systems/world_unload.rs` (or wherever it lives) and confirming the renderer-side `remove_chunk_mesh` now also clears the light volume.

- [ ] **Step 6: Build + run.**

```bash
cargo build --release
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/t5_noon.png --look 45,-15 --time 0.5
```

Open the PNG. **Expect a major visual upgrade**:
- Outdoor scene with smooth light gradients
- Sides of terrain visibly darker than tops (sun direction!)
- Wrap diffuse softens shadows
- If torches are nearby, their light blooms smoothly outward (in color, not just brightness)

- [ ] **Step 7: Commit.**

```bash
git add src/jobs/mod.rs src/ecs/systems/mesh_upload.rs
git commit -m "render: upload light volume on Meshed/Relit completions"
```

---

### Task 6: Re-baseline screenshots + final validation

**Files:**
- Modify: `tests/screenshots/baseline_*.png` (five files)

PR 3 is the first intentional visual change. Old baselines are now stale; capture fresh ones under PR 3's new look.

- [ ] **Step 1: Capture all 5 scenes.**

```bash
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_noon_outdoor.png --look 45,-15 --time 0.5
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_underwater.png --spawn 64,60,-12 --look 0,-10 --time 0.5
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_cave.png --spawn 0,30,0 --look 0,-30 --time 0.5
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_sunset.png --look 90,-10 --time 0.78
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_fog_horizon.png --look 0,0 --time 0.5
```

- [ ] **Step 2: Open each PNG and sanity-check.**

- Noon outdoor: terrain should have visible directional shading. Tops bright, sides medium, undersides dim. Sky in upper portion.
- Underwater: blue tint dominates (composite). Lighting underwater should be dimmer (sky_level is low several blocks under).
- Cave: should be darker than before — caves no longer get sky-light contribution; only block lights matter inside.
- Sunset: warm horizon. Lighting on terrain warm too (sun direction).
- Fog horizon: smooth fog gradient toward horizon.

If something looks broken (e.g. uniform-white terrain, all-black scene), stop and investigate. Likely culprits:
- Light volume not uploaded (Task 5 issue)
- Wrong UVW math (sampling outside the texture)
- `chunk.origin` not bound (group 1 binding 0 wasn't kept)

- [ ] **Step 3: Self-stability check.**

Re-capture noon once more to confirm screenshots are stable:
```bash
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/recap_noon.png --look 45,-15 --time 0.5
python3 tests/screenshots/diff.py tests/screenshots/baseline_noon_outdoor.png /tmp/recap_noon.png
```
Expected: within noise floor.

- [ ] **Step 4: Run all tests.**

```bash
cargo test --release
```
Expected: 230 passed, 0 failed.

- [ ] **Step 5: Live smoke test.**

Run the game without screenshot mode:
```bash
cargo run --release --bin oxium
```
- Walk around. Look at the directional lighting.
- Place a torch. Verify warm light bleeds smoothly outward.
- Walk into water. Underwater grade still works.
- Walk into a cave. Should be much darker than before — no sky leak.
- Toggle through `--time 0.0 / 0.25 / 0.75` between restarts to verify lighting tracks sun position.

- [ ] **Step 6: Commit.**

```bash
git add tests/screenshots/baseline_*.png
git commit -m "test(screenshot): re-baseline scenes under PR 3 per-pixel sampling"
```

---

### Task 7: Wrap up

- [ ] **Step 1: Branch state check.**

```bash
git log main..HEAD --oneline
git status
```
Expected: 6 commits ahead of main, clean working tree.

- [ ] **Step 2: Hand off.**

After verification, invoke `superpowers:finishing-a-development-branch` to merge to local main + prep PR 4 worktree.

---

## Self-review notes

Spec coverage:

| Spec requirement | Plan task |
|------------------|-----------|
| 33³ Rgba8Unorm per-chunk light volume | Task 1 |
| Build blob from DenseChunk + neighbors (1-cell border) | Task 2 |
| Bind layout extended (texture + sampler at group 1) | Task 3 |
| Per-pixel sampling at `world + face_normal * 0.5` | Task 4 |
| Wrap diffuse `(N·L + 0.4) / 1.4` | Task 4 |
| `face_mul` removed; sun_color × sun_lit + sky_amb + block_rgb compositing | Task 4 |
| Vertex light byte unused (vertex layout unchanged) | Task 4 |
| Upload pipeline (Meshed + Relit) | Task 5 |
| Re-baselined screenshots reflect new look | Task 6 |
| **Out of scope for PR 3:** shadow map (PR 5), bloom (PR 4), ambient bounce (PR 6), fog-to-composite move (follow-up), TimeOfDay-driven sun_color/sky_color (PR 8) | — |

Placeholders / type consistency: clean. The `ChunkLightVolume` type, `LIGHT_VOLUME_FORMAT`/`LIGHT_VOLUME_DIM`/`LIGHT_VOLUME_BYTES` constants, and `build_light_volume_blob` name are all stable across tasks.

Known design risks the implementer should be aware of:
1. **Reflection pass uses LDR target + opaque shader.** When the same shader samples a light volume in the reflection pass, the LDR-format reflection target receives HDR linear values. This worked in PR 1 (where the shader output was already tonemapped). Now the shader outputs unclamped linear → reflection texture is LDR. Result: reflections may clip bright values. Acceptable for v1; verify in the smoke test.
2. **Per-draw bind group creation.** At radius 16 with ~4500 visible chunks, building 4500 bind groups per frame is non-trivial CPU work. M4 Max should handle it but profile if FPS drops noticeably. Mitigation: bind-group caching keyed by `(coord, lod, light_volume_revision)` — defer to a follow-up.
3. **Light volume race conditions.** A chunk can receive a Meshed result before its Relit completes (or vice versa). Both paths upload the same blob; whichever lands last wins. The placeholder texture handles the gap. Verify no flicker or black frames during chunk stream-in.
