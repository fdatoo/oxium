//! GPU rendering backend, built on `wgpu` (a portable Vulkan-shaped abstraction).
//!
//! `wgpu` exposes:
//!
//! - `Instance` — discovers GPU adapters.
//! - `Adapter` — a particular physical GPU.
//! - `Device` + `Queue` — the logical interface and its command submission stream.
//! - `Surface` — the swapchain (the OS-window-backed framebuffer).
//!
//! Higher level objects (`RenderPipeline`, `BindGroup`, `Buffer`, `Texture`)
//! are owned per subsystem (sky, opaque voxels, water, cursor, HUD).

pub mod atlas;
pub mod bloom;
pub mod camera;
pub mod font;
pub mod gpu;
pub mod hdr;
pub mod hud;
pub mod light_volume;
pub mod mesh;
pub mod pipelines;
pub mod screenshot;

use std::collections::HashMap;
use std::sync::Arc;
use winit::window::Window;

use crate::mesher::ChunkMesh;
use crate::render::atlas::{build_atlas, upload_atlas, AtlasGpu};
use crate::render::camera::{
    make_camera_bind_group_layout, make_camera_buffer, make_chunk_bind_group_layout, view_proj,
    CameraUniform, ChunkUniform,
};
use crate::render::font::{build_font_atlas, ATLAS_H as FONT_ATLAS_H, ATLAS_W as FONT_ATLAS_W};
use crate::render::gpu::{
    make_depth_sample_texture, make_depth_texture, make_msaa_color_texture,
    make_reflection_color_textures, make_reflection_depth_texture, Gpu,
};
use crate::render::hud::HudFrame;
use crate::render::mesh::{upload_mesh, GpuMesh};
use crate::render::bloom::BloomChain;
use crate::render::pipelines::bloom::{build as build_bloom, BloomPipelines};
use crate::render::pipelines::composite::{build as build_composite, CompositePipeline};
use crate::render::pipelines::cursor::{
    build as build_cursor, make_cursor_bind_group_layout, CursorPipeline,
};
use crate::render::pipelines::hud::{build as build_hud, HudPipeline};
use crate::render::pipelines::opaque::{build as build_opaque, FrontFace, OpaquePipeline};
use crate::render::pipelines::sky::{build as build_sky, SkyPipeline};
use crate::render::pipelines::water::{build as build_water, WaterPipeline};
use crate::voxel::coords::{BlockPos, ChunkCoord};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

/// Extract the 6 view-frustum planes from a column-major view-projection
/// matrix using the Gribb-Hartmann technique. Each returned `Vec4` is
/// `(nx, ny, nz, d)` such that `n.dot(point) + d >= 0` means the point
/// is on the inside (camera-visible) side of that plane.
///
/// Ordering: `[left, right, bottom, top, near, far]`. The near plane
/// uses `row2` (for wgpu's `[0, 1]` depth range, not OpenGL's
/// `[-1, 1]`).
fn extract_frustum_planes(vp: Mat4) -> [Vec4; 6] {
    let m = vp.to_cols_array_2d();
    // glam stores column-major (`m[col][row]`); rebuild rows.
    let row = |r: usize| Vec4::new(m[0][r], m[1][r], m[2][r], m[3][r]);
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r2,      // near (wgpu uses [0, 1] depth)
        r3 - r2, // far
    ]
}

/// Conservative AABB-vs-frustum test using the "n-vertex" trick: for
/// each plane, pick the AABB corner most in the direction of the
/// plane's normal — if even *that* corner is on the inside-negative
/// side, the whole AABB must be outside. Two false-positive cases
/// (chunk straddling a single corner of the frustum) are accepted as
/// the trade for the test being O(6 dot products) per chunk instead
/// of O(48).
fn aabb_in_frustum(planes: &[Vec4; 6], min: Vec3, max: Vec3) -> bool {
    for p in planes {
        let n = p.truncate();
        let positive_vertex = Vec3::new(
            if n.x >= 0.0 { max.x } else { min.x },
            if n.y >= 0.0 { max.y } else { min.y },
            if n.z >= 0.0 { max.z } else { min.z },
        );
        if n.dot(positive_vertex) + p.w < 0.0 {
            return false;
        }
    }
    true
}

/// Top-level rendering object. Owns the GPU state and a hashmap of all
/// loaded chunk meshes keyed by their world coordinate.
///
/// Per-chunk uniform buffers live alongside the meshes: each chunk gets its
/// own `ChunkUniform` (16 bytes) with the chunk's world origin baked in.
/// This is simpler than packing many chunks into one buffer with dynamic
/// offsets and costs negligible memory at our scale (a few hundred chunks ×
/// a 256-byte uniform buffer is well under a megabyte).
pub struct Renderer {
    pub gpu: Gpu,
    /// Live depth attachment for both the opaque and the water render
    /// passes. The opaque pass writes; the water pass tests but
    /// doesn't write. Kept as a `Texture` (not just a view) so its
    /// contents can be copied into `depth_sample_texture` between
    /// passes.
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Offscreen Rgba16Float color target the 3D world passes (sky,
    /// opaque, water, cursor) render into. The composite pass samples
    /// this view and resolves to the swapchain. HUD draws after
    /// composite, directly to the swapchain.
    pub hdr: hdr::HdrTarget,
    /// Sampleable copy of `depth_texture`. The opaque pass's depth
    /// values get blit-copied into this between the opaque and water
    /// passes; the water shader then samples it to compute terrain
    /// depth vs water depth (foam mask + depth tint).
    depth_sample_texture: wgpu::Texture,
    depth_sample_view: wgpu::TextureView,
    /// Bind group exposing `depth_sample_view` at group 3 to the
    /// water pipeline. Recreated alongside the textures on resize.
    water_depth_bg: wgpu::BindGroup,
    /// Reflection-pass variant of `opaque_pipe`. Identical except
    /// `front_face = Cw` — mirroring the camera flips the apparent
    /// winding of every triangle, so without this the regular
    /// pipeline would cull every top-facing surface (mountain tops,
    /// grass patches, etc.) when the mirror eye renders them.
    opaque_pipe_reflection: OpaquePipeline,

    /// Multisampled colour render target for the world pass. The world
    /// pass draws into this `MSAA_SAMPLES`-sample texture; the render
    /// pass's `resolve_target` is the swapchain texture, which wgpu
    /// fills with the resolved single-sample result at end of pass.
    /// Recreated by `resize` alongside the depth texture.
    msaa_color_view: wgpu::TextureView,

    // ── Planar reflection resources. The reflection pass renders the
    // world with a virtual camera mirrored across the water plane
    // and `clip_y_min = SEA_LEVEL` (so only above-water geometry
    // appears) into `reflection_msaa_view`, resolving into
    // `reflection_resolve_texture` for the water shader to sample.
    /// Mirrored-camera uniform buffer. Written once per frame to the
    /// reflected view-projection + flipped sun direction etc.
    reflection_camera_buf: wgpu::Buffer,
    /// Bind group exposing `reflection_camera_buf` at group 0 for the
    /// reflection pass. The same opaque and sky pipelines read from
    /// it — they don't care which buffer backs the group as long as
    /// the layout matches.
    reflection_camera_bg: wgpu::BindGroup,
    /// MSAA colour render target for the reflection pass.
    reflection_msaa_view: wgpu::TextureView,
    /// Single-sample resolve target. The MSAA pass auto-resolves
    /// into this view; the water shader's group-4 bind group points
    /// at it.
    #[allow(dead_code)] // kept-alive owner of `reflection_resolve_view`
    reflection_resolve_texture: wgpu::Texture,
    reflection_resolve_view: wgpu::TextureView,
    /// Dedicated depth attachment for the reflection pass. Can't be
    /// shared with `depth_texture` because that one is still bound
    /// by the concurrent main world pass in the same encoder.
    reflection_depth_view: wgpu::TextureView,
    /// Sampler used by the water shader to read
    /// `reflection_resolve_view`. Linear filtering across the
    /// distorted reflection lookup hides wave-aliased pixel-boundaries.
    reflection_sampler: wgpu::Sampler,
    /// Bind group at index 4 on the water pipeline — exposes the
    /// reflection texture + sampler. Recreated on resize alongside
    /// the textures.
    water_reflection_bg: wgpu::BindGroup,
    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    chunk_bgl: wgpu::BindGroupLayout,
    opaque_pipe: OpaquePipeline,
    /// Transparent water pipeline, drawn after the opaque pass over
    /// the same chunk vertex buffers — the shader filters out
    /// non-water fragments at the top of `fs_main`.
    water_pipe: WaterPipeline,
    /// Sky-gradient pipeline, drawn before opaque each frame. Targets
    /// the HDR MSAA color view.
    sky_pipe: SkyPipeline,
    /// Reflection-pass sky pipeline. Same shader; built against the
    /// surface (swapchain) format because the reflection texture is
    /// LDR — sampled by the water shader as ordinary color.
    sky_pipe_reflection: SkyPipeline,
    /// Composite pipeline: resolves the HDR target to the swapchain.
    /// Currently passthrough; Tasks 5/6 add tonemap + underwater tint.
    composite_pipe: CompositePipeline,
    /// Bloom mip chain (5 levels, ½..1/32 swapchain). Recreated on resize.
    pub bloom: BloomChain,
    /// Threshold / downsample / upsample bloom pipelines. Shared bind-
    /// group layout; instances of `BloomChain::sampler` + per-mip view
    /// fill the layout per draw.
    bloom_pipes: BloomPipelines,
    /// Wireframe cursor pipeline + its uniform/bind group. Drawn last
    /// (over the opaque pass) only when `cursor_visible == true`.
    cursor_pipe: CursorPipeline,
    cursor_buf: wgpu::Buffer,
    cursor_bg: wgpu::BindGroup,
    cursor_visible: bool,

    /// Block texture atlas, bound as group 2 by the opaque pipeline.
    /// Built once at startup from `assets/textures/*.png`; the GPU
    /// resources stay alive for the renderer's lifetime.
    atlas: AtlasGpu,

    /// HUD pipeline + the two bind groups it draws against (one for
    /// the font atlas, one re-using the block atlas for hotbar
    /// icons). The screen-size uniform is rewritten per frame.
    hud_pipe: HudPipeline,
    hud_screen_buf: wgpu::Buffer,
    hud_screen_bg: wgpu::BindGroup,
    hud_font_bg: wgpu::BindGroup,
    hud_atlas_bg: wgpu::BindGroup,
    /// Kept alive so `hud_font_bg`'s texture view stays valid.
    _hud_font_texture: wgpu::Texture,
    _hud_font_view: wgpu::TextureView,
    _hud_sampler: wgpu::Sampler,

    /// Up to three GPU meshes per loaded chunk — one per LOD level
    /// (`[L0, L1, L2]`). The render loop picks which slot to draw based
    /// on the chunk's distance to the camera, falling back to the
    /// nearest available LOD if a job hasn't finished yet.
    chunk_meshes: HashMap<ChunkCoord, [Option<ChunkGpu>; 3]>,
    /// Per-chunk 3D light volume textures, keyed by world chunk coord.
    /// One volume per chunk regardless of how many LOD slots are filled.
    chunk_lights: HashMap<ChunkCoord, light_volume::ChunkLightVolume>,
    /// Linear sampler used at chunk bind group entry 2 (light volume).
    light_sampler: wgpu::Sampler,
    /// Held alive so `placeholder_light_view` stays valid.
    _placeholder_light_tex: wgpu::Texture,
    /// 1×1×1 black light volume used when a chunk's real volume hasn't
    /// uploaded yet (Meshed-before-Relit race). Keeps draw paths from
    /// crashing on missing bind-group entries.
    placeholder_light_view: wgpu::TextureView,
    /// Number of `draw_indexed` calls the last opaque pass issued.
    /// Updated by `encode_opaque_pass`, read by the perf HUD. `Cell`
    /// so the render path can stay `&self` while still recording.
    last_draw_calls: std::cell::Cell<u32>,
    /// Latest underwater factor handed in by `set_underwater`. Folded
    /// into the camera uniform every frame so the shaders can blend
    /// toward a deep-blue tint while the camera is submerged.
    /// Maintained on the renderer side so callers don't have to thread
    /// it through the `render()` call chain.
    underwater_factor: f32,
}

/// Per-chunk GPU resources: the mesh buffers, the chunk-origin uniform,
/// and a cached bind group that references both. The bind group is built
/// once at upload time and reused across all three render passes
/// (opaque, water, reflection). It must be rebuilt whenever the chunk's
/// light volume is (re)uploaded — see `upload_chunk_light_volume`.
struct ChunkGpu {
    mesh: GpuMesh,
    /// Per-LOD chunk uniform (world-space origin).
    ubuf: wgpu::Buffer,
    /// Cached bind group for `chunk_bgl`: ubuf @ 0, light view @ 1,
    /// light sampler @ 2. Used as group 1 in every chunk draw with
    /// a dynamic offset of 0.
    bind_group: wgpu::BindGroup,
    /// Mirrored from `ChunkMesh::has_water` at upload time. The frame
    /// loop ORs this across every visible chunk to decide whether the
    /// planar-reflection pass needs to run — when no on-screen chunk
    /// contains water, the whole reflection render is skipped.
    has_water: bool,
}

impl Renderer {
    /// Initialize the renderer with the chosen swapchain present
    /// mode. Callers pass `Fifo` for v-sync (normal play) or
    /// `Immediate` for "uncapped" perf measurement so HUD FPS
    /// reflects real throughput.
    pub fn new_with_present_mode(window: Arc<Window>, present_mode: wgpu::PresentMode) -> Self {
        let gpu = Gpu::new_with_present_mode(window, present_mode);
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
        // 1×1×1 black 3D texture as the placeholder when a chunk's real
        // light volume hasn't been uploaded yet.
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
        let hdr_target =
            hdr::HdrTarget::new(&gpu.device, gpu.surface_cfg.width, gpu.surface_cfg.height);
        let (depth_texture, depth_view) =
            make_depth_texture(&gpu.device, gpu.surface_cfg.width, gpu.surface_cfg.height);
        let (depth_sample_texture, depth_sample_view) = make_depth_sample_texture(
            &gpu.device,
            gpu.surface_cfg.width,
            gpu.surface_cfg.height,
        );
        // World pass MSAA color attachment is HDR-format so the
        // resolved output goes into the HdrTarget; the composite pass
        // then reads HDR and writes the swapchain. Reflection pass keeps
        // its own surface-format MSAA so water samples LDR reflections.
        let msaa_color_view = make_msaa_color_texture(
            &gpu.device,
            gpu.surface_cfg.width,
            gpu.surface_cfg.height,
            hdr::HDR_FORMAT,
        );
        // Reflection-pass colour + depth attachments.
        let (reflection_msaa_view, reflection_resolve_texture, reflection_resolve_view) =
            make_reflection_color_textures(
                &gpu.device,
                gpu.surface_cfg.width,
                gpu.surface_cfg.height,
                gpu.surface_cfg.format,
            );
        let reflection_depth_view = make_reflection_depth_texture(
            &gpu.device,
            gpu.surface_cfg.width,
            gpu.surface_cfg.height,
        );
        let reflection_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reflection-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let camera_bgl = make_camera_bind_group_layout(&gpu.device);
        let camera_buf = make_camera_buffer(&gpu.device);
        let camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });
        // Separate uniform + bind group for the reflection pass.
        // Layout matches the main camera so the same opaque + sky
        // pipelines can read either when bound at group 0.
        let reflection_camera_buf = make_camera_buffer(&gpu.device);
        let reflection_camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reflection-camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: reflection_camera_buf.as_entire_binding(),
            }],
        });
        let chunk_bgl = make_chunk_bind_group_layout(&gpu.device);

        // Atlas: load block textures from `assets/textures/`, pack them
        // into a single sRGB RGBA8 image, upload to a wgpu texture, and
        // make a bind group. Missing files fall back to magenta tiles
        // (see `build_atlas`), so a startup with no `assets/textures/`
        // directory still renders — just with hot-pink blocks.
        let textures_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("textures");
        let atlas_image = build_atlas(&textures_dir)
            .expect("build_atlas only errors on internal bugs, not missing files");
        let atlas = upload_atlas(&gpu.device, &gpu.queue, &atlas_image);

        // World-pass pipelines target the HDR offscreen view; the
        // reflection pipelines stay at the swapchain (surface) format
        // so the water shader continues to sample its reflection as
        // LDR colour. This split is the same shape as opaque_pipe vs
        // opaque_pipe_reflection.
        let opaque_pipe = build_opaque(
            &gpu.device,
            hdr::HDR_FORMAT,
            &camera_bgl,
            &chunk_bgl,
            &atlas.bind_group_layout,
            FrontFace::Ccw,
        );
        let opaque_pipe_reflection = build_opaque(
            &gpu.device,
            gpu.surface_cfg.format,
            &camera_bgl,
            &chunk_bgl,
            &atlas.bind_group_layout,
            FrontFace::Cw,
        );
        let water_pipe = build_water(
            &gpu.device,
            hdr::HDR_FORMAT,
            &camera_bgl,
            &chunk_bgl,
            &atlas.bind_group_layout,
        );
        let sky_pipe = build_sky(&gpu.device, hdr::HDR_FORMAT, &camera_bgl);
        let sky_pipe_reflection = build_sky(&gpu.device, gpu.surface_cfg.format, &camera_bgl);
        let composite_pipe = build_composite(&gpu.device, gpu.surface_cfg.format);
        let bloom = BloomChain::new(
            &gpu.device,
            gpu.surface_cfg.width,
            gpu.surface_cfg.height,
        );
        let bloom_pipes = build_bloom(&gpu.device, crate::render::bloom::BLOOM_FORMAT);

        // HUD: build the pipeline + upload the font atlas. The font
        // texture is its own resource (R8-style data but stored
        // RGBA8Unorm — see `font.rs`) so the HUD shader can use the
        // same `tex * vertex_color` math for both font and block
        // batches. The block atlas's bind group is *separate* from
        // the opaque pipeline's bind group because the HUD pipeline
        // uses its own bind-group layout (different binding indices).
        let hud_pipe = build_hud(&gpu.device, gpu.surface_cfg.format);
        let hud_screen_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hud-screen-uniform"),
                contents: bytemuck::cast_slice(&[
                    gpu.surface_cfg.width as f32,
                    gpu.surface_cfg.height as f32,
                    0.0,
                    0.0,
                ]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let hud_screen_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud-screen-bg"),
            layout: &hud_pipe.screen_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: hud_screen_buf.as_entire_binding(),
            }],
        });
        let hud_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        // Font atlas texture upload.
        let font_bytes = build_font_atlas();
        let font_size = wgpu::Extent3d {
            width: FONT_ATLAS_W,
            height: FONT_ATLAS_H,
            depth_or_array_layers: 1,
        };
        let font_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud-font-texture"),
            size: font_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Stored as plain `Rgba8Unorm` (not -Srgb) so the white
            // glyph pixels render at their authored brightness — the
            // text shouldn't be gamma-darkened the way world textures
            // are. Output is composited over the (already-srgb-encoded)
            // world by the alpha-blend pipeline state.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &font_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &font_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(FONT_ATLAS_W * 4),
                rows_per_image: Some(FONT_ATLAS_H),
            },
            font_size,
        );
        let font_view = font_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let hud_font_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud-font-bg"),
            layout: &hud_pipe.tex_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&font_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&hud_sampler) },
            ],
        });
        // Block-atlas bind group for the HUD pipeline's layout (the
        // opaque pipeline's bind group can't be reused — its layout
        // matches a *different* pipeline's bind-group layout, and
        // wgpu validates layout identity per draw).
        let atlas_view = atlas
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let hud_atlas_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud-atlas-bg"),
            layout: &hud_pipe.tex_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&atlas_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&hud_sampler) },
            ],
        });

        // Cursor highlight pipeline + buffer.
        let cursor_bgl = make_cursor_bind_group_layout(&gpu.device);
        let cursor_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cursor-uniform"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let cursor_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cursor-bg"),
            layout: &cursor_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cursor_buf.as_entire_binding(),
            }],
        });
        // Cursor is drawn into the world's HDR MSAA view (inside the
        // water pass) so it shares the HDR format.
        let cursor_pipe = build_cursor(
            &gpu.device,
            hdr::HDR_FORMAT,
            &camera_bgl,
            &cursor_bgl,
        );

        // Initial water-pass depth bind group. Recreated on resize.
        let water_depth_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-depth-bg"),
            layout: &water_pipe.depth_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_sample_view),
            }],
        });
        // Initial water-pass reflection bind group. Same lifecycle as
        // the depth one.
        let water_reflection_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-reflection-bg"),
            layout: &water_pipe.reflection_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&reflection_resolve_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&reflection_sampler),
                },
            ],
        });

        Self {
            gpu,
            depth_texture,
            depth_view,
            depth_sample_texture,
            depth_sample_view,
            hdr: hdr_target,
            water_depth_bg,
            msaa_color_view,
            reflection_camera_buf,
            reflection_camera_bg,
            reflection_msaa_view,
            reflection_resolve_texture,
            reflection_resolve_view,
            reflection_depth_view,
            reflection_sampler,
            water_reflection_bg,
            camera_buf,
            camera_bg,
            chunk_bgl,
            opaque_pipe,
            opaque_pipe_reflection,
            water_pipe,
            sky_pipe,
            sky_pipe_reflection,
            composite_pipe,
            bloom,
            bloom_pipes,
            cursor_pipe,
            cursor_buf,
            cursor_bg,
            cursor_visible: false,
            atlas,
            hud_pipe,
            hud_screen_buf,
            hud_screen_bg,
            hud_font_bg,
            hud_atlas_bg,
            _hud_font_texture: font_texture,
            _hud_font_view: font_view,
            _hud_sampler: hud_sampler,
            chunk_meshes: HashMap::new(),
            chunk_lights: HashMap::new(),
            light_sampler,
            _placeholder_light_tex: placeholder_light_tex,
            placeholder_light_view,
            last_draw_calls: std::cell::Cell::new(0),
            underwater_factor: 0.0,
        }
    }

    /// Set the `[0, 1]` underwater factor used by the shaders this and
    /// every following frame. `0` is fully above water; `1` is fully
    /// submerged. Smoothing is the caller's responsibility — pass a
    /// step function for an instant transition, or a single-frame
    /// lerp for the swim-out tint fading.
    pub fn set_underwater(&mut self, factor: f32) {
        self.underwater_factor = factor.clamp(0.0, 1.0);
    }

    /// Update the wireframe cursor target. Pass `None` to hide it (the
    /// player isn't aimed at anything within reach).
    pub fn set_cursor(&mut self, hit: Option<BlockPos>) {
        match hit {
            Some(block) => {
                let v = [
                    block.0.x as f32,
                    block.0.y as f32,
                    block.0.z as f32,
                    1.0,
                ];
                self.gpu
                    .queue
                    .write_buffer(&self.cursor_buf, 0, bytemuck::cast_slice(&v));
                self.cursor_visible = true;
            }
            None => self.cursor_visible = false,
        }
    }

    /// Current swap-chain framebuffer dimensions. Used by the HUD
    /// builder to lay out elements in pixel space.
    pub fn framebuffer_size(&self) -> (u32, u32) {
        (self.gpu.surface_cfg.width, self.gpu.surface_cfg.height)
    }

    /// Reconfigure the surface + depth texture for a new window size.
    pub fn resize(&mut self, w: u32, h: u32) {
        self.gpu.resize(w, h);
        self.hdr.recreate(&self.gpu.device, w, h);
        self.bloom.recreate(&self.gpu.device, w, h);
        let (depth_tex, depth_view) = make_depth_texture(&self.gpu.device, w, h);
        let (depth_sample_tex, depth_sample_view) =
            make_depth_sample_texture(&self.gpu.device, w, h);
        let (refl_msaa, refl_resolve_tex, refl_resolve_view) = make_reflection_color_textures(
            &self.gpu.device,
            w,
            h,
            self.gpu.surface_cfg.format,
        );
        let refl_depth = make_reflection_depth_texture(&self.gpu.device, w, h);
        // Rebuild every bind group that references a (now-stale) view.
        self.water_depth_bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-depth-bg"),
            layout: &self.water_pipe.depth_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_sample_view),
            }],
        });
        self.water_reflection_bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-reflection-bg"),
            layout: &self.water_pipe.reflection_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&refl_resolve_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.reflection_sampler),
                },
            ],
        });
        self.depth_texture = depth_tex;
        self.depth_view = depth_view;
        self.depth_sample_texture = depth_sample_tex;
        self.depth_sample_view = depth_sample_view;
        self.reflection_msaa_view = refl_msaa;
        self.reflection_resolve_texture = refl_resolve_tex;
        self.reflection_resolve_view = refl_resolve_view;
        self.reflection_depth_view = refl_depth;
        self.msaa_color_view =
            make_msaa_color_texture(&self.gpu.device, w, h, hdr::HDR_FORMAT);
        // HUD lays out in pixel space so the screen-size uniform also
        // needs the new dimensions; otherwise the HUD shrinks/expands
        // to fill the old framebuffer rect.
        self.gpu.queue.write_buffer(
            &self.hud_screen_buf,
            0,
            bytemuck::cast_slice(&[w as f32, h as f32, 0.0, 0.0]),
        );
    }

    /// Build the per-chunk bind group used as group 1 in every chunk
    /// draw. Binding 0 is the chunk-origin uniform (dynamic-offset
    /// buffer, size 16), binding 1 is the chunk's light-volume view
    /// (or the 1×1×1 placeholder when no volume has been uploaded
    /// yet), and binding 2 is the shared light sampler.
    fn make_chunk_bind_group(
        &self,
        ubuf: &wgpu::Buffer,
        light_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk-bg"),
            layout: &self.chunk_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: ubuf,
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
        })
    }

    /// Upload (or replace) the GPU mesh for chunk `coord` at LOD `lod`
    /// (0 = full resolution, 1 = 2× downsample, 2 = 4× downsample).
    /// An empty mesh clears just that one LOD slot.
    pub fn upload_chunk_mesh(&mut self, coord: ChunkCoord, lod: u8, mesh: &ChunkMesh) {
        let lod = lod as usize;
        debug_assert!(lod < 3);
        let Some(gpu_mesh) = upload_mesh(&self.gpu.device, mesh) else {
            if let Some(slots) = self.chunk_meshes.get_mut(&coord) {
                slots[lod] = None;
                // If every slot is empty (e.g. all-air chunk), drop the
                // hashmap entry entirely so iteration stays cheap.
                if slots.iter().all(|s| s.is_none()) {
                    self.chunk_meshes.remove(&coord);
                }
            }
            return;
        };
        let origin = coord.origin().0;
        let ubuf = self
            .gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk-uniform"),
                contents: bytemuck::cast_slice(&[ChunkUniform {
                    origin: [origin.x as f32, origin.y as f32, origin.z as f32, 0.0],
                }]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        // Pick the light view first so we don't hold a `&mut` borrow on
        // `self.chunk_meshes` while we still need `&self` to build the
        // bind group.
        let light_view: &wgpu::TextureView = self
            .chunk_lights
            .get(&coord)
            .map(|v| &v.view)
            .unwrap_or(&self.placeholder_light_view);
        let bind_group = self.make_chunk_bind_group(&ubuf, light_view);
        let slots = self
            .chunk_meshes
            .entry(coord)
            .or_insert_with(|| [None, None, None]);
        slots[lod] = Some(ChunkGpu {
            mesh: gpu_mesh,
            ubuf,
            bind_group,
            has_water: mesh.has_water,
        });
    }

    /// Drop *all* LOD meshes for `coord`. Called by `world_unload` after a
    /// chunk leaves the load radius.
    pub fn remove_chunk_mesh(&mut self, coord: ChunkCoord) {
        self.chunk_meshes.remove(&coord);
        self.chunk_lights.remove(&coord);
    }

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
        // The cached per-chunk bind group references the chunk's
        // light-volume `TextureView`, so each LOD slot whose mesh
        // is already uploaded needs its bind group rebuilt to point
        // at the new view. Slots not yet uploaded (mesh hasn't
        // arrived) will pick up the correct view when their mesh
        // upload lands.
        let light_view = &self.chunk_lights.get(&coord).expect("just inserted").view;
        // Build the new bind groups first (needs `&self`), then
        // assign them under `&mut self.chunk_meshes`. We can't call
        // `self.make_chunk_bind_group` while holding a `&mut` into
        // `chunk_meshes`, so collect references via an intermediate
        // pass over the slots.
        if let Some(slots) = self.chunk_meshes.get(&coord) {
            let mut new_bgs: [Option<wgpu::BindGroup>; 3] = [None, None, None];
            for (i, slot) in slots.iter().enumerate() {
                if let Some(cg) = slot {
                    new_bgs[i] = Some(self.make_chunk_bind_group(&cg.ubuf, light_view));
                }
            }
            if let Some(slots_mut) = self.chunk_meshes.get_mut(&coord) {
                for (i, bg_opt) in new_bgs.into_iter().enumerate() {
                    if let (Some(cg), Some(bg)) = (slots_mut[i].as_mut(), bg_opt) {
                        cg.bind_group = bg;
                    }
                }
            }
        }
    }

    /// Drop the per-chunk light volume (called when the chunk unloads).
    pub fn remove_chunk_light_volume(&mut self, coord: ChunkCoord) {
        self.chunk_lights.remove(&coord);
    }

    /// Number of chunk hashmap entries (one per coord, regardless of how
    /// many LOD slots are filled). Exposed for the debug HUD (M10).
    pub fn chunk_mesh_count(&self) -> usize {
        self.chunk_meshes.len()
    }

    /// Number of `draw_indexed` calls the last opaque pass issued.
    /// Updated by `encode_opaque_pass` so the HUD can see it.
    pub fn last_draw_calls(&self) -> u32 {
        self.last_draw_calls.get()
    }

    /// Per-LOD entry counts. Debug helper to see whether LOD jobs are
    /// keeping up with streaming.
    pub fn chunk_mesh_lod_counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for slots in self.chunk_meshes.values() {
            for (i, s) in slots.iter().enumerate() {
                if s.is_some() {
                    counts[i] += 1;
                }
            }
        }
        counts
    }

    /// Pick a LOD level for a chunk at world-space center `chunk_center`
    /// given a camera at `eye`. Closer chunks get LOD0 (full res); the
    /// boundaries (6 / 12 chunks) match the spec's defaults.
    fn pick_lod(eye: Vec3, chunk_center: Vec3) -> usize {
        let d = (chunk_center - eye).length();
        if d < 6.0 * 32.0 {
            0
        } else if d < 12.0 * 32.0 {
            1
        } else {
            2
        }
    }

    /// Draw a single frame. `sun_dir` is the (unit-length) world-space sun
    /// direction and `sun_intensity` is its brightness `[0, 1]`; both come
    /// from the time-of-day system. `time` is seconds since startup and
    /// drives shader-side animation (e.g. water shimmer). The optional
    /// `hud` is drawn as a separate pass over the world view.
    pub fn render(
        &mut self,
        eye: Vec3,
        yaw: f32,
        pitch: f32,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        time: f32,
        hud: Option<&HudFrame>,
    ) -> Result<(), wgpu::SurfaceError> {
        let aspect =
            self.gpu.surface_cfg.width as f32 / self.gpu.surface_cfg.height.max(1) as f32;
        let vp = view_proj(eye, yaw, pitch, 70f32.to_radians(), aspect);
        // Inverse for the sky shader's NDC → world ray reconstruction.
        // Inversion fails for a degenerate matrix; that can only happen
        // with a zero frustum, so we fall back to identity for safety.
        let inv_vp = vp.inverse();
        self.gpu.queue.write_buffer(
            &self.camera_buf,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: vp.to_cols_array_2d(),
                sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
                sun_color: [1.00, 0.96, 0.90, 0.0],
                sky_color: [0.55, 0.70, 0.95, 0.0],
                sun_intensity,
                time,
                underwater_factor: self.underwater_factor,
                clip_y_min: -1_000_000.0,
                eye: [eye.x, eye.y, eye.z, 0.0],
                inv_view_proj: inv_vp.to_cols_array_2d(),
            }]),
        );

        let frustum = extract_frustum_planes(vp);

        // Build the planar-reflection camera uniform via the
        // *reflection matrix* technique. Earlier versions of this
        // code physically moved the camera to a virtual underwater
        // position and built a separate view matrix from there —
        // that rendered the world from a fundamentally different
        // viewpoint and the resulting image did NOT line up in
        // screen space with the main view's projection of
        // water-surface points, so sampling the reflection texture
        // at `clip_pos.xy` produced content that was offset and
        // disagreed with the main view as the camera moved.
        //
        // The standard technique (LearnOpenGL planar-reflection
        // tutorial, Vulkan-Tutorial reflection example, Unreal
        // engine docs) is to keep the main camera's view + proj as-is
        // and *pre-multiply by a reflection matrix* that mirrors
        // world coordinates across the water plane. Water-plane
        // points (y = SEA_LEVEL) are invariant under that mirror so
        // they project to the *same* screen position as without it;
        // above-water content gets mirrored to below-water positions
        // and the main camera renders them at the screen positions
        // they'd actually appear at as reflections. Screen-space
        // sampling at `clip_pos.xy` then locks the reflection to
        // world content.
        //
        // The 4×4 reflection matrix across `y = H`:
        //   [ 1  0  0   0]
        //   [ 0 -1  0  2H]
        //   [ 0  0  1   0]
        //   [ 0  0  0   1]
        let sea_level = crate::worldgen::SEA_LEVEL as f32;
        let refl_mat = glam::Mat4::from_cols(
            glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, -1.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
            glam::Vec4::new(0.0, 2.0 * sea_level, 0.0, 1.0),
        );
        let refl_vp = vp * refl_mat;
        let refl_inv_vp = refl_vp.inverse();
        self.gpu.queue.write_buffer(
            &self.reflection_camera_buf,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: refl_vp.to_cols_array_2d(),
                // Sun direction stays in main-world frame: the inv_vp
                // we wrote above already encodes the reflection
                // (inv(vp * R) = R * inv(vp)), so sky-shader ray
                // reconstruction is reflected automatically and the
                // sun's image lands at the geometrically correct
                // reflected position via the usual dot product.
                sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
                sun_color: [1.00, 0.96, 0.90, 0.0],
                sky_color: [0.55, 0.70, 0.95, 0.0],
                sun_intensity,
                time,
                underwater_factor: 0.0, // reflections don't get the underwater grade
                // Drop below-water world content — the fragment's
                // `v_world` is the *unmirrored* world position, so a
                // mountain at y=100 has v_world.y = 100 (above the
                // SEA_LEVEL clip) and renders; an originally-below-
                // water cave block at y=30 has v_world.y = 30 (below
                // the clip) and discards.
                clip_y_min: sea_level,
                // Eye stays as the *main* eye — fog distance is
                // measured from the actual viewer to the actual
                // (unmirrored) world point.
                eye: [eye.x, eye.y, eye.z, 0.0],
                inv_view_proj: refl_inv_vp.to_cols_array_2d(),
            }]),
        );
        let refl_frustum = extract_frustum_planes(refl_vp);

        // Walk every loaded chunk *once* per frame and pick the visible
        // set + chosen LOD for the main-view frustum. The three render
        // passes (opaque, water, reflection) used to each re-iterate
        // `self.chunk_meshes` (10 k+ entries) and re-run the same
        // distance + frustum + LOD-selection logic — wasted CPU.
        // Walking once and indexing by `&ChunkGpu` in the per-pass loops
        // collapses that work to a single pass and lets us short-circuit
        // the reflection render when nothing visible contains water.
        let (visible_main, any_water_visible) =
            self.build_visible_chunks(eye, &frustum);

        let frame = self.gpu.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // Reflection pass (sky + opaque only, mirrored world via
        // reflection matrix). Culling uses the *main* eye + the
        // reflection-frustum (extracted from `view_proj * R`), so
        // chunks are tested for whether their unmirrored geometry
        // would be visible in the reflection's projection.
        //
        // Skip the entire pass when no on-screen chunk contains water.
        // The water shader's reflection bind group still points at
        // whatever `reflection_resolve_view` held from the last frame,
        // which is fine: the sample only matters when a water fragment
        // is shaded, and by construction no such fragment exists this
        // frame.
        if any_water_visible {
            // The reflection frustum is *different* from the main
            // frustum (it's the mirrored projection), so it culls a
            // different set of chunks. Build a second visible list
            // against it. Only worth doing when we're actually going
            // to render the pass.
            let (visible_reflection, _) =
                self.build_visible_chunks(eye, &refl_frustum);
            self.encode_reflection_pass(&mut enc, &visible_reflection);
        }
        // Main world pass (sky + opaque + water + cursor) into the
        // HDR offscreen target.
        self.encode_opaque_pass(
            &mut enc,
            &self.msaa_color_view,
            &self.hdr.view,
            &visible_main,
        );
        // Bloom chain: builds 5 mips on the HDR target. Result lives in
        // self.bloom.mips[0]; composite reads it in Task 5.
        self.encode_bloom_pass(&mut enc);
        // Composite: read HDR, write to the swapchain. Passthrough
        // today; Tasks 5/6 add tonemap + underwater tint.
        self.encode_composite_pass(&mut enc, &view);
        // HUD: uploaded once per frame into fresh vertex/index
        // buffers (the HUD layout changes every frame as FPS ticks).
        if let Some(hud) = hud {
            self.encode_hud_pass(&mut enc, &view, hud);
        }
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        frame.present();
        Ok(())
    }

    /// Walk `self.chunk_meshes` once and return the visible chunks for
    /// the given camera + frustum, along with each chunk's chosen
    /// `ChunkGpu` (with LOD selection and the "any-LOD fallback"
    /// applied). Also returns whether any of those visible chunks
    /// contain water — used by the frame loop to skip the planar-
    /// reflection pass entirely on water-free frames.
    ///
    /// Why borrow into `chunk_meshes`? The three render passes used to
    /// each independently iterate the whole HashMap and recompute the
    /// same distance + frustum + LOD checks. Borrowing into a small
    /// `Vec<(coord, &ChunkGpu)>` collapses that to a single walk and
    /// lets the per-pass loops do nothing but bind + draw.
    fn build_visible_chunks(
        &self,
        eye: Vec3,
        frustum: &[Vec4; 6],
    ) -> (Vec<(ChunkCoord, &ChunkGpu)>, bool) {
        // Distance cull matches the load radius diagonal — see the long
        // comment in `encode_opaque_pass`'s old loop. Kept here as the
        // single source of truth.
        const CULL_DISTANCE: f32 = 600.0 + 28.0;
        let cull_sq = CULL_DISTANCE * CULL_DISTANCE;
        let mut out: Vec<(ChunkCoord, &ChunkGpu)> =
            Vec::with_capacity(self.chunk_meshes.len());
        let mut any_water = false;
        for (coord, slots) in &self.chunk_meshes {
            let origin = coord.origin().0;
            let chunk_min = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
            let chunk_max = chunk_min + Vec3::splat(32.0);
            let center_f = chunk_min + Vec3::splat(16.0);
            if (center_f - eye).length_squared() > cull_sq {
                continue;
            }
            if !aabb_in_frustum(frustum, chunk_min, chunk_max) {
                continue;
            }
            let preferred = Self::pick_lod(eye, center_f);
            // Try the preferred LOD first; fall back to *any* available
            // slot so a freshly-streamed chunk that only has one LOD
            // built still draws. Same fallback the old per-pass loops
            // used — preserved verbatim so the visible output doesn't
            // change.
            let chosen = slots[preferred]
                .as_ref()
                .or_else(|| slots.iter().flatten().next());
            if let Some(cg) = chosen {
                any_water |= cg.has_water;
                out.push((*coord, cg));
            }
        }
        (out, any_water)
    }

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
        let mut one_pass = |label: &str,
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

    /// Encode the composite pass: sample the HDR target, write to the
    /// swapchain (or screenshot target). Today this is a passthrough;
    /// Tasks 5/6 fold in ACES tonemap and underwater tint. The bind
    /// group is created per-frame because the HDR view is recreated on
    /// resize — caching it would dangle.
    fn encode_composite_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite-bg"),
            layout: &self.composite_pipe.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.hdr.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.composite_pipe.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.camera_buf.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Composite covers every pixel via the fullscreen
                    // triangle, so the clear colour is a defensive
                    // fallback only.
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.composite_pipe.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Encode the HUD pass: alpha-blended 2D overlay, two draws (font
    /// + block-atlas batches). Skipped when both batches are empty.
    fn encode_hud_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        hud: &HudFrame,
    ) {
        use wgpu::util::DeviceExt;
        if hud.text.vertices.is_empty() && hud.icons.vertices.is_empty() {
            return;
        }
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hud-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // `Load` so we composite over the world pass that
                    // already populated this view.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.hud_pipe.pipeline);
        pass.set_bind_group(0, &self.hud_screen_bg, &[]);

        // Helper: stash a batch's CPU buffers into ephemeral wgpu
        // buffers, then draw them. `init_buffer` makes the lifetime
        // straightforward — the buffers are dropped at the end of the
        // pass but the encoded GPU commands hold references to them
        // via the submitted command buffer.
        let mut draw_batch = |batch: &crate::render::hud::HudBatch,
                              bind_group: &wgpu::BindGroup,
                              vbuf_holder: &mut Option<wgpu::Buffer>,
                              ibuf_holder: &mut Option<wgpu::Buffer>| {
            if batch.vertices.is_empty() {
                return;
            }
            let vbuf =
                self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hud-vbuf"),
                    contents: bytemuck::cast_slice(&batch.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ibuf =
                self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hud-ibuf"),
                    contents: bytemuck::cast_slice(&batch.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            // Bind, then keep the buffers alive in the outer scope
            // until the render-pass ends; the borrow checker would
            // otherwise drop them mid-draw.
            *vbuf_holder = Some(vbuf);
            *ibuf_holder = Some(ibuf);
            pass.set_bind_group(1, bind_group, &[]);
            pass.set_vertex_buffer(0, vbuf_holder.as_ref().unwrap().slice(..));
            pass.set_index_buffer(
                ibuf_holder.as_ref().unwrap().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..batch.indices.len() as u32, 0, 0..1);
        };

        let (mut tv, mut ti, mut bv, mut bi) = (None, None, None, None);
        draw_batch(&hud.icons, &self.hud_atlas_bg, &mut tv, &mut ti);
        draw_batch(&hud.text, &self.hud_font_bg, &mut bv, &mut bi);
        // Drop the pass before the closure-captured buffers go out
        // of scope so the encoder finishes recording.
        drop(pass);
        let _ = (tv, ti, bv, bi);
    }

    /// Encode the sky + opaque passes into `enc` against the given color
    /// view + the renderer's depth view. Reused by both the live `render`
    /// and the offscreen screenshot path.
    ///
    /// `eye` is used to pick a LOD level per chunk: closer chunks render
    /// at full resolution, distant ones at LOD1/LOD2.
    /// Render the world from the mirrored camera into
    /// `reflection_resolve_view` (resolved out of `reflection_msaa_view`).
    /// Only sky + opaque geometry is drawn — water never reflects itself,
    /// and the HUD/cursor obviously don't reflect either. The opaque
    /// shader's `clip_y_min = SEA_LEVEL` rule drops anything below water
    /// so the reflected image is the upper world only.
    fn encode_reflection_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        visible: &[(ChunkCoord, &ChunkGpu)],
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("reflection-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.reflection_msaa_view,
                // Resolve straight into the single-sample texture
                // the water shader samples. We never re-use the MSAA
                // contents after the resolve.
                resolve_target: Some(&self.reflection_resolve_view),
                ops: wgpu::Operations {
                    // Sky shader fills every pixel so this clear
                    // colour shouldn't show, but harmless to pick a
                    // sky-ish blue as a safety net.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.20,
                        g: 0.40,
                        b: 0.80,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.reflection_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // 1) Sky — same triangle, but the mirrored inv-view-proj
        // reconstructs rays going DOWN from the mirrored eye toward
        // what was originally above the water. The sun position is
        // also mirrored in the camera uniform so the reflected sun
        // appears at the geometrically correct spot.
        //
        // Uses `sky_pipe_reflection` (surface format) — the reflection
        // texture is LDR because the water shader samples it as plain
        // colour, not HDR linear.
        pass.set_pipeline(&self.sky_pipe_reflection.pipeline);
        pass.set_bind_group(0, &self.reflection_camera_bg, &[]);
        pass.draw(0..3, 0..1);

        // 2) Opaque chunks via the REFLECTION-WINDING pipeline
        // (front_face = Cw). Without this flip, the mirror eye's
        // view sees every world-facing triangle from the back side
        // — back-face cull kicks in and drops the entire visible
        // world, leaving only the sky in the reflection. This was
        // the actual cause of "reflection content shifts wildly
        // with camera motion": the reflection texture was sky-only
        // and small camera moves caused big sky-region shifts in
        // the sampled output.
        pass.set_pipeline(&self.opaque_pipe_reflection.pipeline);
        pass.set_bind_group(0, &self.reflection_camera_bg, &[]);
        pass.set_bind_group(2, &self.atlas.bind_group, &[]);
        // The caller already culled by the reflection frustum and
        // picked an LOD per chunk; just bind + draw using the cached
        // per-chunk bind group.
        for (_, cg) in visible {
            pass.set_bind_group(1, &cg.bind_group, &[0]);
            pass.set_vertex_buffer(0, cg.mesh.vbuf.slice(..));
            pass.set_index_buffer(cg.mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..cg.mesh.index_count, 0, 0..1);
        }
    }

    fn encode_opaque_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        msaa_view: &wgpu::TextureView,
        resolve_view: &wgpu::TextureView,
        visible: &[(ChunkCoord, &ChunkGpu)],
    ) {
        // PASS A: sky + opaque. Writes MSAA colour + MSAA depth.
        // Does NOT resolve yet — the water pass below loads the MSAA
        // samples, blends water on top, and does the final resolve.
        // The depth texture also gets stored so we can copy it into
        // `depth_sample_texture` (read by the water shader for foam
        // and depth-tint) between the two passes.
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sky+opaque-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // The sky pass overwrites every pixel, so this clear
                    // color only shows in degenerate frames (e.g. before
                    // the sky pipeline is even set up).
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // 1) Sky: full-screen triangle at far plane, no depth write.
        pass.set_pipeline(&self.sky_pipe.pipeline);
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.draw(0..3, 0..1);

        // 2) Opaque chunks. The visible-chunk list was built by
        // `build_visible_chunks` in `render()` — see its doc-comment
        // for why a single shared walk replaces the per-pass loops.
        pass.set_pipeline(&self.opaque_pipe.pipeline);
        pass.set_bind_group(0, &self.camera_bg, &[]);
        // Atlas (group 2) is shared by every chunk draw — bind once
        // outside the per-chunk loop. Per-chunk uniform (group 1) still
        // varies per draw and is set inside the loop below.
        pass.set_bind_group(2, &self.atlas.bind_group, &[]);
        let mut draws: u32 = 0;
        for (_, cg) in visible {
            pass.set_bind_group(1, &cg.bind_group, &[0]);
            pass.set_vertex_buffer(0, cg.mesh.vbuf.slice(..));
            pass.set_index_buffer(cg.mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..cg.mesh.index_count, 0, 0..1);
            draws += 1;
        }
        self.last_draw_calls.set(draws);

        // End PASS A so the depth attachment is no longer in use —
        // we can't sample a texture that's bound as an attachment in
        // an active render pass.
        drop(pass);

        // Copy the just-written depth into the sampleable depth
        // texture so the water shader can read terrain depth and
        // compute foam + depth-tint per fragment. MSAA → MSAA
        // 1:1 copy; both textures are 4× sample-count.
        let (w, h) = (self.gpu.surface_cfg.width, self.gpu.surface_cfg.height);
        enc.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::ImageCopyTexture {
                texture: &self.depth_sample_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        // PASS B: water + cursor. Loads MSAA colour and depth from
        // pass A, blends water on top with alpha, then resolves the
        // multisampled colour into `resolve_view` (the swapchain or
        // screenshot target) at end of pass.
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("water-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_view,
                resolve_target: Some(resolve_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    // No depth write from the water pipeline anyway
                    // (depth_write_enabled = false), but we still
                    // need to store so the next frame's clear can
                    // happen at a defined state.
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Water draws. Same vertex buffers as the opaque pass; the
        // water shader discards every non-water fragment. Group 3
        // gives the shader the scene-depth sampler it needs for
        // foam + depth-tint. Reuses the same visible list — chunks
        // without water still go through the shader, but every
        // fragment discards so there's no GPU-side cost beyond the
        // vertex transform.
        pass.set_pipeline(&self.water_pipe.pipeline);
        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(2, &self.atlas.bind_group, &[]);
        pass.set_bind_group(3, &self.water_depth_bg, &[]);
        pass.set_bind_group(4, &self.water_reflection_bg, &[]);
        for (_, cg) in visible {
            pass.set_bind_group(1, &cg.bind_group, &[0]);
            pass.set_vertex_buffer(0, cg.mesh.vbuf.slice(..));
            pass.set_index_buffer(cg.mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..cg.mesh.index_count, 0, 0..1);
        }

        // Cursor wireframe — drawn last so it overlays both opaque
        // and water without depth-fighting against either.
        if self.cursor_visible {
            pass.set_pipeline(&self.cursor_pipe.pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_bind_group(1, &self.cursor_bg, &[]);
            pass.draw(0..24, 0..1);
        }
    }

    /// Render one frame into the supplied texture view + the renderer's
    /// depth attachment. Used by the screenshot path so it can target an
    /// offscreen texture instead of the swap chain.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_view(
        &self,
        target: &wgpu::TextureView,
        eye: Vec3,
        yaw: f32,
        pitch: f32,
        aspect: f32,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        time: f32,
        hud: Option<&HudFrame>,
    ) {
        let vp = view_proj(eye, yaw, pitch, 70f32.to_radians(), aspect);
        let inv_vp = vp.inverse();
        self.gpu.queue.write_buffer(
            &self.camera_buf,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: vp.to_cols_array_2d(),
                sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
                sun_color: [1.00, 0.96, 0.90, 0.0],
                sky_color: [0.55, 0.70, 0.95, 0.0],
                sun_intensity,
                time,
                underwater_factor: self.underwater_factor,
                clip_y_min: -1_000_000.0,
                eye: [eye.x, eye.y, eye.z, 0.0],
                inv_view_proj: inv_vp.to_cols_array_2d(),
            }]),
        );

        let frustum = extract_frustum_planes(vp);
        let (visible_main, _) = self.build_visible_chunks(eye, &frustum);
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // Screenshot targets are sized to match the surface
        // (`capture_offscreen` in `main.rs` clones the
        // `surface_cfg`), so the renderer's existing MSAA colour
        // view is the right shape to resolve into HdrTarget.
        // Future callers that pass an off-size target would need to
        // allocate their own MSAA + HDR intermediates.
        self.encode_opaque_pass(
            &mut enc,
            &self.msaa_color_view,
            &self.hdr.view,
            &visible_main,
        );
        self.encode_composite_pass(&mut enc, target);
        if let Some(hud) = hud {
            self.encode_hud_pass(&mut enc, target, hud);
        }
        self.gpu.queue.submit(std::iter::once(enc.finish()));
    }
}
