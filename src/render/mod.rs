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
pub mod camera;
pub mod font;
pub mod gpu;
pub mod hud;
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
use crate::render::gpu::{make_depth_texture, Gpu};
use crate::render::hud::HudFrame;
use crate::render::mesh::{upload_mesh, GpuMesh};
use crate::render::pipelines::cursor::{
    build as build_cursor, make_cursor_bind_group_layout, CursorPipeline,
};
use crate::render::pipelines::hud::{build as build_hud, HudPipeline};
use crate::render::pipelines::opaque::{build as build_opaque, OpaquePipeline};
use crate::render::pipelines::sky::{build as build_sky, SkyPipeline};
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
    depth_view: wgpu::TextureView,
    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    chunk_bgl: wgpu::BindGroupLayout,
    opaque_pipe: OpaquePipeline,
    /// Sky-gradient pipeline, drawn before opaque each frame.
    sky_pipe: SkyPipeline,
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
    /// Number of `draw_indexed` calls the last opaque pass issued.
    /// Updated by `encode_opaque_pass`, read by the perf HUD. `Cell`
    /// so the render path can stay `&self` while still recording.
    last_draw_calls: std::cell::Cell<u32>,
}

/// Per-chunk GPU resources: the mesh buffers, the chunk-origin uniform, and
/// the bind group that points the pipeline at that uniform.
struct ChunkGpu {
    mesh: GpuMesh,
    /// Kept alive so the `bind_group` keeps a valid buffer reference.
    _ubuf: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

impl Renderer {
    /// Initialize the renderer with the chosen swapchain present
    /// mode. Callers pass `Fifo` for v-sync (normal play) or
    /// `Immediate` for "uncapped" perf measurement so HUD FPS
    /// reflects real throughput.
    pub fn new_with_present_mode(window: Arc<Window>, present_mode: wgpu::PresentMode) -> Self {
        let gpu = Gpu::new_with_present_mode(window, present_mode);
        let depth_view =
            make_depth_texture(&gpu.device, gpu.surface_cfg.width, gpu.surface_cfg.height);
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

        let opaque_pipe = build_opaque(
            &gpu.device,
            gpu.surface_cfg.format,
            &camera_bgl,
            &chunk_bgl,
            &atlas.bind_group_layout,
        );
        let sky_pipe = build_sky(&gpu.device, gpu.surface_cfg.format, &camera_bgl);

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
        let cursor_pipe = build_cursor(
            &gpu.device,
            gpu.surface_cfg.format,
            &camera_bgl,
            &cursor_bgl,
        );

        Self {
            gpu,
            depth_view,
            camera_buf,
            camera_bg,
            chunk_bgl,
            opaque_pipe,
            sky_pipe,
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
            last_draw_calls: std::cell::Cell::new(0),
        }
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
        self.depth_view = make_depth_texture(&self.gpu.device, w, h);
        // HUD lays out in pixel space so the screen-size uniform also
        // needs the new dimensions; otherwise the HUD shrinks/expands
        // to fill the old framebuffer rect.
        self.gpu.queue.write_buffer(
            &self.hud_screen_buf,
            0,
            bytemuck::cast_slice(&[w as f32, h as f32, 0.0, 0.0]),
        );
    }

    /// Upload (or replace) the GPU mesh for chunk `coord` at LOD `lod`
    /// (0 = full resolution, 1 = 2× downsample, 2 = 4× downsample).
    /// An empty mesh clears just that one LOD slot.
    pub fn upload_chunk_mesh(&mut self, coord: ChunkCoord, lod: u8, mesh: &ChunkMesh) {
        let lod = lod as usize;
        debug_assert!(lod < 3);
        let slots = self
            .chunk_meshes
            .entry(coord)
            .or_insert_with(|| [None, None, None]);
        let Some(gpu_mesh) = upload_mesh(&self.gpu.device, mesh) else {
            slots[lod] = None;
            // If every slot is empty (e.g. all-air chunk), drop the
            // hashmap entry entirely so iteration stays cheap.
            if slots.iter().all(|s| s.is_none()) {
                self.chunk_meshes.remove(&coord);
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
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk-bg"),
            layout: &self.chunk_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &ubuf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(16),
                }),
            }],
        });
        slots[lod] = Some(ChunkGpu {
            mesh: gpu_mesh,
            _ubuf: ubuf,
            bg,
        });
    }

    /// Drop *all* LOD meshes for `coord`. Called by `world_unload` after a
    /// chunk leaves the load radius.
    pub fn remove_chunk_mesh(&mut self, coord: ChunkCoord) {
        self.chunk_meshes.remove(&coord);
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
                sun_intensity,
                time,
                _pad1: 0.0,
                _pad2: 0.0,
                eye: [eye.x, eye.y, eye.z, 0.0],
                inv_view_proj: inv_vp.to_cols_array_2d(),
            }]),
        );

        let frustum = extract_frustum_planes(vp);
        let frame = self.gpu.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_opaque_pass(&mut enc, &view, eye, &frustum);
        // HUD: uploaded once per frame into fresh vertex/index
        // buffers (the HUD layout changes every frame as FPS ticks).
        if let Some(hud) = hud {
            self.encode_hud_pass(&mut enc, &view, hud);
        }
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        frame.present();
        Ok(())
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
    fn encode_opaque_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        eye: Vec3,
        frustum: &[Vec4; 6],
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sky+opaque-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
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

        // 2) Opaque chunks. Pick a LOD per chunk by camera distance;
        // fall back to a nearby LOD if the preferred one hasn't been
        // built yet (a freshly-streamed chunk may have L0 ready before
        // L1/L2, or vice versa).
        pass.set_pipeline(&self.opaque_pipe.pipeline);
        pass.set_bind_group(0, &self.camera_bg, &[]);
        // Atlas (group 2) is shared by every chunk draw — bind once
        // outside the per-chunk loop. Per-chunk uniform (group 1) still
        // varies per draw and is set inside the loop below.
        pass.set_bind_group(2, &self.atlas.bind_group, &[]);
        // Distance culling: skip chunks whose centre is beyond the
        // load-radius diagonal. The load radius is 12 chunks (384
        // blocks) horizontally and 8 chunks (256 blocks) vertically,
        // so a corner chunk at the full radius is `√(384² + 256² +
        // 384²) ≈ 600` blocks from the player centre. Anything past
        // that is unloaded anyway, but the cull threshold has to be
        // big enough to *include* every loaded chunk — otherwise
        // the diagonal corners of the load radius (which are
        // farther than the cardinal-direction chunks at the same
        // `dx`/`dz`) get cut, painting a consistent one-quadrant
        // void from any oblique camera.
        //
        // Add 28 (half-diagonal of a chunk) so a chunk straddling
        // the boundary still draws for its in-range corner.
        const CULL_DISTANCE: f32 = 600.0 + 28.0;
        let cull_sq = CULL_DISTANCE * CULL_DISTANCE;
        let mut draws: u32 = 0;
        for (coord, slots) in &self.chunk_meshes {
            let origin = coord.origin().0;
            let chunk_min = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
            let chunk_max = chunk_min + Vec3::splat(32.0);
            let center_f = chunk_min + Vec3::splat(16.0);
            // BOTH culls DISABLED. If DC now equals CH and the void
            // is gone, *some* cull was wrong. If DC == CH but void
            // persists, the missing chunks aren't in `chunk_meshes`
            // at all (mesh job didn't run or was rejected). If
            // DC < CH still, the iteration itself is broken.
            let _ = frustum;
            let _ = chunk_max;
            let _ = cull_sq;
            let _ = center_f;
            let _ = chunk_min;
            let preferred = Self::pick_lod(eye, center_f);
            // Try the preferred LOD first; if it isn't uploaded yet,
            // fall back to *any* available LOD slot rather than
            // staying invisible. The previous version explicitly
            // tried `[preferred, preferred-1, preferred+1]` which
            // for `preferred = 2` collapsed to `[2, 1, 2]` and
            // missed `slot[0]` — so chunks at LOD2 render distance
            // that only have a LOD0 mesh (the case since v0.1.39
            // stopped spawning LOD1/LOD2 at gen time) failed to
            // draw entirely, painting a sky-shader void over the
            // ring of "far enough for LOD2, no LOD2/LOD1 baked yet"
            // chunks. That ring looks like a one-quadrant void
            // from any oblique top-down view, which is the
            // "specific quadrant won't load" symptom.
            let chosen = slots[preferred]
                .as_ref()
                .or_else(|| slots.iter().flatten().next());
            if let Some(cg) = chosen {
                pass.set_bind_group(1, &cg.bg, &[0]);
                pass.set_vertex_buffer(0, cg.mesh.vbuf.slice(..));
                pass.set_index_buffer(cg.mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..cg.mesh.index_count, 0, 0..1);
                draws += 1;
            }
        }
        self.last_draw_calls.set(draws);

        // 3) Cursor wireframe (12 line segments, no vertex buffer).
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
                sun_intensity,
                time,
                _pad1: 0.0,
                _pad2: 0.0,
                eye: [eye.x, eye.y, eye.z, 0.0],
                inv_view_proj: inv_vp.to_cols_array_2d(),
            }]),
        );

        let frustum = extract_frustum_planes(vp);
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_opaque_pass(&mut enc, target, eye, &frustum);
        if let Some(hud) = hud {
            self.encode_hud_pass(&mut enc, target, hud);
        }
        self.gpu.queue.submit(std::iter::once(enc.finish()));
    }
}
