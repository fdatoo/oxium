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
pub mod gpu;
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
use crate::render::gpu::{make_depth_texture, Gpu};
use crate::render::mesh::{upload_mesh, GpuMesh};
use crate::render::pipelines::cursor::{
    build as build_cursor, make_cursor_bind_group_layout, CursorPipeline,
};
use crate::render::pipelines::opaque::{build as build_opaque, OpaquePipeline};
use crate::render::pipelines::sky::{build as build_sky, SkyPipeline};
use crate::voxel::coords::{BlockPos, ChunkCoord};
use glam::Vec3;
use wgpu::util::DeviceExt;

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

    /// Up to three GPU meshes per loaded chunk — one per LOD level
    /// (`[L0, L1, L2]`). The render loop picks which slot to draw based
    /// on the chunk's distance to the camera, falling back to the
    /// nearest available LOD if a job hasn't finished yet.
    chunk_meshes: HashMap<ChunkCoord, [Option<ChunkGpu>; 3]>,
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
    /// Initialize the renderer on the given window.
    pub fn new(window: Arc<Window>) -> Self {
        let gpu = Gpu::new(window);
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
            chunk_meshes: HashMap::new(),
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

    /// Reconfigure the surface + depth texture for a new window size.
    pub fn resize(&mut self, w: u32, h: u32) {
        self.gpu.resize(w, h);
        self.depth_view = make_depth_texture(&self.gpu.device, w, h);
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
    /// drives shader-side animation (e.g. water shimmer).
    pub fn render(
        &self,
        eye: Vec3,
        yaw: f32,
        pitch: f32,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        time: f32,
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

        let frame = self.gpu.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_opaque_pass(&mut enc, &view, eye);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        frame.present();
        Ok(())
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
        for (coord, slots) in &self.chunk_meshes {
            let center = coord.origin().0;
            let center_f = Vec3::new(
                center.x as f32 + 16.0,
                center.y as f32 + 16.0,
                center.z as f32 + 16.0,
            );
            let preferred = Self::pick_lod(eye, center_f);
            // Try preferred → lower-detail neighbour → higher-detail
            // neighbour so the chunk is never invisible when *some* LOD
            // is ready.
            let chosen = slots[preferred]
                .as_ref()
                .or_else(|| slots[preferred.saturating_sub(1)].as_ref())
                .or_else(|| slots[(preferred + 1).min(2)].as_ref());
            if let Some(cg) = chosen {
                pass.set_bind_group(1, &cg.bg, &[0]);
                pass.set_vertex_buffer(0, cg.mesh.vbuf.slice(..));
                pass.set_index_buffer(cg.mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..cg.mesh.index_count, 0, 0..1);
            }
        }

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

        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_opaque_pass(&mut enc, target, eye);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
    }
}
