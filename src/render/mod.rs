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

pub mod camera;
pub mod gpu;
pub mod mesh;
pub mod pipelines;
pub mod screenshot;

use std::sync::Arc;
use winit::window::Window;

use crate::mesher::ChunkMesh;
use crate::render::camera::{
    make_camera_bind_group_layout, make_camera_buffer, make_chunk_bind_group_layout, view_proj,
    CameraUniform, ChunkUniform,
};
use crate::render::gpu::{make_depth_texture, Gpu};
use crate::render::mesh::{upload_mesh, GpuMesh};
use crate::render::pipelines::opaque::{build as build_opaque, OpaquePipeline};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Top-level rendering object. Owns the `wgpu` state, the per-frame
/// uniforms (camera + chunk), the opaque draw pipeline, and (for M1)
/// a single test mesh.
///
/// As later milestones land, the single `test_mesh` slot is replaced by a
/// `HashMap<ChunkCoord, [Option<GpuMesh>; 3]>` (M3+) and additional pipelines
/// (sky, translucent, cursor highlight, HUD).
pub struct Renderer {
    pub gpu: Gpu,
    depth_view: wgpu::TextureView,
    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    chunk_buf: wgpu::Buffer,
    chunk_bg: wgpu::BindGroup,
    opaque_pipe: OpaquePipeline,
    test_mesh: Option<GpuMesh>,
}

impl Renderer {
    /// Initialize the renderer on the given window. Performs adapter, device
    /// and surface setup; builds the opaque pipeline; creates uniform buffers
    /// and depth target sized to the window.
    pub fn new(window: Arc<Window>) -> Self {
        let gpu = Gpu::new(window);

        let depth_view =
            make_depth_texture(&gpu.device, gpu.surface_cfg.width, gpu.surface_cfg.height);

        // Camera uniform: one mat4 per frame, written by `render`.
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

        // Per-chunk uniform: the chunk's world-space origin. M1 only draws
        // one mesh at origin (0,0,0); M3 grows this into a dynamic-offset
        // ring buffer indexed per draw call.
        let chunk_bgl = make_chunk_bind_group_layout(&gpu.device);
        let chunk_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-uniform"),
            contents: bytemuck::cast_slice(&[ChunkUniform { origin: [0.0; 4] }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chunk_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk-bg"),
            layout: &chunk_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                // The layout was declared with `has_dynamic_offset = true` so
                // the binding range is fixed to a single 16-byte slot.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &chunk_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(16),
                }),
            }],
        });

        let opaque_pipe = build_opaque(
            &gpu.device,
            gpu.surface_cfg.format,
            &camera_bgl,
            &chunk_bgl,
        );

        Self {
            gpu,
            depth_view,
            camera_buf,
            camera_bg,
            chunk_buf,
            chunk_bg,
            opaque_pipe,
            test_mesh: None,
        }
    }

    /// Upload a test chunk mesh to the GPU and remember it for `render`.
    /// In M3 this is replaced by `upload_chunk(coord, mesh)`.
    pub fn upload_test_mesh(&mut self, mesh: &ChunkMesh) {
        self.test_mesh = upload_mesh(&self.gpu.device, mesh);
    }

    /// Resize the surface + depth texture in response to a window resize.
    pub fn resize(&mut self, w: u32, h: u32) {
        self.gpu.resize(w, h);
        self.depth_view = make_depth_texture(&self.gpu.device, w, h);
    }

    /// Draw a single frame from the given camera state. Writes per-frame
    /// uniforms, acquires a swap texture, encodes the opaque pass, and
    /// presents.
    pub fn render(&self, eye: Vec3, yaw: f32, pitch: f32) -> Result<(), wgpu::SurfaceError> {
        let aspect =
            self.gpu.surface_cfg.width as f32 / self.gpu.surface_cfg.height.max(1) as f32;
        let vp = view_proj(eye, yaw, pitch, 70f32.to_radians(), aspect);
        self.gpu.queue.write_buffer(
            &self.camera_buf,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: vp.to_cols_array_2d(),
            }]),
        );
        self.gpu.queue.write_buffer(
            &self.chunk_buf,
            0,
            bytemuck::cast_slice(&[ChunkUniform {
                origin: [0.0, 0.0, 0.0, 0.0],
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

        self.encode_opaque_pass(&mut enc, &view);

        self.gpu.queue.submit(std::iter::once(enc.finish()));
        frame.present();
        Ok(())
    }

    /// Encode the opaque draw pass into `enc`, writing to `color_view` and
    /// the renderer's own depth view. Shared between window rendering and
    /// the offscreen screenshot path so they stay visually identical.
    fn encode_opaque_pass(&self, enc: &mut wgpu::CommandEncoder, color_view: &wgpu::TextureView) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("opaque-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Sky-blue clear so an "empty" frame is visually
                    // distinguishable from a black-window crash.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.55,
                        g: 0.78,
                        b: 1.0,
                        a: 1.0,
                    }),
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

        if let Some(m) = &self.test_mesh {
            pass.set_pipeline(&self.opaque_pipe.pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            // The chunk bind group was declared with a dynamic offset (so it
            // can be reused for many chunks in M3); for M1's single chunk
            // the offset is always 0.
            pass.set_bind_group(1, &self.chunk_bg, &[0]);
            pass.set_vertex_buffer(0, m.vbuf.slice(..));
            pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..1);
        }
    }

    /// Render exactly one frame from `(eye, yaw, pitch)` straight into the
    /// supplied texture *view*, with the renderer's depth attachment. Used
    /// by the screenshot path, which needs a `COPY_SRC`-capable texture
    /// rather than a swap-chain frame.
    pub fn render_to_view(
        &self,
        target: &wgpu::TextureView,
        eye: Vec3,
        yaw: f32,
        pitch: f32,
        aspect: f32,
    ) {
        let vp = view_proj(eye, yaw, pitch, 70f32.to_radians(), aspect);
        self.gpu.queue.write_buffer(
            &self.camera_buf,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: vp.to_cols_array_2d(),
            }]),
        );
        self.gpu.queue.write_buffer(
            &self.chunk_buf,
            0,
            bytemuck::cast_slice(&[ChunkUniform {
                origin: [0.0, 0.0, 0.0, 0.0],
            }]),
        );

        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_opaque_pass(&mut enc, target);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
    }
}
