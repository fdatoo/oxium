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

use std::collections::HashMap;
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
use crate::voxel::coords::ChunkCoord;
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

    /// One GPU mesh + a per-chunk uniform buffer per loaded chunk (LOD0
    /// only — LODs 1 and 2 arrive in M8).
    chunk_meshes: HashMap<ChunkCoord, ChunkGpu>,
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
            chunk_bgl,
            opaque_pipe,
            chunk_meshes: HashMap::new(),
        }
    }

    /// Reconfigure the surface + depth texture for a new window size.
    pub fn resize(&mut self, w: u32, h: u32) {
        self.gpu.resize(w, h);
        self.depth_view = make_depth_texture(&self.gpu.device, w, h);
    }

    /// Upload (or replace) the GPU mesh for chunk `coord`. If the mesh is
    /// empty (no visible faces), removes any existing entry — useful so
    /// remeshing an all-air chunk doesn't leave a stale draw call behind.
    pub fn upload_chunk_mesh(&mut self, coord: ChunkCoord, mesh: &ChunkMesh) {
        let Some(gpu_mesh) = upload_mesh(&self.gpu.device, mesh) else {
            self.chunk_meshes.remove(&coord);
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
        self.chunk_meshes.insert(
            coord,
            ChunkGpu {
                mesh: gpu_mesh,
                _ubuf: ubuf,
                bg,
            },
        );
    }

    /// Drop the GPU mesh + uniform for `coord`. Called by `world_unload`
    /// after a chunk leaves the load radius.
    pub fn remove_chunk_mesh(&mut self, coord: ChunkCoord) {
        self.chunk_meshes.remove(&coord);
    }

    /// Number of chunk meshes currently held by the GPU. Exposed for the
    /// debug HUD (M10).
    pub fn chunk_mesh_count(&self) -> usize {
        self.chunk_meshes.len()
    }

    /// Draw a single frame.
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

    /// Encode the opaque pass into `enc` against the given color view +
    /// the renderer's depth view. Reused by both the live `render` and
    /// the offscreen screenshot path.
    fn encode_opaque_pass(&self, enc: &mut wgpu::CommandEncoder, color_view: &wgpu::TextureView) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("opaque-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
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

        pass.set_pipeline(&self.opaque_pipe.pipeline);
        pass.set_bind_group(0, &self.camera_bg, &[]);
        // Each chunk has its own bind group at offset 0. M3 doesn't sort or
        // frustum-cull yet — both arrive in M8 — so we just iterate the
        // hashmap in arbitrary order.
        for cg in self.chunk_meshes.values() {
            pass.set_bind_group(1, &cg.bg, &[0]);
            pass.set_vertex_buffer(0, cg.mesh.vbuf.slice(..));
            pass.set_index_buffer(cg.mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..cg.mesh.index_count, 0, 0..1);
        }
    }

    /// Render one frame into the supplied texture view + the renderer's
    /// depth attachment. Used by the screenshot path so it can target an
    /// offscreen texture instead of the swap chain.
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

        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_opaque_pass(&mut enc, target);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
    }
}
