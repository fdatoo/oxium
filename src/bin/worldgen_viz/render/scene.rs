//! 3D scene pipeline: takes any `Camera` and a collection of mesh buffers.

use crate::camera::Camera;
use bytemuck::{Pod, Zeroable};
use oxium::voxel::coords::ChunkCoord;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    /// xyz = selected chunk coord (as floats — keeps the uniform
    /// layout aligned without int-vector padding gymnastics).
    /// w = 1.0 if a chunk is currently pinned, 0.0 otherwise.
    selected_chunk: [f32; 4],
    /// x = elapsed seconds since startup. Drives the pulse on the
    /// pinned-chunk tint. yzw reserved.
    time: [f32; 4],
}

pub struct ChunkGpuBuffers {
    pub vbo: wgpu::Buffer,
    pub ibo: wgpu::Buffer,
    pub index_count: u32,
}

pub struct SceneRenderer {
    pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    depth_format: wgpu::TextureFormat,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    chunks: HashMap<ChunkCoord, ChunkGpuBuffers>,
    /// Mirror of the most-recently-uploaded camera matrix. Used by
    /// the render pass for CPU-side frustum culling — `chunks` may
    /// hold thousands of chunks accumulated as the user pans around;
    /// drawing them all every frame is wasted GPU work. We keep them
    /// in memory (so they pop back in instantly when the camera
    /// turns) but only issue draw calls for those whose 32³ AABB
    /// intersects the camera frustum.
    last_view_proj: glam::Mat4,
}

impl SceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // The camera UBO now feeds both stages: vertex uses
                // view_proj, fragment uses selected_chunk + time for
                // the pinned-chunk pulse tint.
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera ubo"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bg"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viz shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viz pl"),
            bind_group_layouts: &[&camera_layout],
            push_constant_ranges: &[],
        });
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viz pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // The viz mesher emits all face quads in CW order
                // viewed from outside (each face's first-triangle
                // cross product points INTO the cube). Mark CW as
                // front-facing and cull back-faces: with cull_mode:
                // None and both sides drawn, coplanar front+back of
                // every face fight for the depth test on every
                // pixel — a pronounced z-fighting that shows up
                // especially with the cutaway peeled open.
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let depth_view = create_depth(device, depth_format, width, height);
        Self {
            pipeline,
            depth_view,
            depth_format,
            camera_buffer,
            camera_bind_group,
            chunks: HashMap::new(),
            last_view_proj: glam::Mat4::IDENTITY,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth_view = create_depth(device, self.depth_format, width, height);
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn update_camera(
        &mut self,
        queue: &wgpu::Queue,
        cam: &dyn Camera,
        aspect: f32,
        selected_chunk: Option<ChunkCoord>,
        time_seconds: f32,
        cutaway_max_y: f32,
    ) {
        let vp = cam.view_proj(aspect);
        let selected = match selected_chunk {
            Some(c) => [c.0.x as f32, c.0.y as f32, c.0.z as f32, 1.0],
            None => [0.0, 0.0, 0.0, 0.0],
        };
        let u = CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            selected_chunk: selected,
            time: [time_seconds, cutaway_max_y, 0.0, 0.0],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[u]));
        self.last_view_proj = vp;
    }

    /// Upload (or replace) the GPU buffers for one chunk's mesh.
    pub fn upload_chunk(
        &mut self,
        device: &wgpu::Device,
        coord: ChunkCoord,
        vertices: &[Vertex],
        indices: &[u32],
    ) {
        if indices.is_empty() {
            self.chunks.remove(&coord);
            return;
        }
        let vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk vbo"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk ibo"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        self.chunks.insert(
            coord,
            ChunkGpuBuffers {
                vbo,
                ibo,
                index_count: indices.len() as u32,
            },
        );
    }

    pub fn drop_chunk(&mut self, coord: ChunkCoord) {
        self.chunks.remove(&coord);
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    /// Coords of every chunk currently holding GPU buffers. Used by
    /// the post-edit refill path: each gets re-requested so its mesh
    /// updates against the new config without the screen ever
    /// flashing blank.
    pub fn chunk_coords(&self) -> Vec<ChunkCoord> {
        self.chunks.keys().copied().collect()
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for (coord, buf) in &self.chunks {
            if !chunk_in_frustum(self.last_view_proj, *coord) {
                continue;
            }
            pass.set_vertex_buffer(0, buf.vbo.slice(..));
            pass.set_index_buffer(buf.ibo.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..buf.index_count, 0, 0..1);
        }
    }

    /// Number of chunks currently in the GPU set. For status display.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Number of chunks the frustum culler would draw right now.
    /// O(n) — used by the status bar, not the per-frame render path
    /// (which already inlines the test).
    pub fn visible_chunk_count(&self) -> usize {
        self.chunks
            .keys()
            .filter(|c| chunk_in_frustum(self.last_view_proj, **c))
            .count()
    }

}

/// CPU-side frustum culling for a 32³ chunk. Projects all 8 AABB
/// corners through `view_proj`; if every corner is outside the same
/// frustum plane (left / right / bottom / top / near / far), the
/// chunk is fully outside and we skip the draw.
///
/// This is the standard "Lengyel" trick. It has a known false-positive
/// case (a chunk diagonally crossing the frustum can have all 8
/// corners outside without itself being outside), but for our small
/// chunks-vs-large-frustum ratio that case is extremely rare and the
/// cost of admitting one extra chunk to the draw queue is small.
fn chunk_in_frustum(view_proj: glam::Mat4, coord: ChunkCoord) -> bool {
    use oxium::voxel::coords::CHUNK_DIM_U;
    let dim = CHUNK_DIM_U as i32;
    let min = (coord.0 * dim).as_vec3();
    let dim_f = dim as f32;
    let corners = [
        glam::Vec3::new(min.x, min.y, min.z),
        glam::Vec3::new(min.x + dim_f, min.y, min.z),
        glam::Vec3::new(min.x, min.y + dim_f, min.z),
        glam::Vec3::new(min.x + dim_f, min.y + dim_f, min.z),
        glam::Vec3::new(min.x, min.y, min.z + dim_f),
        glam::Vec3::new(min.x + dim_f, min.y, min.z + dim_f),
        glam::Vec3::new(min.x, min.y + dim_f, min.z + dim_f),
        glam::Vec3::new(min.x + dim_f, min.y + dim_f, min.z + dim_f),
    ];
    let mut left = 0;
    let mut right = 0;
    let mut bottom = 0;
    let mut top = 0;
    let mut near_plane = 0;
    let mut far_plane = 0;
    for c in &corners {
        let clip = view_proj * glam::Vec4::new(c.x, c.y, c.z, 1.0);
        if clip.x < -clip.w {
            left += 1;
        }
        if clip.x > clip.w {
            right += 1;
        }
        if clip.y < -clip.w {
            bottom += 1;
        }
        if clip.y > clip.w {
            top += 1;
        }
        if clip.z < 0.0 {
            near_plane += 1;
        }
        if clip.z > clip.w {
            far_plane += 1;
        }
    }
    !(left == 8 || right == 8 || bottom == 8 || top == 8 || near_plane == 8 || far_plane == 8)
}

fn create_depth(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    w: u32,
    h: u32,
) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viz depth"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}
