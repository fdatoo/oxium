//! Transparent water pipeline.
//!
//! Identical vertex layout, bind groups, and primitive state to the
//! opaque pipeline — both pipelines draw the same chunk vertex buffers,
//! just over different fragment-shader paths. The differences live in:
//!
//! * **Blend state:** standard "source-over" alpha blend so water
//!   surfaces composite on top of the already-rendered opaque world.
//! * **Depth write disabled:** depth *test* still on (water hides behind
//!   closer opaque blocks), but writing depth would prevent overlapping
//!   water surfaces from blending correctly when the camera looks
//!   through one body of water into another.
//! * **Shader:** `water.wgsl` discards non-water fragments and emits
//!   vertex wave displacement + fresnel + sun specular for the rest.

use crate::render::gpu::DEPTH_FORMAT;

pub struct WaterPipeline {
    pub pipeline: wgpu::RenderPipeline,
}

const SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/shaders/water.wgsl"
));

pub fn build(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    camera_bgl: &wgpu::BindGroupLayout,
    chunk_bgl: &wgpu::BindGroupLayout,
    atlas_bgl: &wgpu::BindGroupLayout,
) -> WaterPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("water-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("water-layout"),
        bind_group_layouts: &[camera_bgl, chunk_bgl, atlas_bgl],
        push_constant_ranges: &[],
    });

    // Same vertex layout as the opaque pipeline — both pipelines bind
    // the chunk's single vertex buffer.
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::mesher::Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Uint8x4,
            },
            wgpu::VertexAttribute {
                offset: 4,
                shader_location: 2,
                format: wgpu::VertexFormat::Unorm8x4,
            },
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 3,
                format: wgpu::VertexFormat::Uint8x4,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 4,
                format: wgpu::VertexFormat::Uint8x4,
            },
        ],
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("water-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                // Source-over alpha blend: `final = src.rgb * src.a +
                // dst.rgb * (1 - src.a)`. The fragment shader emits a
                // pre-multiplied-style colour with alpha controlling
                // how much of the world below shows through.
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            // Water surfaces are double-sided in spirit (player can
            // look up from underwater and see the underside), but
            // back-face cull is still fine because the mesher emits
            // both top and bottom faces of a water column when both
            // are exposed. Saves overdraw vs `cull_mode: None`.
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            // Depth *test* on so water is occluded by closer opaque
            // geometry, but no depth *write* — multiple water layers
            // blend cleanly without the first one sealing the
            // second's pixels off.
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    WaterPipeline { pipeline }
}
