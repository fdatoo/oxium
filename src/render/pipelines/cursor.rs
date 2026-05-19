//! Wireframe cursor highlight pipeline.
//!
//! Renders 12 line segments — the edges of the targeted block — with
//! `LineList` topology. Depth test is `LessEqual` (so the lines win
//! ties against the block's own faces) but depth-writes are off so we
//! don't disturb the depth buffer for any subsequent passes.

use crate::render::gpu::{DEPTH_FORMAT, MSAA_SAMPLES};

/// Compile-time-embedded WGSL source for the cursor shader.
const SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/shaders/cursor.wgsl"
));

/// Owned pipeline.
pub struct CursorPipeline {
    pub pipeline: wgpu::RenderPipeline,
}

/// Build the cursor pipeline.
pub fn build(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bgl: &wgpu::BindGroupLayout,
    cursor_bgl: &wgpu::BindGroupLayout,
) -> CursorPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cursor-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cursor-layout"),
        bind_group_layouts: &[camera_bgl, cursor_bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cursor-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });
    CursorPipeline { pipeline }
}

/// Uniform layout for the cursor pipeline: a single `vec4<f32>` carrying
/// the block's `(min.x, min.y, min.z, _)` corner.
pub fn make_cursor_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cursor-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(16),
            },
            count: None,
        }],
    })
}
