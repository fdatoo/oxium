//! Procedural sky pipeline.
//!
//! Draws a single full-screen triangle that paints the background using the
//! sky shader (in `assets/shaders/sky.wgsl`). Runs *before* the opaque pass
//! so opaque geometry overwrites it where present.
//!
//! The pipeline shares the camera bind group with the opaque pipeline so
//! `sun_intensity` is just there.

use crate::render::gpu::{DEPTH_FORMAT, MSAA_SAMPLES};

/// Compile-time-embedded WGSL source for the sky shader.
const SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/shaders/sky.wgsl"
));

/// Owned pipeline + label. Drawn with `draw(0..3, 0..1)` — no vertex
/// buffer needed because the shader generates positions from
/// `@builtin(vertex_index)`.
pub struct SkyPipeline {
    pub pipeline: wgpu::RenderPipeline,
}

/// Build the sky pipeline.
pub fn build(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bgl: &wgpu::BindGroupLayout,
) -> SkyPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sky-layout"),
        bind_group_layouts: &[camera_bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sky-pipeline"),
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
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            // Do not write depth — the sky sits at the far plane and we
            // want opaque geometry to overwrite it without depth conflict.
            depth_write_enabled: false,
            // LessEqual so a near-far quad at the same depth wins; together
            // with `depth_write_enabled = false` this leaves the depth
            // buffer clear-ready for the subsequent opaque pass.
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
    SkyPipeline { pipeline }
}
