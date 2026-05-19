//! HUD 2D overlay pipeline.
//!
//! One pipeline serves both HUD batches (text + icons). The bind
//! group at `@group(1)` is swapped between draws to point at the
//! font atlas or the block atlas respectively. Alpha blending is on
//! so font glyphs and selection borders composite over the world
//! pass beneath. Depth test is off — the HUD lives strictly above
//! everything regardless of the world geometry's depth values.

use crate::render::hud::HudVertex;

/// Wraps the built HUD pipeline + the screen-size bind group layout
/// (group 0) and the texture bind group layout (group 1).
pub struct HudPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub screen_bgl: wgpu::BindGroupLayout,
    pub tex_bgl: wgpu::BindGroupLayout,
}

const SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/shaders/hud.wgsl"
));

/// Build the HUD pipeline. Called once at startup.
pub fn build(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> HudPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hud-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    // group(0) = screen-size uniform (one vec4<f32>).
    let screen_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hud-screen-bgl"),
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
    });
    // group(1) = atlas texture + sampler. Both batches use this
    // layout but with different texture views bound per draw.
    let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hud-tex-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hud-layout"),
        bind_group_layouts: &[&screen_bgl, &tex_bgl],
        push_constant_ranges: &[],
    });

    // Vertex layout: pos_px (Float32x2) + uv (Float32x2) + color
    // (Unorm8x4) = 8 + 8 + 4 = 20 bytes. wgpu rounds the stride to
    // the natural attribute alignment but our explicit
    // `array_stride` keeps things deterministic.
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<HudVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 2,
                format: wgpu::VertexFormat::Unorm8x4,
            },
        ],
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("hud-pipeline"),
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
                // Premultiplied-alpha style blend: src ∗ src.a +
                // dst ∗ (1 − src.a). Standard "over" for HUD glyphs.
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // No back-face cull: HUD quads might be authored either
            // winding and we'd rather render both than silently lose
            // glyphs to culling.
            cull_mode: None,
            ..Default::default()
        },
        // No depth test, no depth attachment — HUD is rendered in its
        // own pass that targets the colour view only.
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    HudPipeline {
        pipeline,
        screen_bgl,
        tex_bgl,
    }
}
