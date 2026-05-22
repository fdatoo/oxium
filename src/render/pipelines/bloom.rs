//! Bloom pipelines — one per fragment entry point in `bloom.wgsl`.
//!
//! Threshold + downsample use a single-target color attachment with no
//! blending. Upsample uses an additive blend so each upsample pass
//! accumulates onto whatever the downsample chain wrote to the
//! destination mip — the "blur spread" look.
//!
//! All three share one bind-group layout (a 2D texture + linear
//! sampler) and one pipeline layout, so callers can swap pipelines
//! without rebuilding bind groups.

const SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/shaders/bloom.wgsl"
));

pub struct BloomPipelines {
    pub bgl: wgpu::BindGroupLayout,
    pub threshold: wgpu::RenderPipeline,
    pub downsample: wgpu::RenderPipeline,
    pub upsample: wgpu::RenderPipeline,
}

pub fn build(device: &wgpu::Device, bloom_format: wgpu::TextureFormat) -> BloomPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bloom-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bloom-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bloom-layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    // Helper: build a pipeline targeting `bloom_format` with a given
    // fragment entry point and blend state.
    let make_pipeline = |label: &str, entry: &str, blend: Option<wgpu::BlendState>| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: bloom_format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    };

    let threshold = make_pipeline("bloom-threshold", "fs_threshold", None);
    let downsample = make_pipeline("bloom-downsample", "fs_downsample", None);
    // Upsample: additive blend so we accumulate onto the destination
    // mip's existing content. `SrcAlpha` lets future work tweak per-
    // mip weights via alpha; we write alpha = 1.0 in the shader, so
    // the current behavior is one-to-one additive.
    let upsample_blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    };
    let upsample = make_pipeline("bloom-upsample", "fs_upsample", Some(upsample_blend));

    BloomPipelines {
        bgl,
        threshold,
        downsample,
        upsample,
    }
}
