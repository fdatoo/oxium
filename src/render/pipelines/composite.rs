//! Composite pass: samples the HDR offscreen target and writes to the
//! swapchain. Tonemap + underwater grade are folded in by later tasks
//! of PR 1; bloom + volumetrics in subsequent PRs.
//!
//! The pipeline takes no vertex buffer — `vs_main` emits a fullscreen
//! triangle from `vertex_index`. No depth attachment, no MSAA (the
//! HDR target was already resolved by the world passes).

/// Compile-time-embedded WGSL source for the composite shader.
const SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/shaders/composite.wgsl"
));

/// Owned pipeline + the bind-group layout for sampling the HDR view +
/// the sampler used to read it. Pipelines pre-create the sampler once
/// and reuse it across resizes.
pub struct CompositePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
}

/// Build the composite pipeline targeting the swapchain `format`. The
/// HDR target itself is `Rgba16Float` (see `render::hdr::HDR_FORMAT`)
/// but that's the *input* — `format` is the swapchain output.
pub fn build(device: &wgpu::Device, format: wgpu::TextureFormat) -> CompositePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("composite-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });
    // Rgba16Float without the `float32-filterable` device feature is
    // only sampleable as non-filtering. Nearest sampling matches the
    // 1:1 fullscreen blit we're doing anyway.
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite-bgl"),
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
            // Camera uniform — supplies `time` and `underwater_factor`
            // to the underwater tint added in PR 1 Task 6.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("composite-layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("composite-pipeline"),
        layout: Some(&pipeline_layout),
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
                blend: None,
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
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("composite-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    CompositePipeline {
        pipeline,
        bgl,
        sampler,
    }
}
