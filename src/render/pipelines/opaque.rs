//! Opaque voxel pipeline.
//!
//! Draws fully-opaque chunk faces with our 16-byte packed vertex format.
//! The wgsl shader source is embedded at compile time via `include_str!` so
//! the binary is self-contained (no separate shader files at runtime).

use crate::render::gpu::{DEPTH_FORMAT, MSAA_SAMPLES};

/// Wraps the built `wgpu::RenderPipeline` for opaque chunk drawing.
pub struct OpaquePipeline {
    pub pipeline: wgpu::RenderPipeline,
}

/// Which face winding the pipeline treats as "front-facing" for
/// back-face culling. The reflection pass needs the opposite of the
/// main pass because mirroring the camera flips the apparent
/// chirality of every triangle.
#[derive(Debug, Clone, Copy)]
pub enum FrontFace {
    Ccw,
    Cw,
}

impl FrontFace {
    fn to_wgpu(self) -> wgpu::FrontFace {
        match self {
            FrontFace::Ccw => wgpu::FrontFace::Ccw,
            FrontFace::Cw => wgpu::FrontFace::Cw,
        }
    }
}

/// Compile-time-embedded WGSL source for the opaque shader. Using
/// `CARGO_MANIFEST_DIR` makes the path independent of the binary's runtime
/// working directory.
const SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/shaders/opaque.wgsl"
));

/// Build the opaque render pipeline. Called once at startup.
///
/// `front_face` lets the caller request the reflection-pass variant
/// (`FrontFace::Cw`). Because mirroring the camera across the water
/// plane flips the apparent handedness of every triangle, what was
/// CCW from the main camera's POV is CW from the mirror eye's POV —
/// so a Ccw-cull-back pipeline used for the reflection would cull
/// every mountain top face and the reflection would only contain
/// the sky. The reflection variant uses Cw winding to keep top
/// faces visible.
pub fn build(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    camera_bgl: &wgpu::BindGroupLayout,
    chunk_bgl: &wgpu::BindGroupLayout,
    atlas_bgl: &wgpu::BindGroupLayout,
    front_face: FrontFace,
) -> OpaquePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("opaque-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("opaque-layout"),
        // group(0) = camera, group(1) = chunk uniform, group(2) = atlas
        // (texture + sampler shared by every chunk draw).
        bind_group_layouts: &[camera_bgl, chunk_bgl, atlas_bgl],
        push_constant_ranges: &[],
    });

    // Vertex layout packs the 16-byte `Vertex` struct into four GPU-side
    // 4-byte tuples so the shader can pull them as `vec4<u32>` slots:
    //
    //   offset 0  → pos.xyz + ao                          (Uint8x4)
    //   offset 4  → color RGBA tint                       (Unorm8x4 → [0,1])
    //   offset 8  → normal_face + light + 2-byte pad      (Uint8x4)
    //   offset 12 → tile_index + u_tile + v_tile + pad    (Uint8x4)
    //
    // Shader location numbers below match the wgsl `@location` attributes.
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
        label: Some("opaque-pipeline"),
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
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // Mesher emits CCW-from-outside quads. The reflection pass
            // wants Cw because its mirror camera sees the world from
            // the opposite side.
            front_face: front_face.to_wgpu(),
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
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

    OpaquePipeline { pipeline }
}
