//! Per-frame and per-draw uniform buffers.
//!
//! Two uniforms live here:
//!
//! - [`CameraUniform`] — the single combined view-projection matrix used by
//!   every opaque draw. Updated once per frame.
//! - [`ChunkUniform`] — per-chunk world-space origin used by the vertex
//!   shader to lift local 0..32 voxel coordinates into world space. Updated
//!   per draw call (or once when only one chunk is drawn, as in M1).
//!
//! Keeping these tiny and immutable per-frame is on purpose: a small static
//! uniform layout makes the bind-group pipeline cheap.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// View-projection + lighting context shared with every opaque/sky draw.
///
/// Memory layout (std140-friendly, all members 16-byte aligned, total 176
/// bytes — a multiple of 16):
///
/// | Offset | Size | Field           |
/// |-------:|-----:|-----------------|
/// |     0  |  64  | `view_proj`     |
/// |    64  |  16  | `sun_dir`       |
/// |    80  |   4  | `sun_intensity` |
/// |    84  |  12  | trailing scalar padding (`_pad0..2`) |
/// |    96  |  16  | `eye`           |
/// |   112  |  64  | `inv_view_proj` |
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    /// Unit-length sun direction in world space. `w` is unused padding.
    pub sun_dir: [f32; 4],
    /// Scalar sun brightness, 0..=1. The shader multiplies the per-vertex
    /// sky-light channel by this so torches still glow in the dark.
    pub sun_intensity: f32,
    /// Seconds since startup. Drives shader-side animation (currently the
    /// water-surface shimmer in the opaque fragment shader). f32 has
    /// enough precision for the first ~hour of play before the
    /// fractional part loses resolution — that's plenty for "subtle
    /// ripples" detail.
    pub time: f32,
    /// `0.0` ⇒ camera is in air; `1.0` ⇒ submerged in water; fractional
    /// values appear during the swim-out transition.  Every fragment
    /// shader mixes its final colour toward a deep-blue water tint
    /// scaled by this factor — the cheap way to get the "you're
    /// underwater" colour grade without a dedicated post pass.
    pub underwater_factor: f32,
    /// Minimum world-space Y a fragment may have before being
    /// rendered. Defaults to a deep negative for the main world
    /// pass; set to `SEA_LEVEL` for the reflection pass so anything
    /// below the water plane is clipped out of the reflection (you
    /// don't see underwater geometry reflected in the surface).
    pub clip_y_min: f32,
    /// Camera (eye) world-space position. `w` unused. Used by:
    /// - opaque fog to compute fragment distance from camera
    /// - sky sun-disc to compute world-space ray direction
    pub eye: [f32; 4],
    /// Inverse of `view_proj`. The sky shader uses it to reconstruct a
    /// world-space ray direction from NDC, so the sun disc can be drawn
    /// in the correct world direction regardless of camera orientation.
    pub inv_view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    /// Default identity transform with an overhead sun. Used as initial
    /// buffer contents before the first frame's data is written.
    pub fn identity() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            sun_dir: [0.0, 1.0, 0.0, 0.0],
            sun_intensity: 1.0,
            time: 0.0,
            underwater_factor: 0.0,
            clip_y_min: -1_000_000.0,
            eye: [0.0; 4],
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

/// Bind-group layout for the camera uniform. One uniform at binding 0,
/// visible from *both* shader stages: the vertex stage uses `view_proj`,
/// the fragment stage uses `sun_dir`/`sun_intensity` (sky shader) and
/// will use `sun_dir` for diffuse shading once M5+ lighting kicks in.
pub fn make_camera_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Allocate the (initially-identity) camera uniform buffer.
pub fn make_camera_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera-uniform"),
        contents: bytemuck::cast_slice(&[CameraUniform::identity()]),
        // UNIFORM = bindable as a uniform; COPY_DST = `queue.write_buffer`.
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Build a standard right-handed perspective + look-at view-projection
/// matrix.
///
/// `yaw` rotates around the world's `+Y` axis (0 looks toward `+X`);
/// `pitch` is the up/down rotation, clamped by the caller to ±89° to avoid
/// gimbal-lock at the poles.
///
/// `fov_y` is in *radians*, vertical field-of-view.
pub fn view_proj(eye: Vec3, yaw: f32, pitch: f32, fov_y: f32, aspect: f32) -> Mat4 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    // The forward vector matches the convention used by the input system:
    // yaw=0, pitch=0 → look down +X.
    let forward = Vec3::new(cy * cp, sp, sy * cp);
    let view = Mat4::look_at_rh(eye, eye + forward, Vec3::Y);
    let proj = Mat4::perspective_rh(fov_y, aspect, 0.05, 1000.0);
    proj * view
}

/// Per-chunk uniform. `origin.xyz` is the world-space position of the
/// chunk's `(0, 0, 0)` corner; the shader adds it to each vertex's local
/// position to obtain world coordinates. `w` is padding because std140
/// uniform rules require vec3 to be aligned to 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ChunkUniform {
    pub origin: [f32; 4],
}

/// Bind-group layout for the chunk uniform.
///
/// We mark it as having a *dynamic offset* even though M1 only uses one
/// chunk: that way the bind-group layout can be reused as-is when M3 packs
/// many chunk uniforms into a single buffer and addresses them with a
/// per-draw dynamic offset.
pub fn make_chunk_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chunk-bgl"),
        entries: &[
            // Binding 0 — per-chunk world-space origin uniform (unchanged).
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                },
                count: None,
            },
            // Binding 1 — 3D light volume texture.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            // Binding 2 — linear sampler for the light volume.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}
