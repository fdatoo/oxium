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

/// Single mat4 view-projection. Stored as `[[f32; 4]; 4]` (column-major)
/// to match wgsl's `mat4x4<f32>` memory layout exactly under bytemuck.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    /// Default identity transform; useful as initial buffer contents before
    /// the first frame's view-proj is computed.
    pub fn identity() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

/// Bind-group layout for the camera uniform. One uniform at binding 0,
/// visible only from the vertex stage (the fragment doesn't transform
/// anything by view-proj).
pub fn make_camera_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
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
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                // Minimum binding size = sizeof(ChunkUniform) = 16 bytes.
                min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
            },
            count: None,
        }],
    })
}
