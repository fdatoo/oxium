//! GPU-side mesh handles. A [`GpuMesh`] is just the vertex buffer + index
//! buffer pair plus the index count; nothing else is needed for a draw call.

use crate::mesher::ChunkMesh;
use wgpu::util::DeviceExt;

/// One drawable chunk on the GPU.
pub struct GpuMesh {
    /// Vertex buffer — interleaved [`crate::mesher::Vertex`] values.
    pub vbuf: wgpu::Buffer,
    /// Index buffer — `u32` triangle indices.
    pub ibuf: wgpu::Buffer,
    /// Number of indices in `ibuf`. Used by `draw_indexed`.
    pub index_count: u32,
}

/// Upload a CPU-side mesh to the GPU. Returns `None` for empty meshes —
/// we don't want to allocate zero-byte buffers because wgpu treats that as
/// an error.
pub fn upload_mesh(device: &wgpu::Device, mesh: &ChunkMesh) -> Option<GpuMesh> {
    if mesh.indices.is_empty() {
        return None;
    }
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk-vbuf"),
        contents: bytemuck::cast_slice(&mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk-ibuf"),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Some(GpuMesh {
        vbuf,
        ibuf,
        index_count: mesh.indices.len() as u32,
    })
}
