//! 3D scene pipeline. Vertex format + wgpu plumbing land here in Task 6.

use bytemuck::{Pod, Zeroable};

/// PR 1 vertex format: position + baked color. PR 3 splits color into a
/// parallel buffer to support paint-mode toggling without remesh.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}
