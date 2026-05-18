//! Voxel chunk meshing: produces vertex + index buffers for the renderer.
//!
//! M1 ships only [`naive::mesh_chunk_no_neighbors`] — one quad per visible
//! block face — which is correct but emits many more vertices than needed.
//! M4 introduces the greedy mesher in `greedy.rs`, which fuses coplanar
//! same-appearance faces into rectangles and is what the live game uses.
//!
//! The naive mesher is kept after M4 as a debug fallback / golden-test
//! reference (a greedy mesh and a naive mesh should be visually identical).

pub mod naive;

use bytemuck::{Pod, Zeroable};

/// One of six axis-aligned cube face directions. The numeric discriminants
/// match the `normal_face` byte stored in [`Vertex`] and are read by the
/// vertex shader to pick face-direction tinting.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Face {
    /// All six faces in a fixed, stable order (matches the discriminants).
    pub fn all() -> [Face; 6] {
        [
            Face::PosX,
            Face::NegX,
            Face::PosY,
            Face::NegY,
            Face::PosZ,
            Face::NegZ,
        ]
    }

    /// Outward-pointing unit normal for this face, as `[dx, dy, dz]`.
    pub fn normal(self) -> [i32; 3] {
        match self {
            Face::PosX => [1, 0, 0],
            Face::NegX => [-1, 0, 0],
            Face::PosY => [0, 1, 0],
            Face::NegY => [0, -1, 0],
            Face::PosZ => [0, 0, 1],
            Face::NegZ => [0, 0, -1],
        }
    }
}

/// 16-byte packed per-vertex payload.
///
/// Why so packed? At our render distance we'll produce millions of vertices.
/// Each saved byte multiplies. The fields are deliberately laid out so the
/// vertex shader can read them as three `vec4<u32>`s (see `opaque.wgsl`):
///
/// | Offset | Bytes | Field             | Notes                              |
/// |--------|-------|-------------------|------------------------------------|
/// |   0    |   3   | `pos`             | local chunk coords `0..=32`        |
/// |   3    |   1   | `ao`              | 0..=3, ambient-occlusion darkness  |
/// |   4    |   4   | `color`           | RGBA, normalised `[0,1]` in shader |
/// |   8    |   1   | `normal_face`     | [`Face`] discriminant              |
/// |   9    |   1   | `light`           | low 4 = sky, high 4 = block        |
/// |  10    |   2   | `_pad`            | alignment to 4-byte tuples         |
///
/// Total: 16 bytes — matches the GPU's preferred `vec4<u32>` access pattern.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [u8; 3],
    pub ao: u8,
    pub color: [u8; 4],
    pub normal_face: u8,
    pub light: u8,
    pub _pad: [u8; 2],
}

/// CPU-side mesh data destined for the GPU. Owned by the worker thread that
/// built it; ownership transfers to the main thread when uploaded via
/// [`crate::render::mesh::upload_mesh`].
pub struct ChunkMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    /// Allocate an empty mesh; both buffers are `Vec::new()` (no allocation
    /// until the first push).
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }
}
