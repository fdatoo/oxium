//! Voxel chunk meshing: produces vertex + index buffers for the renderer.
//!
//! M1 ships only [`naive::mesh_chunk_no_neighbors`] — one quad per visible
//! block face — which is correct but emits many more vertices than needed.
//! M4 introduces the greedy mesher in `greedy.rs`, which fuses coplanar
//! same-appearance faces into rectangles and is what the live game uses.
//!
//! The naive mesher is kept after M4 as a debug fallback / golden-test
//! reference (a greedy mesh and a naive mesh should be visually identical).

pub mod ao;
pub mod greedy;
pub mod lod;
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
/// vertex shader can read them as four `vec4<u32>`s (see `opaque.wgsl`):
///
/// | Offset | Bytes | Field         | Notes                                  |
/// |--------|-------|---------------|----------------------------------------|
/// |   0    |   3   | `pos`         | local chunk coords `0..=32`            |
/// |   3    |   1   | `ao`          | 0..=3, ambient-occlusion darkness      |
/// |   4    |   4   | `color`       | RGBA tint, normalised `[0,1]` in shader|
/// |   8    |   1   | `normal_face` | [`Face`] discriminant                  |
/// |   9    |   1   | `light`       | low 4 = sky, high 4 = block            |
/// |  10    |   2   | `_pad`        | alignment to 4-byte tuples             |
/// |  12    |   1   | `tile_index`  | atlas tile (0..=15); 0xFF = untextured |
/// |  13    |   1   | `u_tile`      | corner U in *tile units* (0..=32)      |
/// |  14    |   1   | `v_tile`      | corner V in *tile units* (0..=32)      |
/// |  15    |   1   | `_pad2`       | alignment to 4-byte tuples             |
///
/// Total: 16 bytes — matches the GPU's preferred `vec4<u32>` access pattern.
///
/// "Tile units" means: 1.0 = one whole tile width. Across a greedy
/// `w × h` quad the corner UVs span `(0,0)..(w,h)`, the rasteriser
/// interpolates linearly, and the fragment shader calls `fract` to wrap
/// each integer cell back to the tile's `[0, 1)` range. That gives
/// per-block tile repetition for free without splitting greedy quads.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [u8; 3],
    pub ao: u8,
    pub color: [u8; 4],
    pub normal_face: u8,
    pub light: u8,
    pub _pad: [u8; 2],
    pub tile_index: u8,
    pub u_tile: u8,
    pub v_tile: u8,
    pub _pad2: u8,
}

/// Sentinel `tile_index` value: "this vertex has no texture, use the
/// vertex colour straight". Read by the shader; written by the mesher
/// for blocks like `Air` and `Torch` that don't have an atlas binding.
pub const UNTEXTURED_TILE: u8 = 0xFF;

/// CPU-side mesh data destined for the GPU. Owned by the worker thread that
/// built it; ownership transfers to the main thread when uploaded via
/// [`crate::render::mesh::upload_mesh`].
pub struct ChunkMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Set by the mesher when the source chunk contains any `Block::Water`
    /// cell. The renderer copies this flag onto its per-chunk GPU record
    /// and uses it to decide whether the planar-reflection pass is
    /// needed: if no on-screen chunk contains water, the (expensive) full-
    /// world reflection render can be skipped entirely.
    pub has_water: bool,
}

impl ChunkMesh {
    /// Allocate an empty mesh; both buffers are `Vec::new()` (no allocation
    /// until the first push).
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            has_water: false,
        }
    }
}
