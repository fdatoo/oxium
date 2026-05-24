//! Public mesh data types shared by mesh builders and the renderer.

use bytemuck::{Pod, Zeroable};

/// One of six axis-aligned cube face directions.
///
/// The numeric discriminants match the `normal_face` byte stored in
/// [`Vertex`] and are read by the vertex shader to pick face-direction
/// tinting.
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
    /// All six faces in a fixed, stable order.
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
/// The fields are deliberately laid out so the vertex shader can read them as
/// four `vec4<u32>`s:
///
/// | Offset | Bytes | Field         | Notes                                  |
/// |--------|-------|---------------|----------------------------------------|
/// |   0    |   3   | `pos`         | local chunk coords `0..=32`            |
/// |   3    |   1   | `ao`          | 0..=3, ambient-occlusion darkness      |
/// |   4    |   4   | `color`       | RGBA tint, normalised `[0,1]` in shader|
/// |   8    |   1   | `normal_face` | [`Face`] discriminant                  |
/// |   9    |   1   | `light`       | high 4 = sky, low 4 = block            |
/// |  10    |   2   | `_pad`        | alignment to 4-byte tuples             |
/// |  12    |   1   | `tile_index`  | atlas tile (0..=15); 0xFF = untextured |
/// |  13    |   1   | `u_tile`      | corner U in tile units                 |
/// |  14    |   1   | `v_tile`      | corner V in tile units                 |
/// |  15    |   1   | `_pad2`       | alignment to 4-byte tuples             |
///
/// "Tile units" means one integer step equals one atlas tile width. Greedy
/// quads can span many blocks while the fragment shader wraps with `fract`.
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

/// Sentinel `tile_index` value: no texture, use vertex colour directly.
pub const UNTEXTURED_TILE: u8 = 0xFF;

/// CPU-side mesh data destined for the GPU.
pub struct ChunkMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// True when the source chunk contains any water cell. The renderer uses
    /// this to skip the planar-reflection pass when no visible chunk needs it.
    pub has_water: bool,
}

impl ChunkMesh {
    /// Allocate an empty mesh; both buffers are `Vec::new()`.
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            has_water: false,
        }
    }
}
