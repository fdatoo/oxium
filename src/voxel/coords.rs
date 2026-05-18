//! Three coordinate spaces that are easy to confuse and so wrapped in
//! distinct newtypes:
//!
//! - [`BlockPos`] — a *world*-space integer block coordinate. Can be negative.
//! - [`ChunkCoord`] — `BlockPos.div_euclid(32)`. Identifies which chunk a
//!   block lives in.
//! - [`LocalPos`] — block position *within* a chunk, always in `0..32`.
//!
//! The reason we don't just use `IVec3` everywhere is that division rounds
//! toward zero by default in Rust, which silently corrupts negative-coordinate
//! math. `div_euclid` is the right primitive (it rounds *toward negative
//! infinity*), but it's only one call to forget — so the conversion is gated
//! behind these types.

use glam::{IVec3, UVec3};

/// Side length of a chunk, in blocks. Cubic; chosen for 3D-noise/cave friendliness.
pub const CHUNK_DIM: i32 = 32;

/// Same as [`CHUNK_DIM`] but unsigned, for cases where the signed value would
/// only ever be widened to `u32` anyway (e.g. iterating local coords).
pub const CHUNK_DIM_U: u32 = 32;

/// World-space block coordinates. Integer, signed, unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockPos(pub IVec3);

/// Chunk grid coordinates — `BlockPos.div_euclid(32)`. The chunk at the
/// world origin is `ChunkCoord(IVec3::ZERO)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord(pub IVec3);

/// Block position within a chunk. Each axis is `0..CHUNK_DIM_U`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalPos(pub UVec3);

impl BlockPos {
    /// Which chunk does this block live in? Uses **floor division**, so
    /// `BlockPos(-1)` lives in `ChunkCoord(-1)`, not `ChunkCoord(0)`.
    pub fn to_chunk(self) -> ChunkCoord {
        ChunkCoord(IVec3::new(
            self.0.x.div_euclid(CHUNK_DIM),
            self.0.y.div_euclid(CHUNK_DIM),
            self.0.z.div_euclid(CHUNK_DIM),
        ))
    }

    /// What is this block's position *within its own chunk*? Always in
    /// `0..32` on each axis, even for negative `BlockPos`.
    pub fn to_local(self) -> LocalPos {
        LocalPos(UVec3::new(
            self.0.x.rem_euclid(CHUNK_DIM) as u32,
            self.0.y.rem_euclid(CHUNK_DIM) as u32,
            self.0.z.rem_euclid(CHUNK_DIM) as u32,
        ))
    }
}

impl ChunkCoord {
    /// The world-space block at the chunk's `(0, 0, 0)` local corner.
    pub fn origin(self) -> BlockPos {
        BlockPos(self.0 * CHUNK_DIM)
    }
}

impl LocalPos {
    /// Flatten a `LocalPos` into a 0..32_768 array index. Order: x fastest,
    /// then y, then z. Keeping the layout x-fastest matches the typical
    /// `for z { for y { for x ... } }` iteration the meshers want.
    #[inline]
    pub fn to_index(self) -> usize {
        let UVec3 { x, y, z } = self.0;
        (x + y * CHUNK_DIM_U + z * CHUNK_DIM_U * CHUNK_DIM_U) as usize
    }

    /// Inverse of [`to_index`](Self::to_index).
    pub fn from_index(idx: usize) -> Self {
        let i = idx as u32;
        let x = i % CHUNK_DIM_U;
        let y = (i / CHUNK_DIM_U) % CHUNK_DIM_U;
        let z = i / (CHUNK_DIM_U * CHUNK_DIM_U);
        LocalPos(UVec3::new(x, y, z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_to_chunk_positive() {
        assert_eq!(BlockPos(IVec3::new(0, 0, 0)).to_chunk(), ChunkCoord(IVec3::ZERO));
        assert_eq!(BlockPos(IVec3::new(31, 0, 0)).to_chunk(), ChunkCoord(IVec3::ZERO));
        assert_eq!(BlockPos(IVec3::new(32, 0, 0)).to_chunk(), ChunkCoord(IVec3::new(1, 0, 0)));
    }

    #[test]
    fn block_to_chunk_negative() {
        // floor-div, NOT truncation: -1 → chunk -1, not chunk 0
        assert_eq!(BlockPos(IVec3::new(-1, 0, 0)).to_chunk(), ChunkCoord(IVec3::new(-1, 0, 0)));
        assert_eq!(BlockPos(IVec3::new(-32, 0, 0)).to_chunk(), ChunkCoord(IVec3::new(-1, 0, 0)));
        assert_eq!(BlockPos(IVec3::new(-33, 0, 0)).to_chunk(), ChunkCoord(IVec3::new(-2, 0, 0)));
    }

    #[test]
    fn block_to_local_negative() {
        assert_eq!(BlockPos(IVec3::new(-1, 0, 0)).to_local(), LocalPos(UVec3::new(31, 0, 0)));
        assert_eq!(BlockPos(IVec3::new(-32, 0, 0)).to_local(), LocalPos(UVec3::ZERO));
    }

    #[test]
    fn index_round_trip() {
        for &(x, y, z) in &[(0, 0, 0), (31, 31, 31), (1, 2, 3), (15, 7, 22)] {
            let p = LocalPos(UVec3::new(x, y, z));
            assert_eq!(LocalPos::from_index(p.to_index()), p, "{:?}", (x, y, z));
        }
    }

    #[test]
    fn chunk_origin_round_trip() {
        let c = ChunkCoord(IVec3::new(-3, 2, 5));
        assert_eq!(c.origin().to_chunk(), c);
    }
}
