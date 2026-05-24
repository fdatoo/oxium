//! Per-chunk quart-resolution 2D cache.
//!
//! Stores one value per `QUART_PER_CHUNK × QUART_PER_CHUNK` cell of
//! a chunk. Used to amortize expensive 2D field evaluations
//! (continentalness, erosion, offset, factor, jaggedness) across
//! the ~1000 voxels in each 4×4 column quarter.
//!
//! A "quart" is a 4×4 block column cell, matching Minecraft's `QuartPos`
//! coordinate space. Within a 32-block chunk there are 8×8 = 64 quarts.
//! Expensive 2D noise channels (those driven by `FlatCache` markers in
//! the density graph) are evaluated once per quart and shared across all
//! 16 block-columns and 32 vertical slices in that quart — a 512×
//! amortisation factor per channel relative to per-voxel evaluation.

use crate::voxel::coords::CHUNK_DIM_U;

/// 4 blocks per quart (matches Minecraft `QuartPos`).
pub const QUART_SIZE: u32 = 4;
/// Quarts per chunk side: 32 / 4 = 8.
pub const QUART_PER_CHUNK: u32 = CHUNK_DIM_U / QUART_SIZE;

/// 2D cache of `T` over the chunk footprint, sampled at quart
/// resolution. Total `QUART_PER_CHUNK²` = 64 entries.
pub struct FlatCache2D<T: Copy> {
    values: [Option<T>; (QUART_PER_CHUNK * QUART_PER_CHUNK) as usize],
}

impl<T: Copy> FlatCache2D<T> {
    pub fn new() -> Self {
        Self {
            values: [None; (QUART_PER_CHUNK * QUART_PER_CHUNK) as usize],
        }
    }

    /// Returns the cached value at quart `(qx, qz)` (both in
    /// `0..QUART_PER_CHUNK`), computing and storing it on miss.
    pub fn get_or_compute<F>(&mut self, qx: u32, qz: u32, mut compute: F) -> T
    where
        F: FnMut() -> T,
    {
        let idx = (qx + qz * QUART_PER_CHUNK) as usize;
        if let Some(v) = self.values[idx] {
            return v;
        }
        let v = compute();
        self.values[idx] = Some(v);
        v
    }

    /// Maps a block-space (lx, lz) inside the chunk to the
    /// corresponding (qx, qz).
    pub fn block_to_quart(lx: u32, lz: u32) -> (u32, u32) {
        (lx / QUART_SIZE, lz / QUART_SIZE)
    }
}

impl<T: Copy> Default for FlatCache2D<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_returns_same_value_without_recompute() {
        let mut cache = FlatCache2D::<f32>::new();
        let mut calls = 0;
        let v1 = cache.get_or_compute(3, 5, || {
            calls += 1;
            7.5
        });
        let v2 = cache.get_or_compute(3, 5, || {
            calls += 1;
            999.0 // would diverge if recomputed
        });
        assert_eq!(v1, 7.5);
        assert_eq!(v2, 7.5);
        assert_eq!(calls, 1, "compute closure should run exactly once");
    }

    #[test]
    fn different_quarts_compute_independently() {
        let mut cache = FlatCache2D::<i32>::new();
        let a = cache.get_or_compute(0, 0, || 1);
        let b = cache.get_or_compute(1, 0, || 2);
        let c = cache.get_or_compute(0, 1, || 3);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    }

    #[test]
    fn block_to_quart_groups_by_four() {
        assert_eq!(FlatCache2D::<f32>::block_to_quart(0, 0), (0, 0));
        assert_eq!(FlatCache2D::<f32>::block_to_quart(3, 3), (0, 0));
        assert_eq!(FlatCache2D::<f32>::block_to_quart(4, 0), (1, 0));
        assert_eq!(FlatCache2D::<f32>::block_to_quart(31, 31), (7, 7));
    }
}
