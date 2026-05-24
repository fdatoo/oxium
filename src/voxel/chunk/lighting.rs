//! Chunk-local light packing and worldgen-provided light inputs.

use crate::voxel::block::BlockRegistry;
use crate::voxel::coords::{ChunkCoord, LocalPos};
use glam::UVec3;

use super::{CHUNK_AREA, DenseChunk};

pub const LIGHT_INPUT_UNKNOWN_Y: i16 = i16::MIN;

/// Four-bit light level, invariant `0..=15`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct LightLevel(u8);

impl LightLevel {
    pub const ZERO: Self = Self(0);
    pub const MAX: Self = Self(15);

    /// Build a light level if `value` fits in four bits.
    #[inline]
    pub fn new(value: u8) -> Option<Self> {
        (value < 16).then_some(Self(value))
    }

    /// Build a light level from a value already expected to be four-bit.
    ///
    /// The debug assertion catches invariant violations during development;
    /// the mask preserves the historical release behavior of the old helpers.
    #[inline]
    pub fn from_nibble(value: u8) -> Self {
        debug_assert!(value < 16, "light level out of range (max 15)");
        Self(value & 0x0F)
    }

    #[inline]
    pub fn get(self) -> u8 {
        self.0
    }
}

/// Packed RGB block light: `(R << 8) | (G << 4) | B`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PackedRgbLight(u16);

impl PackedRgbLight {
    #[inline]
    pub fn new(r: LightLevel, g: LightLevel, b: LightLevel) -> Self {
        Self(((r.get() as u16) << 8) | ((g.get() as u16) << 4) | b.get() as u16)
    }

    /// Wrap an already-packed raw cell from chunk storage.
    #[inline]
    pub fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    #[inline]
    pub fn raw(self) -> u16 {
        self.0
    }

    #[inline]
    pub fn channels(self) -> (LightLevel, LightLevel, LightLevel) {
        (
            LightLevel::from_nibble(((self.0 >> 8) & 0x0F) as u8),
            LightLevel::from_nibble(((self.0 >> 4) & 0x0F) as u8),
            LightLevel::from_nibble((self.0 & 0x0F) as u8),
        )
    }

    #[inline]
    pub fn channels_u8(self) -> (u8, u8, u8) {
        let (r, g, b) = self.channels();
        (r.get(), g.get(), b.get())
    }

    #[inline]
    pub fn brightness(self) -> LightLevel {
        let (r, g, b) = self.channels();
        r.max(g).max(b)
    }
}

/// Pack `(R, G, B)` channels (each 0..=15) into `DenseChunk::block_rgb`.
#[inline]
pub fn pack_rgb(r: u8, g: u8, b: u8) -> u16 {
    PackedRgbLight::new(
        LightLevel::from_nibble(r),
        LightLevel::from_nibble(g),
        LightLevel::from_nibble(b),
    )
    .raw()
}

/// Inverse of [`pack_rgb`].
#[inline]
pub fn unpack_rgb(cell: u16) -> (u8, u8, u8) {
    PackedRgbLight::from_raw(cell).channels_u8()
}

/// Scalar brightness for legacy paths that need one block-light value.
#[inline]
pub fn rgb_brightness(cell: u16) -> u8 {
    PackedRgbLight::from_raw(cell).brightness().get()
}

/// Worldgen/block-derived lighting metadata for one chunk.
///
/// This is the contract between chunk generation and lighting. The light
/// worker still computes per-voxel light, but it no longer has to guess
/// whether a missing +Y chunk means "open sky" or "unknown vertical context".
#[derive(Debug, Clone)]
pub struct ChunkLightInputs {
    /// First opaque voxel encountered scanning the local column top-down, as
    /// world Y. [`LIGHT_INPUT_UNKNOWN_Y`] means no opaque voxel inside this
    /// chunk column.
    pub first_opaque_y: Box<[i16; CHUNK_AREA]>,
    /// Highest generated terrain/surface Y known for this world column.
    pub surface_y: Box<[i16; CHUNK_AREA]>,
    /// Sky level entering this chunk's top face when no +Y lit neighbor is
    /// available. `0` means the top context is unknown or blocked.
    pub top_sky: Box<[u8; CHUNK_AREA]>,
    /// Number of emissive voxels in this chunk.
    pub emissive_count: u16,
}

impl Default for ChunkLightInputs {
    fn default() -> Self {
        Self {
            first_opaque_y: Box::new([LIGHT_INPUT_UNKNOWN_Y; CHUNK_AREA]),
            surface_y: Box::new([LIGHT_INPUT_UNKNOWN_Y; CHUNK_AREA]),
            top_sky: Box::new([0; CHUNK_AREA]),
            emissive_count: 0,
        }
    }
}

impl ChunkLightInputs {
    pub fn from_dense(dense: &DenseChunk, coord: ChunkCoord, registry: &BlockRegistry) -> Self {
        Self::from_dense_with_surface(dense, coord, registry, |_, _| None)
    }

    pub fn from_dense_with_surface<F>(
        dense: &DenseChunk,
        coord: ChunkCoord,
        registry: &BlockRegistry,
        mut surface_y: F,
    ) -> Self
    where
        F: FnMut(u32, u32) -> Option<i32>,
    {
        let mut out = Self::default();
        let origin_y = coord.origin().0.y;
        let top_y = origin_y + 32;
        let mut emissive_count: u32 = 0;

        for z in 0..32u32 {
            for x in 0..32u32 {
                let column = (z * 32 + x) as usize;
                if let Some(surface) = surface_y(x, z) {
                    out.surface_y[column] = surface.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    if surface < top_y {
                        out.top_sky[column] = 15;
                    }
                }

                for y in (0..32u32).rev() {
                    let idx = LocalPos(UVec3::new(x, y, z)).to_index();
                    let info = registry.info(dense.blocks[idx]);
                    if info.emission != [0, 0, 0] {
                        emissive_count += 1;
                    }
                    if info.opaque && out.first_opaque_y[column] == LIGHT_INPUT_UNKNOWN_Y {
                        out.first_opaque_y[column] =
                            (origin_y + y as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    }
                }
            }
        }

        out.emissive_count = emissive_count.min(u16::MAX as u32) as u16;
        out
    }

    pub fn rebuild_preserving_surface(
        &self,
        dense: &DenseChunk,
        coord: ChunkCoord,
        registry: &BlockRegistry,
    ) -> Self {
        Self::from_dense_with_surface(dense, coord, registry, |x, z| {
            let idx = (z * 32 + x) as usize;
            let y = self.surface_y[idx];
            (y != LIGHT_INPUT_UNKNOWN_Y).then_some(y as i32)
        })
    }

    #[inline]
    pub fn top_sky_at(&self, x: u32, z: u32) -> u8 {
        self.top_sky[(z * 32 + x) as usize] & 0x0F
    }
}
