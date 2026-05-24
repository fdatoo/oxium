//! Palette-compressed chunk storage.

use crate::voxel::block::Block;
use crate::voxel::coords::LocalPos;
use crate::voxel::packed::Packed4Bit;
use serde::{Deserialize, Serialize};

use super::{CHUNK_VOL, DenseChunk, pack_rgb, unpack_rgb};

/// Canonical in-RAM and on-disk chunk form: a palette of distinct blocks plus
/// a 4-bit-per-voxel index array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalettedChunk {
    /// Distinct block kinds present in this chunk, indexed by `indices`.
    pub palette: Vec<Block>,
    /// 4 bits per voxel: an index into `palette`.
    pub indices: Packed4Bit,
    /// Sky-light values, 0..=15.
    pub sky_light: Packed4Bit,
    /// Red channel of per-voxel block light (0..=15).
    pub block_red: Packed4Bit,
    /// Green channel of per-voxel block light (0..=15).
    pub block_green: Packed4Bit,
    /// Blue channel of per-voxel block light (0..=15).
    pub block_blue: Packed4Bit,
}

impl PalettedChunk {
    /// All-air chunk: single-entry palette, all indices zero, no light.
    pub fn all_air() -> Self {
        Self {
            palette: vec![Block::Air],
            indices: Packed4Bit::zeros(CHUNK_VOL),
            sky_light: Packed4Bit::zeros(CHUNK_VOL),
            block_red: Packed4Bit::zeros(CHUNK_VOL),
            block_green: Packed4Bit::zeros(CHUNK_VOL),
            block_blue: Packed4Bit::zeros(CHUNK_VOL),
        }
    }

    /// Build a `PalettedChunk` from a `DenseChunk`.
    pub fn compress(dense: &DenseChunk) -> Self {
        let mut palette: Vec<Block> = Vec::with_capacity(8);
        let mut lookup = [u8::MAX; crate::voxel::block::BLOCK_COUNT];
        let mut indices = Packed4Bit::zeros(CHUNK_VOL);

        for i in 0..CHUNK_VOL {
            let b = dense.blocks[i];
            let slot = b as u8 as usize;
            let idx = if lookup[slot] != u8::MAX {
                lookup[slot]
            } else {
                assert!(
                    palette.len() < 16,
                    "M3: palette exceeded 16 entries; widen to BitPackedArray later"
                );
                let new = palette.len() as u8;
                palette.push(b);
                lookup[slot] = new;
                new
            };
            indices.set(i, idx);
        }

        let mut sky = Packed4Bit::zeros(CHUNK_VOL);
        let mut r = Packed4Bit::zeros(CHUNK_VOL);
        let mut g = Packed4Bit::zeros(CHUNK_VOL);
        let mut b = Packed4Bit::zeros(CHUNK_VOL);
        for i in 0..CHUNK_VOL {
            sky.set(i, dense.sky_light[i] & 0x0F);
            let (rr, gg, bb) = unpack_rgb(dense.block_rgb[i]);
            r.set(i, rr);
            g.set(i, gg);
            b.set(i, bb);
        }

        Self {
            palette,
            indices,
            sky_light: sky,
            block_red: r,
            block_green: g,
            block_blue: b,
        }
    }

    /// Inverse of [`Self::compress`]: produce a fresh `DenseChunk`.
    pub fn decompress(&self) -> DenseChunk {
        let mut blocks = Box::new([Block::Air; CHUNK_VOL]);
        for i in 0..CHUNK_VOL {
            let palette_idx = self.indices.get(i) as usize;
            blocks[i] = self.palette[palette_idx];
        }
        let mut sky = Box::new([0u8; CHUNK_VOL]);
        let mut block_rgb = Box::new([0u16; CHUNK_VOL]);
        for i in 0..CHUNK_VOL {
            sky[i] = self.sky_light.get(i);
            block_rgb[i] = pack_rgb(
                self.block_red.get(i),
                self.block_green.get(i),
                self.block_blue.get(i),
            );
        }
        DenseChunk {
            blocks,
            sky_light: sky,
            block_rgb,
        }
    }

    /// Read a single block by local position without a full decompress.
    pub fn get(&self, p: LocalPos) -> Block {
        self.palette[self.indices.get(p.to_index()) as usize]
    }

    #[inline]
    pub fn block_at(&self, idx: usize) -> Block {
        self.palette[self.indices.get(idx) as usize]
    }

    #[inline]
    pub fn sky_light_at(&self, idx: usize) -> u8 {
        self.sky_light.get(idx)
    }

    #[inline]
    pub fn block_rgb_at(&self, idx: usize) -> (u8, u8, u8) {
        (
            self.block_red.get(idx),
            self.block_green.get(idx),
            self.block_blue.get(idx),
        )
    }

    #[inline]
    pub fn set_sky_light_at(&mut self, idx: usize, value: u8) {
        self.sky_light.set(idx, value);
    }

    #[inline]
    pub fn set_block_rgb_at(&mut self, idx: usize, r: u8, g: u8, b: u8) {
        self.block_red.set(idx, r);
        self.block_green.set(idx, g);
        self.block_blue.set(idx, b);
    }
}

/// Legacy v1 paletted-chunk layout used before RGB block light.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PalettedChunkV1 {
    pub palette: Vec<Block>,
    pub indices: Packed4Bit,
    pub sky_light: Packed4Bit,
    pub block_light: Packed4Bit,
}

impl From<PalettedChunkV1> for PalettedChunk {
    fn from(v1: PalettedChunkV1) -> Self {
        Self {
            palette: v1.palette,
            indices: v1.indices,
            sky_light: v1.sky_light,
            block_red: v1.block_light.clone(),
            block_green: v1.block_light.clone(),
            block_blue: v1.block_light,
        }
    }
}
