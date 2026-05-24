//! Dense, fully-unpacked chunk storage.

use crate::voxel::block::Block;
use crate::voxel::coords::LocalPos;

use super::CHUNK_VOL;

/// Fully-expanded view of one chunk.
///
/// The dense form is allocated on the heap so the 160 KB payload does not blow
/// up stack frames. Meshing and lighting jobs own dense chunks while they work,
/// then compress back to [`super::PalettedChunk`] for long-term storage.
pub struct DenseChunk {
    /// Flat 32^3 block array, indexed by [`LocalPos::to_index`].
    pub blocks: Box<[Block; CHUNK_VOL]>,
    /// Sky-light values 0..=15.
    pub sky_light: Box<[u8; CHUNK_VOL]>,
    /// Per-voxel packed block-light: `(R << 8) | (G << 4) | B`.
    pub block_rgb: Box<[u16; CHUNK_VOL]>,
}

impl DenseChunk {
    /// Build a chunk where every voxel is the same block. Light arrays start at 0.
    pub fn new_filled(block: Block) -> Self {
        Self {
            blocks: Box::new([block; CHUNK_VOL]),
            sky_light: Box::new([0u8; CHUNK_VOL]),
            block_rgb: Box::new([0u16; CHUNK_VOL]),
        }
    }

    /// Shorthand for a chunk full of [`Block::Air`].
    pub fn empty() -> Self {
        Self::new_filled(Block::Air)
    }

    /// Read the block at a local position.
    #[inline]
    pub fn get(&self, p: LocalPos) -> Block {
        self.blocks[p.to_index()]
    }

    /// Write a block at a local position.
    #[inline]
    pub fn set(&mut self, p: LocalPos, b: Block) {
        self.blocks[p.to_index()] = b;
    }
}
