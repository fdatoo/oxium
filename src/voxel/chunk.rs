//! [`DenseChunk`] — the hot, transient, fully-unpacked form of a chunk.
//!
//! `DenseChunk` is *never* the long-term in-memory storage: keeping every
//! loaded chunk as 128 KB of flat arrays would burn ~2 GB at our target
//! render distance. Instead, chunks live as a palette-compressed
//! `PalettedChunk` (see `voxel::paletted` in M3) and are decompressed into a
//! `DenseChunk` only for hot, transient work — meshing, lighting recomputes,
//! and edit batches.
//!
//! The dense form is allocated on the heap (`Box<[T; N]>`) so that the 128 KB
//! payload doesn't blow up stack frames, and so that the mesher/lighter can
//! own a chunk during a job without paying any borrow cost on the world.

use crate::voxel::block::Block;
use crate::voxel::coords::LocalPos;
use crate::voxel::packed::Packed4Bit;
use serde::{Deserialize, Serialize};

/// Number of voxels in one chunk: 32 × 32 × 32 = 32 768.
pub const CHUNK_VOL: usize = 32 * 32 * 32;

/// Fully-expanded view of one chunk. ~128 KB:
/// 64 KB blocks + 32 KB sky-light + 32 KB block-light.
///
/// The light arrays only use the bottom 4 bits per byte (range `0..=15`);
/// they sit unpacked here for fast random access during the BFS, then get
/// packed back into a `Packed4Bit` when stored long-term.
pub struct DenseChunk {
    /// Flat 32³ block array, indexed by [`LocalPos::to_index`].
    pub blocks: Box<[Block; CHUNK_VOL]>,
    /// Sky-light values 0..=15, populated by the sky-light BFS.
    pub sky_light: Box<[u8; CHUNK_VOL]>,
    /// Block-light values 0..=15, populated by the block-light BFS.
    pub block_light: Box<[u8; CHUNK_VOL]>,
}

impl DenseChunk {
    /// Build a chunk where every voxel is the same block. Light arrays start at 0.
    pub fn new_filled(block: Block) -> Self {
        Self {
            blocks: Box::new([block; CHUNK_VOL]),
            sky_light: Box::new([0u8; CHUNK_VOL]),
            block_light: Box::new([0u8; CHUNK_VOL]),
        }
    }

    /// Shorthand for a chunk full of [`Block::Air`].
    pub fn empty() -> Self {
        Self::new_filled(Block::Air)
    }

    /// Read the block at a local position. `LocalPos` is range-checked to
    /// `0..32` at construction, so the index is always in-bounds.
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

/// Canonical *in-RAM and on-disk* chunk form: a palette of distinct blocks
/// plus a 4-bit-per-voxel index array. Compresses a typical chunk from
/// ~128 KB (dense) to ~50 KB — the difference between 2 GB and 850 MB at
/// our target render distance.
///
/// `palette[indices.get(i)] == block at voxel i`.
///
/// M3 limits the palette to 16 entries (so each index fits in 4 bits).
/// A future widen to 8 bits per voxel happens automatically if any chunk
/// needs >16 distinct blocks — we'd promote `indices` to a richer
/// `BitPackedArray` then. v0 chunks empirically average 3–6 palette
/// entries, so the 4-bit cap is comfortable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalettedChunk {
    /// Distinct block kinds present in this chunk, indexed by `indices`.
    pub palette: Vec<Block>,
    /// 4 bits per voxel — an index into `palette`.
    pub indices: Packed4Bit,
    /// Sky-light values, 0..=15.
    pub sky_light: Packed4Bit,
    /// Block-light values, 0..=15.
    pub block_light: Packed4Bit,
}

impl PalettedChunk {
    /// All-air chunk: single-entry palette, all indices zero, no light.
    /// Used as the initial state for newly-introduced chunk slots.
    pub fn all_air() -> Self {
        Self {
            palette: vec![Block::Air],
            indices: Packed4Bit::zeros(CHUNK_VOL),
            sky_light: Packed4Bit::zeros(CHUNK_VOL),
            block_light: Packed4Bit::zeros(CHUNK_VOL),
        }
    }

    /// Build a `PalettedChunk` from a `DenseChunk`. Palette is built in
    /// first-seen order; both light arrays are bit-packed.
    pub fn compress(dense: &DenseChunk) -> Self {
        // `lookup` maps `Block as usize` → palette index. We use `u8::MAX`
        // as the "unseen" sentinel because palette indices are 0..=15.
        let mut palette: Vec<Block> = Vec::with_capacity(8);
        let mut lookup = [u8::MAX; super::block::BLOCK_COUNT];
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
        let mut blk = Packed4Bit::zeros(CHUNK_VOL);
        for i in 0..CHUNK_VOL {
            sky.set(i, dense.sky_light[i] & 0x0F);
            blk.set(i, dense.block_light[i] & 0x0F);
        }

        Self {
            palette,
            indices,
            sky_light: sky,
            block_light: blk,
        }
    }

    /// Inverse of [`compress`](Self::compress) — produce a fresh `DenseChunk`.
    /// The mesher and lighting passes always work on dense data, so this is
    /// called once at the start of any hot job and recompressed at the end.
    pub fn decompress(&self) -> DenseChunk {
        let mut blocks = Box::new([Block::Air; CHUNK_VOL]);
        for i in 0..CHUNK_VOL {
            let palette_idx = self.indices.get(i) as usize;
            blocks[i] = self.palette[palette_idx];
        }
        let mut sky = Box::new([0u8; CHUNK_VOL]);
        let mut blk = Box::new([0u8; CHUNK_VOL]);
        for i in 0..CHUNK_VOL {
            sky[i] = self.sky_light.get(i);
            blk[i] = self.block_light.get(i);
        }
        DenseChunk {
            blocks,
            sky_light: sky,
            block_light: blk,
        }
    }

    /// Read a single block by local position without going through a full
    /// decompress. Useful for the world's `get_block` query path.
    pub fn get(&self, p: LocalPos) -> Block {
        self.palette[self.indices.get(p.to_index()) as usize]
    }
}

/// Lifecycle marker for a chunk slot. The state machine is the engine's
/// rule for *what can happen next* to a chunk:
///
/// `Empty` → `Generating` → `Generated` → `Meshing` → `Ready`
///
/// (Edits move `Ready` back to `Meshing`; light dirty moves back to
/// `Generated` and re-runs lighting.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkState {
    #[default]
    Empty,
    Generating,
    Generated,
    Meshing,
    Ready,
}

/// Tracks which expensive recomputes the chunk owes. Edits set these;
/// jobs clear them after running.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkDirty {
    pub mesh: bool,
    pub light: bool,
}

/// Read-only references to (up to) the six neighbouring chunks in
/// [`crate::mesher::Face`] order. Used by both the lighting BFS and the
/// neighbour-aware mesher so they can sample one block off the chunk's
/// edge.
pub struct Neighbors<'a> {
    pub chunks: [Option<&'a DenseChunk>; 6],
}

/// Per-chunk bookkeeping. Lives alongside the [`PalettedChunk`] in
/// `ChunkSlot::Stored`.
///
/// `state` and `dirty` drive the streaming/relight scheduler; `modified`
/// is read by the persistence layer to decide whether to flush the chunk
/// on unload/autosave. The renderer holds its own
/// `HashMap<ChunkCoord, [Option<ChunkGpu>; 3]>` for LOD mesh storage —
/// no separate `MeshHandle` is needed in v0.
#[derive(Debug, Default)]
pub struct ChunkMeta {
    pub state: ChunkState,
    pub dirty: ChunkDirty,
    /// True if this chunk has been edited by the player since load.
    /// Persistence uses this to skip saving unmodified, regenerable chunks.
    pub modified: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::coords::CHUNK_DIM_U;
    use glam::UVec3;

    #[test]
    fn new_filled_returns_all_block() {
        let c = DenseChunk::new_filled(Block::Stone);
        assert!(c.blocks.iter().all(|&b| b == Block::Stone));
    }

    #[test]
    fn empty_is_all_air() {
        let c = DenseChunk::empty();
        assert!(c.blocks.iter().all(|&b| b == Block::Air));
    }

    #[test]
    fn set_get_round_trip() {
        let mut c = DenseChunk::empty();
        let p = LocalPos(UVec3::new(5, 10, 20));
        c.set(p, Block::Dirt);
        assert_eq!(c.get(p), Block::Dirt);
    }

    #[test]
    fn boundary_indices_in_range() {
        let p = LocalPos(UVec3::new(CHUNK_DIM_U - 1, CHUNK_DIM_U - 1, CHUNK_DIM_U - 1));
        assert_eq!(p.to_index(), CHUNK_VOL - 1);
    }

    #[test]
    fn paletted_all_air_round_trip() {
        let p = PalettedChunk::all_air();
        let d = p.decompress();
        assert!(d.blocks.iter().all(|&b| b == Block::Air));
    }

    #[test]
    fn paletted_compress_decompress_preserves_blocks() {
        let mut d = DenseChunk::empty();
        d.set(LocalPos(UVec3::new(0, 0, 0)), Block::Stone);
        d.set(LocalPos(UVec3::new(31, 31, 31)), Block::Grass);
        d.set(LocalPos(UVec3::new(5, 10, 20)), Block::Water);
        d.sky_light[100] = 0xA;
        d.block_light[200] = 0x7;

        let p = PalettedChunk::compress(&d);
        let d2 = p.decompress();

        assert_eq!(
            d2.blocks[LocalPos(UVec3::new(0, 0, 0)).to_index()],
            Block::Stone
        );
        assert_eq!(
            d2.blocks[LocalPos(UVec3::new(31, 31, 31)).to_index()],
            Block::Grass
        );
        assert_eq!(
            d2.blocks[LocalPos(UVec3::new(5, 10, 20)).to_index()],
            Block::Water
        );
        assert_eq!(d2.sky_light[100], 0xA);
        assert_eq!(d2.block_light[200], 0x7);
    }

    #[test]
    fn paletted_compress_dedupes_palette() {
        let d = DenseChunk::new_filled(Block::Stone);
        let p = PalettedChunk::compress(&d);
        assert_eq!(p.palette.len(), 1);
    }
}
