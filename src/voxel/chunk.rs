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

/// Pack `(R, G, B)` channels (each 0..=15) into the u16 layout used
/// by `DenseChunk::block_rgb`. Out-of-range inputs are masked to 4 bits.
#[inline]
pub fn pack_rgb(r: u8, g: u8, b: u8) -> u16 {
    debug_assert!(r < 16 && g < 16 && b < 16, "pack_rgb: channel out of range (max 15)");
    ((r as u16 & 0x0F) << 8) | ((g as u16 & 0x0F) << 4) | (b as u16 & 0x0F)
}

/// Inverse of `pack_rgb`: extract R/G/B (each 0..=15) from a packed cell.
#[inline]
pub fn unpack_rgb(cell: u16) -> (u8, u8, u8) {
    let r = ((cell >> 8) & 0x0F) as u8;
    let g = ((cell >> 4) & 0x0F) as u8;
    let b = (cell & 0x0F) as u8;
    (r, g, b)
}

/// Scalar brightness for a packed cell, used by the mesher/legacy shader
/// path until PR 3 swaps to a 3D light volume.
#[inline]
pub fn rgb_brightness(cell: u16) -> u8 {
    let (r, g, b) = unpack_rgb(cell);
    r.max(g).max(b)
}

/// Build the 33³ Rgba8Unorm blob for a chunk's GPU light volume. Each
/// axis spans 0..=32 — index 32 reads from the +X/+Y/+Z neighbor's
/// index 0 so trilinear sampling at the chunk's far face sees valid
/// neighbor values (no clamp artefact). Cells where the neighbor isn't
/// loaded clamp to the local boundary value.
///
/// Layout per voxel:
///   R = block_red   / 15
///   G = block_green / 15
///   B = block_blue  / 15
///   A = sky_light   / 15
pub fn build_light_volume_blob(
    dense: &DenseChunk,
    neighbors: &Neighbors,
) -> Box<[u8; 33 * 33 * 33 * 4]> {
    let mut buf = vec![0u8; 33 * 33 * 33 * 4].into_boxed_slice();
    let out: &mut [u8; 33 * 33 * 33 * 4] =
        buf.as_mut().try_into().expect("size mismatch");
    // Scale 0..=15 → 0..=255 with rounding so 15 maps to 255 exactly.
    let scale = |v: u8| ((v as u32 * 255 + 7) / 15) as u8;
    for z in 0..33 {
        for y in 0..33 {
            for x in 0..33 {
                let (r, g, b, a) = sample_for_blob(dense, neighbors, x, y, z);
                let idx = (z * 33 * 33 + y * 33 + x) * 4;
                out[idx]     = scale(r);
                out[idx + 1] = scale(g);
                out[idx + 2] = scale(b);
                out[idx + 3] = scale(a);
            }
        }
    }
    buf.try_into().expect("size mismatch")
}

/// Sample (R, G, B, sky) at local index (x, y, z) where each axis is
/// 0..=32. Indices 0..=31 read from this chunk; index 32 reads from the
/// +X/+Y/+Z neighbor's index 0. If the neighbor isn't loaded, returns
/// the local boundary cell (clamped to index 31).
fn sample_for_blob(
    dense: &DenseChunk,
    neighbors: &Neighbors,
    x: usize,
    y: usize,
    z: usize,
) -> (u8, u8, u8, u8) {
    use crate::mesher::Face;
    let (chunk_src, lx, ly, lz): (&DenseChunk, usize, usize, usize) = if x == 32 {
        match neighbors.chunks[Face::PosX as usize] {
            Some(n) => (n, 0, y.min(31), z.min(31)),
            None    => (dense, 31, y.min(31), z.min(31)),
        }
    } else if y == 32 {
        match neighbors.chunks[Face::PosY as usize] {
            Some(n) => (n, x.min(31), 0, z.min(31)),
            None    => (dense, x.min(31), 31, z.min(31)),
        }
    } else if z == 32 {
        match neighbors.chunks[Face::PosZ as usize] {
            Some(n) => (n, x.min(31), y.min(31), 0),
            None    => (dense, x.min(31), y.min(31), 31),
        }
    } else {
        (dense, x, y, z)
    };
    let idx = crate::voxel::coords::LocalPos(glam::UVec3::new(lx as u32, ly as u32, lz as u32))
        .to_index();
    let (r, g, b) = unpack_rgb(chunk_src.block_rgb[idx]);
    let a = chunk_src.sky_light[idx] & 0x0F;
    (r, g, b, a)
}

/// Fully-expanded view of one chunk. ~160 KB:
/// 64 KB blocks + 32 KB sky-light + 64 KB block-rgb.
///
/// The light arrays only use the bottom 4 bits per byte (range `0..=15`);
/// they sit unpacked here for fast random access during the BFS, then get
/// packed back into a `Packed4Bit` when stored long-term.
pub struct DenseChunk {
    /// Flat 32³ block array, indexed by [`LocalPos::to_index`].
    pub blocks: Box<[Block; CHUNK_VOL]>,
    /// Sky-light values 0..=15, populated by the sky-light BFS.
    pub sky_light: Box<[u8; CHUNK_VOL]>,
    /// Per-voxel packed block-light: `(R << 8) | (G << 4) | B`,
    /// each channel 4 bits (0..=15). Top 4 bits unused. Populated by
    /// the colored block-light BFS in `lighting::recompute_chunk`.
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
    /// Red channel of per-voxel block light (0..=15).
    pub block_red: Packed4Bit,
    /// Green channel of per-voxel block light (0..=15).
    pub block_green: Packed4Bit,
    /// Blue channel of per-voxel block light (0..=15).
    pub block_blue: Packed4Bit,
}

impl PalettedChunk {
    /// All-air chunk: single-entry palette, all indices zero, no light.
    /// Used as the initial state for newly-introduced chunk slots.
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
    /// Monotonic version of the chunk's *data* — incremented every
    /// time `set_block` or a relight swap changes the `PalettedChunk`.
    /// Mesh jobs snapshot this at spawn time and carry it in their
    /// result; the upload path rejects results whose version is
    /// older than the chunk's current `mesh_version` so a slow
    /// streaming mesh job can't overwrite the fresh edit-triggered
    /// mesh that completed first (the visible "block flickers back
    /// for a moment" artefact).
    pub mesh_version: u64,
}

/// Legacy v1 paletted-chunk layout used by region files written before
/// the colored-block-light upgrade. Only deserialized — never written.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PalettedChunkV1 {
    pub palette: Vec<Block>,
    pub indices: Packed4Bit,
    pub sky_light: Packed4Bit,
    pub block_light: Packed4Bit,
}

impl From<PalettedChunkV1> for PalettedChunk {
    /// Convert v1 (single-channel) to v2 (RGB) by mirroring the brightness
    /// into all three channels. Old saves render the same as today
    /// (max(R,G,B) = old block_light) until chunks are re-relit.
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
        d.block_rgb[200] = pack_rgb(0x7, 0x0, 0x0);

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
        assert_eq!(d2.block_rgb[200], pack_rgb(0x7, 0x0, 0x0));
    }

    #[test]
    fn paletted_compress_dedupes_palette() {
        let d = DenseChunk::new_filled(Block::Stone);
        let p = PalettedChunk::compress(&d);
        assert_eq!(p.palette.len(), 1);
    }

    #[test]
    fn light_volume_blob_size_and_layout() {
        let mut d = DenseChunk::empty();
        d.sky_light[0] = 15;
        d.block_rgb[0] = pack_rgb(15, 0, 0);
        let n = Neighbors { chunks: [None; 6] };
        let blob = build_light_volume_blob(&d, &n);
        assert_eq!(blob.len(), 33 * 33 * 33 * 4);
        // First voxel (0,0,0): R should be ~255 (from block_rgb's R=15), A also ~255.
        assert!(blob[0] >= 240, "R channel scaled wrong: {}", blob[0]);
        assert_eq!(blob[1], 0);
        assert_eq!(blob[2], 0);
        assert!(blob[3] >= 240, "A channel scaled wrong: {}", blob[3]);
    }
}
