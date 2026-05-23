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
use crate::voxel::block::BlockRegistry;
use crate::voxel::coords::{ChunkCoord, LocalPos};
use crate::voxel::packed::Packed4Bit;
use serde::{Deserialize, Serialize};

/// Number of voxels in one chunk: 32 × 32 × 32 = 32 768.
pub const CHUNK_VOL: usize = 32 * 32 * 32;
pub const CHUNK_AREA: usize = 32 * 32;

pub const LIGHT_INPUT_UNKNOWN_Y: i16 = i16::MIN;

/// Pack `(R, G, B)` channels (each 0..=15) into the u16 layout used
/// by `DenseChunk::block_rgb`. Out-of-range inputs are masked to 4 bits.
#[inline]
pub fn pack_rgb(r: u8, g: u8, b: u8) -> u16 {
    debug_assert!(
        r < 16 && g < 16 && b < 16,
        "pack_rgb: channel out of range (max 15)"
    );
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

/// Worldgen/block-derived lighting metadata for one chunk.
///
/// This is the contract between chunk generation and lighting. The light
/// worker still computes per-voxel light, but it no longer has to guess
/// whether a missing +Y chunk means "open sky" or "unknown vertical context".
#[derive(Debug, Clone)]
pub struct ChunkLightInputs {
    /// First opaque voxel encountered scanning the local column top-down,
    /// as world Y. [`LIGHT_INPUT_UNKNOWN_Y`] means no opaque voxel inside
    /// this chunk column.
    pub first_opaque_y: Box<[i16; CHUNK_AREA]>,
    /// Highest generated terrain/surface Y known for this world column.
    /// Missing for disk-only chunks when no generator metadata was supplied.
    pub surface_y: Box<[i16; CHUNK_AREA]>,
    /// Sky level entering this chunk's top face when no +Y lit neighbor is
    /// available. `0` means the top context is unknown or blocked.
    pub top_sky: Box<[u8; CHUNK_AREA]>,
    /// Number of emissive voxels in this chunk, useful for diagnostics and
    /// future queue seeding.
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
                    let idx = crate::voxel::coords::LocalPos(glam::UVec3::new(x, y, z)).to_index();
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
    let out: &mut [u8; 33 * 33 * 33 * 4] = buf.as_mut().try_into().expect("size mismatch");
    // Scale 0..=15 → 0..=255 with rounding so 15 maps to 255 exactly.
    let scale = |v: u8| ((v as u32 * 255 + 7) / 15) as u8;
    for z in 0..33 {
        for y in 0..33 {
            for x in 0..33 {
                let (r, g, b, a) = sample_for_blob(dense, neighbors, x, y, z);
                let idx = (z * 33 * 33 + y * 33 + x) * 4;
                out[idx] = scale(r);
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
///
/// Opaque cells get a "halo" fill: their stored value is the max of
/// their 6 axial neighbors' values. The DenseChunk's BFS stores 0 in
/// opaque cells (light doesn't penetrate them), but when the shader
/// samples a face corner the trilinear filter pulls in the diagonally
/// adjacent opaque cell — without the halo, that 0 darkens the corner
/// even though the corner sits right next to a fully-lit air cell.
fn sample_for_blob(
    dense: &DenseChunk,
    neighbors: &Neighbors,
    x: usize,
    y: usize,
    z: usize,
) -> (u8, u8, u8, u8) {
    let (chunk_src, lx, ly, lz) = resolve_cell(dense, neighbors, x, y, z);
    let idx = crate::voxel::coords::LocalPos(glam::UVec3::new(lx as u32, ly as u32, lz as u32))
        .to_index();
    let block = chunk_src.blocks[idx];
    let (mut r, mut g, mut b) = unpack_rgb(chunk_src.block_rgb[idx]);
    let mut a = chunk_src.sky_light[idx] & 0x0F;

    // Halo fill for opaque cells. Only Air propagates light through
    // the BFS; Water and Leaves attenuate but still hold non-zero
    // values. We treat anything other than Air as "doesn't naturally
    // store usable light", and for those cells we look outward for a
    // brighter neighbor. (Water/Leaves rarely matter for the corner
    // artefact because their own stored light is already representative.)
    if block != Block::Air {
        // Walk the 6 axial neighbors in this chunk + neighbor borrow.
        // The query handles the +X/+Y/+Z and 0-edge cases (it walks the
        // 0..=32 grid, so an axis underflow / overflow reads -X/-Y/-Z
        // neighbors when available).
        for (dx, dy, dz) in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            let nz = z as isize + dz;
            // Bounds: -1..=32 inclusive. Index -1 reads the negative
            // face neighbour when available so chunk-edge negative faces
            // get the same air-side halo as +X/+Y/+Z faces.
            if nx < -1 || nx > 32 || ny < -1 || ny > 32 || nz < -1 || nz > 32 {
                continue;
            }
            let Some((ns, nlx, nly, nlz)) = resolve_cell_signed(dense, neighbors, nx, ny, nz)
            else {
                continue;
            };
            let nidx = crate::voxel::coords::LocalPos(glam::UVec3::new(
                nlx as u32, nly as u32, nlz as u32,
            ))
            .to_index();
            let (nr, ng, nb) = unpack_rgb(ns.block_rgb[nidx]);
            let na = ns.sky_light[nidx] & 0x0F;
            r = r.max(nr);
            g = g.max(ng);
            b = b.max(nb);
            a = a.max(na);
        }
    }
    (r, g, b, a)
}

fn resolve_cell_signed<'a>(
    dense: &'a DenseChunk,
    neighbors: &'a Neighbors,
    x: isize,
    y: isize,
    z: isize,
) -> Option<(&'a DenseChunk, usize, usize, usize)> {
    use crate::mesher::Face;
    let negative_axes = (x < 0) as u8 + (y < 0) as u8 + (z < 0) as u8;
    if negative_axes > 1 {
        return None;
    }
    if x < 0 {
        return Some(match neighbors.chunks[Face::NegX as usize] {
            Some(n) => (n, 31, y.clamp(0, 31) as usize, z.clamp(0, 31) as usize),
            None => (dense, 0, y.clamp(0, 31) as usize, z.clamp(0, 31) as usize),
        });
    }
    if y < 0 {
        return Some(match neighbors.chunks[Face::NegY as usize] {
            Some(n) => (n, x.clamp(0, 31) as usize, 31, z.clamp(0, 31) as usize),
            None => (dense, x.clamp(0, 31) as usize, 0, z.clamp(0, 31) as usize),
        });
    }
    if z < 0 {
        return Some(match neighbors.chunks[Face::NegZ as usize] {
            Some(n) => (n, x.clamp(0, 31) as usize, y.clamp(0, 31) as usize, 31),
            None => (dense, x.clamp(0, 31) as usize, y.clamp(0, 31) as usize, 0),
        });
    }
    Some(resolve_cell(
        dense, neighbors, x as usize, y as usize, z as usize,
    ))
}

/// Resolve a 0..=32 query coord into the appropriate `DenseChunk` and
/// local 0..=31 index. Index 32 on any axis crosses into the +X/+Y/+Z
/// neighbor's index 0; missing neighbors clamp to the local boundary
/// (index 31).
fn resolve_cell<'a>(
    dense: &'a DenseChunk,
    neighbors: &'a Neighbors,
    x: usize,
    y: usize,
    z: usize,
) -> (&'a DenseChunk, usize, usize, usize) {
    use crate::mesher::Face;
    if x == 32 {
        match neighbors.chunks[Face::PosX as usize] {
            Some(n) => (n, 0, y.min(31), z.min(31)),
            None => (dense, 31, y.min(31), z.min(31)),
        }
    } else if y == 32 {
        match neighbors.chunks[Face::PosY as usize] {
            Some(n) => (n, x.min(31), 0, z.min(31)),
            None => (dense, x.min(31), 31, z.min(31)),
        }
    } else if z == 32 {
        match neighbors.chunks[Face::PosZ as usize] {
            Some(n) => (n, x.min(31), y.min(31), 0),
            None => (dense, x.min(31), y.min(31), 31),
        }
    } else {
        (dense, x, y, z)
    }
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

    /// Read the block at a flat index `0..CHUNK_VOL` without decompressing.
    /// One palette indirection per call; allocates nothing.
    #[inline]
    pub fn block_at(&self, idx: usize) -> Block {
        self.palette[self.indices.get(idx) as usize]
    }

    /// Read the sky-light level at a flat index `0..CHUNK_VOL` without
    /// decompressing. Returns 0..=15 directly from the packed nibble.
    #[inline]
    pub fn sky_light_at(&self, idx: usize) -> u8 {
        self.sky_light.get(idx)
    }

    /// Read the (R, G, B) block-light tuple at a flat index `0..CHUNK_VOL`
    /// without decompressing. Each channel is 0..=15.
    #[inline]
    pub fn block_rgb_at(&self, idx: usize) -> (u8, u8, u8) {
        (
            self.block_red.get(idx),
            self.block_green.get(idx),
            self.block_blue.get(idx),
        )
    }

    /// Write the sky-light level at a flat index. Caller must hold a
    /// unique `&mut self` (e.g., via `Arc::make_mut`).
    #[inline]
    pub fn set_sky_light_at(&mut self, idx: usize, value: u8) {
        self.sky_light.set(idx, value);
    }

    /// Write the (R, G, B) block-light tuple at a flat index. Caller must
    /// hold a unique `&mut self`.
    #[inline]
    pub fn set_block_rgb_at(&mut self, idx: usize, r: u8, g: u8, b: u8) {
        self.block_red.set(idx, r);
        self.block_green.set(idx, g);
        self.block_blue.set(idx, b);
    }
}

/// Lifecycle marker for a chunk slot. The state machine is the world's
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

/// Six-bit mask keyed by [`crate::mesher::Face`] discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FaceMask(u8);

impl FaceMask {
    pub const NONE: Self = Self(0);

    #[inline]
    pub fn set(&mut self, face: crate::mesher::Face) {
        self.0 |= 1 << face as u8;
    }

    #[inline]
    pub fn contains(self, face: crate::mesher::Face) -> bool {
        self.0 & (1 << face as u8) != 0
    }
}

impl From<[bool; 6]> for FaceMask {
    fn from(value: [bool; 6]) -> Self {
        let mut mask = Self::NONE;
        for (i, changed) in value.iter().enumerate() {
            if *changed {
                mask.0 |= 1 << i;
            }
        }
        mask
    }
}

/// Chunk-owned lighting lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LightState {
    #[default]
    Unlit,
    Queued,
    Lighting {
        version: u64,
    },
    Lit {
        version: u64,
    },
    NeedsBorderReconcile,
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
    /// Monotonic version of block data relevant to lighting. Relight
    /// jobs snapshot this at dispatch and stale results are dropped.
    pub light_version: u64,
    pub light_state: LightState,
    pub unresolved_borders: FaceMask,
    /// Per-column world-Y of the lowest sky-source cell. Built by
    /// `crate::lighting::ChunkSkyLightSources::build_from_dense` at
    /// `World::insert` time and rebuilt whenever the chunk's blocks
    /// change. Kept as cheap metadata for future sky-source optimizations.
    ///
    /// Defaults to a heightmap full of `NO_SOURCE_FLOOR`, which is
    /// the safe value for a freshly-defaulted `ChunkMeta` — no
    /// floor means "treat every cell as a potential source"
    /// (matches today's BFS column-drop default of `light = 15`).
    pub sky_sources: crate::lighting::ChunkSkyLightSources,
    /// Generator/block manifest consumed by the relight worker.
    pub light_inputs: ChunkLightInputs,
    /// True when a relight commit wrote new voxel light. The current
    /// commit path uploads the returned light blob immediately; this flag
    /// remains useful for diagnostics and future deferred upload paths.
    pub light_gpu_dirty: bool,
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
        let p = LocalPos(UVec3::new(
            CHUNK_DIM_U - 1,
            CHUNK_DIM_U - 1,
            CHUNK_DIM_U - 1,
        ));
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

    #[test]
    fn light_volume_halo_reads_negative_face_neighbor() {
        let mut d = DenseChunk::empty();
        let edge = LocalPos(UVec3::new(0, 10, 10));
        d.set(edge, Block::Stone);

        let mut neg_x = DenseChunk::empty();
        let neighbor_air = LocalPos(UVec3::new(31, 10, 10));
        neg_x.sky_light[neighbor_air.to_index()] = 15;

        let n = Neighbors {
            chunks: [None, Some(&neg_x), None, None, None, None],
        };
        let blob = build_light_volume_blob(&d, &n);
        let blob_idx = (10 * 33 * 33 + 10 * 33) * 4;
        assert!(
            blob[blob_idx + 3] >= 240,
            "negative boundary halo did not import sky light: {}",
            blob[blob_idx + 3]
        );
    }

    #[test]
    fn paletted_block_at_matches_decompressed_get() {
        let mut d = DenseChunk::empty();
        d.set(LocalPos(UVec3::new(0, 0, 0)), Block::Stone);
        d.set(LocalPos(UVec3::new(31, 31, 31)), Block::Water);
        d.set(LocalPos(UVec3::new(5, 10, 20)), Block::Torch);
        let p = PalettedChunk::compress(&d);
        for i in 0..CHUNK_VOL {
            assert_eq!(p.block_at(i), d.blocks[i], "block_at mismatch at idx {i}");
        }
    }

    #[test]
    fn paletted_sky_light_at_matches_decompressed() {
        let mut d = DenseChunk::empty();
        d.sky_light[0] = 15;
        d.sky_light[100] = 7;
        d.sky_light[CHUNK_VOL - 1] = 3;
        let p = PalettedChunk::compress(&d);
        assert_eq!(p.sky_light_at(0), 15);
        assert_eq!(p.sky_light_at(100), 7);
        assert_eq!(p.sky_light_at(CHUNK_VOL - 1), 3);
        assert_eq!(p.sky_light_at(50), 0); // untouched
    }

    #[test]
    fn paletted_block_rgb_at_matches_decompressed() {
        let mut d = DenseChunk::empty();
        d.block_rgb[10] = pack_rgb(15, 7, 3);
        d.block_rgb[200] = pack_rgb(0, 8, 12);
        let p = PalettedChunk::compress(&d);
        assert_eq!(p.block_rgb_at(10), (15, 7, 3));
        assert_eq!(p.block_rgb_at(200), (0, 8, 12));
        assert_eq!(p.block_rgb_at(11), (0, 0, 0)); // untouched
    }

    #[test]
    fn paletted_set_sky_light_at_round_trip() {
        let mut p = PalettedChunk::all_air();
        p.set_sky_light_at(5, 12);
        p.set_sky_light_at(6, 8);
        assert_eq!(p.sky_light_at(5), 12);
        assert_eq!(p.sky_light_at(6), 8);
        assert_eq!(p.sky_light_at(7), 0); // untouched
    }

    #[test]
    fn paletted_set_block_rgb_at_round_trip() {
        let mut p = PalettedChunk::all_air();
        p.set_block_rgb_at(42, 11, 9, 5);
        assert_eq!(p.block_rgb_at(42), (11, 9, 5));
        assert_eq!(p.block_rgb_at(43), (0, 0, 0)); // untouched
        p.set_block_rgb_at(42, 0, 0, 0);
        assert_eq!(p.block_rgb_at(42), (0, 0, 0));
    }
}
