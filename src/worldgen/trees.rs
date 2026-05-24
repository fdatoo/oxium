//! Tree placement: per-cell deterministic rolls, `TreeKind::{Oak, Palm}`,
//! palm-shape stamping.
//!
//! Tree placement is **cell-based**: the world is divided into
//! `TREE_CELL_SIZE × TREE_CELL_SIZE` (8×8) column cells. Each cell
//! independently rolls a biome-weighted probability (`TREE_RATE_*`)
//! to decide whether a tree should appear there, and if so, jitters
//! the trunk position deterministically within a `TREE_MARGIN` border
//! band. Cells in adjacent chunks are also evaluated so cross-chunk
//! trees (trunk in a neighbour, leaves in this chunk) are placed
//! correctly.
//!
//! ### What lives here vs mod.rs
//!
//! The **data types** and **helper functions** live here:
//! - [`Tree`]: per-cell tree placement metadata (origin, height, kind).
//! - [`try_set_air`]: place a block if the voxel is Air and inside the chunk.
//! - [`tree_hash`]: deterministic (seed, x, z, salt) → u32 mixer.
//!
//! The **Generator impl methods** (`add_trees`, `tree_in_cell_with_regions`,
//! `stamp_tree`) live in `trees_impl.rs`. They access private `Generator`
//! fields (`self.seed`, `self.heightmap`, `self.density`) and call the
//! helpers above via `trees::try_set_air` / `trees::tree_hash`.
//!
//! ### Design notes
//!
//! - **Determinism:** `tree_hash(seed, cell_x, cell_z, salt)` drives all
//!   per-cell rolls so the same seed always produces the same forest layout.
//! - **Cross-chunk correctness:** `fill_chunk` expands the scan radius by
//!   `TREE_MARGIN` extra cells on each side so frond overhangs and canopy
//!   slabs originating outside the chunk boundary are still placed.
//! - **Beach palms:** the `on_beach` shortcut in `tree_in_cell_with_regions`
//!   divides palm rate by 4 so tropical beaches read as *scattered* palms
//!   rather than dense clumps.
//!
//! See `docs/book/content/part-4-chunk-fill/4.9-trees.mdx` for the visual
//! design rationale and `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`
//! for the biome × tree-rate table.

use crate::voxel::block::Block;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{CHUNK_DIM, ChunkCoord, LocalPos};
use glam::UVec3;

use crate::worldgen::biome::TreeKind;

/// Per-cell tree placement metadata. Constructed by
/// `Generator::tree_in_cell_with_regions`; consumed by
/// `Generator::stamp_tree` to write trunk + canopy blocks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tree {
    /// World-space X coordinate of the trunk base.
    pub(crate) wx: i32,
    /// World-space Z coordinate of the trunk base.
    pub(crate) wz: i32,
    /// World-space Y of the topmost solid block under the trunk.
    /// The trunk itself starts at `base_y + 1`.
    pub(crate) base_y: i32,
    /// Number of `Wood` blocks above `base_y` (inclusive).
    /// Oak: 4–6; Palm: 7–9.
    pub(crate) trunk_h: i32,
    /// Canopy shape — [`TreeKind::Oak`] for temperate/boreal,
    /// [`TreeKind::Palm`] for tropical.
    pub(crate) kind: TreeKind,
}

/// Write `b` at world coords `(wx, wy, wz)` if they fall inside
/// `coord`'s 32³ volume *and* the existing block is `Air`.
///
/// Both conditions are required:
/// - The bounds check stops a tree from writing into a neighbouring chunk.
/// - The air check stops the trunk from carving through hills and stops
///   adjacent chunks' calls from overwriting each other.
pub(crate) fn try_set_air(
    coord: ChunkCoord,
    out: &mut DenseChunk,
    wx: i32,
    wy: i32,
    wz: i32,
    b: Block,
) {
    let chunk_origin = coord.origin().0;
    let lx = wx - chunk_origin.x;
    let ly = wy - chunk_origin.y;
    let lz = wz - chunk_origin.z;
    if lx < 0 || ly < 0 || lz < 0 || lx >= CHUNK_DIM || ly >= CHUNK_DIM || lz >= CHUNK_DIM {
        return;
    }
    let lp = LocalPos(UVec3::new(lx as u32, ly as u32, lz as u32));
    if out.get(lp) != Block::Air {
        return;
    }
    out.set(lp, b);
}

/// Deterministic per-cell hash: `(seed, x, z, salt) → u32`.
///
/// Uses the xor-shift / golden-ratio multiply pattern common in
/// shader noise functions. Not cryptographic — good enough for
/// tree placement where the only requirement is low visual
/// correlation between nearby cells.
///
/// `salt` separates different rolls in the same cell (trunk-X jitter,
/// trunk-Z jitter, height roll, tree-vs-no-tree roll).
pub(crate) fn tree_hash(seed: u64, x: i32, z: i32, salt: u32) -> u32 {
    let mut h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (x as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h = h.rotate_left(13);
    h ^= (z as i64 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h = h.rotate_left(17);
    h ^= (salt as u64).wrapping_mul(0xCC9E_2D51_1B87_3593);
    ((h ^ (h >> 33)) as u32) ^ ((h >> 16) as u32)
}
