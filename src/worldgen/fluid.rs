//! Connected-component flood-fill that settles aquifer-placed fluid.
//!
//! The aquifer system places `Water` or `Lava` blocks one voxel at a
//! time. When a fluid voxel lands inside a carved cave with no solid
//! beneath it (and no other supporting fluid neighbour), it would
//! "float in midair" in our static voxel world. In Minecraft water
//! flows downward; we don't have runtime fluid mechanics yet, so a
//! one-shot flood-fill at chunk-build time settles the placement:
//! unsupported fluid bodies are demoted to `Air`.
//!
//! The function operates on a per-chunk boolean mask
//! (`aquifer_mask`) that records which voxels came from the aquifer
//! path vs the ocean/lake surface flood (which is always supported by
//! definition and must not be touched). Only masked voxels
//! participate in flood-fill; non-masked fluid is treated as a
//! *support source* for adjacent masked fluid (e.g. an aquifer pool
//! touching the ocean is supported by the ocean body).
//!
//! ## Support rules (per-voxel, bottom-up propagation)
//!
//! Each masked fluid voxel is **supported** iff the voxel directly
//! beneath it is one of:
//!
//! * A solid block, OR
//! * A non-masked fluid voxel (the body below us is an external
//!   supported body — ocean, lake, future runtime-placed source),
//!   OR
//! * Another masked fluid voxel that was itself determined to be
//!   supported (the support chain propagates upward through fluid
//!   columns), OR
//! * Off the bottom of the chunk (`y = 0` voxels are
//!   conservatively supported; the chunk below presumably contains
//!   the floor).
//!
//! Lateral and overhead neighbours do NOT support: rock walls and
//! ceilings can't hold water up, and laterally-adjacent fluid
//! bodies don't either — they'd simply level out by flowing.
//!
//! Unsupported masked voxels are demoted to `Air`. The bottom-up
//! traversal means every voxel's support status is known before any
//! voxel above it is evaluated, so a single pass suffices — no
//! iteration to convergence needed.
//!
//! ## Reuse for future fluid mechanics
//!
//! The same flood-fill primitive is intended to back runtime fluid
//! operations (block-broken events that disturb a body, source
//! placement that spreads outward, etc.). The signature deliberately
//! takes a generic "membership mask" + a chunk; a future runtime
//! wrapper can build the mask differently (e.g. "all water in this
//! affected region") without modifying the algorithm.
//!
//! ## Complexity
//!
//! `O(chunk_volume)` — every voxel is visited at most once across
//! all flood-fills in the chunk. Memory is one `bool` per voxel
//! (≈ 32 KB for a 32³ chunk) plus a reused per-component stack.

use crate::voxel::block::Block;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{CHUNK_DIM_U, LocalPos};
use glam::UVec3;

/// Settle aquifer-placed fluid in `chunk`. Walks every masked
/// `Water`/`Lava` voxel bottom-up and demotes any voxel whose
/// support chain (recursively, through fluid columns) does not
/// resolve to a solid block, an external fluid body, or the chunk
/// floor. Non-masked fluid voxels are never touched.
///
/// Panics in debug builds if `aquifer_mask.len() != CHUNK_DIM_U³`.
pub fn settle_fluid(chunk: &mut DenseChunk, aquifer_mask: &[bool]) {
    let chunk_volume = (CHUNK_DIM_U as usize).pow(3);
    debug_assert_eq!(aquifer_mask.len(), chunk_volume);

    // Phase 1: bottom-up support propagation. Each masked fluid
    // voxel's support depends only on the voxel directly below it,
    // and that voxel was visited in a previous iteration of the
    // outer Y loop — so a single pass suffices.
    let mut supported = vec![false; chunk_volume];
    for y in 0..CHUNK_DIM_U {
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let idx = chunk_idx(x, y, z);
                if !aquifer_mask[idx] {
                    continue;
                }
                if !is_fluid(chunk.get(LocalPos(UVec3::new(x, y, z)))) {
                    // Defensive: masked voxel that isn't fluid
                    // (shouldn't normally happen).
                    continue;
                }

                if y == 0 {
                    // Chunk floor: conservatively assume the chunk
                    // below provides a floor. Avoids cross-chunk
                    // cascades when a water body straddles the seam.
                    supported[idx] = true;
                    continue;
                }

                let below_idx = chunk_idx(x, y - 1, z);
                let below = chunk.get(LocalPos(UVec3::new(x, y - 1, z)));
                supported[idx] = if matches!(below, Block::Air) {
                    // Air below: no support.
                    false
                } else if is_fluid(below) {
                    if aquifer_mask[below_idx] {
                        // Masked fluid below: support chains
                        // upward iff the below voxel is itself
                        // supported.
                        supported[below_idx]
                    } else {
                        // Non-masked fluid below (ocean, lake,
                        // future runtime-placed source): always
                        // counts as a supporting body.
                        true
                    }
                } else {
                    // Solid block below: classical floor.
                    true
                };
            }
        }
    }

    // Phase 2: demote unsupported masked fluid in place.
    for y in 0..CHUNK_DIM_U {
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let idx = chunk_idx(x, y, z);
                if !aquifer_mask[idx] || supported[idx] {
                    continue;
                }
                let pos = LocalPos(UVec3::new(x, y, z));
                if is_fluid(chunk.get(pos)) {
                    chunk.set(pos, Block::Air);
                }
            }
        }
    }
}

#[inline]
fn chunk_idx(x: u32, y: u32, z: u32) -> usize {
    let dim = CHUNK_DIM_U as usize;
    (x as usize) + dim * (y as usize) + dim * dim * (z as usize)
}

#[inline]
fn is_fluid(b: Block) -> bool {
    matches!(b, Block::Water | Block::Lava)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk() -> DenseChunk {
        DenseChunk::empty()
    }

    fn set_mask(mask: &mut [bool], x: u32, y: u32, z: u32) {
        mask[chunk_idx(x, y, z)] = true;
    }

    fn make_mask() -> Vec<bool> {
        vec![false; (CHUNK_DIM_U as usize).pow(3)]
    }

    /// Lone water voxel suspended in air (no neighbours, no boundary).
    /// Should be demoted.
    #[test]
    fn isolated_floating_voxel_is_demoted() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, y, z) = (15, 15, 15); // middle of chunk
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
        set_mask(&mut mask, x, y, z);

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Air);
    }

    /// Water voxel with solid block below — supported, kept.
    #[test]
    fn voxel_with_solid_below_is_kept() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, y, z) = (15, 15, 15);
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
        chunk.set(LocalPos(UVec3::new(x, y - 1, z)), Block::Stone);
        set_mask(&mut mask, x, y, z);

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Water);
    }

    /// Horizontal slab of water — three voxels side by side in air,
    /// no support anywhere. Entire component demoted as a group.
    #[test]
    fn floating_horizontal_slab_demoted_as_a_unit() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let y = 15;
        let z = 15;
        for x in 14..=16 {
            chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
            set_mask(&mut mask, x, y, z);
        }

        settle_fluid(&mut chunk, &mask);

        for x in 14..=16 {
            assert_eq!(
                chunk.get(LocalPos(UVec3::new(x, y, z))),
                Block::Air,
                "slab voxel x={x} should be demoted"
            );
        }
    }

    /// Slab of three voxels with a single solid only beneath the
    /// rightmost. Per-voxel support means only the voxel directly
    /// above the solid is kept; the two hangers drain. This is the
    /// regression test for the "cave ceiling water" bug — previously
    /// one rooted voxel kept the whole component.
    #[test]
    fn slab_with_one_support_keeps_only_the_supported_voxel() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let y = 15;
        let z = 15;
        for x in 14..=16 {
            chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
            set_mask(&mut mask, x, y, z);
        }
        chunk.set(LocalPos(UVec3::new(16, y - 1, z)), Block::Stone);

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(14, y, z))), Block::Air);
        assert_eq!(chunk.get(LocalPos(UVec3::new(15, y, z))), Block::Air);
        assert_eq!(chunk.get(LocalPos(UVec3::new(16, y, z))), Block::Water);
    }

    /// Aquifer voxel adjacent (horizontally) to a non-masked Water
    /// voxel but with Air below — the lateral connection does NOT
    /// support. Physically: the aquifer water would drain into
    /// the ocean / down through the air pocket, not hover.
    #[test]
    fn lateral_ocean_does_not_support_aquifer_voxel() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, y, z) = (15, 15, 15);
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
        set_mask(&mut mask, x, y, z); // aquifer

        chunk.set(LocalPos(UVec3::new(x + 1, y, z)), Block::Water);
        // NOT masked — represents ocean / lake fluid, but only
        // lateral; voxel below (15, 14, 15) is Air.

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Air);
    }

    /// Aquifer voxel directly above a non-masked Water voxel — the
    /// below-fluid IS a supported body, so the aquifer voxel is
    /// kept (it physically rests on the ocean's column).
    #[test]
    fn aquifer_voxel_above_ocean_is_kept() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, y, z) = (15, 15, 15);
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
        set_mask(&mut mask, x, y, z); // aquifer

        chunk.set(LocalPos(UVec3::new(x, y - 1, z)), Block::Water);
        // Below is non-masked ocean water → supports

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Water);
    }

    /// Voxel on chunk floor (`y = 0`) — supported via the below
    /// boundary (we assume the chunk below provides a floor).
    #[test]
    fn floor_boundary_voxel_is_kept() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, y, z) = (15, 0, 15);
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
        set_mask(&mut mask, x, y, z);

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Water);
    }

    /// Voxel against a lateral chunk wall — NOT supported. Lateral
    /// boundaries don't confer support; only the below boundary does.
    #[test]
    fn lateral_boundary_voxel_is_demoted() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, y, z) = (0, 15, 15); // left wall, mid-Y
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
        set_mask(&mut mask, x, y, z);

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Air);
    }

    /// Water hanging from a rock ceiling — the rock above must NOT
    /// count as support. This is the regression: previously any
    /// non-Air neighbour was supporting, so cave-ceiling water
    /// bodies stayed put.
    #[test]
    fn water_with_only_rock_above_is_demoted() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, y, z) = (15, 15, 15);
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
        chunk.set(LocalPos(UVec3::new(x, y + 1, z)), Block::Stone); // ceiling
        set_mask(&mut mask, x, y, z);

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Air);
    }

    /// Water with stone walls on either side but air below and no
    /// floor anywhere — the side walls must NOT confer support.
    #[test]
    fn water_with_only_lateral_rock_is_demoted() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, y, z) = (15, 15, 15);
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
        chunk.set(LocalPos(UVec3::new(x - 1, y, z)), Block::Stone);
        chunk.set(LocalPos(UVec3::new(x + 1, y, z)), Block::Stone);
        set_mask(&mut mask, x, y, z);

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Air);
    }

    /// Non-masked floating water (e.g. ocean voxel above carved
    /// seabed) must NOT be touched by the settle pass.
    #[test]
    fn non_masked_floating_water_is_preserved() {
        let mut chunk = make_chunk();
        let mask = make_mask(); // empty mask

        let (x, y, z) = (15, 15, 15);
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
        // mask[idx] = false — not aquifer water

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Water);
    }

    /// Vertical column of 4 voxels with stone at the bottom —
    /// supported via the stone, whole column kept.
    #[test]
    fn vertical_column_with_floor_kept() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, z) = (15, 15);
        chunk.set(LocalPos(UVec3::new(x, 10, z)), Block::Stone);
        for y in 11..=14 {
            chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
            set_mask(&mut mask, x, y, z);
        }

        settle_fluid(&mut chunk, &mask);

        for y in 11..=14 {
            assert_eq!(
                chunk.get(LocalPos(UVec3::new(x, y, z))),
                Block::Water,
                "column voxel y={y} kept"
            );
        }
    }

    /// Vertical column of 4 voxels with NO floor — air everywhere
    /// around. Entire column demoted.
    #[test]
    fn vertical_column_no_floor_demoted() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, z) = (15, 15);
        for y in 11..=14 {
            chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Water);
            set_mask(&mut mask, x, y, z);
        }

        settle_fluid(&mut chunk, &mask);

        for y in 11..=14 {
            assert_eq!(
                chunk.get(LocalPos(UVec3::new(x, y, z))),
                Block::Air,
                "column voxel y={y} demoted"
            );
        }
    }

    /// Lava behaves identically to water — same flood-fill.
    #[test]
    fn lava_floating_is_demoted() {
        let mut chunk = make_chunk();
        let mut mask = make_mask();
        let (x, y, z) = (15, 15, 15);
        chunk.set(LocalPos(UVec3::new(x, y, z)), Block::Lava);
        set_mask(&mut mask, x, y, z);

        settle_fluid(&mut chunk, &mask);

        assert_eq!(chunk.get(LocalPos(UVec3::new(x, y, z))), Block::Air);
    }
}
