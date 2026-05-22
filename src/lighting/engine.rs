//! `LightEngine` — the graph-based light propagator. PR2 ships only the
//! data structures and a no-op `tick`; PR3 wires propagation.
//!
//! A `LightEngine` owns four `ChannelEngine`s — one for sky light, three
//! for the R/G/B block-light channels. Each `ChannelEngine` holds:
//!   - `block_nodes_to_check`: positions whose underlying block changed
//!     since the last tick; the tick drains this set first, expanding
//!     each entry into the right combination of increase/decrease ops.
//!   - `increase`: a `BucketQueue` of "spread this level outward" ops.
//!   - `decrease`: a `BucketQueue` of "tear down this contribution" ops.
//!
//! See `docs/superpowers/specs/2026-05-21-lighting-graph-engine-design.md`
//! — "Architecture" + "Data model" sections.

use crate::lighting::queue::BucketQueue;
use crate::voxel::coords::BlockPos;
use std::collections::HashSet;

/// One channel's pending work: edits to absorb + two propagation queues.
/// PR2 contains no logic; the fields are populated and drained by PR3.
#[derive(Debug, Default)]
pub struct ChannelEngine {
    pub block_nodes_to_check: HashSet<BlockPos>,
    pub increase: BucketQueue,
    pub decrease: BucketQueue,
}

impl ChannelEngine {
    /// True iff there is no pending work in any of the three fields.
    /// Used by `LightEngine::is_idle` and by tests.
    pub fn is_idle(&self) -> bool {
        self.block_nodes_to_check.is_empty()
            && self.increase.is_empty()
            && self.decrease.is_empty()
    }
}

/// The four-channel graph engine: sky + R + G + B. RGB channels are
/// stored as a 3-element array indexed by `RgbChannel`; the indices
/// are stable so PR3 can iterate them uniformly.
#[derive(Debug)]
pub struct LightEngine {
    pub sky: ChannelEngine,
    pub block_rgb: [ChannelEngine; 3],
    /// Side-table populated by `on_block_changed`, drained by `tick`.
    /// Maps each changed position to its (old_block, new_block) tuple so
    /// the tick can compute the right combination of increase/decrease ops
    /// per channel without re-querying World state mid-tick.
    pub pending_block_changes:
        std::collections::HashMap<crate::voxel::coords::BlockPos, (crate::voxel::block::Block, crate::voxel::block::Block)>,
}

impl Default for LightEngine {
    fn default() -> Self {
        Self {
            sky: ChannelEngine::default(),
            // [ChannelEngine; 3] doesn't auto-derive Default (ChannelEngine
            // isn't Copy because of HashSet/VecDeque), so build via from_fn.
            block_rgb: std::array::from_fn(|_| ChannelEngine::default()),
            pending_block_changes: std::collections::HashMap::new(),
        }
    }
}

/// Index for the three block-light channels. PR3 uses this to drive the
/// uniform per-channel propagation loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RgbChannel {
    R = 0,
    G = 1,
    B = 2,
}

impl From<RgbChannel> for usize {
    fn from(c: RgbChannel) -> usize {
        c as u8 as usize
    }
}

impl RgbChannel {
    /// All three channels in canonical order (R, G, B). Used by the
    /// tick loop to iterate channels uniformly.
    pub const ALL: [RgbChannel; 3] = [RgbChannel::R, RgbChannel::G, RgbChannel::B];
}

impl LightEngine {
    /// True iff every channel reports `is_idle`.
    pub fn is_idle(&self) -> bool {
        self.pending_block_changes.is_empty()
            && self.sky.is_idle()
            && self.block_rgb.iter().all(ChannelEngine::is_idle)
    }

    /// Record a block change for the tick to process. Stores the
    /// (old, new) tuple in `pending_block_changes` and inserts the
    /// position into each channel's `block_nodes_to_check` so the tick
    /// knows to look at this position on every channel.
    pub fn enqueue_block_change(
        &mut self,
        pos: crate::voxel::coords::BlockPos,
        old_block: crate::voxel::block::Block,
        new_block: crate::voxel::block::Block,
    ) {
        self.pending_block_changes.insert(pos, (old_block, new_block));
        self.sky.block_nodes_to_check.insert(pos);
        for ch in 0..3 {
            self.block_rgb[ch].block_nodes_to_check.insert(pos);
        }
    }

    /// Drain up to `budget` queued nodes from the engine's queues. PR2
    /// skeleton: no-op. PR3 will:
    /// 1. Take `&mut ChunkStore` (or split the borrow from `&mut World`)
    ///    and a `&BlockRegistry` so it can read opacity / emission.
    /// 2. For each channel: drain `block_nodes_to_check`, then the
    ///    decrease queue, then the increase queue, up to a shared budget.
    /// 3. Mark touched chunks `light_gpu_dirty` as it writes light values.
    ///
    /// The signature here ships as `(&mut self, budget)` because the real
    /// `&mut World` argument introduces an aliasing issue (`light_engine`
    /// is a field of `World`); PR3 resolves it at the call site by either
    /// making `tick` a method on `World` or by passing a destructured
    /// chunk store. Either way, PR2's signature is provisional — code
    /// that calls `tick` today only does so from tests.
    pub fn tick(&mut self, _budget: usize) {
        // PR2 skeleton: deliberately empty. The body lands in PR3.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_engine_is_idle_on_every_channel() {
        let e = LightEngine::default();
        assert!(e.is_idle(), "fresh engine should have no pending work");
        assert!(e.sky.is_idle());
        for ch in &e.block_rgb {
            assert!(ch.is_idle());
        }
    }

    #[test]
    fn tick_on_empty_engine_is_a_noop() {
        // PR2's contract: tick on an idle engine returns immediately and
        // leaves the engine idle. The body landing in PR3 must continue
        // to satisfy this when its queues are empty.
        let mut e = LightEngine::default();
        e.tick(10_000);
        assert!(e.is_idle(), "tick on idle engine must remain idle");
    }

    #[test]
    fn rgb_channel_into_usize_uses_discriminant() {
        assert_eq!(usize::from(RgbChannel::R), 0);
        assert_eq!(usize::from(RgbChannel::G), 1);
        assert_eq!(usize::from(RgbChannel::B), 2);
    }

    #[test]
    fn rgb_channel_all_lists_three_in_order() {
        assert_eq!(RgbChannel::ALL, [RgbChannel::R, RgbChannel::G, RgbChannel::B]);
        assert_eq!(RgbChannel::ALL.len(), 3);
    }

    #[test]
    fn rgb_channel_indexes_into_block_rgb_array() {
        let e = LightEngine::default();
        // The whole point: e.block_rgb[ch.into()] should work for any ch.
        for ch in RgbChannel::ALL {
            assert!(e.block_rgb[usize::from(ch)].is_idle());
        }
    }

    #[test]
    fn enqueue_block_change_populates_side_table_and_all_channels() {
        use crate::voxel::block::Block;
        use crate::voxel::coords::BlockPos;
        use glam::IVec3;
        let mut e = LightEngine::default();
        assert!(e.is_idle());

        let pos = BlockPos(IVec3::new(1, 2, 3));
        e.enqueue_block_change(pos, Block::Air, Block::Stone);

        assert!(!e.is_idle(), "engine should not be idle after enqueue");
        assert_eq!(e.pending_block_changes.get(&pos), Some(&(Block::Air, Block::Stone)));
        assert!(e.sky.block_nodes_to_check.contains(&pos));
        for ch in &e.block_rgb {
            assert!(ch.block_nodes_to_check.contains(&pos), "all RGB channels should see the change");
        }
    }

    #[test]
    fn enqueue_block_change_overwrites_repeat_at_same_pos() {
        use crate::voxel::block::Block;
        use crate::voxel::coords::BlockPos;
        use glam::IVec3;
        let mut e = LightEngine::default();
        let pos = BlockPos(IVec3::ZERO);
        e.enqueue_block_change(pos, Block::Air, Block::Stone);
        e.enqueue_block_change(pos, Block::Stone, Block::Air);
        // The second call's (old, new) wins — the engine only ever sees one
        // composite delta per pos per tick.
        assert_eq!(e.pending_block_changes.get(&pos), Some(&(Block::Stone, Block::Air)));
        // Set still contains pos exactly once (it's a HashSet).
        assert_eq!(e.sky.block_nodes_to_check.len(), 1);
    }
}
