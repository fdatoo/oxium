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
}

impl Default for LightEngine {
    fn default() -> Self {
        Self {
            sky: ChannelEngine::default(),
            // [ChannelEngine; 3] doesn't auto-derive Default (ChannelEngine
            // isn't Copy because of HashSet/VecDeque), so build via from_fn.
            block_rgb: std::array::from_fn(|_| ChannelEngine::default()),
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

impl LightEngine {
    /// True iff every channel reports `is_idle`.
    pub fn is_idle(&self) -> bool {
        self.sky.is_idle() && self.block_rgb.iter().all(ChannelEngine::is_idle)
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
}
