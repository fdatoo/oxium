//! Magnitude-bucketed FIFO queue used by the graph-engine propagation phases.
//!
//! Light propagation values are bounded to `0..=15` and monotonic within a
//! single phase (increase spreads higher levels first, decrease tears down
//! lower-or-equal cells), so processing the highest-level entry first is the
//! correct strategy. A bucket sort over 16 FIFOs is optimal: O(1) push,
//! O(1) `pop_highest` via a `u16` mask, no heap, no allocations beyond the
//! per-bucket `VecDeque`'s amortised growth.
//!
//! See `docs/superpowers/specs/2026-05-21-lighting-graph-engine-design.md`
//! — "BucketQueue" section.

use crate::voxel::coords::{BlockPos, ChunkCoord};
use std::collections::VecDeque;

/// A single propagation operation: "consider spreading `from_level` outward
/// from `pos`, except into faces blocked by `propagation_mask`."
///
/// `propagation_mask` has 6 bits, one per face in `crate::mesher::Face`
/// order. Bit `i` set = "do not propagate in face `i`'s direction"
/// (i.e., the back-face of the cell we just came from). `mask == 0`
/// means "all 6 faces allowed" and is the value used when enqueuing a
/// source (torch, sky-source cell).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueEntry {
    pub pos: BlockPos,
    pub from_level: u8,
    pub propagation_mask: u8,
}

/// 16 FIFO buckets indexed by `from_level` (0..=15). `nonempty_mask` bit `i`
/// is set iff `buckets[i]` is non-empty, so `pop_highest` can find the next
/// bucket to drain in O(1) via `15 - nonempty_mask.leading_zeros()`.
#[derive(Debug)]
pub struct BucketQueue {
    buckets: [VecDeque<QueueEntry>; 16],
    nonempty_mask: u16,
}

impl Default for BucketQueue {
    fn default() -> Self {
        Self {
            buckets: std::array::from_fn(|_| VecDeque::new()),
            nonempty_mask: 0,
        }
    }
}

impl BucketQueue {
    /// Push an entry into the bucket for its `from_level`. Levels above 15
    /// are clamped to 15 via a `debug_assert` — the algorithm should never
    /// pass an out-of-range level, but a clamp avoids panicking release
    /// builds if a bug slips through.
    pub fn push(&mut self, entry: QueueEntry) {
        debug_assert!(
            entry.from_level <= 15,
            "BucketQueue::push: from_level {} > 15",
            entry.from_level,
        );
        let bucket = (entry.from_level as usize).min(15);
        self.buckets[bucket].push_back(entry);
        self.nonempty_mask |= 1u16 << bucket;
    }

    /// Pop the entry from the highest-level non-empty bucket, FIFO within
    /// the bucket. Returns `None` if the queue is empty.
    pub fn pop_highest(&mut self) -> Option<QueueEntry> {
        if self.nonempty_mask == 0 {
            return None;
        }
        // Highest set bit: `15 - leading_zeros` works because `leading_zeros`
        // on a u16 returns 16 only when the value is 0 (which we already
        // checked). For any non-zero u16, leading_zeros is in 0..=15.
        let bit = 15 - self.nonempty_mask.leading_zeros() as usize;
        let entry = self.buckets[bit].pop_front().expect(
            "BucketQueue: nonempty_mask claims bucket non-empty but pop_front returned None",
        );
        if self.buckets[bit].is_empty() {
            self.nonempty_mask &= !(1u16 << bit);
        }
        Some(entry)
    }

    /// Remove every queued entry whose `pos` lies in the given chunk.
    /// Used by `LightEngine::on_chunk_unloaded` (PR3) to drop dangling
    /// references to chunks that are no longer in the world. After this
    /// call, `nonempty_mask` is reconciled with the remaining bucket
    /// contents (a bucket that became empty has its bit cleared).
    pub fn purge_chunk(&mut self, coord: ChunkCoord) {
        for (bucket_i, bucket) in self.buckets.iter_mut().enumerate() {
            bucket.retain(|e| e.pos.to_chunk() != coord);
            if bucket.is_empty() {
                self.nonempty_mask &= !(1u16 << bucket_i);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nonempty_mask == 0
    }

    /// Total entries across all buckets. O(16) — fine for tests, avoid in
    /// hot paths.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    fn entry(x: i32, y: i32, z: i32, level: u8) -> QueueEntry {
        QueueEntry {
            pos: BlockPos(IVec3::new(x, y, z)),
            from_level: level,
            propagation_mask: 0,
        }
    }

    #[test]
    fn new_queue_is_empty() {
        let q = BucketQueue::default();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn pop_empty_returns_none() {
        let mut q = BucketQueue::default();
        assert_eq!(q.pop_highest(), None);
    }

    #[test]
    fn push_then_pop_round_trips() {
        let mut q = BucketQueue::default();
        let e = entry(1, 2, 3, 10);
        q.push(e);
        assert!(!q.is_empty());
        assert_eq!(q.pop_highest(), Some(e));
        assert!(q.is_empty());
        assert_eq!(q.pop_highest(), None);
    }

    #[test]
    fn pop_highest_returns_highest_level_first() {
        // Push in mixed order; pop should come back in strictly
        // decreasing-level order.
        let mut q = BucketQueue::default();
        let levels = [3u8, 15, 7, 1, 12, 0, 8, 15];
        for (i, &l) in levels.iter().enumerate() {
            q.push(entry(i as i32, 0, 0, l));
        }
        let mut popped = Vec::new();
        while let Some(e) = q.pop_highest() {
            popped.push(e.from_level);
        }
        // Sorted descending — but the two 15s should both come out first
        // (FIFO within bucket).
        let mut expected: Vec<u8> = levels.to_vec();
        expected.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(popped, expected);
    }

    #[test]
    fn same_level_pops_in_fifo_order() {
        let mut q = BucketQueue::default();
        let a = entry(0, 0, 0, 5);
        let b = entry(1, 0, 0, 5);
        let c = entry(2, 0, 0, 5);
        q.push(a);
        q.push(b);
        q.push(c);
        assert_eq!(q.pop_highest(), Some(a));
        assert_eq!(q.pop_highest(), Some(b));
        assert_eq!(q.pop_highest(), Some(c));
    }

    #[test]
    fn nonempty_mask_clears_when_bucket_drains() {
        // Push 2 entries at level 10 and 1 at level 5. Drain. After each
        // pop, check `is_empty` is correct.
        let mut q = BucketQueue::default();
        q.push(entry(0, 0, 0, 10));
        q.push(entry(1, 0, 0, 10));
        q.push(entry(2, 0, 0, 5));
        assert!(!q.is_empty());
        assert_eq!(q.pop_highest().unwrap().from_level, 10);
        assert!(!q.is_empty());
        assert_eq!(q.pop_highest().unwrap().from_level, 10);
        assert!(!q.is_empty()); // bucket 10 now empty, but bucket 5 has one
        assert_eq!(q.pop_highest().unwrap().from_level, 5);
        assert!(q.is_empty());
    }

    #[test]
    fn purge_chunk_removes_only_matching_entries() {
        let mut q = BucketQueue::default();
        // chunk (0,0,0) covers x,y,z in 0..32.
        // chunk (1,0,0) covers x in 32..64.
        let in_chunk_0 = entry(5, 5, 5, 8);
        let in_chunk_1 = entry(40, 5, 5, 8);
        let also_chunk_0 = entry(15, 0, 0, 3);
        q.push(in_chunk_0);
        q.push(in_chunk_1);
        q.push(also_chunk_0);
        assert_eq!(q.len(), 3);

        q.purge_chunk(ChunkCoord(IVec3::ZERO));

        // Only the chunk-1 entry should remain.
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop_highest(), Some(in_chunk_1));
        assert!(q.is_empty());
    }

    #[test]
    fn purge_chunk_reconciles_nonempty_mask() {
        // Push one entry per bucket from 5..15, all in chunk (0,0,0).
        // Purge chunk (0,0,0). Mask should be 0; is_empty true.
        let mut q = BucketQueue::default();
        for level in 5..=15 {
            q.push(entry(5, 5, 5, level));
        }
        assert_eq!(q.len(), 11);
        q.purge_chunk(ChunkCoord(IVec3::ZERO));
        assert!(q.is_empty(), "mask should be fully cleared after purge of all-matching chunk");
        assert_eq!(q.pop_highest(), None);
    }
}
