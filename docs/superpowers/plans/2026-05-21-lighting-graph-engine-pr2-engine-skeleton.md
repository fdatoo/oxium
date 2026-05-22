# Lighting Graph Engine PR 2 — BucketQueue + ChannelEngine Skeleton

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two core data structures the graph engine will use — a magnitude-bucketed propagation queue (`BucketQueue`) and the per-channel + top-level engine structs (`ChannelEngine`, `LightEngine`) with a no-op `tick`. Wire a `light_engine` field into `World`. **Do not change any propagation behaviour** — the existing BFS keeps running. PR3 is the cutover that hooks the engine up.

**Architecture:** `src/lighting/queue.rs` owns `BucketQueue` (16 magnitude buckets indexed 0..=15, FIFO within each bucket, `pop_highest` returns the highest-level entry in O(1) via a `u16` `nonempty_mask`). `src/lighting/engine.rs` owns `ChannelEngine` (one per channel: pending-change set + increase queue + decrease queue) and `LightEngine` (sky channel + RGB channels). `World` gains a `pub light_engine: LightEngine` field. Nothing in PR2 reads or writes the queues outside tests.

**Tech stack:** Rust, std collections. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-21-lighting-graph-engine-design.md` (Data model + Algorithm sections describe the final state; PR2 implements only the empty data structures + skeleton).

**Risk:** Low. Pure-additive: no existing call site is modified except `World::new` (adds one default-initialized field).

---

## Files

**Create:**
- `src/lighting/queue.rs` — `BucketQueue`, `QueueEntry`, push/`pop_highest`/`purge_chunk`/`is_empty` + 8 unit tests
- `src/lighting/engine.rs` — `ChannelEngine`, `LightEngine`, no-op `tick`, `Default` impls + 2 unit tests

**Modify:**
- `src/lighting/mod.rs` — `pub mod queue;` + `pub mod engine;` + re-exports
- `src/voxel/world.rs` — `pub light_engine: LightEngine` on `World`, initialised in `World::new`; one integration test

**Do not touch in this PR:**
- The existing BFS (`recompute_chunk`, `sky_light`, `block_rgb`, `seed_from_neighbors`, `snapshot_face_boundaries`) in `src/lighting/mod.rs`
- `ChunkMeta.sky_sources` — PR3 decides whether to migrate that into `LightEngine.sky_sources` or keep it dual-homed
- `World::set_block`, `World::mark_below_dirty`, `mesh_upload.rs`, `JobResult::Relit`, `relight_pump`

---

## Design notes for the implementer

### Why a bucket queue, not a FIFO

Propagation level is bounded (`0..=15`) and monotonic within a single phase: increase only spreads to higher cells; decrease only tears down lower-or-equal cells. So we can pop in highest-level-first order without a heap — just an array of 16 FIFOs indexed by level. `pop_highest` finds the highest non-empty bucket in O(1) via `15 - nonempty_mask.leading_zeros()`. Cache-coherent, deterministic order, no allocations beyond the per-bucket `VecDeque`'s amortised growth.

### Why level 0 silently lives in bucket 0

Pushing `from_level == 0` would propagate nothing useful (`prop = level - 1 - cost` is already < 0 even at no cost). The queue accepts it without complaint — bucket 0 just collects no-op entries that pop and do nothing. Simpler than special-casing the push API. The real propagation phases in PR3 will check `level > 1` before pushing to neighbours, so bucket 0 should normally stay empty.

### Why `tick` takes only `budget` in PR2

The spec calls for `pub fn tick(world: &mut World, registry: &BlockRegistry, budget: usize)`, but in Rust that signature has an aliasing problem (`light_engine` is a field of `World`). PR3 — the cutover — will resolve it by either making `tick` a method on `World` that destructures internally, or by passing `&mut ChunkStore` rather than `&mut World`. For PR2 we ship the simplest signature that compiles and is honest about being a no-op: `pub fn tick(&mut self, budget: usize)`. PR3 changes the signature when adding the body.

### Why `LightEngine` has no `sky_sources` field in PR2

PR1 stored `ChunkSkyLightSources` on `ChunkMeta`. The spec ultimately wants it inside `LightEngine` (`HashMap<ChunkCoord, ChunkSkyLightSources>`). Migrating in PR2 would mean either dropping the PR1 field (breaking PR1's tests) or carrying both (pointless duplication). PR3 — when the engine actually reads sky-source data — is the right time to decide whether to migrate, dual-home, or keep on `ChunkMeta`. PR2 leaves PR1's placement alone.

### Why `QueueEntry` stores `pos: BlockPos`, not a packed long like Minecraft

Minecraft packs `(x, y, z, level, direction-mask)` into a `long` for fastutil queue density. We use `BlockPos` (12 bytes via `IVec3`) + two `u8`s. Per-entry memory is ~16 bytes after alignment — fine for our scale (worst-case queue depth ~50k = ~800 KB, all on the heap inside `VecDeque`). The packed-long encoding is a future optimisation if profiling demands it; PR2 prioritises clarity.

---

## Tasks

### Task 1: `BucketQueue` + `QueueEntry`

**Files:**
- Create: `src/lighting/queue.rs`
- Modify: `src/lighting/mod.rs`

- [ ] **Step 1: Create `src/lighting/queue.rs` with the queue + tests.**

Write the file exactly as below. It defines `QueueEntry`, `BucketQueue`, push/pop/purge/is_empty, and 8 unit tests.

```rust
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
```

- [ ] **Step 2: Wire the module into `src/lighting/mod.rs`.**

Find the block currently containing `pub mod sky_sources;` and `pub use sky_sources::{ChunkSkyLightSources, NO_SOURCE_FLOOR};` (around lines 20-22 after PR1 landed). Add `pub mod queue;` and a re-export for `QueueEntry` immediately after.

The final block should look like:

```rust
pub mod queue;
pub mod sky_sources;

pub use queue::{BucketQueue, QueueEntry};
pub use sky_sources::{ChunkSkyLightSources, NO_SOURCE_FLOOR};
```

Order: `pub mod` declarations alphabetised; `pub use` declarations alphabetised. Keep them as two separate blocks.

- [ ] **Step 3: Run the new module's tests.**

Run: `cargo test --lib lighting::queue::tests -- --nocapture`
Expected: all 8 tests pass.

- [ ] **Step 4: Run the full lib suite.**

Run: `cargo test --lib 2>&1 | tail -5`
Expected: `XXX passed; 0 failed; 4 ignored` where `XXX` is the post-PR1 baseline (228 or 230 if your working tree still has `mark_below_dirty_*` tests) plus 8 new from this task.

- [ ] **Step 5: Commit.**

```bash
git add src/lighting/queue.rs src/lighting/mod.rs
git commit -m "feat(lighting): BucketQueue + QueueEntry (graph-engine PR2, part 1/3)

16 magnitude-bucketed FIFOs indexed by from_level (0..=15). pop_highest
is O(1) via a u16 nonempty_mask. push/purge_chunk also O(1)/O(buckets).
Pure data structure with no callers; engine skeleton (part 2) and
World wiring (part 3) come next."
```

---

### Task 2: `ChannelEngine` + `LightEngine` skeleton

**Files:**
- Create: `src/lighting/engine.rs`
- Modify: `src/lighting/mod.rs`

- [ ] **Step 1: Create `src/lighting/engine.rs` with the engine skeleton + tests.**

Write the file exactly as below.

```rust
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
```

- [ ] **Step 2: Wire the module into `src/lighting/mod.rs`.**

Update the `pub mod` / `pub use` block from Task 1 to add `engine`. After this edit the block reads:

```rust
pub mod engine;
pub mod queue;
pub mod sky_sources;

pub use engine::{ChannelEngine, LightEngine, RgbChannel};
pub use queue::{BucketQueue, QueueEntry};
pub use sky_sources::{ChunkSkyLightSources, NO_SOURCE_FLOOR};
```

(Keep both blocks alphabetised.)

- [ ] **Step 3: Run the new module's tests.**

Run: `cargo test --lib lighting::engine::tests -- --nocapture`
Expected: both tests pass.

- [ ] **Step 4: Run the full lib suite.**

Run: `cargo test --lib 2>&1 | tail -5`
Expected: still no failures, count is up by 2 from Task 1's baseline.

- [ ] **Step 5: Commit.**

```bash
git add src/lighting/engine.rs src/lighting/mod.rs
git commit -m "feat(lighting): ChannelEngine + LightEngine skeleton (graph-engine PR2, part 2/3)

LightEngine has one ChannelEngine for sky and three for RGB block light.
Each ChannelEngine bundles a HashSet<BlockPos> of pending block changes
and two BucketQueues (increase + decrease). tick(budget) is a no-op
placeholder — body lands in PR3 along with the real (&mut World,
&BlockRegistry, budget) signature.

Not yet wired into World; that's part 3."
```

---

### Task 3: Add `light_engine` field to `World`

**Files:**
- Modify: `src/voxel/world.rs`

- [ ] **Step 1: Write the failing integration test.**

Append to the `tests` module at the bottom of `src/voxel/world.rs`:

```rust
    /// World::new must construct a default LightEngine alongside the
    /// chunks map and registry. The engine is idle (no queued work)
    /// on a fresh world — PR3 will start feeding it.
    #[test]
    fn new_world_has_idle_light_engine() {
        let w = World::new(42);
        assert!(
            w.light_engine.is_idle(),
            "freshly-constructed World must have an idle LightEngine",
        );
    }
```

- [ ] **Step 2: Run the failing test.**

Run: `cargo test --lib voxel::world::tests::new_world_has_idle_light_engine -- --nocapture`
Expected: FAIL with a compile error — `no field 'light_engine' on type 'World'`.

- [ ] **Step 3: Add the field to `World` and initialise it in `World::new`.**

Find the `World` struct (around `src/voxel/world.rs:36`):

```rust
pub struct World {
    pub chunks: HashMap<ChunkCoord, ChunkSlot>,
    pub registry: BlockRegistry,
    pub seed: u64,
}
```

Replace with:

```rust
pub struct World {
    pub chunks: HashMap<ChunkCoord, ChunkSlot>,
    pub registry: BlockRegistry,
    pub seed: u64,
    /// Per-voxel graph light propagator. PR2 ships this as an idle
    /// skeleton; nothing calls `light_engine.tick` yet. PR3 wires
    /// `set_block` and chunk-load to enqueue work here, and the
    /// frame loop ticks the engine each frame.
    pub light_engine: crate::lighting::LightEngine,
}
```

Then update `World::new` (immediately below the struct, around `src/voxel/world.rs:45`) from:

```rust
    pub fn new(seed: u64) -> Self {
        Self {
            chunks: HashMap::new(),
            registry: BlockRegistry::new(),
            seed,
        }
    }
```

to:

```rust
    pub fn new(seed: u64) -> Self {
        Self {
            chunks: HashMap::new(),
            registry: BlockRegistry::new(),
            seed,
            light_engine: crate::lighting::LightEngine::default(),
        }
    }
```

- [ ] **Step 4: Run the previously-failing test.**

Run: `cargo test --lib voxel::world::tests::new_world_has_idle_light_engine -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run the full suite.**

Run: `cargo test --lib 2>&1 | tail -5`
Expected: full suite passes; test count is up by 1 from Task 2's baseline.

- [ ] **Step 6: Verify nothing else in the codebase constructs `World` directly.**

Run: `grep -rn "World {" src/ | grep -v "test"`
Expected: no struct-literal constructions of `World` outside of `World::new`. If there are any, they'll fail to compile because the `light_engine` field is now required; either fix them (route through `World::new`) or add `light_engine: LightEngine::default()` to the literal. (Spot-check: in the post-PR1 codebase the only direct construction is in `World::new`. If `cargo build --lib` is clean after Step 3, no further edits are needed.)

- [ ] **Step 7: Commit.**

```bash
git add src/voxel/world.rs
git commit -m "feat(lighting): World gains light_engine field (graph-engine PR2, part 3/3)

pub light_engine: LightEngine on World, initialised to default by
World::new. The engine is idle and untouched — PR3 wires set_block,
chunk-load, and the per-frame tick to it.

This concludes PR2: queue + engine skeleton complete, ready for PR3
to add the propagation body and the cutover from today's BFS."
```

---

## Verification

After all three tasks:

- [ ] **Run the full lib suite once more.**

Run: `cargo test --lib`
Expected: clean pass. New tests added across PR2: 8 (queue) + 2 (engine) + 1 (world) = **11 new tests**.

- [ ] **Confirm zero behaviour change.**

Run: `cargo run --release` and walk around for ~30 seconds. Lighting, streaming, edits, rendering — all visually identical to before PR2 (and to before PR1; the visual artifact in the ocean is still there and that's expected — PR3 is the cutover).

Reason: PR2 only **adds** data structures and an empty engine. `LightEngine.tick` is a no-op, no one calls it, and the existing BFS continues to run.

- [ ] **Check `git log` shows three focused commits.**

Run: `git log --oneline -3`
Expected:
```
<sha> feat(lighting): World gains light_engine field (graph-engine PR2, part 3/3)
<sha> feat(lighting): ChannelEngine + LightEngine skeleton (graph-engine PR2, part 2/3)
<sha> feat(lighting): BucketQueue + QueueEntry (graph-engine PR2, part 1/3)
```

---

## What PR3 will do (preview, not part of this PR)

- Implement `LightEngine::tick` — three phases per channel: drain `block_nodes_to_check`, drain decrease queue, drain increase queue, all bounded by a shared `budget`.
- Resolve the `tick` signature: change to `pub fn tick(world: &mut World, registry: &BlockRegistry, budget: usize)` via a method on `World` that destructures internally, or refactor so the engine takes only the chunk store.
- Add `World::on_block_changed(pos, old_block, new_block)` and `World::on_chunk_loaded(coord)` as the producer entry points; call them from `set_block` and `mesh_upload`'s `JobResult::Generated`/`LoadedFromDisk` handlers.
- Delete the old BFS plumbing: `lighting::recompute_chunk`, `sky_light`, `block_rgb`, `seed_from_neighbors`, `snapshot_face_boundaries`, `Jobs::spawn_relight`, `JobResult::Relit`, `mesh_upload::relight_pump`, `World::mark_below_dirty`, `ChunkDirty::light`. Gate the delete behind a `legacy-lighting` Cargo feature for one release as a safety net.
- Decide `sky_sources` placement: keep on `ChunkMeta`, migrate to `LightEngine.sky_sources: HashMap<ChunkCoord, ChunkSkyLightSources>`, or dual-home. Per the PR1 final review, this decision belongs to PR3.

PR3 is the high-risk PR. PR2's job is to make PR3's diff as small as possible by pre-building the data structures.
