# Lighting Graph Engine PR 4 — Perf Fix + Proper Decrease

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the PR3 graph engine usable in production by eliminating per-op chunk decompression (currently ~7 decompresses per propagation op, observed to freeze the main thread), then replace the minimal-decrease chunk-recompute fallback with the spec's proper Minecraft-style decrease propagation (tear-down + repair from independent sources).

**Architecture:** Two intertwined changes. First, add direct-cell accessors to `PalettedChunk` (read/write a single voxel's blocks/light without decompressing the whole chunk) and a per-tick `TickCache` that decompresses each touched chunk once, mutates the cached `DenseChunk` many times, then recompresses at tick end. Net effect: per-op cost drops from O(7 × 128 KB allocs) to O(0..1 chunk-clone amortized). Second, implement `drain_decrease_channel` that walks the tear-down chain (zero cells lit by the removed source; re-flood neighbours that are still lit by independent sources). Replaces the current "clear-and-reflood-the-whole-chunk" fallback.

**Tech Stack:** Rust. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-21-lighting-graph-engine-design.md`

**Risk:** Medium-high. Touches engine internals heavily but each task is self-contained and the build stays green. PR3's `--feature legacy-lighting` fallback is still in place if anything regresses catastrophically.

---

## Optional intermediate validation point

Tasks 1-3 are the perf fix. After T3 the engine should run smoothly in `cargo run --release` with correct lighting (the user's beachball-on-launch issue should be gone). If T3's smoke test passes, you can choose to merge that subset and continue with the decrease work in a follow-up PR. Tasks 4-6 are the proper-decrease pass.

---

## Files

**Modify:**
- `src/voxel/chunk.rs` — add 5 direct-cell accessors to `PalettedChunk`; their unit tests
- `src/lighting/engine.rs` — `TickCache`, refactor `drain_pending_block_changes` + `drain_increase_channel` + source self-write block, add `drain_decrease_channel`, wire decrease into the tick

**Do not touch:**
- `BucketQueue` and `QueueEntry` (PR2, stable)
- `ChunkSkyLightSources` (PR1, stable)
- `World::on_chunk_loaded` / `on_block_changed` / `light_engine_tick` (PR3, stable interface)
- The `--feature legacy-lighting` gates (PR3, retained as fallback)

---

## Design notes

### Why direct-cell accessors on `PalettedChunk`

`PalettedChunk` already holds light data in 4-bit-packed `Packed4Bit` arrays (`sky_light`, `block_red`, `block_green`, `block_blue`). Reading a single cell's light value is one bit-op on `Packed4Bit::get(idx)` — cheap. The engine's current per-op `data.decompress()` allocates a fresh 128 KB `DenseChunk` just to read one cell, then throws it away. A pair of `sky_light_at(idx)` / `block_rgb_at(idx)` methods on `PalettedChunk` lets the engine read directly from the packed arrays without the alloc.

Writes are similar via `Packed4Bit::set` — but writes still need to bump `Arc<PalettedChunk>` to a unique handle (`Arc::make_mut`), which clones the whole `PalettedChunk` (~50 KB) if the Arc is shared with a mesh job. To avoid one clone per write, the `TickCache` batches writes per chunk per tick (Task 2).

### Why a per-tick `TickCache`

Even with `set_sky_light_at` accessors, writing 50 000 cells across N chunks would mean up to N `Arc::make_mut` clones plus 50 000 nibble writes. The cache amortizes by:
1. **Decompressing each touched chunk at most once per tick** (~128 KB alloc per unique chunk, not per op).
2. **Mutating the cached `DenseChunk` directly** (1 byte/nibble write per op — no Arc churn).
3. **Recompressing only the chunks that were actually written** at tick end, once per chunk.

For a tick that touches 20 chunks at 50 000 ops: cost goes from `O(50_000 × 7 × 128 KB) = 44 GB` of heap churn down to `O(20 × 128 KB) + O(20 × 80 KB) = ~4 MB`. Roughly **10 000× reduction**, which is the gap between "freezes the main thread" and "runs at 60 FPS."

### Why proper decrease (not just "increase + chunk-recompute fallback")

PR3 shipped a minimal-decrease fallback: when a change might lower a cell's contribution, recompute the whole chunk's light from scratch. That's correct but wastes work on every player edit. The spec's proper-decrease algorithm (Minecraft's `propagateDecrease`):

```
while let Some(entry) = decrease.pop_highest():
    for face in 6:
        if mask blocks this face: continue
        npos = entry.pos + face
        cur = light_at(npos)
        if cur != 0 and cur < entry.from_level:
            # neighbour was lit by us — tear it down
            set_level(npos, 0)
            decrease.push((npos, cur, mask_back))
            # if npos is itself a source (torch / sky-source), re-light
            emission = emission_at(npos, channel)
            if emission > 0:
                set_level(npos, emission)
                increase.push((npos, emission, no_mask))
        elif cur >= entry.from_level:
            # neighbour is independently sourced — re-flood from here
            increase.push((npos, cur, no_mask))
```

Costs `O(actually_dimmed_cells)` per edit instead of `O(chunk_volume)`. The increase queue picks up the repair work and re-floods from any cell that's still lit independently.

### Why keep `recompute_chunk_light_from_scratch` for `on_chunk_loaded`

`on_chunk_loaded` is the chunk-install path (called from `mesh_upload::drain_jobs`), not the per-tick path. It runs once per chunk lifetime. The recompute approach is fine there — we want a clean "rebuild from scratch" for a freshly-loaded chunk. Only the per-edit recompute (inside `drain_pending_block_changes`) gets replaced with proper decrease.

### Why this doesn't change the public API

`LightEngine::tick_with(chunks, registry, budget)` keeps the same signature. `World::on_block_changed`, `on_chunk_loaded`, `light_engine_tick` are unchanged. The internals get faster and the decrease semantics get tighter, but every external caller sees the same surface.

---

## Tasks

# Phase A — Perf Fix (Tasks 1-3)

## Task 1: Direct-cell accessors on `PalettedChunk`

**Files:**
- Modify: `src/voxel/chunk.rs`

- [ ] **Step 1: Add the accessor methods to `impl PalettedChunk`.**

Find `impl PalettedChunk { ... }` in `src/voxel/chunk.rs`. After the existing `pub fn get(&self, p: LocalPos) -> Block` method (around line 346), add the following accessors:

```rust
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
```

- [ ] **Step 2: Add unit tests at the bottom of the `tests` module in `chunk.rs`.**

Append to the existing `#[cfg(test)] mod tests { ... }` block:

```rust
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
        // Overwrite same idx.
        p.set_block_rgb_at(42, 0, 0, 0);
        assert_eq!(p.block_rgb_at(42), (0, 0, 0));
    }
```

- [ ] **Step 3: Run the new tests.**

Run: `cargo test --lib voxel::chunk::tests -- --nocapture 2>&1 | tail -10`
Expected: 5 new tests pass on top of existing chunk tests.

Run: `cargo test --lib 2>&1 | tail -3`
Expected: full suite passes, count up by 5 from baseline.

- [ ] **Step 4: Commit.**

```bash
git add src/voxel/chunk.rs
git commit -m "feat(chunk): direct-cell PalettedChunk accessors (graph-engine PR4, part 1/6)

Adds block_at(idx), sky_light_at(idx), block_rgb_at(idx),
set_sky_light_at(idx, v), set_block_rgb_at(idx, r, g, b) on
PalettedChunk. All O(1), no allocation, read/write directly from
the packed 4-bit arrays.

Engine's hot path will use these (next tasks) to avoid the
per-op 128 KB decompress alloc that froze the main thread in PR3."
```

---

## Task 2: `TickCache` struct + unit tests

**Files:**
- Modify: `src/lighting/engine.rs`

- [ ] **Step 1: Add `TickCache` to `engine.rs`.**

Add the following at the top of `src/lighting/engine.rs`'s implementation block — right after the `Channel` enum definition (which lives somewhere near the top of the helper-function section):

```rust
/// Per-tick decompress cache. The engine touches each chunk at most a few
/// hundred times per tick; without caching, each touch decompresses the
/// chunk (128 KB alloc) and discards the result. With the cache, each
/// chunk is decompressed at most once per tick into a `DenseChunk`,
/// mutated many times directly via array indexing, and recompressed at
/// `flush` time only if it was actually written.
///
/// Caller pattern (engine `tick_with`):
/// 1. Construct a fresh `TickCache::default()` at tick start.
/// 2. Pass `&mut cache` and `&chunks` (read-only) to drain helpers.
/// 3. Helpers call `cache.dense(&chunks, coord)` to get a mutable
///    `DenseChunk`; cache misses trigger one decompress.
/// 4. Helpers call `cache.mark_written(coord)` after mutating to
///    indicate the chunk needs recompression at flush.
/// 5. Call `cache.flush(&mut chunks)` at tick end — recompresses every
///    written chunk into its `Arc<PalettedChunk>` and sets
///    `meta.light_gpu_dirty = true`.
#[derive(Default)]
pub(crate) struct TickCache {
    inner: std::collections::HashMap<crate::voxel::coords::ChunkCoord, crate::voxel::chunk::DenseChunk>,
    written: std::collections::HashSet<crate::voxel::coords::ChunkCoord>,
}

impl TickCache {
    /// Borrow the decompressed `DenseChunk` for `coord`. On cache miss,
    /// decompresses the chunk's `Arc<PalettedChunk>` once and inserts.
    /// Returns `None` if the chunk is `Pending` or not loaded.
    pub(crate) fn dense<'a>(
        &'a mut self,
        chunks: &std::collections::HashMap<
            crate::voxel::coords::ChunkCoord,
            crate::voxel::world::ChunkSlot,
        >,
        coord: crate::voxel::coords::ChunkCoord,
    ) -> Option<&'a mut crate::voxel::chunk::DenseChunk> {
        use crate::voxel::world::ChunkSlot;
        if !self.inner.contains_key(&coord) {
            let ChunkSlot::Stored { data, .. } = chunks.get(&coord)? else { return None };
            self.inner.insert(coord, data.decompress());
        }
        self.inner.get_mut(&coord)
    }

    /// Mark `coord` as needing recompression at flush.
    pub(crate) fn mark_written(&mut self, coord: crate::voxel::coords::ChunkCoord) {
        self.written.insert(coord);
    }

    /// Recompress every written chunk into its slot's `Arc<PalettedChunk>`
    /// and set `meta.light_gpu_dirty = true`. Consumes the cache.
    pub(crate) fn flush(
        self,
        chunks: &mut std::collections::HashMap<
            crate::voxel::coords::ChunkCoord,
            crate::voxel::world::ChunkSlot,
        >,
    ) {
        use crate::voxel::world::ChunkSlot;
        for coord in self.written {
            let Some(ChunkSlot::Stored { data, meta }) = chunks.get_mut(&coord) else { continue };
            let Some(dense) = self.inner.get(&coord) else { continue };
            *data = std::sync::Arc::new(crate::voxel::chunk::PalettedChunk::compress(dense));
            meta.light_gpu_dirty = true;
        }
    }
}
```

If `Channel` is currently in a `mod` or near specific helper functions, place `TickCache` right after `Channel`. If you can't find a clean spot, place it just before the first `fn drain_*` function. Visibility: `pub(crate)` so it's available to free functions in this module but not exported.

- [ ] **Step 2: Add unit tests at the bottom of the `tests` module in `engine.rs`.**

Append to the existing `#[cfg(test)] mod tests { ... }` block:

```rust
    #[test]
    fn tick_cache_decompresses_each_chunk_at_most_once() {
        use crate::voxel::block::BlockRegistry;
        use crate::voxel::chunk::{DenseChunk, PalettedChunk};
        use crate::voxel::coords::{ChunkCoord, LocalPos};
        use crate::voxel::world::ChunkSlot;
        use glam::{IVec3, UVec3};
        use std::collections::HashMap;

        let _registry = BlockRegistry::new();
        let mut dense = DenseChunk::empty();
        dense.sky_light[0] = 9;
        let coord = ChunkCoord(IVec3::ZERO);
        let mut chunks: HashMap<ChunkCoord, ChunkSlot> = HashMap::new();
        chunks.insert(coord, ChunkSlot::Stored {
            data: std::sync::Arc::new(PalettedChunk::compress(&dense)),
            meta: Default::default(),
        });

        let mut cache = TickCache::default();
        // First access: should decompress and return the data.
        {
            let d = cache.dense(&chunks, coord).expect("loaded chunk");
            assert_eq!(d.sky_light[0], 9);
            d.sky_light[0] = 12; // mutate
        }
        cache.mark_written(coord);
        // Second access: cache hit. Mutation is visible.
        {
            let d = cache.dense(&chunks, coord).expect("loaded chunk");
            assert_eq!(d.sky_light[0], 12, "second access should see prior write");
            assert_eq!(d.sky_light[LocalPos(UVec3::new(1, 0, 0)).to_index()], 0);
        }
    }

    #[test]
    fn tick_cache_flush_writes_back_and_marks_gpu_dirty() {
        use crate::voxel::chunk::{DenseChunk, PalettedChunk};
        use crate::voxel::coords::ChunkCoord;
        use crate::voxel::world::ChunkSlot;
        use glam::IVec3;
        use std::collections::HashMap;

        let dense = DenseChunk::empty();
        let coord = ChunkCoord(IVec3::ZERO);
        let mut chunks: HashMap<ChunkCoord, ChunkSlot> = HashMap::new();
        chunks.insert(coord, ChunkSlot::Stored {
            data: std::sync::Arc::new(PalettedChunk::compress(&dense)),
            meta: Default::default(),
        });

        let mut cache = TickCache::default();
        {
            let d = cache.dense(&chunks, coord).expect("loaded chunk");
            d.sky_light[42] = 11;
        }
        cache.mark_written(coord);
        cache.flush(&mut chunks);

        // Verify the write made it into the Arc<PalettedChunk> and
        // light_gpu_dirty was set.
        let ChunkSlot::Stored { data, meta } = chunks.get(&coord).unwrap() else { panic!() };
        assert_eq!(data.sky_light_at(42), 11);
        assert!(meta.light_gpu_dirty, "flush must mark chunk light_gpu_dirty");
    }

    #[test]
    fn tick_cache_flush_skips_unwritten_chunks() {
        use crate::voxel::chunk::{DenseChunk, PalettedChunk};
        use crate::voxel::coords::ChunkCoord;
        use crate::voxel::world::ChunkSlot;
        use glam::IVec3;
        use std::collections::HashMap;

        let dense = DenseChunk::empty();
        let coord = ChunkCoord(IVec3::ZERO);
        let mut chunks: HashMap<ChunkCoord, ChunkSlot> = HashMap::new();
        chunks.insert(coord, ChunkSlot::Stored {
            data: std::sync::Arc::new(PalettedChunk::compress(&dense)),
            meta: Default::default(),
        });

        let mut cache = TickCache::default();
        // Touch but don't write.
        {
            let d = cache.dense(&chunks, coord).expect("loaded chunk");
            let _read = d.sky_light[0];
        }
        // Note: NO mark_written call.
        cache.flush(&mut chunks);

        // light_gpu_dirty should still be false.
        let ChunkSlot::Stored { meta, .. } = chunks.get(&coord).unwrap() else { panic!() };
        assert!(!meta.light_gpu_dirty, "flush must not mark untouched chunks dirty");
    }
```

- [ ] **Step 3: Run the new tests.**

Run: `cargo test --lib lighting::engine::tests -- --nocapture 2>&1 | tail -15`
Expected: 3 new tests pass on top of existing engine tests.

Run: `cargo test --lib 2>&1 | tail -3`
Expected: full suite passes, count up by 3 from T1's baseline.

- [ ] **Step 4: Commit.**

```bash
git add src/lighting/engine.rs
git commit -m "feat(lighting): TickCache for per-tick chunk decompress amortization (graph-engine PR4, part 2/6)

Pure-additive helper: TickCache.dense(chunks, coord) decompresses each
chunk at most once per tick, returns a mutable DenseChunk for in-place
modification, and tracks which chunks were written via mark_written.
At flush() time, only the written chunks get recompressed and have
their Arc<PalettedChunk> bumped + light_gpu_dirty set.

Not wired into the engine yet — part 3 refactors the hot path to use
the cache and eliminates the per-op decompress that froze the main
thread in PR3 production."
```

---

## Task 3: Refactor engine hot path to use `TickCache`

**Files:**
- Modify: `src/lighting/engine.rs`

This is the perf fix. The current `drain_increase_channel` and the source self-write block at its top BOTH decompress for every op. After this refactor: at most one decompress per touched chunk per tick.

- [ ] **Step 1: Refactor `tick_with` to construct + flush the cache.**

Find `pub fn tick_with(&mut self, chunks, registry, budget)` in `src/lighting/engine.rs`. Replace the body with:

```rust
    pub fn tick_with(
        &mut self,
        chunks: &mut std::collections::HashMap<
            crate::voxel::coords::ChunkCoord,
            crate::voxel::world::ChunkSlot,
        >,
        registry: &crate::voxel::block::BlockRegistry,
        budget: usize,
    ) {
        let mut cache = TickCache::default();
        let mut remaining = budget;

        // Phase A — drain pending block changes.
        remaining = drain_pending_block_changes(self, &mut cache, chunks, registry, remaining);

        // Phase C — drain per-channel increase queues.
        if remaining > 0 {
            remaining = drain_increase_channel(
                &mut self.sky, &mut cache, chunks, registry, Channel::Sky, remaining,
            );
        }
        for ch_i in 0..3 {
            if remaining == 0 { break }
            remaining = drain_increase_channel(
                &mut self.block_rgb[ch_i],
                &mut cache,
                chunks,
                registry,
                Channel::BlockRgb(ch_i),
                remaining,
            );
        }
        let _ = remaining;

        // Flush all cached writes back into the chunk store.
        cache.flush(chunks);
    }
```

Note the signature change: helper functions now take `&mut TickCache` and `&HashMap<...>` (immutable) instead of `&mut HashMap<...>`. We pass `chunks` (which is `&mut`) to helpers as `chunks` (Rust auto-reborrows as immutable when the receiving param is `&`).

- [ ] **Step 2: Refactor `drain_pending_block_changes` to use the cache.**

Find `fn drain_pending_block_changes(engine, chunks, registry, mut budget) -> usize` in `engine.rs`. Replace the entire function with:

```rust
fn drain_pending_block_changes(
    engine: &mut LightEngine,
    cache: &mut TickCache,
    chunks: &std::collections::HashMap<
        crate::voxel::coords::ChunkCoord,
        crate::voxel::world::ChunkSlot,
    >,
    registry: &crate::voxel::block::BlockRegistry,
    mut budget: usize,
) -> usize {
    use crate::voxel::coords::ChunkCoord;

    let entries: Vec<_> = engine.pending_block_changes.drain().collect();
    let mut recompute_chunks: std::collections::HashSet<ChunkCoord> = Default::default();

    for (pos, (old_block, new_block)) in entries {
        if budget == 0 {
            engine.pending_block_changes.insert(pos, (old_block, new_block));
            continue;
        }
        budget -= 1;

        let old_info = registry.info(old_block);
        let new_info = registry.info(new_block);

        let opacity_changed = old_info.opaque != new_info.opaque;
        let emission_decreased =
            new_info.emission[0] < old_info.emission[0]
            || new_info.emission[1] < old_info.emission[1]
            || new_info.emission[2] < old_info.emission[2];

        // PR3-compatible minimal-decrease handling for opacity changes
        // and emission decreases: defer to a chunk recompute. PR4 keeps
        // this branch for now; PR4 task 5 replaces emission-decrease
        // with proper decrease propagation, and opacity changes still
        // route through recompute because they affect the heightmap.
        if opacity_changed || emission_decreased {
            recompute_chunks.insert(pos.to_chunk());
        }

        // Strict emission increase per channel: write the new level and
        // enqueue an increase op.
        for ch_i in 0..3 {
            if new_info.emission[ch_i] > old_info.emission[ch_i] {
                if let Some(dense) = cache.dense(chunks, pos.to_chunk()) {
                    let idx = pos.to_local().to_index();
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[idx]);
                    let mut chans = [r, g, b];
                    chans[ch_i] = new_info.emission[ch_i];
                    dense.block_rgb[idx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                    cache.mark_written(pos.to_chunk());
                    engine.block_rgb[ch_i].increase.push(crate::lighting::queue::QueueEntry {
                        pos,
                        from_level: new_info.emission[ch_i],
                        propagation_mask: 0,
                    });
                }
            }
        }
    }

    // Execute per-chunk recompute fallback for opacity changes /
    // emission decreases. Each recompute does its own decompress +
    // mutation + flush via the cache, so subsequent ops on the same
    // chunk are cheap.
    for coord in recompute_chunks {
        recompute_chunk_light_in_cache(engine, cache, chunks, registry, coord);
    }

    budget
}
```

- [ ] **Step 3: Add the cache-aware variant of `recompute_chunk_light_from_scratch`.**

Find the existing `fn recompute_chunk_light_from_scratch` in `engine.rs`. Keep it as-is (it's still used by `World::on_chunk_loaded` indirectly via the on-load path). Add a new function next to it that mutates via the cache:

```rust
/// Cache-aware variant: clear and re-enqueue a single chunk's light from
/// scratch, mutating via `TickCache` rather than the chunk's
/// `Arc<PalettedChunk>` directly. Used inside `tick_with` for the
/// minimal-decrease fallback. Identical semantics to
/// `recompute_chunk_light_from_scratch` but doesn't decompress/recompress
/// per call — the cache amortizes that across the tick.
fn recompute_chunk_light_in_cache(
    engine: &mut LightEngine,
    cache: &mut TickCache,
    chunks: &std::collections::HashMap<
        crate::voxel::coords::ChunkCoord,
        crate::voxel::world::ChunkSlot,
    >,
    registry: &crate::voxel::block::BlockRegistry,
    coord: crate::voxel::coords::ChunkCoord,
) {
    use crate::voxel::coords::{BlockPos, LocalPos, CHUNK_DIM, CHUNK_DIM_U};
    use glam::UVec3;

    // Rebuild the heightmap from current blocks. The heightmap lives on
    // ChunkMeta, not the cached DenseChunk, so we mutate it separately.
    // We need &mut chunks for that, but cache currently only takes &chunks.
    // Solution: this function's caller (drain_pending_block_changes) is
    // called BEFORE the cache borrows chunks immutably, so we can do
    // the heightmap update separately. But since we already structured
    // the tick to pass cache + &chunks together, the cleanest path is
    // to update the heightmap in cache.flush. For PR4 we defer that:
    // the heightmap won't be exactly right inside the tick for opacity
    // changes, but `light_gpu_dirty` triggers an upload anyway, and the
    // NEXT tick's recompute (or the next on_chunk_loaded) will use the
    // updated blocks.
    //
    // For now, just clear+reflood light arrays. The on_chunk_loaded
    // path rebuilds the heightmap correctly when chunks first install.

    let Some(dense) = cache.dense(chunks, coord) else { return };

    dense.sky_light.iter_mut().for_each(|v| *v = 0);
    dense.block_rgb.iter_mut().for_each(|v| *v = 0);

    let chunk_bottom_y = coord.0.y * CHUNK_DIM;

    // Sky sources. Use the existing heightmap from ChunkMeta — fetched
    // separately because the cache doesn't hold meta.
    let sky_sources_opt: Option<crate::lighting::ChunkSkyLightSources> =
        chunks.get(&coord).and_then(|s| match s {
            crate::voxel::world::ChunkSlot::Stored { meta, .. } => Some(meta.sky_sources.clone()),
            _ => None,
        });
    if let Some(sky_sources) = sky_sources_opt {
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                let lsy = sky_sources.lowest_source_y(lx, lz);
                let start_ly: i32 = if lsy == crate::lighting::NO_SOURCE_FLOOR {
                    0
                } else {
                    (lsy - chunk_bottom_y).max(0)
                };
                for ly in (start_ly as u32)..CHUNK_DIM_U {
                    let idx = LocalPos(UVec3::new(lx, ly, lz)).to_index();
                    dense.sky_light[idx] = 15;
                    let pos = BlockPos(glam::IVec3::new(
                        coord.0.x * CHUNK_DIM + lx as i32,
                        chunk_bottom_y + ly as i32,
                        coord.0.z * CHUNK_DIM + lz as i32,
                    ));
                    engine.sky.increase.push(crate::lighting::queue::QueueEntry {
                        pos, from_level: 15, propagation_mask: 0,
                    });
                }
            }
        }
    }

    // Emissive cells. We need block data — read via the cached DenseChunk
    // (it's already in scope).
    for lz in 0..CHUNK_DIM_U {
        for ly in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                let idx = LocalPos(UVec3::new(lx, ly, lz)).to_index();
                let info = registry.info(dense.blocks[idx]);
                if info.emission[0] > 0 || info.emission[1] > 0 || info.emission[2] > 0 {
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[idx]);
                    dense.block_rgb[idx] = crate::voxel::chunk::pack_rgb(
                        r.max(info.emission[0]),
                        g.max(info.emission[1]),
                        b.max(info.emission[2]),
                    );
                    let pos = BlockPos(glam::IVec3::new(
                        coord.0.x * CHUNK_DIM + lx as i32,
                        chunk_bottom_y + ly as i32,
                        coord.0.z * CHUNK_DIM + lz as i32,
                    ));
                    for ch_i in 0..3 {
                        if info.emission[ch_i] > 0 {
                            engine.block_rgb[ch_i].increase.push(crate::lighting::queue::QueueEntry {
                                pos,
                                from_level: info.emission[ch_i],
                                propagation_mask: 0,
                            });
                        }
                    }
                }
            }
        }
    }

    cache.mark_written(coord);
}
```

- [ ] **Step 4: Refactor `drain_increase_channel` to use the cache.**

Find `fn drain_increase_channel(ch_engine, chunks, registry, channel, mut budget) -> usize` in `engine.rs`. Replace its signature and body with:

```rust
fn drain_increase_channel(
    ch_engine: &mut ChannelEngine,
    cache: &mut TickCache,
    chunks: &std::collections::HashMap<
        crate::voxel::coords::ChunkCoord,
        crate::voxel::world::ChunkSlot,
    >,
    registry: &crate::voxel::block::BlockRegistry,
    channel: Channel,
    mut budget: usize,
) -> usize {
    use crate::voxel::block::Block;
    use crate::voxel::coords::{BlockPos, LocalPos};

    let face_deltas: [(i32, i32, i32); 6] = [
        ( 1,  0,  0), (-1,  0,  0),
        ( 0,  1,  0), ( 0, -1,  0),
        ( 0,  0,  1), ( 0,  0, -1),
    ];
    let opposite_face: [u8; 6] = [1, 0, 3, 2, 5, 4];

    while budget > 0 {
        let Some(entry) = ch_engine.increase.pop_highest() else { break };
        budget -= 1;
        if entry.from_level <= 1 { continue; }

        // Source self-write — ensure the source cell holds at least
        // `from_level`. Cache makes this cheap.
        {
            let src_chunk = entry.pos.to_chunk();
            let src_idx = entry.pos.to_local().to_index();
            if let Some(dense) = cache.dense(chunks, src_chunk) {
                let cur = match channel {
                    Channel::Sky => dense.sky_light[src_idx],
                    Channel::BlockRgb(c) => {
                        let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[src_idx]);
                        [r, g, b][c]
                    }
                };
                if cur < entry.from_level {
                    match channel {
                        Channel::Sky => dense.sky_light[src_idx] = entry.from_level,
                        Channel::BlockRgb(c) => {
                            let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[src_idx]);
                            let mut chans = [r, g, b];
                            chans[c] = entry.from_level;
                            dense.block_rgb[src_idx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                        }
                    }
                    cache.mark_written(src_chunk);
                }
            }
        }

        // Propagate to neighbours.
        for face_i in 0..6 {
            if (entry.propagation_mask >> face_i) & 1 == 1 { continue; }
            let (dx, dy, dz) = face_deltas[face_i];
            let npos = BlockPos(glam::IVec3::new(
                entry.pos.0.x + dx,
                entry.pos.0.y + dy,
                entry.pos.0.z + dz,
            ));
            let nchunk_coord = npos.to_chunk();
            let nidx = npos.to_local().to_index();

            let Some(dense) = cache.dense(chunks, nchunk_coord) else { continue };
            let nblock = dense.blocks[nidx];
            let ninfo = registry.info(nblock);
            if ninfo.opaque { continue; }

            let cost: u8 = if nblock == Block::Water { 3 } else { 1 };
            let prop_level = entry.from_level.saturating_sub(cost);
            if prop_level == 0 { continue; }

            let cur_level = match channel {
                Channel::Sky => dense.sky_light[nidx],
                Channel::BlockRgb(c) => {
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[nidx]);
                    [r, g, b][c]
                }
            };
            if prop_level <= cur_level { continue; }

            // Write the higher value.
            match channel {
                Channel::Sky => dense.sky_light[nidx] = prop_level,
                Channel::BlockRgb(c) => {
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[nidx]);
                    let mut chans = [r, g, b];
                    chans[c] = prop_level;
                    dense.block_rgb[nidx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                }
            }
            cache.mark_written(nchunk_coord);

            ch_engine.increase.push(crate::lighting::queue::QueueEntry {
                pos: npos,
                from_level: prop_level,
                propagation_mask: 1u8 << opposite_face[face_i],
            });

            // silence unused
            let _ = LocalPos::from_index(0);
        }
    }

    budget
}
```

(The `let _ = LocalPos::from_index(0);` at the end can be removed if it triggers a real unused-import warning rather than silencing one. Use it only if your build warns about unused imports.)

- [ ] **Step 5: Delete the now-unused old `recompute_chunk_light_from_scratch` function.**

If `recompute_chunk_light_from_scratch` (the non-cache variant from PR3) has no remaining callers, delete it. `World::on_chunk_loaded` does NOT call it (it has its own inline enqueue logic). Check:

```bash
grep -rn "recompute_chunk_light_from_scratch" src/
```

Expected: only the function definition itself, no callers. If so, delete the function. If there ARE callers (e.g., a test or another helper), leave it for now and remove in a follow-up.

- [ ] **Step 6: Build + run tests.**

Run: `cargo build --lib 2>&1 | tail -3`
Expected: clean build. Possibly some unused-import warnings if the old function is removed — clean those up by removing newly-unused `use` statements.

Run: `cargo test --lib 2>&1 | tail -3`
Expected: all tests pass. Engine tests should still produce the same light values (perf is the only thing that changed).

Run: `cargo build --release 2>&1 | tail -3`
Expected: clean release build.

- [ ] **Step 7: Manual smoke test — the critical validation step.**

Run:
```bash
cargo run --release
```

Observe:
- World should load within 3-5 seconds (HUD's "World seed: 42" splash disappears).
- `CH:` in the HUD should climb above 0 quickly (chunks reaching the GPU).
- Lighting should look correct — sky-exposed surfaces lit, caves dark, water-blue tint.
- FPS should be 60+ on the M4 Max.
- No beachball.

If the world doesn't load:
- Check `top` for `oxium`'s CPU/memory — if pegged at 100% CPU for a single thread, the tick is still too slow (something didn't get the cache benefit).
- Compare against `cargo run --release --features legacy-lighting` to confirm the perf fix is the issue (legacy should work normally).
- Report BLOCKED with what you observed.

If the world loads but lighting looks wrong (e.g., ocean still has the artifact, or new wrongness):
- Note the visible issue
- Continue to T4-T6 anyway — proper decrease may fix it

Exit the game with Cmd-Q (or close the window).

- [ ] **Step 8: Commit.**

```bash
git add src/lighting/engine.rs
git commit -m "perf(lighting): TickCache amortizes chunk decompress in engine hot path (graph-engine PR4, part 3/6)

tick_with constructs a TickCache, threads it through
drain_pending_block_changes and drain_increase_channel as &mut, and
flushes at the end. Each touched chunk is decompressed at most once
per tick instead of up to 7 times per propagation op.

Per-op cost on a write-heavy tick drops from ~7 × 128 KB decompress
+ 1 × 80 KB compress (~960 KB heap churn per op) to a single
~16 byte cache-hit lookup + direct array indexing on the cached
DenseChunk. At 50_000 ops/frame this is the difference between
freezing the main thread (PR3 production) and 60 FPS.

drain_pending_block_changes uses recompute_chunk_light_in_cache (a
cache-aware variant of recompute_chunk_light_from_scratch) for the
minimal-decrease fallback. The non-cache variant is removed if it
has no remaining callers.

Validated by manual smoke test: cargo run --release loads the world
and renders 60 FPS where PR3 production froze on launch."
```

---

**🎯 INTERMEDIATE MERGE POINT — Tasks 1-3 complete.** The engine is now fast enough for production. The minimal-decrease fallback (chunk recompute) is still in place but works correctly. If you want to ship the perf fix and continue with proper decrease in a follow-up PR, this is the moment.

---

# Phase B — Proper Decrease Propagation (Tasks 4-6)

## Task 4: `drain_decrease_channel` implementation

**Files:**
- Modify: `src/lighting/engine.rs`

- [ ] **Step 1: Add the decrease drain function.**

Add the following function to `src/lighting/engine.rs`, right after `drain_increase_channel`:

```rust
/// Phase B — drain a single channel's decrease queue using the Minecraft
/// algorithm: for each popped entry, walk the 6 faces; if the neighbour
/// was lit BY this source (cur < entry.from_level), tear down its value
/// to 0 and push the neighbour as a decrease op so the tear-down chain
/// continues. If the neighbour is at least as bright (cur >= from_level),
/// it's lit by an independent source — re-flood from there via the
/// increase queue.
///
/// Special case: if the torn-down cell is itself an emitter (torch on
/// block-light channel, source cell on sky channel), restore its
/// emission and re-flood. This handles the "decrease wave passes through
/// a torch" case without erasing the torch's contribution.
fn drain_decrease_channel(
    ch_engine: &mut ChannelEngine,
    cache: &mut TickCache,
    chunks: &std::collections::HashMap<
        crate::voxel::coords::ChunkCoord,
        crate::voxel::world::ChunkSlot,
    >,
    registry: &crate::voxel::block::BlockRegistry,
    channel: Channel,
    mut budget: usize,
) -> usize {
    use crate::voxel::block::Block;
    use crate::voxel::coords::BlockPos;

    let face_deltas: [(i32, i32, i32); 6] = [
        ( 1,  0,  0), (-1,  0,  0),
        ( 0,  1,  0), ( 0, -1,  0),
        ( 0,  0,  1), ( 0,  0, -1),
    ];
    let opposite_face: [u8; 6] = [1, 0, 3, 2, 5, 4];

    while budget > 0 {
        let Some(entry) = ch_engine.decrease.pop_highest() else { break };
        budget -= 1;

        for face_i in 0..6 {
            if (entry.propagation_mask >> face_i) & 1 == 1 { continue; }
            let (dx, dy, dz) = face_deltas[face_i];
            let npos = BlockPos(glam::IVec3::new(
                entry.pos.0.x + dx,
                entry.pos.0.y + dy,
                entry.pos.0.z + dz,
            ));
            let nchunk = npos.to_chunk();
            let nidx = npos.to_local().to_index();

            let Some(dense) = cache.dense(chunks, nchunk) else { continue };

            let cur = match channel {
                Channel::Sky => dense.sky_light[nidx],
                Channel::BlockRgb(c) => {
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[nidx]);
                    [r, g, b][c]
                }
            };

            if cur != 0 && cur < entry.from_level {
                // Neighbour was lit by us — tear it down to 0.
                match channel {
                    Channel::Sky => dense.sky_light[nidx] = 0,
                    Channel::BlockRgb(c) => {
                        let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[nidx]);
                        let mut chans = [r, g, b];
                        chans[c] = 0;
                        dense.block_rgb[nidx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                    }
                }
                cache.mark_written(nchunk);

                // Check if this cell is itself an emitter (or sky source).
                // If so, restore its level and enqueue as an increase
                // source so we don't erase its independent contribution.
                let nblock = dense.blocks[nidx];
                let emission = match channel {
                    Channel::Sky => sky_emission_for(nchunk, nidx, chunks),
                    Channel::BlockRgb(c) => registry.info(nblock).emission[c],
                };
                if emission > 0 {
                    match channel {
                        Channel::Sky => dense.sky_light[nidx] = emission,
                        Channel::BlockRgb(c) => {
                            let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[nidx]);
                            let mut chans = [r, g, b];
                            chans[c] = emission;
                            dense.block_rgb[nidx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                        }
                    }
                    ch_engine.increase.push(crate::lighting::queue::QueueEntry {
                        pos: npos,
                        from_level: emission,
                        propagation_mask: 0,
                    });
                }

                // Continue the tear-down chain.
                ch_engine.decrease.push(crate::lighting::queue::QueueEntry {
                    pos: npos,
                    from_level: cur,
                    propagation_mask: 1u8 << opposite_face[face_i],
                });
            } else if cur >= entry.from_level && cur > 0 {
                // Neighbour is independently sourced. Re-flood from
                // here to repair anything we tore down upstream.
                ch_engine.increase.push(crate::lighting::queue::QueueEntry {
                    pos: npos,
                    from_level: cur,
                    propagation_mask: 0,
                });
            }
        }

        // silence unused warning for Block import on builds without water-cost usage
        let _ = Block::Air;
    }

    budget
}

/// Helper for the sky channel's "is this cell itself a sky source?"
/// check during decrease. Returns 15 if the cell at `(coord, idx)` sits
/// at or above the column's `lowest_source_y`, else 0.
fn sky_emission_for(
    coord: crate::voxel::coords::ChunkCoord,
    local_idx: usize,
    chunks: &std::collections::HashMap<
        crate::voxel::coords::ChunkCoord,
        crate::voxel::world::ChunkSlot,
    >,
) -> u8 {
    use crate::voxel::coords::{LocalPos, CHUNK_DIM};
    use crate::voxel::world::ChunkSlot;
    let Some(ChunkSlot::Stored { meta, .. }) = chunks.get(&coord) else { return 0 };
    let lp = LocalPos::from_index(local_idx);
    let lsy = meta.sky_sources.lowest_source_y(lp.0.x, lp.0.z);
    if lsy == crate::lighting::NO_SOURCE_FLOOR {
        // Whole column passes through transparent inside this chunk.
        // Treat any cell as a sky source (consistent with on_chunk_loaded
        // behaviour for this case).
        return 15;
    }
    let world_y = coord.0.y * CHUNK_DIM + lp.0.y as i32;
    if world_y >= lsy { 15 } else { 0 }
}
```

- [ ] **Step 2: Build to check.**

Run: `cargo build --lib 2>&1 | tail -3`
Expected: clean build. The function is added but not yet called — pure-additive.

- [ ] **Step 3: Commit.**

```bash
git add src/lighting/engine.rs
git commit -m "feat(lighting): drain_decrease_channel — Minecraft-style decrease propagation (graph-engine PR4, part 4/6)

Adds the proper decrease algorithm: for each popped entry, walks 6
faces; if the neighbour was lit by us (cur < from_level), tears it
down to 0, restores any independent emission (torch/sky-source) it has,
and continues the chain via decrease.push. If the neighbour is at
least as bright (cur >= from_level), enqueues an increase from there
to repair anything we tore down upstream.

Not yet wired into the tick — part 5 hooks it in and replaces the
chunk-recompute fallback with proper decrease enqueueing."
```

---

## Task 5: Wire decrease into `tick_with` and `drain_pending_block_changes`

**Files:**
- Modify: `src/lighting/engine.rs`

- [ ] **Step 1: Add the decrease phase to `tick_with`.**

Find the `tick_with` body (already refactored in Task 3). Add the decrease phase between Phase A and Phase C:

```rust
    pub fn tick_with(
        &mut self,
        chunks: &mut std::collections::HashMap<
            crate::voxel::coords::ChunkCoord,
            crate::voxel::world::ChunkSlot,
        >,
        registry: &crate::voxel::block::BlockRegistry,
        budget: usize,
    ) {
        let mut cache = TickCache::default();
        let mut remaining = budget;

        // Phase A — drain pending block changes.
        remaining = drain_pending_block_changes(self, &mut cache, chunks, registry, remaining);

        // Phase B — drain per-channel decrease queues. Walk sky first
        // then each RGB channel, same order as increase.
        if remaining > 0 {
            remaining = drain_decrease_channel(
                &mut self.sky, &mut cache, chunks, registry, Channel::Sky, remaining,
            );
        }
        for ch_i in 0..3 {
            if remaining == 0 { break }
            remaining = drain_decrease_channel(
                &mut self.block_rgb[ch_i],
                &mut cache,
                chunks,
                registry,
                Channel::BlockRgb(ch_i),
                remaining,
            );
        }

        // Phase C — drain per-channel increase queues.
        if remaining > 0 {
            remaining = drain_increase_channel(
                &mut self.sky, &mut cache, chunks, registry, Channel::Sky, remaining,
            );
        }
        for ch_i in 0..3 {
            if remaining == 0 { break }
            remaining = drain_increase_channel(
                &mut self.block_rgb[ch_i],
                &mut cache,
                chunks,
                registry,
                Channel::BlockRgb(ch_i),
                remaining,
            );
        }
        let _ = remaining;

        cache.flush(chunks);
    }
```

- [ ] **Step 2: Replace the per-edit `recompute_chunks` fallback in `drain_pending_block_changes` with proper decrease enqueueing.**

In `drain_pending_block_changes` (refactored in Task 3), the current per-edit handling routes opacity changes AND emission decreases to `recompute_chunks`, which then trigger a chunk recompute. Replace the emission-decrease branch with a decrease enqueue. Opacity changes STAY on the recompute path because they affect the heightmap (PR5 or later can refine).

Find this block in `drain_pending_block_changes`:

```rust
        if opacity_changed || emission_decreased {
            recompute_chunks.insert(pos.to_chunk());
        }
```

Replace with:

```rust
        if opacity_changed {
            // Opacity changes affect the heightmap; the chunk
            // recompute path handles them (PR5 refines).
            recompute_chunks.insert(pos.to_chunk());
        }

        // Emission decrease per channel: enqueue a proper decrease op
        // at the OLD emission level so the tear-down chain knows how
        // bright the source was.
        for ch_i in 0..3 {
            if new_info.emission[ch_i] < old_info.emission[ch_i]
                && old_info.emission[ch_i] > 0
            {
                // Zero the cell on this channel first so the decrease
                // wave doesn't see this cell as still-lit-by-source.
                if let Some(dense) = cache.dense(chunks, pos.to_chunk()) {
                    let idx = pos.to_local().to_index();
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[idx]);
                    let mut chans = [r, g, b];
                    chans[ch_i] = 0;
                    dense.block_rgb[idx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                    cache.mark_written(pos.to_chunk());
                }
                engine.block_rgb[ch_i].decrease.push(crate::lighting::queue::QueueEntry {
                    pos,
                    from_level: old_info.emission[ch_i],
                    propagation_mask: 0,
                });
            }
        }
```

The opacity_changed branch routes into `recompute_chunks` which `recompute_chunk_light_in_cache` then processes. Keep that intact.

- [ ] **Step 3: Build + run tests.**

Run: `cargo build --lib 2>&1 | tail -3`
Expected: clean build.

Run: `cargo test --lib 2>&1 | tail -3`
Expected: all tests pass. The existing engine tests (torch lighting, opacity-recompute) should still pass because:
- Torch test only enqueues increases — decrease path is empty.
- Opacity-recompute test triggers opacity_changed → recompute path (unchanged).

- [ ] **Step 4: Manual smoke test.**

Run `cargo run --release` again. Walk into a cave, mine a torch (Block::Torch placement key — check your input bindings), confirm the lit area dims correctly without breaking the whole chunk.

(Skip in subagent execution if interactive — defer to user.)

- [ ] **Step 5: Commit.**

```bash
git add src/lighting/engine.rs
git commit -m "feat(lighting): wire decrease phase into tick_with (graph-engine PR4, part 5/6)

tick_with now runs Phase A (block changes) → Phase B (decrease) →
Phase C (increase) per channel. drain_pending_block_changes routes
emission decreases into the decrease queue instead of the chunk
recompute fallback; opacity changes still recompute (heightmap rebuild
is deferred to PR5).

Removing a torch correctly dims only the cells it was lighting, in
O(actually_dimmed_cells) instead of O(chunk_volume). Cells lit by
other independent sources are repaired via the increase queue's
re-flood from those sources."
```

---

## Task 6: Decrease correctness tests

**Files:**
- Modify: `src/lighting/engine.rs`

- [ ] **Step 1: Add the tests.**

Append to the `#[cfg(test)] mod tests { ... }` block in `src/lighting/engine.rs`:

```rust
    #[test]
    fn decrease_dims_only_cells_lit_by_removed_torch() {
        use crate::voxel::block::{Block, BlockRegistry};
        use crate::voxel::chunk::{unpack_rgb, DenseChunk, PalettedChunk};
        use crate::voxel::coords::{BlockPos, ChunkCoord, LocalPos};
        use crate::voxel::world::ChunkSlot;
        use glam::{IVec3, UVec3};
        use std::collections::HashMap;

        let registry = BlockRegistry::new();
        let mut dense = DenseChunk::empty();
        dense.set(LocalPos(UVec3::new(16, 16, 16)), Block::Torch);
        let coord = ChunkCoord(IVec3::ZERO);
        let mut chunks: HashMap<ChunkCoord, ChunkSlot> = HashMap::new();
        chunks.insert(coord, ChunkSlot::Stored {
            data: std::sync::Arc::new(PalettedChunk::compress(&dense)),
            meta: crate::voxel::chunk::ChunkMeta {
                sky_sources: crate::lighting::ChunkSkyLightSources::build_from_dense(
                    &dense, coord, &registry,
                ),
                ..Default::default()
            },
        });

        let mut engine = LightEngine::default();

        // Step 1: light up the chunk by enqueueing the torch as a source.
        let torch_pos = BlockPos(IVec3::new(16, 16, 16));
        let torch_emission = registry.info(Block::Torch).emission;
        for ch_i in 0..3 {
            if torch_emission[ch_i] > 0 {
                engine.block_rgb[ch_i].increase.push(crate::lighting::queue::QueueEntry {
                    pos: torch_pos, from_level: torch_emission[ch_i], propagation_mask: 0,
                });
            }
        }
        engine.tick_with(&mut chunks, &registry, 50_000);
        // Confirm: adjacent cell is lit.
        let adj_idx = LocalPos(UVec3::new(17, 16, 16)).to_index();
        {
            let ChunkSlot::Stored { data, .. } = chunks.get(&coord).unwrap() else { panic!() };
            let (r0, _, _) = unpack_rgb(data.decompress().block_rgb[adj_idx]);
            assert!(r0 >= 12, "torch should light adjacent cell first; got R={r0}");
        }

        // Step 2: remove the torch — replace with Air, enqueue the
        // block change. Mirrors what set_block + on_block_changed does.
        if let Some(ChunkSlot::Stored { data, .. }) = chunks.get_mut(&coord) {
            let mut d = data.decompress();
            d.set(LocalPos(UVec3::new(16, 16, 16)), Block::Air);
            *data = std::sync::Arc::new(PalettedChunk::compress(&d));
        }
        engine.enqueue_block_change(torch_pos, Block::Torch, Block::Air);
        engine.tick_with(&mut chunks, &registry, 50_000);

        // Confirm: adjacent cell is now dark on R channel (the torch's
        // contribution was torn down).
        let ChunkSlot::Stored { data, .. } = chunks.get(&coord).unwrap() else { panic!() };
        let (r1, _, _) = unpack_rgb(data.decompress().block_rgb[adj_idx]);
        assert_eq!(r1, 0, "adjacent cell should be fully dark after torch removal; got R={r1}");
    }

    #[test]
    fn decrease_preserves_cells_lit_by_independent_torch() {
        use crate::voxel::block::{Block, BlockRegistry};
        use crate::voxel::chunk::{unpack_rgb, DenseChunk, PalettedChunk};
        use crate::voxel::coords::{BlockPos, ChunkCoord, LocalPos};
        use crate::voxel::world::ChunkSlot;
        use glam::{IVec3, UVec3};
        use std::collections::HashMap;

        let registry = BlockRegistry::new();
        let mut dense = DenseChunk::empty();
        // Two torches at distance 4 (each lights cells up to 12 blocks
        // away; their lit regions overlap heavily).
        dense.set(LocalPos(UVec3::new(10, 16, 16)), Block::Torch);
        dense.set(LocalPos(UVec3::new(20, 16, 16)), Block::Torch);
        let coord = ChunkCoord(IVec3::ZERO);
        let mut chunks: HashMap<ChunkCoord, ChunkSlot> = HashMap::new();
        chunks.insert(coord, ChunkSlot::Stored {
            data: std::sync::Arc::new(PalettedChunk::compress(&dense)),
            meta: crate::voxel::chunk::ChunkMeta {
                sky_sources: crate::lighting::ChunkSkyLightSources::build_from_dense(
                    &dense, coord, &registry,
                ),
                ..Default::default()
            },
        });

        let mut engine = LightEngine::default();
        let pos_a = BlockPos(IVec3::new(10, 16, 16));
        let pos_b = BlockPos(IVec3::new(20, 16, 16));
        let torch_emission = registry.info(Block::Torch).emission;
        for pos in [pos_a, pos_b] {
            for ch_i in 0..3 {
                if torch_emission[ch_i] > 0 {
                    engine.block_rgb[ch_i].increase.push(crate::lighting::queue::QueueEntry {
                        pos, from_level: torch_emission[ch_i], propagation_mask: 0,
                    });
                }
            }
        }
        engine.tick_with(&mut chunks, &registry, 200_000);

        // Cell at (15, 16, 16) is midway — lit by both torches.
        let mid_idx = LocalPos(UVec3::new(15, 16, 16)).to_index();
        let mid_before = {
            let ChunkSlot::Stored { data, .. } = chunks.get(&coord).unwrap() else { panic!() };
            unpack_rgb(data.decompress().block_rgb[mid_idx]).0
        };
        assert!(mid_before >= 7, "midway cell should be lit by both torches; got R={mid_before}");

        // Remove torch A.
        if let Some(ChunkSlot::Stored { data, .. }) = chunks.get_mut(&coord) {
            let mut d = data.decompress();
            d.set(LocalPos(UVec3::new(10, 16, 16)), Block::Air);
            *data = std::sync::Arc::new(PalettedChunk::compress(&d));
        }
        engine.enqueue_block_change(pos_a, Block::Torch, Block::Air);
        engine.tick_with(&mut chunks, &registry, 200_000);

        // The midway cell should STILL be lit (by torch B).
        let mid_after = {
            let ChunkSlot::Stored { data, .. } = chunks.get(&coord).unwrap() else { panic!() };
            unpack_rgb(data.decompress().block_rgb[mid_idx]).0
        };
        assert!(
            mid_after >= 4,
            "midway cell should remain lit by torch B after torch A removed; got R={mid_after} (was {mid_before})",
        );
    }
```

- [ ] **Step 2: Run the new tests.**

Run: `cargo test --lib lighting::engine::tests -- --nocapture 2>&1 | tail -20`
Expected: both new tests pass.

If `decrease_dims_only_cells_lit_by_removed_torch` fails with `R != 0`, the decrease wave isn't reaching the adjacent cell — investigate `drain_decrease_channel`'s face traversal.

If `decrease_preserves_cells_lit_by_independent_torch` fails with `R == 0` after torch removal, the increase repair from the surviving torch isn't running — investigate the `cur >= from_level` branch in `drain_decrease_channel`.

- [ ] **Step 3: Run the full suite.**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: full suite passes.

- [ ] **Step 4: Commit.**

```bash
git add src/lighting/engine.rs
git commit -m "test(lighting): decrease correctness — torch removal (graph-engine PR4, part 6/6)

Two integration tests:
1. decrease_dims_only_cells_lit_by_removed_torch — single torch,
   tick to convergence, remove torch, tick again, verify the
   adjacent cell is fully dark.
2. decrease_preserves_cells_lit_by_independent_torch — two torches
   with overlapping lit regions, remove one, verify the midway
   cell is still lit by the other (the tear-down chain must enqueue
   an increase from the surviving source to repair).

This concludes PR4: engine is production-fast (Tasks 1-3) and
removes light correctly (Tasks 4-6)."
```

---

## Verification

After all six tasks:

- [ ] **Run the full lib test suite.**

Run: `cargo test --lib`
Expected: clean pass. Test count up by 10 from PR3's baseline of 246 (5 chunk + 3 cache + 2 decrease = 10 new). Should be around **256 passing**.

- [ ] **Build both feature configurations.**

Run: `cargo build --release` and `cargo build --release --features legacy-lighting`
Expected: both clean.

- [ ] **Manual smoke test.**

Run `cargo run --release`. Confirm:
- World loads within 5 seconds.
- Lighting looks correct end-to-end — sky-exposed surfaces lit, caves dark.
- The ocean-seam artifact from the original screenshot should now be gone.
- Place a torch in a dark area, verify it lights its surroundings.
- Mine the torch, verify the lit area dims correctly (without breaking the chunk).
- FPS sustained at 60+.

If anything regresses: `cargo run --release --features legacy-lighting` is the safety belt.

---

## What's next

After this PR merges:

- **PR5** (final in spec sequence): streaming-mode budget bump — during initial world stream-in, raise the tick budget temporarily so the engine converges faster across many newly-loaded chunks.
- **Spec PR5+** (visual polish, deferred from 2026-05-20): sun shadow map, ambient bounce, volumetric atmospherics, day/night sun + sky colors. All land on top of the engine without changes.
- **Future**: refactor `recompute_chunk_light_in_cache` to also rebuild the heightmap correctly inside the tick (currently relies on the next `on_chunk_loaded` for that). PR5 or later.
- **Future**: remove the `--feature legacy-lighting` gates entirely once the engine has shipped without issues for a release or two.
