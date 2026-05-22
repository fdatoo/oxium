# Lighting Graph Engine PR 3 — Cutover (Engine Implementation + Production Wiring)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `LightEngine::tick` (increase phase + minimal "clear-and-reflood" decrease), wire it into `World` (`on_chunk_loaded`, `on_block_changed`, `light_engine_tick`), wire those into production call sites (mesh_upload, set_block, App::update), switch GPU upload to read from engine writes, and gate the old BFS behind `--feature legacy-lighting` as a runtime fallback. After this PR the engine is the default light propagator; visual output changes (ocean-seam artifacts disappear; chunks loaded out of order produce correct sky values).

**Architecture:** Engine reads block/opacity data from `World.chunks`, reads sky-source heightmaps from `ChunkMeta.sky_sources` (populated in PR1), and writes light values into `DenseChunk.sky_light`/`block_rgb`. The borrow-aliasing problem (`light_engine` is a field of `World`) is solved by `World::light_engine_tick`, which destructures `&mut self` into `(&mut chunks, &mut light_engine, &registry)` before calling the engine. `mesh_upload` switches from scanning `dirty.light` to scanning a new `ChunkMeta.light_gpu_dirty` flag the engine sets on writes.

**Tech stack:** Rust. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-21-lighting-graph-engine-design.md`

**Risk:** **High.** This is the biggest PR of the 5-PR sequence. It deletes hundreds of lines of cascade plumbing and changes the canonical source of light values. The `--feature legacy-lighting` flag exists as a safety net.

---

## Optional intermediate merge point

Tasks 1-5 are **pure-additive**: they implement the engine and add dormant methods on `World`, but no production code path invokes any of them. After Task 5 the engine is fully tested but disconnected — visual behaviour and the existing BFS are unchanged. If you want to validate the engine's correctness before flipping the production switch, **you can merge after Task 5** and continue with Tasks 6-9 in a follow-up PR.

Tasks 6-9 are the actual cutover. They wire the engine into mesh_upload, set_block, the frame loop, and the GPU upload path, then gate the old BFS behind the feature flag. **Visual output changes when Task 9 lands.**

---

## Files

**Modify:**
- `src/lighting/engine.rs` — implement `tick`, add `pending_block_changes`, `From<RgbChannel> for usize`, `RgbChannel::all()`
- `src/voxel/chunk.rs` — add `pub light_gpu_dirty: bool` to `ChunkMeta`
- `src/voxel/world.rs` — add `on_chunk_loaded`, `on_block_changed`, `light_engine_tick` methods; wire `set_block` to call `on_block_changed`
- `src/ecs/systems/mesh_upload.rs` — call `world.on_chunk_loaded` after install in `Generated` / `LoadedFromDisk`; add `upload_dirty_light_volumes`; delete or gate the `Relit` handler and `relight_pump`
- `src/app.rs` — add `world.light_engine_tick(budget)` call to the frame loop; delete or gate the `relight_pump` call
- `Cargo.toml` — add `[features] legacy-lighting = []`
- `src/lighting/mod.rs` — gate `recompute_chunk`, `sky_light`, `block_rgb`, `seed_from_neighbors`, `snapshot_face_boundaries`, `BfsChannel` behind `#[cfg(feature = "legacy-lighting")]`
- `src/jobs/mod.rs` — gate `JobResult::Relit` and `spawn_relight` behind the feature
- `src/voxel/world.rs` — gate `mark_below_dirty` behind the feature
- `src/voxel/chunk.rs` — gate `ChunkDirty::light` behind the feature

**Do not touch:**
- `ChunkSkyLightSources` (PR1, stable)
- `BucketQueue` (PR2, stable)
- The existing BFS function bodies — they get gated by feature flag, not modified.

---

## Design notes

### Why `World::light_engine_tick` (not `LightEngine::tick(&mut World)`)

`light_engine` is a field of `World`, so a method on `LightEngine` cannot take `&mut World` — that would require `&mut self.light_engine` and `&mut self` simultaneously, which Rust forbids. The fix is a method **on `World`** that destructures `&mut self`:

```rust
impl World {
    pub fn light_engine_tick(&mut self, budget: usize) {
        let Self { chunks, registry, light_engine, .. } = self;
        light_engine.tick_with(chunks, registry, budget);
    }
}
```

`LightEngine::tick_with(chunks, registry, budget)` takes the destructured pieces and operates on them. The old `LightEngine::tick(&mut self, _budget)` from PR2 stays for any tests that don't want a full World; PR3 adds `tick_with` as the production entry point.

### Why minimal decrease (clear-and-reflood) for PR3

The spec's "PR4: decrease queue correctness pass" implements the proper Minecraft-style decrease (tear-down chain + repair from independent sources). For PR3, when the engine sees a decrease op (source removed, opacity increased), it falls back to recomputing the affected chunk's light from scratch using the engine's own increase machinery — clear that chunk's sky_light and block_rgb to zero, re-enqueue every sky source and emissive cell in the chunk, then let the increase queue rebuild. This is wasteful per-edit but correct, and removes the need for the full decrease algorithm in PR3.

### Why `ChunkMeta.light_gpu_dirty` (new flag, not reuse `dirty.light`)

`dirty.light` today means "this chunk owes a BFS recompute". After the cutover, that meaning goes away — the engine writes incrementally rather than recomputing. The new `light_gpu_dirty` flag means specifically "the engine wrote to this chunk's light arrays since the last GPU upload; re-upload the 3D texture." Different semantic, different name. `dirty.light` survives behind the `legacy-lighting` feature; in the default build it's gated out.

### Why keep `sky_sources` on `ChunkMeta` (not migrate to `LightEngine.sky_sources`)

The spec's data model puts `sky_sources: HashMap<ChunkCoord, ChunkSkyLightSources>` on `LightEngine`. PR1 put it on `ChunkMeta` and the PR1 final review explicitly deferred this decision to PR3. PR3 keeps it on `ChunkMeta` — the engine reads it via `&ChunkMeta`. Rationale: `ChunkMeta` is already the per-chunk metadata holder, `World::insert` already populates it (PR1), and moving it to `LightEngine` requires duplicating the populate/cleanup paths with no concrete benefit. If a future refactor wants a flat per-coord HashMap, the migration is one PR.

---

# Phase A — Engine implementation (Tasks 1-5, pure-additive)

## Task 1: `From<RgbChannel> for usize` + `RgbChannel::all()`

**Files:**
- Modify: `src/lighting/engine.rs`

- [ ] **Step 1: Add the `From` impl and `all()` iterator to `RgbChannel`.**

Find the `impl LightEngine` block in `src/lighting/engine.rs`. Above it (right after the `RgbChannel` enum definition, around line 70-80), add:

```rust
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
```

- [ ] **Step 2: Add unit tests at the bottom of the `tests` module.**

In the `#[cfg(test)] mod tests { ... }` block at the bottom of `engine.rs`, add:

```rust
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
```

- [ ] **Step 3: Run tests.**

Run: `cargo test --lib lighting::engine::tests`
Expected: 5 passed, 0 failed (2 from PR2 + 3 new).

- [ ] **Step 4: Commit.**

```bash
git add src/lighting/engine.rs
git commit -m "feat(lighting): From<RgbChannel> for usize + ::ALL iterator (graph-engine PR3, part 1/9)

Lets the tick loop index block_rgb[ch.into()] uniformly. ::ALL is the
canonical [R, G, B] order used by per-channel iteration in tick.
Flagged as carryover from PR2 final review."
```

---

## Task 2: `pending_block_changes` side-table on `LightEngine`

**Files:**
- Modify: `src/lighting/engine.rs`

- [ ] **Step 1: Add the field + a helper to enqueue a change.**

Find the `pub struct LightEngine { ... }` block. Replace it with:

```rust
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
```

Update `impl Default for LightEngine` to initialise the new field:

```rust
impl Default for LightEngine {
    fn default() -> Self {
        Self {
            sky: ChannelEngine::default(),
            block_rgb: std::array::from_fn(|_| ChannelEngine::default()),
            pending_block_changes: std::collections::HashMap::new(),
        }
    }
}
```

Add a method to `impl LightEngine`:

```rust
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
```

Update `LightEngine::is_idle` to also consider `pending_block_changes`:

```rust
    pub fn is_idle(&self) -> bool {
        self.pending_block_changes.is_empty()
            && self.sky.is_idle()
            && self.block_rgb.iter().all(ChannelEngine::is_idle)
    }
```

- [ ] **Step 2: Add unit tests.**

Append to the `tests` module:

```rust
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
```

- [ ] **Step 3: Run tests.**

Run: `cargo test --lib lighting::engine::tests`
Expected: 7 passed (5 from Task 1 + 2 new).

- [ ] **Step 4: Commit.**

```bash
git add src/lighting/engine.rs
git commit -m "feat(lighting): pending_block_changes side-table (graph-engine PR3, part 2/9)

LightEngine gains a HashMap<BlockPos, (Block, Block)> side-table
populated by enqueue_block_change. The tick body (next task) consumes
it to compute the right increase/decrease ops per channel without
re-querying World state mid-iteration."
```

---

## Task 3: Implement `LightEngine::tick_with` — increase phase

**Files:**
- Modify: `src/lighting/engine.rs`

This task implements the core propagation algorithm. The decrease handling is minimal (clear-and-reflood the affected chunk) and gets refined in Task 4.

- [ ] **Step 1: Add the `tick_with` method and its helpers.**

Add the following at the bottom of `impl LightEngine` (right before the `tests` module):

```rust
    /// Production tick entry point. Drains pending block changes and
    /// propagation queues, writing light values directly into the
    /// chunks. `chunks` and `registry` are passed in by
    /// `World::light_engine_tick` after destructuring `&mut World` —
    /// the destructure side-steps the field-aliasing problem (the
    /// engine itself is a field of World).
    ///
    /// Processes at most `budget` total operations across all phases
    /// and channels. If `budget` is exhausted mid-channel, the
    /// remaining work persists in the queues and resumes next tick.
    ///
    /// PR3 ships with **minimal decrease**: when a change reduces a
    /// cell's contribution, we fall back to recomputing the chunk's
    /// light from scratch using the increase machinery. PR4 will
    /// replace this with proper Minecraft-style decrease propagation.
    pub fn tick_with(
        &mut self,
        chunks: &mut std::collections::HashMap<
            crate::voxel::coords::ChunkCoord,
            crate::voxel::world::ChunkSlot,
        >,
        registry: &crate::voxel::block::BlockRegistry,
        budget: usize,
    ) {
        let mut remaining = budget;

        // Phase A — drain pending block changes across all channels.
        // For each changed position, decide if it's a strict increase
        // (only new emission > old emission, no opacity change), a
        // strict decrease (emission down or opacity up), or both;
        // then either enqueue increase ops or mark the affected chunk
        // for full recompute (minimal-decrease fallback).
        remaining = drain_pending_block_changes(self, chunks, registry, remaining);

        // Phase B — drain the per-channel increase queues. Walk
        // sky first then each RGB channel.
        if remaining > 0 {
            remaining = drain_increase_channel(&mut self.sky, chunks, registry, Channel::Sky, remaining);
        }
        for ch_i in 0..3 {
            if remaining == 0 { break }
            remaining = drain_increase_channel(
                &mut self.block_rgb[ch_i],
                chunks,
                registry,
                Channel::BlockRgb(ch_i),
                remaining,
            );
        }
        let _ = remaining;
    }
```

Then add the supporting types and helper functions at the bottom of the file (above the `tests` module). These are private to the module — they implement the engine's internals.

```rust
/// Which channel a tick step is processing. Used by the propagation
/// helpers to decide how to read/write the right slot of the chunk's
/// light arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Sky,
    BlockRgb(usize),  // 0=R, 1=G, 2=B
}

/// Phase A. Drains `engine.pending_block_changes` and, for each entry,
/// updates the chunk's `sky_sources` heightmap if opacity changed, then
/// either enqueues increase ops (new emission, new sky-source exposure)
/// or marks the chunk for full recompute (minimal-decrease fallback).
/// Returns the remaining budget.
fn drain_pending_block_changes(
    engine: &mut LightEngine,
    chunks: &mut std::collections::HashMap<
        crate::voxel::coords::ChunkCoord,
        crate::voxel::world::ChunkSlot,
    >,
    registry: &crate::voxel::block::BlockRegistry,
    mut budget: usize,
) -> usize {
    use crate::voxel::block::Block;
    use crate::voxel::coords::ChunkCoord;
    use crate::voxel::world::ChunkSlot;

    // Snapshot the keys so we can mutate `engine.pending_block_changes`
    // and `chunks` during iteration. The map itself is cleared at the
    // end of this function regardless of which entries we actually
    // process — the engine sees each change at most once.
    let entries: Vec<_> = engine.pending_block_changes.drain().collect();

    // Track chunks that need full recompute (minimal-decrease fallback).
    let mut recompute_chunks: std::collections::HashSet<ChunkCoord> = Default::default();

    for (pos, (old_block, new_block)) in entries {
        if budget == 0 {
            // Re-insert what we haven't processed. (Pessimistic: a
            // remaining entry might just be re-queued cleanly, but we
            // preserve it as-is so the next tick sees it.)
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
        let opacity_increased = !old_info.opaque && new_info.opaque;

        // ---- Decide handling per channel ----

        // Sky channel: opacity changes affect the heightmap and may
        // require recompute (a roof was added → some cells go from
        // source to non-source; a roof was removed → some cells become
        // sources). For PR3 minimal decrease, both directions trigger
        // a chunk recompute.
        if opacity_changed {
            recompute_chunks.insert(pos.to_chunk());
        }

        // Block-light channels: emission delta drives behaviour.
        // Strict increase (any channel's emission rose): enqueue an
        // increase op on that channel at this pos at the new level.
        // Strict decrease: chunk recompute.
        if emission_decreased {
            recompute_chunks.insert(pos.to_chunk());
        }
        for ch_i in 0..3 {
            if new_info.emission[ch_i] > old_info.emission[ch_i] {
                // Write the new level into the chunk's storage and
                // enqueue an increase op.
                if let Some(ChunkSlot::Stored { data, meta }) =
                    chunks.get_mut(&pos.to_chunk())
                {
                    let mut dense = data.decompress();
                    let idx = pos.to_local().to_index();
                    // Pack the new emission as the cell's level.
                    let cur = crate::voxel::chunk::unpack_rgb(dense.block_rgb[idx]);
                    let new_packed = crate::voxel::chunk::pack_rgb(
                        if ch_i == 0 { new_info.emission[0] } else { cur.0 },
                        if ch_i == 1 { new_info.emission[1] } else { cur.1 },
                        if ch_i == 2 { new_info.emission[2] } else { cur.2 },
                    );
                    dense.block_rgb[idx] = new_packed;
                    *data = std::sync::Arc::new(crate::voxel::chunk::PalettedChunk::compress(&dense));
                    meta.light_gpu_dirty = true;
                    engine.block_rgb[ch_i].increase.push(crate::lighting::queue::QueueEntry {
                        pos,
                        from_level: new_info.emission[ch_i],
                        propagation_mask: 0,
                    });
                }
            }
        }

        // Opacity decrease (was opaque, now transparent) on a non-air
        // block: nothing new to do for the increase phase — the
        // recompute handles it via heightmap rebuild + re-enqueue.
        // (Strictly, a new sky-source cell could be enqueued here too,
        // but the recompute path is simpler and PR4 will refine.)
        let _ = opacity_increased;
        // Note: emission==0 default case is the common "place a stone"
        // path. Stone is opaque, so opacity_changed triggers recompute.
        let _ = (new_block, old_block, Block::Air);  // silence unused warnings
    }

    // Execute the recompute fallback for every affected chunk.
    for coord in recompute_chunks {
        if budget == 0 {
            // Bail and let next tick resume — but we've already drained
            // pending_block_changes for this tick, so the recompute
            // won't be retriggered automatically. To keep correctness
            // simple, do the recompute anyway and only check budget
            // between chunks.
            // (Defer: PR4 will revisit budgeting under heavy edit load.)
        }
        budget = budget.saturating_sub(estimate_chunk_recompute_cost());
        recompute_chunk_light_from_scratch(engine, chunks, registry, coord);
    }

    budget
}

/// Rough cost estimate (in node-ops) for one chunk's full recompute.
/// Used to decrement the tick budget proportionally.
fn estimate_chunk_recompute_cost() -> usize {
    // 32^3 cells × 4 channels × ~3 visits each = ~400k worst case.
    // Tick budget is consumed but not enforced strictly (the recompute
    // is atomic per chunk). Returning a representative number keeps
    // the tick from over-committing in a single call.
    400_000
}

/// Clear and re-enqueue a single chunk's light from scratch. This is
/// the PR3 minimal-decrease fallback: when any change might have reduced
/// a cell's contribution, we nuke the chunk's light arrays and let the
/// increase queue rebuild them from the chunk's source cells (sky
/// sources + emissive blocks).
fn recompute_chunk_light_from_scratch(
    engine: &mut LightEngine,
    chunks: &mut std::collections::HashMap<
        crate::voxel::coords::ChunkCoord,
        crate::voxel::world::ChunkSlot,
    >,
    registry: &crate::voxel::block::BlockRegistry,
    coord: crate::voxel::coords::ChunkCoord,
) {
    use crate::voxel::chunk::PalettedChunk;
    use crate::voxel::coords::{LocalPos, CHUNK_DIM_U};
    use crate::voxel::world::ChunkSlot;
    use glam::UVec3;

    let Some(ChunkSlot::Stored { data, meta }) = chunks.get_mut(&coord) else { return };

    // Rebuild the sky-source heightmap from the chunk's current blocks.
    let mut dense = data.decompress();
    meta.sky_sources = crate::lighting::ChunkSkyLightSources::build_from_dense(
        &dense, coord, registry,
    );

    // Clear all light arrays.
    dense.sky_light.iter_mut().for_each(|v| *v = 0);
    dense.block_rgb.iter_mut().for_each(|v| *v = 0);

    // Enqueue every sky-source cell at level 15.
    let chunk_bottom_y = coord.0.y * crate::voxel::coords::CHUNK_DIM;
    for lz in 0..CHUNK_DIM_U {
        for lx in 0..CHUNK_DIM_U {
            let lsy = meta.sky_sources.lowest_source_y(lx, lz);
            if lsy == crate::lighting::NO_SOURCE_FLOOR {
                // Whole column has no opaque block in this chunk; all
                // cells are sources at world-Y >= some unknown floor
                // somewhere below or above. For PR3 we treat the
                // entire column as a source within this chunk.
                for ly in 0..CHUNK_DIM_U {
                    let pos_world_y = chunk_bottom_y + ly as i32;
                    enqueue_sky_source(engine, &mut dense, coord, lx, ly as u32, lz, pos_world_y);
                }
            } else {
                // Enqueue cells from lsy up to chunk top as sources.
                // (Within this chunk only — cells above chunk top live
                // in the +Y chunk, which has its own heightmap.)
                let lsy_local = (lsy - chunk_bottom_y).max(0);
                if lsy_local < CHUNK_DIM_U as i32 {
                    for ly in (lsy_local as u32)..CHUNK_DIM_U {
                        let pos_world_y = chunk_bottom_y + ly as i32;
                        enqueue_sky_source(engine, &mut dense, coord, lx, ly, lz, pos_world_y);
                    }
                }
            }
        }
    }

    // Enqueue every emissive cell at its emission level.
    for lz in 0..CHUNK_DIM_U {
        for ly in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                let lp = LocalPos(UVec3::new(lx, ly, lz));
                let idx = lp.to_index();
                let info = registry.info(dense.blocks[idx]);
                if info.emission[0] > 0 || info.emission[1] > 0 || info.emission[2] > 0 {
                    let pos = crate::voxel::coords::BlockPos(
                        glam::IVec3::new(
                            coord.0.x * crate::voxel::coords::CHUNK_DIM + lx as i32,
                            coord.0.y * crate::voxel::coords::CHUNK_DIM + ly as i32,
                            coord.0.z * crate::voxel::coords::CHUNK_DIM + lz as i32,
                        ),
                    );
                    // Write the emission directly into block_rgb so the
                    // cell holds at least its own contribution.
                    let cur = crate::voxel::chunk::unpack_rgb(dense.block_rgb[idx]);
                    dense.block_rgb[idx] = crate::voxel::chunk::pack_rgb(
                        cur.0.max(info.emission[0]),
                        cur.1.max(info.emission[1]),
                        cur.2.max(info.emission[2]),
                    );
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

    *data = std::sync::Arc::new(PalettedChunk::compress(&dense));
    meta.light_gpu_dirty = true;
}

/// Write level=15 into the chunk's sky_light at `(lx, ly, lz)` and
/// enqueue an increase op so spread happens through the queue.
fn enqueue_sky_source(
    engine: &mut LightEngine,
    dense: &mut crate::voxel::chunk::DenseChunk,
    coord: crate::voxel::coords::ChunkCoord,
    lx: u32,
    ly: u32,
    lz: u32,
    pos_world_y: i32,
) {
    use crate::voxel::coords::{BlockPos, LocalPos, CHUNK_DIM};
    let pos = BlockPos(glam::IVec3::new(
        coord.0.x * CHUNK_DIM + lx as i32,
        pos_world_y,
        coord.0.z * CHUNK_DIM + lz as i32,
    ));
    let idx = LocalPos(glam::UVec3::new(lx, ly, lz)).to_index();
    dense.sky_light[idx] = 15;
    engine.sky.increase.push(crate::lighting::queue::QueueEntry {
        pos,
        from_level: 15,
        propagation_mask: 0,
    });
}

/// Phase C — drain a single channel's increase queue, propagating
/// values outward by 1 per air step (3 in water), per Minecraft's
/// classic light-flood rules. Stops when the budget is exhausted or
/// the queue is empty. Returns the remaining budget.
fn drain_increase_channel(
    ch_engine: &mut ChannelEngine,
    chunks: &mut std::collections::HashMap<
        crate::voxel::coords::ChunkCoord,
        crate::voxel::world::ChunkSlot,
    >,
    registry: &crate::voxel::block::BlockRegistry,
    channel: Channel,
    mut budget: usize,
) -> usize {
    use crate::voxel::block::Block;
    use crate::voxel::coords::{BlockPos, LocalPos};
    use crate::voxel::world::ChunkSlot;

    // 6-face deltas in the same order as `crate::mesher::Face::all()`
    // discriminants — bit i of propagation_mask corresponds to face i.
    let face_deltas: [(i32, i32, i32); 6] = [
        ( 1,  0,  0),  // 0: PosX
        (-1,  0,  0),  // 1: NegX
        ( 0,  1,  0),  // 2: PosY
        ( 0, -1,  0),  // 3: NegY
        ( 0,  0,  1),  // 4: PosZ
        ( 0,  0, -1),  // 5: NegZ
    ];
    // Each face's opposite (back-face mask bit to set when pushing onward).
    let opposite_face: [u8; 6] = [1, 0, 3, 2, 5, 4];

    while budget > 0 {
        let Some(entry) = ch_engine.increase.pop_highest() else { break };
        budget -= 1;
        if entry.from_level <= 1 { continue; }

        for face_i in 0..6 {
            if (entry.propagation_mask >> face_i) & 1 == 1 { continue; }
            let (dx, dy, dz) = face_deltas[face_i];
            let npos = BlockPos(glam::IVec3::new(
                entry.pos.0.x + dx,
                entry.pos.0.y + dy,
                entry.pos.0.z + dz,
            ));
            let nchunk_coord = npos.to_chunk();
            let nlocal = npos.to_local();
            let nidx = nlocal.to_index();

            // Look up the neighbour's chunk. If absent or pending, skip —
            // the engine will pick up that chunk's edge when it loads
            // (on_chunk_loaded enqueues neighbour boundary cells).
            let Some(ChunkSlot::Stored { data, meta }) = chunks.get_mut(&nchunk_coord) else {
                continue;
            };

            let mut dense = data.decompress();
            let nblock = dense.blocks[nidx];
            let ninfo = registry.info(nblock);
            if ninfo.opaque { continue; }

            // Per-step cost: 1 in air, 3 in water (matches today's BFS).
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
                Channel::Sky => {
                    dense.sky_light[nidx] = prop_level;
                }
                Channel::BlockRgb(c) => {
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[nidx]);
                    let mut chans = [r, g, b];
                    chans[c] = prop_level;
                    dense.block_rgb[nidx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                }
            }
            *data = std::sync::Arc::new(crate::voxel::chunk::PalettedChunk::compress(&dense));
            meta.light_gpu_dirty = true;

            // Enqueue onward propagation, blocking back-face.
            ch_engine.increase.push(crate::lighting::queue::QueueEntry {
                pos: npos,
                from_level: prop_level,
                propagation_mask: 1u8 << opposite_face[face_i],
            });
        }
    }

    budget
}
```

- [ ] **Step 2: Add unit tests for `tick_with` on a synthetic 1-chunk world.**

Append to the `tests` module:

```rust
    #[test]
    fn tick_with_propagates_torch_light_within_a_chunk() {
        use crate::voxel::block::{Block, BlockRegistry};
        use crate::voxel::chunk::{pack_rgb, unpack_rgb, DenseChunk, PalettedChunk};
        use crate::voxel::coords::{BlockPos, ChunkCoord, LocalPos};
        use crate::voxel::world::ChunkSlot;
        use glam::{IVec3, UVec3};
        use std::collections::HashMap;

        let registry = BlockRegistry::new();
        // Build a chunk full of air with one torch at (16, 16, 16) local.
        let mut dense = DenseChunk::empty();
        dense.set(LocalPos(UVec3::new(16, 16, 16)), Block::Torch);
        let chunk = PalettedChunk::compress(&dense);
        let coord = ChunkCoord(IVec3::ZERO);

        let mut chunks: HashMap<ChunkCoord, ChunkSlot> = HashMap::new();
        chunks.insert(coord, ChunkSlot::Stored {
            data: std::sync::Arc::new(chunk),
            meta: crate::voxel::chunk::ChunkMeta {
                sky_sources: crate::lighting::ChunkSkyLightSources::build_from_dense(
                    &dense, coord, &registry,
                ),
                ..Default::default()
            },
        });

        let mut engine = LightEngine::default();
        let torch_pos = BlockPos(IVec3::new(16, 16, 16));
        // Simulate "the chunk just loaded": enqueue the torch as a source.
        let torch_info = registry.info(Block::Torch);
        engine.block_rgb[0].increase.push(crate::lighting::queue::QueueEntry {
            pos: torch_pos, from_level: torch_info.emission[0], propagation_mask: 0,
        });
        engine.block_rgb[1].increase.push(crate::lighting::queue::QueueEntry {
            pos: torch_pos, from_level: torch_info.emission[1], propagation_mask: 0,
        });
        engine.block_rgb[2].increase.push(crate::lighting::queue::QueueEntry {
            pos: torch_pos, from_level: torch_info.emission[2], propagation_mask: 0,
        });

        // Tick to convergence.
        engine.tick_with(&mut chunks, &registry, 50_000);
        assert!(engine.sky.is_idle());
        for ch in &engine.block_rgb {
            assert!(ch.is_idle(), "RGB channels should drain");
        }

        // Verify the torch cell and an adjacent cell have RGB values.
        let ChunkSlot::Stored { data, .. } = chunks.get(&coord).unwrap() else { panic!() };
        let dense = data.decompress();
        let center_idx = LocalPos(UVec3::new(16, 16, 16)).to_index();
        let adj_idx = LocalPos(UVec3::new(17, 16, 16)).to_index();
        let (cr, _cg, _cb) = unpack_rgb(dense.block_rgb[center_idx]);
        let (ar, _ag, _ab) = unpack_rgb(dense.block_rgb[adj_idx]);
        // Torch emission is [13, 13, 13]. Center should hold ~13, adj at least 12.
        assert!(cr >= 13, "torch cell R should hold its own emission: got {}", cr);
        assert!(ar >= 11, "adjacent cell R should be lit: got {}", ar);
        let _ = pack_rgb(0, 0, 0);  // silence unused import on warn build
    }

    #[test]
    fn tick_with_idle_engine_is_a_noop() {
        use crate::voxel::block::BlockRegistry;
        use crate::voxel::coords::ChunkCoord;
        use crate::voxel::world::ChunkSlot;
        use std::collections::HashMap;

        let registry = BlockRegistry::new();
        let mut chunks: HashMap<ChunkCoord, ChunkSlot> = HashMap::new();
        let mut engine = LightEngine::default();
        engine.tick_with(&mut chunks, &registry, 10_000);
        assert!(engine.is_idle());
    }
```

- [ ] **Step 3: Run tests.**

Run: `cargo test --lib lighting::engine::tests`
Expected: 9 passed (7 from Tasks 1-2 + 2 new). If the torch test fails on adjacency, raise the budget — the queue may not have drained.

- [ ] **Step 4: Commit.**

```bash
git add src/lighting/engine.rs
git commit -m "feat(lighting): tick_with — increase phase + minimal decrease (graph-engine PR3, part 3/9)

LightEngine::tick_with(chunks, registry, budget) drains
pending_block_changes (Phase A), then per-channel increase queues
(Phase C). Minimal-decrease fallback recomputes whole chunks for any
change that could lower a cell's contribution; PR4 replaces this with
proper Minecraft-style decrease propagation.

Internal helpers: drain_pending_block_changes, drain_increase_channel,
recompute_chunk_light_from_scratch, enqueue_sky_source. Private to the
module; not exposed in mod.rs. Unit tests verify a torch at chunk
center lights its adjacent cell correctly."
```

---

## Task 4: Verify the minimal decrease path

This task adds a synthetic-world test that exercises the recompute fallback (a block change that reduces emission or adds opacity should result in correct values via the recompute path). No new production code; tests only.

**Files:**
- Modify: `src/lighting/engine.rs` (test module only)

- [ ] **Step 1: Add the test.**

Append to the `tests` module:

```rust
    #[test]
    fn opacity_increase_triggers_chunk_recompute_via_pending_change() {
        use crate::voxel::block::{Block, BlockRegistry};
        use crate::voxel::chunk::{DenseChunk, PalettedChunk};
        use crate::voxel::coords::{BlockPos, ChunkCoord, LocalPos};
        use crate::voxel::world::ChunkSlot;
        use glam::{IVec3, UVec3};
        use std::collections::HashMap;

        let registry = BlockRegistry::new();
        // Pre-lit chunk: pretend sky has fully propagated (set sky_light = 15
        // at and above y=10, 0 below). One air block sits at (16, 10, 16).
        let mut dense = DenseChunk::empty();
        for ly in 10..32 {
            for lz in 0..32 {
                for lx in 0..32 {
                    dense.sky_light[LocalPos(UVec3::new(lx, ly, lz)).to_index()] = 15;
                }
            }
        }
        let coord = ChunkCoord(IVec3::ZERO);
        let chunk = PalettedChunk::compress(&dense);
        let mut chunks: HashMap<ChunkCoord, ChunkSlot> = HashMap::new();
        chunks.insert(coord, ChunkSlot::Stored {
            data: std::sync::Arc::new(chunk),
            meta: crate::voxel::chunk::ChunkMeta {
                sky_sources: crate::lighting::ChunkSkyLightSources::build_from_dense(
                    &dense, coord, &registry,
                ),
                ..Default::default()
            },
        });

        let mut engine = LightEngine::default();
        // Simulate placing stone at (16, 20, 16): opacity changes 0 → 1.
        // This should trigger a chunk recompute that rebuilds the heightmap
        // and re-floods light. After tick, cells below the stone should
        // be dark (sky can't reach them).
        let pos = BlockPos(IVec3::new(16, 20, 16));
        // Update the chunk's block first (mirroring what set_block does).
        if let Some(ChunkSlot::Stored { data, .. }) = chunks.get_mut(&coord) {
            let mut dense = data.decompress();
            dense.set(LocalPos(UVec3::new(16, 20, 16)), Block::Stone);
            *data = std::sync::Arc::new(PalettedChunk::compress(&dense));
        }
        engine.enqueue_block_change(pos, Block::Air, Block::Stone);
        engine.tick_with(&mut chunks, &registry, 50_000);

        // Verify cell directly below stone (16, 19, 16) is no longer at 15
        // (it should be dark because stone blocks the column).
        let ChunkSlot::Stored { data, meta } = chunks.get(&coord).unwrap() else { panic!() };
        let dense = data.decompress();
        let below_idx = LocalPos(UVec3::new(16, 19, 16)).to_index();
        // The lowest_source_y for column (16, 16) should now be 21 (cell
        // immediately above stone y=20).
        assert_eq!(meta.sky_sources.lowest_source_y(16, 16), 21);
        // Cell below stone should be lit only by lateral propagation
        // from neighbouring columns — strictly less than 15.
        assert!(
            dense.sky_light[below_idx] < 15,
            "cell below new stone should be shaded; got {}",
            dense.sky_light[below_idx],
        );
    }
```

- [ ] **Step 2: Run tests.**

Run: `cargo test --lib lighting::engine::tests`
Expected: 10 passed (9 from Tasks 1-3 + 1 new).

- [ ] **Step 3: Commit.**

```bash
git add src/lighting/engine.rs
git commit -m "test(lighting): opacity-increase triggers chunk recompute (graph-engine PR3, part 4/9)

Adds synthetic-world test verifying the minimal-decrease fallback: a
new opaque block in the middle of a lit column causes a chunk
recompute that rebuilds the heightmap and re-floods sky light, leaving
the cells below the new stone shaded."
```

---

## Task 5: `World` methods — `on_chunk_loaded`, `on_block_changed`, `light_engine_tick` (dormant)

**Files:**
- Modify: `src/voxel/world.rs`

These methods exist and are unit-tested, but no production caller invokes them yet. Phase B (Tasks 6-9) wires the callers.

- [ ] **Step 1: Add the methods to `impl World`.**

Find `impl World { ... }` in `src/voxel/world.rs`. Add the following three methods right after `World::insert`:

```rust
    /// Engine entry point for "this block just changed". Called from
    /// `set_block`. Records the (old, new) tuple in the engine's
    /// pending-changes side-table and queues the position on every
    /// channel; the next `light_engine_tick` consumes it.
    pub fn on_block_changed(
        &mut self,
        pos: crate::voxel::coords::BlockPos,
        old_block: crate::voxel::block::Block,
        new_block: crate::voxel::block::Block,
    ) {
        self.light_engine.enqueue_block_change(pos, old_block, new_block);
    }

    /// Engine entry point for "a chunk just installed". Called from
    /// the `JobResult::Generated` and `JobResult::LoadedFromDisk`
    /// handlers in `mesh_upload`. Enqueues every sky-source cell and
    /// every emissive block in the chunk as increase ops, plus each
    /// neighbour's boundary cells (so the engine can spread our
    /// freshly-loaded chunk's light across the seam without a
    /// special seed pass).
    pub fn on_chunk_loaded(&mut self, coord: crate::voxel::coords::ChunkCoord) {
        use crate::voxel::block::Block;
        use crate::voxel::coords::{BlockPos, LocalPos, CHUNK_DIM, CHUNK_DIM_U};
        use crate::voxel::world::ChunkSlot;
        use glam::{IVec3, UVec3};

        // Snapshot the chunk's blocks + heightmap for enqueuing.
        let Some(ChunkSlot::Stored { data, meta }) = self.chunks.get(&coord) else { return };
        let dense = data.decompress();
        let sky_sources = meta.sky_sources.clone();
        let chunk_bottom_y = coord.0.y * CHUNK_DIM;

        // Enqueue every sky-source cell at level 15.
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                let lsy = sky_sources.lowest_source_y(lx, lz);
                let start_ly: i32 = if lsy == crate::lighting::NO_SOURCE_FLOOR {
                    0
                } else {
                    (lsy - chunk_bottom_y).max(0)
                };
                for ly in (start_ly as u32)..CHUNK_DIM_U {
                    let pos = BlockPos(IVec3::new(
                        coord.0.x * CHUNK_DIM + lx as i32,
                        chunk_bottom_y + ly as i32,
                        coord.0.z * CHUNK_DIM + lz as i32,
                    ));
                    self.light_engine.sky.increase.push(crate::lighting::queue::QueueEntry {
                        pos, from_level: 15, propagation_mask: 0,
                    });
                }
            }
        }

        // Enqueue every emissive cell at its emission level.
        for lz in 0..CHUNK_DIM_U {
            for ly in 0..CHUNK_DIM_U {
                for lx in 0..CHUNK_DIM_U {
                    let idx = LocalPos(UVec3::new(lx, ly, lz)).to_index();
                    let info = self.registry.info(dense.blocks[idx]);
                    if info.emission[0] > 0 || info.emission[1] > 0 || info.emission[2] > 0 {
                        let pos = BlockPos(IVec3::new(
                            coord.0.x * CHUNK_DIM + lx as i32,
                            chunk_bottom_y + ly as i32,
                            coord.0.z * CHUNK_DIM + lz as i32,
                        ));
                        for ch_i in 0..3 {
                            if info.emission[ch_i] > 0 {
                                self.light_engine.block_rgb[ch_i].increase.push(
                                    crate::lighting::queue::QueueEntry {
                                        pos,
                                        from_level: info.emission[ch_i],
                                        propagation_mask: 0,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }

        // Re-enqueue face-adjacent neighbour boundary cells so the engine
        // spreads their existing light across the new seam. For each face,
        // walk the 32×32 boundary slice on the neighbour's side and push
        // its current sky_light + block_rgb levels as increase ops.
        let face_offsets: [IVec3; 6] = [
            IVec3::new( 1, 0, 0), IVec3::new(-1, 0, 0),
            IVec3::new( 0, 1, 0), IVec3::new( 0,-1, 0),
            IVec3::new( 0, 0, 1), IVec3::new( 0, 0,-1),
        ];
        for face_off in face_offsets {
            let nc = crate::voxel::coords::ChunkCoord(coord.0 + face_off);
            let Some(ChunkSlot::Stored { data: ndata, .. }) = self.chunks.get(&nc) else { continue };
            let ndense = ndata.decompress();
            // Walk the slice of the neighbour adjacent to the seam.
            // For face = +X, neighbour's slice is at lx=0; for -X, lx=31; etc.
            // We push EVERY cell in the neighbour's slice — the queue's
            // bucket sort ensures redundant pushes coalesce in the right order.
            for u in 0..CHUNK_DIM_U {
                for v in 0..CHUNK_DIM_U {
                    let (nlx, nly, nlz) = match face_off {
                        IVec3 { x:  1, y: 0, z: 0 } => (0,                 v, u),
                        IVec3 { x: -1, y: 0, z: 0 } => (CHUNK_DIM_U - 1,   v, u),
                        IVec3 { x: 0, y:  1, z: 0 } => (u,                 0,                 v),
                        IVec3 { x: 0, y: -1, z: 0 } => (u,                 CHUNK_DIM_U - 1,   v),
                        IVec3 { x: 0, y: 0, z:  1 } => (u, v, 0),
                        IVec3 { x: 0, y: 0, z: -1 } => (u, v, CHUNK_DIM_U - 1),
                        _ => unreachable!(),
                    };
                    let nidx = LocalPos(UVec3::new(nlx, nly, nlz)).to_index();
                    let sky = ndense.sky_light[nidx];
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(ndense.block_rgb[nidx]);
                    let npos = BlockPos(IVec3::new(
                        nc.0.x * CHUNK_DIM + nlx as i32,
                        nc.0.y * CHUNK_DIM + nly as i32,
                        nc.0.z * CHUNK_DIM + nlz as i32,
                    ));
                    if sky > 0 {
                        self.light_engine.sky.increase.push(crate::lighting::queue::QueueEntry {
                            pos: npos, from_level: sky, propagation_mask: 0,
                        });
                    }
                    for (ch_i, level) in [r, g, b].iter().enumerate() {
                        if *level > 0 {
                            self.light_engine.block_rgb[ch_i].increase.push(
                                crate::lighting::queue::QueueEntry {
                                    pos: npos, from_level: *level, propagation_mask: 0,
                                },
                            );
                        }
                    }
                }
            }
            let _ = Block::Air;  // silence unused warning
        }
    }

    /// Engine entry point for "drain up to `budget` ops this frame".
    /// Called from `App::update` once per frame. Destructures `&mut self`
    /// to split the field-aliasing problem (`light_engine` is a field
    /// of `World`, so a method on `LightEngine` can't take `&mut World`).
    pub fn light_engine_tick(&mut self, budget: usize) {
        let Self { chunks, registry, light_engine, .. } = self;
        light_engine.tick_with(chunks, registry, budget);
    }
```

- [ ] **Step 2: Add unit tests at the bottom of the `tests` module.**

Append to the existing tests module:

```rust
    #[test]
    fn on_block_changed_queues_work_in_engine() {
        use crate::voxel::block::Block;
        use crate::voxel::coords::BlockPos;
        let mut w = World::new(42);
        assert!(w.light_engine.is_idle());
        w.on_block_changed(BlockPos(IVec3::new(0, 0, 0)), Block::Air, Block::Torch);
        assert!(!w.light_engine.is_idle());
        assert!(w.light_engine.pending_block_changes.contains_key(&BlockPos(IVec3::new(0, 0, 0))));
    }

    #[test]
    fn on_chunk_loaded_enqueues_sky_sources_for_air_chunk() {
        use crate::voxel::chunk::DenseChunk;
        let mut w = World::new(42);
        let chunk = PalettedChunk::compress(&DenseChunk::empty());
        let coord = ChunkCoord(IVec3::ZERO);
        w.insert(coord, chunk);
        w.on_chunk_loaded(coord);
        // All-air chunk: every cell is a sky source within the chunk;
        // 32^3 = 32768 cells should land in the sky channel's queue.
        assert!(!w.light_engine.sky.increase.is_empty());
    }

    #[test]
    fn light_engine_tick_drains_queued_work() {
        use crate::voxel::block::Block;
        use crate::voxel::chunk::DenseChunk;
        use crate::voxel::coords::{BlockPos, LocalPos};
        use glam::UVec3;
        let mut w = World::new(42);
        // Build a chunk with a torch at center.
        let mut dense = DenseChunk::empty();
        dense.set(LocalPos(UVec3::new(16, 16, 16)), Block::Torch);
        let chunk = PalettedChunk::compress(&dense);
        let coord = ChunkCoord(IVec3::ZERO);
        w.insert(coord, chunk);
        w.on_chunk_loaded(coord);
        assert!(!w.light_engine.is_idle());
        w.light_engine_tick(50_000);
        assert!(w.light_engine.is_idle(), "tick should drain all queued work given big enough budget");
        // The chunk should have light_gpu_dirty set since the engine wrote
        // to it.
        if let Some(ChunkSlot::Stored { meta, .. }) = w.chunks.get(&coord) {
            assert!(meta.light_gpu_dirty, "engine writes should set light_gpu_dirty");
        }
    }
```

Note: the third test references `meta.light_gpu_dirty`. This field is added in Task 6 below; this test will fail to compile until Task 6 lands. **Defer running this specific test** until Task 6 is also implemented — or temporarily comment its `light_gpu_dirty` assertion and uncomment it after Task 6.

- [ ] **Step 3: Run the first two tests now (third deferred until Task 6).**

Run: `cargo test --lib voxel::world::tests::on_block_changed_queues_work_in_engine`
Run: `cargo test --lib voxel::world::tests::on_chunk_loaded_enqueues_sky_sources_for_air_chunk`
Expected: both pass.

`cargo test --lib voxel::world::tests::light_engine_tick_drains_queued_work` will fail to compile because `light_gpu_dirty` doesn't exist yet — that's expected; Task 6 adds it.

If the compile fails preventing other tests from running, comment out the `meta.light_gpu_dirty` assertion in `light_engine_tick_drains_queued_work` for now and add a `// TODO(Task 6): uncomment after light_gpu_dirty exists` marker. Uncomment it in Task 6.

- [ ] **Step 4: Commit.**

```bash
git add src/voxel/world.rs
git commit -m "feat(lighting): World methods for engine entry points (graph-engine PR3, part 5/9)

Three new methods on World, all dormant (no production caller yet):
- on_block_changed(pos, old, new) — queues a block change
- on_chunk_loaded(coord) — enqueues sky sources, emissives, and
  neighbour boundary cells when a chunk installs
- light_engine_tick(budget) — destructures &mut self to side-step the
  field-aliasing problem and drives one tick of the engine

Phase B (Tasks 6-9) wires the production callers. Engine is fully
implemented and unit-tested but not yet authoritative."
```

---

**🎯 MERGE POINT — Tasks 1-5 complete.** The engine is fully implemented and dormant. Visual behaviour and the existing BFS are unchanged. You can stop here and merge if you want to validate before flipping the production switch.

---

# Phase B — Cutover (Tasks 6-9, visible behaviour changes at Task 9)

## Task 6: Add `ChunkMeta.light_gpu_dirty` flag

**Files:**
- Modify: `src/voxel/chunk.rs`
- Modify: `src/voxel/world.rs` (uncomment the `light_gpu_dirty` assertion in the test)

- [ ] **Step 1: Add the field to `ChunkMeta`.**

Find `pub struct ChunkMeta { ... }` (around line 393 in chunk.rs). Add a new field after `sky_sources`:

```rust
    /// True when the engine has written to this chunk's `sky_light` or
    /// `block_rgb` since the last GPU upload of the light volume. The
    /// `upload_dirty_light_volumes` pass in `mesh_upload` scans this
    /// flag each frame and re-uploads + clears for any chunk that's
    /// flagged.
    pub light_gpu_dirty: bool,
```

(`bool` defaults to `false`, so no `Default` impl change needed.)

- [ ] **Step 2: Uncomment the assertion in `light_engine_tick_drains_queued_work` (if you commented it in Task 5).**

In `src/voxel/world.rs`, find the test `light_engine_tick_drains_queued_work` and uncomment any lines referencing `meta.light_gpu_dirty`.

- [ ] **Step 3: Run tests.**

Run: `cargo test --lib`
Expected: all tests pass, including the previously-deferred `light_engine_tick_drains_queued_work`.

- [ ] **Step 4: Commit.**

```bash
git add src/voxel/chunk.rs src/voxel/world.rs
git commit -m "feat(lighting): ChunkMeta.light_gpu_dirty flag (graph-engine PR3, part 6/9)

New per-chunk boolean. Engine sets it when writing to the chunk's
light arrays; mesh_upload's upload_dirty_light_volumes (Task 8) reads
+ clears it to re-upload the 3D texture.

Distinct from dirty.light (legacy BFS recompute flag, retained for
the legacy-lighting feature)."
```

---

## Task 7: Wire `on_block_changed` into `World::set_block`

**Files:**
- Modify: `src/voxel/world.rs`

- [ ] **Step 1: Add the call inside `set_block`.**

Find `pub fn set_block(&mut self, pos: BlockPos, new_block: Block) -> Vec<ChunkCoord>` in `src/voxel/world.rs`. Find the place where it reads `old_block` and writes `new_block` (currently:

```rust
        let mut dense = data.decompress();
        dense.set(local, new_block);
        *data = std::sync::Arc::new(PalettedChunk::compress(&dense));
```

Change to capture `old_block` first and notify the engine:

```rust
        let mut dense = data.decompress();
        let old_block = dense.get(local);
        if old_block == new_block {
            return vec![];
        }
        dense.set(local, new_block);
        *data = std::sync::Arc::new(PalettedChunk::compress(&dense));

        // Notify the graph engine. Engine tick (next frame) processes
        // the change. Today's BFS path (dirty.light below) continues
        // to run too — it's still authoritative until Task 9 flips
        // the GPU upload source. After the cutover, the engine is the
        // sole writer.
        self.light_engine.enqueue_block_change(pos, old_block, new_block);
```

- [ ] **Step 2: Verify.**

Run: `cargo test --lib`
Expected: all tests pass. `set_block` integration tests will exercise the new code path (the engine's pending_block_changes accumulates per edit, but tick is only run via App::update / explicit test calls, so for tests that don't tick, the engine just stays non-idle harmlessly).

- [ ] **Step 3: Commit.**

```bash
git add src/voxel/world.rs
git commit -m "feat(lighting): set_block notifies the graph engine (graph-engine PR3, part 7/9)

set_block now calls light_engine.enqueue_block_change after the
PalettedChunk update. Returns early if old==new (avoids enqueuing
no-op edits). BFS dirty.light flag is still set in parallel — engine
becomes authoritative only after Task 9's GPU upload cutover."
```

---

## Task 8: Wire `on_chunk_loaded` into `mesh_upload::drain_jobs`; add `upload_dirty_light_volumes`; add the frame-tick call

**Files:**
- Modify: `src/ecs/systems/mesh_upload.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Add `on_chunk_loaded` calls in `drain_jobs`.**

Find the `JobResult::Generated` arm (around line 59 of `mesh_upload.rs`). After `world.insert(coord, data);` and before the existing `world.mark_below_dirty(coord);`, add:

```rust
                // Hand the new chunk to the graph engine. on_chunk_loaded
                // enqueues sky sources, emissives, and neighbour boundary
                // cells; the engine's next tick spreads them.
                world.on_chunk_loaded(coord);
```

Find the `JobResult::LoadedFromDisk { coord, data: Some(data) }` arm (around line 222). After `world.insert(coord, data);`, add the same call:

```rust
                    world.on_chunk_loaded(coord);
```

- [ ] **Step 2: Add `upload_dirty_light_volumes`.**

At the bottom of `mesh_upload.rs` (after `relight_pump`), add a new function:

```rust
/// Scan loaded chunks for `light_gpu_dirty` and re-upload their 3D
/// light textures. Bounded to `UPLOAD_BUDGET` chunks per frame so a
/// big convergence wave doesn't saturate PCIe bandwidth.
pub fn upload_dirty_light_volumes(
    world: &mut crate::voxel::world::World,
    renderer: &mut crate::render::Renderer,
) {
    use crate::voxel::world::ChunkSlot;
    const UPLOAD_BUDGET: usize = 32;
    let mut uploaded = 0;
    // Collect the dirty coords first; rebuilding the volume needs to
    // gather_neighbors which borrows the world immutably.
    let dirty: Vec<_> = world.chunks.iter()
        .filter_map(|(c, slot)| match slot {
            ChunkSlot::Stored { meta, .. } if meta.light_gpu_dirty => Some(*c),
            _ => None,
        })
        .take(UPLOAD_BUDGET)
        .collect();
    for coord in dirty {
        let neighbors = gather_neighbors(world, coord);
        let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get_mut(&coord) else { continue };
        let dense = data.decompress();
        let blob = crate::voxel::chunk::build_light_volume_blob(&dense, &neighbors);
        renderer.upload_chunk_light_volume(coord, &blob);
        meta.light_gpu_dirty = false;
        uploaded += 1;
    }
    let _ = uploaded;
}
```

- [ ] **Step 3: Call `upload_dirty_light_volumes` + `light_engine_tick` from `App::update`.**

Find the frame loop in `src/app.rs` (around line 396, near the existing `drain_jobs` / `drain_persistence` / `relight_pump` calls). After `drain_persistence` and before (or after) `relight_pump`, add:

```rust
            time(prof, "light_engine_tick", || {
                self.world.light_engine_tick(50_000);
            });
            time(prof, "upload_dirty_light_volumes", || {
                crate::ecs::systems::mesh_upload::upload_dirty_light_volumes(
                    &mut self.world,
                    &mut self.renderer,
                );
            });
```

- [ ] **Step 4: Build and test.**

Run: `cargo build --lib`
Expected: clean build.

Run: `cargo test --lib`
Expected: all tests pass. (The existing BFS still runs in parallel — `dirty.light` is still set; `relight_pump` still pumps it. The engine writes are uploaded via `light_gpu_dirty` but the BFS writes are uploaded via the existing path. Both compete; in practice the engine's writes happen after the BFS's for any given chunk, so the engine's values win on the GPU. This is fine for the dual-running transition state.)

- [ ] **Step 5: Manual smoke test.**

Run: `cargo run --release` and walk around for 30 seconds. The lighting should look the same as before (or slightly different — both systems are writing). No crashes, no major visual regressions.

- [ ] **Step 6: Commit.**

```bash
git add src/ecs/systems/mesh_upload.rs src/app.rs
git commit -m "feat(lighting): wire engine into production frame loop (graph-engine PR3, part 8/9)

drain_jobs's Generated and LoadedFromDisk arms now call
world.on_chunk_loaded(coord) to feed the engine. App::update calls
world.light_engine_tick(50000) each frame, then
upload_dirty_light_volumes scans light_gpu_dirty and re-uploads the
3D light textures touched by the engine.

The old BFS still runs in parallel; relight_pump and mark_below_dirty
remain active. The engine writes are visible on the GPU but the BFS
writes win for any chunk it processes after the engine. Task 9 is the
clean cutover that deletes the BFS path."
```

---

## Task 9: Cutover — `--feature legacy-lighting`, gate the BFS

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/lighting/mod.rs`
- Modify: `src/voxel/chunk.rs`
- Modify: `src/voxel/world.rs`
- Modify: `src/jobs/mod.rs`
- Modify: `src/ecs/systems/mesh_upload.rs`
- Modify: `src/app.rs`

This task makes the engine authoritative. The old BFS, cascade plumbing, and `Relit` job path move behind a `--feature legacy-lighting` flag that defaults off.

- [ ] **Step 1: Add the feature to `Cargo.toml`.**

Find the `[features]` section (or create one) and add:

```toml
[features]
default = []

# Opt-in fallback to the legacy per-chunk BFS lighting. Off by default;
# the graph-engine LightEngine is authoritative in the default build.
# Build with `cargo build --features legacy-lighting` to fall back if
# the engine has a regression.
legacy-lighting = []
```

- [ ] **Step 2: Gate the BFS in `src/lighting/mod.rs`.**

Wrap the entire body of `recompute_chunk`, `sky_light`, `block_rgb`, `seed_from_neighbors`, `snapshot_face_boundaries`, `BfsChannel`, `bfs_spread_sky`, `bfs_spread_rgb`, `mirror_boundary`, and the `D` constant in:

```rust
#[cfg(feature = "legacy-lighting")]
// ... existing function/const/enum
```

Easiest: put `#[cfg(feature = "legacy-lighting")]` at the top of each item. Or wrap a block. The `#[cfg(test)] mod tests` block at the bottom that exercises these functions also needs the gate (or move tests into a `#[cfg(all(test, feature = "legacy-lighting"))]` block).

- [ ] **Step 3: Gate `ChunkDirty::light` in `src/voxel/chunk.rs`.**

Find `pub struct ChunkDirty { ... }`. Change `pub light: bool` to:

```rust
    #[cfg(feature = "legacy-lighting")]
    pub light: bool,
```

The `Default` derive still works because `bool` defaults to `false` and the field is only present under the feature.

- [ ] **Step 4: Gate `World::mark_below_dirty` in `src/voxel/world.rs`.**

Add `#[cfg(feature = "legacy-lighting")]` above `pub fn mark_below_dirty(...)` and above its tests `mark_below_dirty_flips_below_chunk_flag` and `mark_below_dirty_no_op_when_below_absent` if those exist in the worktree.

Also in `set_block`, find any line that sets `meta.dirty.light = true;` and wrap it:

```rust
        #[cfg(feature = "legacy-lighting")]
        { meta.dirty.light = true; }
```

- [ ] **Step 5: Gate `JobResult::Relit` and `spawn_relight` in `src/jobs/mod.rs`.**

Add `#[cfg(feature = "legacy-lighting")]` above `JobResult::Relit { ... }` (the variant) and above `pub fn spawn_relight(...)`.

- [ ] **Step 6: Gate the `Relit` handler and `relight_pump` in `src/ecs/systems/mesh_upload.rs`.**

Add `#[cfg(feature = "legacy-lighting")]` above the `JobResult::Relit { ... } => { ... }` arm in `drain_jobs`. Same for `pub fn relight_pump(...)`.

Also gate `world.mark_below_dirty(coord)` in the `Generated` arm:

```rust
                #[cfg(feature = "legacy-lighting")]
                world.mark_below_dirty(coord);
```

- [ ] **Step 7: Gate the `relight_pump` call in `src/app.rs`.**

Wrap the existing `relight_pump` call in:

```rust
            #[cfg(feature = "legacy-lighting")]
            { self.perf.light_queue = time(prof, "relight_pump", || {
                crate::ecs::systems::mesh_upload::relight_pump(
                    &mut self.world,
                    &self.jobs,
                    &self.registry,
                )
            }) as u32 as _; }
```

- [ ] **Step 8: Verify the default build is clean.**

Run: `cargo build --lib`
Expected: clean build, no errors.

Run: `cargo test --lib`
Expected: all engine tests pass; many pre-existing BFS tests are now excluded (compile-time gated). Test count drops by however many tests live in the gated `lighting::mod.rs::tests` and `voxel::world::tests` `mark_below_dirty_*` modules.

If pre-existing screenshot baseline tests now fail, those failures are expected — the engine produces different (correct) values. Accept the failures or refresh the baselines in a follow-up commit (see "After this PR" below).

- [ ] **Step 9: Verify the `--features legacy-lighting` build also compiles.**

Run: `cargo build --lib --features legacy-lighting`
Expected: clean build. (Doesn't have to be runnable correctly — this is a compile-only check that the gated code is still well-formed.)

- [ ] **Step 10: Manual smoke test of the default build.**

Run: `cargo run --release`

Verify:
- World renders.
- Lighting looks plausible (sky exposure correct on outdoor surfaces; torches glow).
- Walking into a cave produces darkness.
- The ocean artifact from the user's earlier screenshot should be **gone or significantly reduced** — sky values inside water should be correct, no chunk-seam discontinuities.

If lighting is wildly broken, run `cargo run --release --features legacy-lighting` to fall back to the BFS for comparison. Diagnose and fix.

- [ ] **Step 11: Commit.**

```bash
git add Cargo.toml src/lighting/mod.rs src/voxel/chunk.rs src/voxel/world.rs src/jobs/mod.rs src/ecs/systems/mesh_upload.rs src/app.rs
git commit -m "feat(lighting): cutover — engine is authoritative, BFS opt-in (graph-engine PR3, part 9/9)

Default build: LightEngine is the only light propagator.
- recompute_chunk, sky_light, block_rgb, seed_from_neighbors,
  snapshot_face_boundaries, BfsChannel, bfs_spread_* gated behind
  --feature legacy-lighting.
- ChunkDirty::light, World::mark_below_dirty, JobResult::Relit,
  spawn_relight, relight_pump, the Relit drain arm, and the
  set_block dirty.light setter all gated.

--features legacy-lighting recompiles the old BFS as a runtime
fallback. Keep for one release as a safety net.

Visible behaviour change: chunk-seam artifacts disappear; out-of-order
chunk loads produce correct sky values; cells under deep water no
longer freeze at 15. Screenshot baselines may need a refresh in a
follow-up commit."
```

---

## Verification

- [ ] **Run the full lib test suite.**

Run: `cargo test --lib`
Expected: pass. Test count will be lower than pre-PR3 because BFS-specific tests are now gated.

- [ ] **Verify both feature configurations compile.**

Run: `cargo build --lib && cargo build --lib --features legacy-lighting`
Expected: both pass.

- [ ] **Visual smoke test.**

Run: `cargo run --release` and validate the ocean-seam fix.

---

## What's next

After this PR merges:

- **Screenshot baselines** will need refreshing in a follow-up commit. Run the screenshot harness, compare new outputs to old baselines, eyeball each diff for "did the engine fix something or break something?", commit the new baselines.
- **PR4** (next in the sequence) implements proper Minecraft-style **decrease queue correctness**. Today's minimal-decrease (chunk recompute) is correct but wasteful per-edit; PR4 replaces it with the tear-down + repair-from-independent-sources algorithm from the spec.
- **PR5** (final in the sequence) adds the **streaming-mode budget bump** — during initial world load, raise the tick budget temporarily so the engine converges faster.
- Eventually, **the `legacy-lighting` feature** can be removed entirely once the engine has proven itself for a release or two.
