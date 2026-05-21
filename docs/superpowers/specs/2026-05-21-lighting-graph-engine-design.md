# Lighting Graph Engine — Design Spec

**Date:** 2026-05-21
**Status:** Approved, ready for implementation planning
**Supersedes:** The CPU-propagation portion of [`2026-05-20-lighting-design.md`](2026-05-20-lighting-design.md) (sections "Module shape" partly, "CPU light propagation" fully). The visual/GPU pipeline of that doc — HDR, bloom, per-pixel volume sampling, shadow map, ambient bounce, volumetrics, day/night colors — is unaffected. PRs 1-4 of that spec are merged; PRs 5-8 remain and land on top of this engine unchanged.

## Summary

Replace oxium's per-chunk batch BFS lighting with a single global graph engine that propagates light per-voxel, incrementally. The new engine handles each block change in `O(actually_affected_cells)` work rather than re-running a full chunk BFS, propagates correctly across chunk boundaries without seam-seed handoff, and converges on a known-good fixed point regardless of chunk load order.

The engine is modelled on Minecraft's `LightEngine` (separate increase/decrease queues per channel, heightmap-driven sky source tracking) with one efficiency adopted from Terasology (magnitude-bucket queues). All chunk-batch cascade plumbing — `dirty.light` flag, `mark_below_dirty`, `snapshot_face_boundaries`, `seed_from_neighbors`, `relight_pump`, `JobResult::Relit` — is deleted.

Primary motivation: today's per-chunk BFS produces wrong, inconsistent, and unstably-converging sky-light values, especially at chunk seams and across out-of-order chunk loads. The bandaids in the codebase (rate-limited cascade, in-flight dirty mark preservation, special-case `mark_below_dirty`) all treat symptoms of the same architectural problem — chunks are the wrong unit of work for light propagation.

## Goals and non-goals

**Goals:**

- Sky and block light values are correct everywhere within bounded convergence time after every block change, regardless of chunk load order or edit batch size.
- Chunk seams produce no light discontinuity at rest.
- Per-edit cost scales with the number of cells whose value actually changes (typically tens), not the volume of the chunks touched (32³ × however many neighbours).
- Removing a light source produces correct dimming without re-flooding the chunk from scratch.
- All four channels (sky + R + G + B) propagate independently — a red light can flood past a green attenuator without the green channel constraining it.
- The visual pipeline (PR5 shadow map, PR6 ambient bounce, PR7 volumetrics, PR8 day/night colors) lands on top of this engine with no changes required.

**Non-goals:**

- Visual / GPU pipeline redesign — covered by the existing 2026-05-20 spec.
- Sub-frame interactivity for huge bulk edits (explosions, world-mods) — convergence over multiple ticks is acceptable; we cap per-tick work.
- True global illumination, light bouncing off coloured surfaces, screen-space techniques — out of scope, deferred indefinitely.
- Multithreaded propagation across multiple worker threads — single dedicated tick is sufficient at our scale.

## Why redesign rather than patch

The current per-chunk BFS has three structural problems that compound as more features land on top:

1. **Chunk-as-unit of work amplifies cost.** A single block edit at a chunk boundary marks `dirty.light` on up to 7 chunks (self + 6 face neighbours). Each chunk is then fully cleared and reflooded, costing ~250 µs per chunk. The `relight_pump` rate-limits to 16 chunks/frame to avoid blowing the worker pool, but this means a cascade through a 16-chunk-wide region takes ~1 second to visibly settle, during which intermediate frames render inconsistent values.

2. **Boundary handoff is lossy and asymmetric.** `seed_from_neighbors` reads `neighbour_cell - 1` once at the start of a chunk's BFS. If the neighbour's BFS later increases its boundary value (because *its* neighbour changed), this chunk has no idea — the cascade has to ping-pong via `snapshot_face_boundaries` diffing, which only triggers if the *exact* cell changed. Subtle convergence bugs result; the codebase has multiple comments documenting these (`World::mark_below_dirty`, the `JobResult::Relit` "preserve in-flight dirty marks" branch).

3. **Sky column drop assumes the +Y neighbour is loaded.** When chunks load out of order (the common case under parallel `spawn_load`), a chunk computes its sky values assuming full sun from above. When the real +Y chunk arrives later carrying e.g. deep water (which attenuates sky to 0), the stale values are frozen in place. `mark_below_dirty` is the bandaid; it works only when chunks arrive in the right pattern.

These are not bug-fixable in isolation. They are properties of "the unit of work is a chunk, not a voxel."

## Reference architectures consulted

| | **Minecraft 1.20+** | **Terasology** | **Oxium today** |
|---|---|---|---|
| Unit of work | Per-voxel, fully incremental | Per-voxel, batched in 3×3×3 chunk view | Per-chunk batch BFS |
| Add/remove model | Separate increase + decrease queues | Single reduce + single increase pass | Full clear + reflood per dirty chunk |
| Sky model | `ChunkSkyLightSources` heightmap drives source add/remove | `sunlightRegen` per-voxel max-depth channel | Column drop using +Y neighbour's bottom row at gen time |
| Cross-chunk | Implicit (one global graph) | `LocalChunkView` (27-chunk coord space) | Explicit "snapshot → seed → diff → cascade dirty flag" |
| Queue impl | FIFO long queue | 16 magnitude buckets (one per level) | `VecDeque` per chunk |

This design adopts:
- Minecraft's **per-voxel incremental graph with separate increase/decrease queues** (correctness, surgical updates).
- Minecraft's **`ChunkSkyLightSources` heightmap pattern** (correct sky behaviour under chunk load reorderings, roof changes).
- Terasology's **16-bucket magnitude-ordered queue** (cache-friendly propagation, no heap operations).

Terasology's `sunlightRegen` channel is *not* adopted in v1 — Minecraft's heightmap-driven source model already fixes the bug class we hit. Revisit if heightmap-only sky misbehaves in practice.

## Architecture

### Module shape

```
src/lighting/
├── mod.rs          # public API: LightEngine, world hooks (on_block_changed, on_chunk_loaded, tick)
├── engine.rs       # LightEngine struct, per-channel ChannelEngine, tick orchestration
├── propagate.rs    # increase + decrease propagation per channel
├── sky_sources.rs  # ChunkSkyLightSources heightmap per chunk
├── queue.rs        # BucketQueue (16 magnitude buckets, u16 nonempty mask)
└── tests/          # unit + integration tests (see Testing section)
```

### Threading model

**Main-thread tick.** The engine runs once per frame from `App::update`, with a bounded node budget (default 10k nodes/tick). No locking; the engine holds `&mut World` during its tick.

Rationale:
- Per-node cost with a tight bucket-queue loop is ~5-10 ns (block lookup, level compare, opacity check, queue push). 10k nodes/tick is ~0.1 ms — fits trivially under our frame budget.
- The main thread already mutates the World for `set_block`; the engine joining that ownership pattern is dramatically simpler than the alternative (`Arc<RwLock>`-per-chunk + a dedicated worker thread).
- During initial world stream-in, the budget can temporarily rise (e.g., 100k nodes/tick) since visual perfection isn't important until the player is interacting.

The previous BFS-on-rayon model exists because today's per-chunk BFS is too coarse to run on the main thread. The new engine is fine-grained enough that this concern disappears.

### Where the cascade plumbing went

| Old | New |
|---|---|
| `ChunkMeta::dirty::light` flag | Gone — engine owns "what needs recomputing" |
| `World::mark_below_dirty` | Gone — heightmap update on chunk load handles it |
| `lighting::seed_from_neighbors` | Gone — propagation crosses chunks naturally |
| `lighting::snapshot_face_boundaries` | Gone — no boundary diffing |
| `mesh_upload::relight_pump` | Gone |
| `JobResult::Relit` and its drain branch | Gone |
| `Jobs::spawn_relight` | Gone |
| `ChunkDirty::light` field | Replaced by `ChunkMeta::light_gpu_dirty` (set by engine, cleared by GPU upload) |

## Data model

### `LightEngine`

```rust
// src/lighting/engine.rs
pub struct LightEngine {
    sky:       ChannelEngine,
    block_rgb: [ChannelEngine; 3],  // R, G, B propagate independently
    sky_sources: HashMap<ChunkCoord, ChunkSkyLightSources>,
}

struct ChannelEngine {
    block_nodes_to_check: HashSet<BlockPos>,
    increase: BucketQueue,
    decrease: BucketQueue,
}
```

`block_nodes_to_check` is the set of positions whose underlying block changed since the last tick. Drained at tick start; each entry expands into the correct increase and/or decrease queue operations.

### `BucketQueue`

```rust
// src/lighting/queue.rs
pub struct BucketQueue {
    buckets: [VecDeque<QueueEntry>; 16],   // index = light level
    nonempty_mask: u16,                    // bit i = (buckets[i] not empty)
}

struct QueueEntry {
    pos: BlockPos,
    from_level: u8,
    propagation_mask: u8,  // 6 bits, one per face; 1 = blocked (don't propagate back)
}

impl BucketQueue {
    fn push(&mut self, entry: QueueEntry) { /* push to buckets[level], set mask bit */ }
    fn pop_highest(&mut self) -> Option<QueueEntry> {
        // 15 - nonempty_mask.leading_zeros() gives next non-empty bucket
        // pop_front from it; clear mask bit if it became empty
    }
}
```

`pop_highest` is O(1). Bucket sort works because light values are bounded and discrete (`0..=15`), and propagation only ever decreases the value being spread — so processing in highest-first order means a cell is never popped at a stale level. Same property Minecraft relies on, implemented more cache-coherently.

### `ChunkSkyLightSources`

```rust
// src/lighting/sky_sources.rs
pub struct ChunkSkyLightSources {
    // For each (x, z) column local to this chunk: the world-Y of the lowest
    // cell that is a sky source — i.e., one above the topmost opaque block
    // (so the cell itself is non-opaque and has nothing opaque above it).
    // Cells at y >= lowest_source_y are sources (sky = 15). Cells at
    // y < lowest_source_y propagate via the engine.
    // i32::MIN means "no opaque block in this column above the world floor"
    // (the whole column is sources, including down through this chunk).
    lowest_source_y: [i32; (CHUNK_DIM_U * CHUNK_DIM_U) as usize],
}

impl ChunkSkyLightSources {
    /// Build from scratch by scanning the chunk + chunks above (in the loaded set).
    pub fn build(world: &World, coord: ChunkCoord) -> Self { ... }

    /// Update when a block at (lx, ly, lz) changed opacity.
    /// Returns the (old, new) lowest_source_y for that column so the engine
    /// can emit the right increase/decrease ops on the cells that changed
    /// from source to non-source or vice versa.
    pub fn update(&mut self, world: &World, coord: ChunkCoord,
                  lx: u32, lz: u32, ly: i32) -> (i32, i32) { ... }

    pub fn lowest_source_y(&self, lx: u32, lz: u32) -> i32 { ... }
}
```

Memory: 32 × 32 × 4 bytes = 4 KB per chunk. At render radius 16 (~17k chunks) → ~70 MB. Trivial.

### What stays in `DenseChunk` (unchanged)

```rust
pub sky_light: Box<[u8; CHUNK_VOL]>,    // 4 bits used; engine writes here
pub block_rgb: Box<[u16; CHUNK_VOL]>,   // 4 bits per RGB channel; engine writes here
pub blocks:    Box<[u16; CHUNK_VOL]>,
```

Storage layout intentionally untouched — GPU upload (`render/light_volume.rs`) and persistence (`persistence/region.rs`) work as today.

### `ChunkMeta` changes

```rust
// REMOVED
pub dirty: ChunkDirty { mesh: bool, light: bool }   // light field gone

// ADDED
pub light_gpu_dirty: bool   // engine sets; mesh_upload clears after re-uploading 3D texture
// sky source heightmap lives in LightEngine.sky_sources, not in ChunkMeta
```

`ChunkDirty` keeps the `mesh` field for the existing mesh-rebuild flow.

## Algorithm

### Tick orchestration

```rust
pub fn tick(world: &mut World, registry: &BlockRegistry, mut budget: usize) {
    for channel in [Sky, R, G, B] {
        if budget == 0 { break }
        let used = drain_block_nodes_to_check(channel, world, registry, budget);
        budget -= used;
        if budget == 0 { break }
        let used = drain_decrease(channel, world, registry, budget);
        budget -= used;
        if budget == 0 { break }
        let used = drain_increase(channel, world, registry, budget);
        budget -= used;
    }
}
```

If `budget` is exhausted mid-channel, the engine resumes from where it left off next tick — queues are persistent across ticks. Convergence is guaranteed: every increase op monotonically raises a cell value, every decrease op monotonically lowers one, both are bounded by 0..=15, so the queues empty in finite time.

### Phase A — drain `block_nodes_to_check`

Each entry is a block position that changed since last tick. The engine reads the previous and current block state (the previous emission/opacity is cached at the moment of the change — see `World::set_block` below) and emits the right queue ops:

```
for pos in block_nodes_to_check.drain():
    (old_block, new_block) = pos.change_record  // cached at set_block time

    # ----- Sky channel -----
    if opacity changed at pos:
        (old_lsy, new_lsy) = sky_sources.update(world, pos.chunk(), pos.lx, pos.lz, pos.world_y)
        if new_lsy > old_lsy:
            # roof added — cells from old_lsy..=new_lsy-1 were sources, now aren't
            for y in old_lsy..new_lsy:
                level = chunk.sky_light_at(x, y, z)   # was 15
                set_level(sky, (x, y, z), 0)
                enqueue_decrease(sky, (x, y, z), 15)
        elif new_lsy < old_lsy:
            # roof removed — cells from new_lsy..old_lsy are now sources
            for y in new_lsy..=old_lsy:
                set_level(sky, (x, y, z), 15)
                enqueue_increase(sky, (x, y, z), 15)

    if new_block became opaque:
        # pull all light out of this cell, on all channels
        old_sky = chunk.sky_light_at(pos)
        set_level(sky, pos, 0)
        enqueue_decrease(sky, pos, old_sky)
        for c in RGB:
            old = chunk.block_rgb_at(pos, c)
            set_level(block_rgb[c], pos, 0)
            enqueue_decrease(block_rgb[c], pos, old)

    # ----- Block-light (per RGB channel) -----
    for c in RGB:
        if old_block.emission[c] != new_block.emission[c]:
            if new_block.emission[c] > old_block.emission[c]:
                set_level(block_rgb[c], pos, new_block.emission[c])
                enqueue_increase(block_rgb[c], pos, new_block.emission[c])
            else:
                # source weakened or removed
                enqueue_decrease(block_rgb[c], pos, old_block.emission[c])
                if new_block.emission[c] > 0:
                    set_level(block_rgb[c], pos, new_block.emission[c])
                    enqueue_increase(block_rgb[c], pos, new_block.emission[c])
```

### Phase B — drain decrease queue

```
while let Some(entry) = decrease.pop_highest():
    for face in 6:
        if entry.propagation_mask has face blocked: continue
        npos = entry.pos.offset(face)
        if npos out of loaded chunks: continue
        cur = read_level(channel, npos)
        if cur != 0 and cur < entry.from_level:
            # This neighbour was lit by us. Tear down its contribution.
            set_level(channel, npos, 0)
            mark_chunk_gpu_dirty(npos.chunk())
            # CRITICAL: if this cell is itself an emitter on this channel,
            # the tear-down just dimmed a source. Restore it as a fresh
            # increase source so we don't lose its contribution.
            emission = world.block(npos).emission_on(channel)
            if emission > 0:
                set_level(channel, npos, emission)
                increase.push(QueueEntry { pos: npos, from_level: emission,
                                            propagation_mask: 0 })
            decrease.push(QueueEntry { pos: npos, from_level: cur,
                                       propagation_mask: bit_for(opposite(face)) })
        elif cur >= entry.from_level:
            # Neighbour is at least as bright independently — re-flood from here
            # to repair anything we tore down.
            increase.push(QueueEntry { pos: npos, from_level: cur, propagation_mask: 0 })
```

The emission re-check matters when the decrease wave passes through a cell that contains a torch. Without it, a torch placed in the path of an unrelated light removal would have its contribution erased. Sky channel: `emission_on(Sky)` is 15 if `npos.world_y >= lowest_source_y(npos.x, npos.z)` else 0 — this restores sky sources naturally when a decrease wave passes through them.

This is the crucial difference from a naive BFS: decrease propagates the *removal* of a contribution, not just a value change. Cells lit by independent sources keep their light; cells lit only by the tear-down source go dark and trigger further decreases.

### Phase C — drain increase queue

```
while let Some(entry) = increase.pop_highest():
    for face in 6:
        if entry.propagation_mask has face blocked: continue
        npos = entry.pos.offset(face)
        if npos out of loaded chunks: continue
        block = world.block(npos)
        cost = propagation_cost(block, face, channel)
        if cost > entry.from_level: continue
        prop_level = entry.from_level - cost
        cur = read_level(channel, npos)
        if prop_level > cur:
            set_level(channel, npos, prop_level)
            mark_chunk_gpu_dirty(npos.chunk())
            if prop_level > 1:
                increase.push(QueueEntry { pos: npos, from_level: prop_level,
                                           propagation_mask: bit_for(opposite(face)) })
```

`propagation_cost`:
- `Block::Air`: 1
- `Block::Water`: 3 (today's value)
- `info.opaque == true`: ∞ (loop skips)
- Default non-opaque non-air: 1

`propagation_mask`'s "block back-face" bit prevents the push-pop ping-pong from re-visiting the cell we just came from. (Forward progress is still possible from any other face.)

### `read_level` / `set_level`

Both look up the right `DenseChunk` via `world.chunks.get_mut(&pos.chunk())`, then read/write the right slot of `sky_light` or unpack/repack the right nibble of `block_rgb`. For channel == sky, the `read_level` for cells at or above `lowest_source_y` (with no opaque block at the position itself) returns 15 unconditionally — this lets propagation cross unloaded chunks above the lowest source as if they were full sun.

## API

### `World::set_block`

```rust
pub fn set_block(&mut self, pos: BlockPos, new_block: Block) -> Vec<ChunkCoord> {
    let coord = pos.to_chunk();
    let local = pos.to_local();
    let Some(ChunkSlot::Stored { data, meta }) = self.chunks.get_mut(&coord) else {
        return vec![];
    };

    let mut dense = data.decompress();
    let old_block = dense.get(local);
    if old_block == new_block { return vec![] }
    dense.set(local, new_block);
    *data = Arc::new(PalettedChunk::compress(&dense));

    meta.dirty.mesh = true;
    meta.modified = true;
    meta.mesh_version = meta.mesh_version.wrapping_add(1);

    // Lighting: one call. No cascade.
    self.light_engine.on_block_changed(pos, old_block, new_block);

    // Border-edit re-mesh of face-adjacent neighbours (unchanged from today).
    let mut dirty = vec![coord];
    push_boundary_mesh_neighbours(&mut dirty, self, coord, local);
    dirty
}
```

`light_engine.on_block_changed` does only one thing: stores `(old_block, new_block)` in a tiny side-table keyed by `pos`, then inserts `pos` into each channel's `block_nodes_to_check`. All work happens in the tick.

### Chunk load (`JobResult::Generated`, `JobResult::LoadedFromDisk`)

```rust
JobResult::Generated { coord, data } => {
    world.insert(coord, data);
    world.on_chunk_loaded(coord);   // method on World; destructures internally to split the borrow
    // mesh spawning unchanged
}
```

`World::on_chunk_loaded(coord)` splits its own borrow into `chunks` and `light_engine` and calls the engine with a `&ChunkStore` view it can read from while mutating its own queues:

1. Build `ChunkSkyLightSources::build(world, coord)` and store it in `engine.sky_sources`.
2. Enqueue every emissive block in the chunk as an increase source on the appropriate RGB channels.
3. For each `(x, z)` column, enqueue every cell at or above `lowest_source_y` as an `increase(sky, 15)` source.
4. For each face-adjacent already-loaded neighbour, re-enqueue the boundary cells of *that neighbour* as increase sources on every channel. (Their stored values are unchanged; the engine will spread them across the seam without a special seed pass.)
5. Mark `meta.light_gpu_dirty = true` on the new chunk so it uploads.

The chunk becomes visible to the GPU with whatever light values are stored (loaded from disk if `LoadedFromDisk`, or empty if `Generated`). The engine converges the values over the following ticks. First-impression visual fidelity:
- `LoadedFromDisk`: values come from disk; engine re-enqueues sources to fix any staleness from neighbours.
- `Generated`: values start at 0; engine fills in over a few ticks. First frame may be dark (worst case ~150 ms / 10 ticks at 60 fps); subsequent frames converge.

### Chunk unload

```rust
pub fn on_chunk_unloaded(&mut self, coord: ChunkCoord) {
    self.sky_sources.remove(&coord);
    for ch in [&mut self.sky, &mut self.block_rgb[0], &mut self.block_rgb[1], &mut self.block_rgb[2]] {
        ch.block_nodes_to_check.retain(|p| p.to_chunk() != coord);
        ch.increase.purge_chunk(coord);
        ch.decrease.purge_chunk(coord);
    }
}
```

`BucketQueue::purge_chunk` walks each bucket's `VecDeque` and removes entries in the unloaded chunk. O(queue size), rare event.

### GPU upload integration

`mesh_upload.rs` adds a scan:

```rust
pub fn upload_dirty_light_volumes(
    world: &mut World, renderer: &mut Renderer, budget: usize,
) {
    let mut uploaded = 0;
    for (coord, slot) in world.chunks.iter_mut() {
        if uploaded >= budget { break }
        if let ChunkSlot::Stored { data, meta } = slot {
            if meta.light_gpu_dirty {
                let neighbours = gather_neighbors(world, *coord);  // for the +1 border
                let blob = build_light_volume(data, &neighbours);
                renderer.upload_chunk_light_volume(*coord, &blob);
                meta.light_gpu_dirty = false;
                uploaded += 1;
            }
        }
    }
}
```

Budget (default 32 chunks/frame) prevents a big convergence wave from saturating PCIe upload bandwidth. `build_light_volume` is the existing function in `render/light_volume.rs`, unchanged.

### Persistence

Unchanged. `sky_light` and `block_rgb` arrays are still serialized with each chunk in `persistence/region.rs`. On load (`JobResult::LoadedFromDisk`), the saved values are trusted as-is and the engine `on_chunk_loaded` runs the same neighbour re-enqueue as for `Generated` — this corrects any staleness introduced by neighbours having different blocks now than when this chunk was last saved.

A `lighting_version: u8` byte in the chunk save header lets us bump version when propagation rules change (e.g., water cost adjusted). On load with mismatched version, treat the saved light arrays as zero and let the engine re-derive everything.

## PR sequence

| # | PR | Scope | Visual diff | Risk |
|---|---|---|---|---|
| 1 | Sky source heightmap data structure | Add `ChunkSkyLightSources`, populate on chunk load alongside the existing BFS (which still runs). Expose `lowest_source_y(x, z)`. Not wired to lighting. | None — pure addition. | Low |
| 2 | Bucket queue + ChannelEngine skeleton | Add `queue.rs`, `engine.rs` with empty `tick()`. `World` holds a `LightEngine` field but `set_block` still calls today's BFS. Unit tests for queue ordering and chunk-purge. | None. | Low |
| 3 | Cutover (big-bang behind feature flag) | Replace BFS with engine. Delete `seed_from_neighbors`, `snapshot_face_boundaries`, `mark_below_dirty`, `relight_pump`, `JobResult::Relit`, `spawn_relight`. Wire `set_block` → `on_block_changed`, `Generated`/`Loaded` → `on_chunk_loaded`. Engine `tick()` runs from `App::update`. **Gated by `--feature legacy-lighting`** which routes back to the old BFS for one release as a safety net. | Many screenshot baselines refresh (values are now actually correct). Sky values in caves under deep water no longer freeze at 15. | **High** |
| 4 | Decrease queue correctness pass | PR3 implements increase + a minimal decrease (clear-and-reflood-on-source-removal). PR4 fills in proper Minecraft-style decrease propagation (tear down the chain, repair from independent sources). | Removing a torch correctly dims only the cells it was lighting. | Medium |
| 5 | Streaming-mode budget bump | Per-tick budget becomes configurable; during initial world stream-in (say, while >100 chunks are still pending mesh), bump from 10k to 100k nodes/tick. | Faster initial-load light convergence. | Low |

PR3 is the de-risk point. The legacy-lighting feature flag lets us revert at runtime if a regression slips through screenshot tests.

After PR5, the existing 2026-05-20 spec's PR5-8 (sun shadow map, ambient bounce, volumetrics, day/night colors) can land unchanged on top of this engine — they consume `sky_level` and `block_rgb` values from the per-chunk 3D texture, which is now populated correctly.

## Testing strategy

| Layer | Tests |
|---|---|
| `queue.rs` | Bucket queue push/pop order; nonempty_mask integrity; chunk purge correctness; multi-channel queue independence |
| `sky_sources.rs` | `build` produces the correct lowest_source_y for synthetic chunks; `update` returns correct `(old, new)` for opacity changes; correct behaviour at chunk top boundary (look up into +Y neighbour) |
| `engine.rs` propagate | Synthetic 1-chunk world: torch placed/removed produces expected light pattern; opaque block placed in lit area drops light correctly; RGB channels propagate independently (red torch doesn't emit green) |
| `lighting/tests/multi_chunk.rs` (integration) | Multi-chunk scenarios — 3-chunk-wide tunnel keeps gradient; chunks loaded out of order produce same final state as in-order load; cross-chunk torch lights the correct cells in the neighbour; roof added in chunk A correctly darkens column in chunk B |
| Headless screenshot harness | PR3 refreshes existing baselines (acknowledged). New baselines added in PR3: `out_of_order_chunk_load_converges`, `roof_added_then_removed_converges`, `cross_chunk_torch`, `deep_water_no_frozen_15` |
| Regression tests for past bugs | One per documented incident: `+Y_loaded_late_does_not_freeze_15` (was the `mark_below_dirty` bug); `boundary_edit_no_relight_cascade` (was the per-edit 4-neighbour cascade); `in_flight_dirty_mark_preservation_obsolete` (was the `JobResult::Relit` race) — each test sets up the original failure scenario and asserts the engine handles it |
| Convergence property test | Enqueue N random block changes on a synthetic world; tick to convergence; assert total node-ops ≤ `C × (changed_cells × max_level)` for a small constant C; assert no oscillation (a second tick with empty queues is a no-op) |

CI: any PR touching `lighting/`, `voxel/world.rs`, `voxel/chunk.rs`, or `mesh_upload.rs` runs the screenshot harness; diffs >2% flag for manual review (above the 6% noise floor of chunk streaming — see [Screenshot diff tool memory](.. /../../.claude/projects/-Users-fdatoo-Developer-oxium/memory/project_screenshot_diff_tool.md)).

## Risks and mitigations

**PR3 big-bang cutover.** Replaces ~400 lines of subtle code with ~600 lines of different subtle code. Many screenshot baselines will diff because values were wrong before. *Mitigation:* `--feature legacy-lighting` keeps the old BFS as a runtime fallback for one release. PR3 ships with a comparison test that runs the engine against the legacy BFS on the same synthetic world and asserts the engine's values are at least as correct (greater-or-equal sky at exposed cells; equal RGB at points dominated by a single source).

**Heightmap rebuild cost at chunk load.** Scanning 32×32×32 for opaque blocks is 32K opacity lookups per chunk. At streaming rate ~50 chunks/sec → 1.6M lookups/sec → ~50 µs/chunk. Acceptable.

**Initial chunk load enqueue spike.** A fresh chunk enqueues all its sky sources (up to 32 × 32 × column-height-above-source cells) plus emissives. For 100 chunks loading at once → ~100k enqueues. Default tick budget (10k nodes) caps frame impact; convergence over ~10 ticks (~167 ms @ 60 fps). *Mitigation:* PR5's streaming-mode budget bump.

**Saved light + engine version drift.** Changing propagation rules later (e.g., adjusting water cost) makes saved values inconsistent with the engine. *Mitigation:* `lighting_version` byte in save header; mismatch triggers full re-derivation via `on_chunk_loaded` (overwrites stale values).

**Decrease queue depends on every block mutation going through `set_block`.** If a code path mutates `dense.blocks` directly without calling `set_block` (or doesn't notify the engine), light values stay wrong forever. *Mitigation:* PR3 audit — every block mutation in the codebase must call `light_engine.on_block_changed`; chunk-load uses `on_chunk_loaded` instead. Add `#[deprecated]` annotations or grep checks to enforce.

**Multi-tick visible convergence.** A big change (cave-in, large structure placement) produces light fluctuation over ~5 frames before stable. Today's batch model produces a single "snap" instead. *Mitigation:* this is arguably more correct (light propagates at finite speed in the algorithm); if it looks wrong in practice, raise per-tick budget for that situation.

**Convergence pathologies.** Decrease that triggers increase that triggers decrease, etc. The algorithm provably terminates (each cell's level is bounded; each op moves a cell's value monotonically per phase) but worst-case ops per single block change could be high. *Mitigation:* the convergence property test bounds the ratio of ops to changed cells; failures in CI surface algorithmic regressions.

## Open questions

Two minor questions to resolve during implementation, neither blocking:

1. **Tick position in the frame.** Engine tick before or after `set_block` calls? "After" means lighting is one frame behind edits (matches today). "Before" means the player's last frame's lighting catches up after their current edit lands. Probably "after"; revisit if perceptual lag is annoying.
2. **`propagation_mask` width.** 6 bits (one per face, "block back-face") or 7 bits (Minecraft uses an extra bit to encode "this is the source, propagate in all directions"). The 7th bit may be unnecessary given how we enqueue sources (mask = 0 = no faces blocked); decide in PR2.

No other open questions. Every decision has a concrete answer.
