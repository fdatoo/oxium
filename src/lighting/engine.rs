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
        // Pre-pass: rebuild sky_sources heightmaps for opacity-changed chunks.
        // Requires &mut chunks; must happen before the TickCache borrows chunks
        // immutably via dense().
        {
            use crate::voxel::world::ChunkSlot;
            let opacity_coords: std::collections::HashSet<_> = self.pending_block_changes.iter()
                .filter_map(|(pos, (old, new))| {
                    if registry.info(*old).opaque != registry.info(*new).opaque {
                        Some(pos.to_chunk())
                    } else {
                        None
                    }
                })
                .collect();
            for coord in opacity_coords {
                if let Some(ChunkSlot::Stored { data, meta }) = chunks.get_mut(&coord) {
                    let dense = data.decompress();
                    meta.sky_sources = crate::lighting::ChunkSkyLightSources::build_from_dense(
                        &dense, coord, registry,
                    );
                }
            }
        }

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
}

/// Which channel a tick step is processing. Used by the propagation
/// helpers to decide how to read/write the right slot of the chunk's
/// light arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Sky,
    BlockRgb(usize),  // 0=R, 1=G, 2=B
}

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

/// Phase A. Drains `engine.pending_block_changes` and, for each entry,
/// updates the chunk's `sky_sources` heightmap if opacity changed, then
/// either enqueues increase ops (new emission, new sky-source exposure)
/// or marks the chunk for full recompute (minimal-decrease fallback).
/// Returns the remaining budget.
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
                let chunk = pos.to_chunk();
                let idx = pos.to_local().to_index();
                let written = {
                    if let Some(dense) = cache.dense(chunks, chunk) {
                        let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[idx]);
                        let mut chans = [r, g, b];
                        chans[ch_i] = new_info.emission[ch_i];
                        dense.block_rgb[idx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                        true
                    } else {
                        false
                    }
                };
                if written {
                    cache.mark_written(chunk);
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

/// Cache-aware variant: clear and re-enqueue a single chunk's light from
/// scratch, mutating via `TickCache` rather than the chunk's
/// `Arc<PalettedChunk>` directly. Used inside `tick_with` for the
/// minimal-decrease fallback.
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

    let chunk_bottom_y = coord.0.y * CHUNK_DIM;

    // Clone sky_sources from meta before borrowing dense (chunks is immutable).
    let sky_sources_opt: Option<crate::lighting::ChunkSkyLightSources> =
        chunks.get(&coord).and_then(|s| match s {
            crate::voxel::world::ChunkSlot::Stored { meta, .. } => Some(meta.sky_sources.clone()),
            _ => None,
        });

    // Scope the dense borrow so we can call cache.mark_written afterward.
    {
        let Some(dense) = cache.dense(chunks, coord) else { return };

        dense.sky_light.iter_mut().for_each(|v| *v = 0);
        dense.block_rgb.iter_mut().for_each(|v| *v = 0);

        if let Some(ref sky_sources) = sky_sources_opt {
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
        // dense is dropped here, releasing the &mut cache borrow
    }

    cache.mark_written(coord);
}

/// Phase C — drain a single channel's increase queue, propagating
/// values outward by 1 per air step (3 in water). Uses `TickCache` to
/// avoid per-op chunk decompression. Returns the remaining budget.
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
    use crate::voxel::coords::BlockPos;

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
            let needs_write = {
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
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            };
            if needs_write {
                cache.mark_written(src_chunk);
            }
        }

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

            // Single cache.dense call: read, check, write all in one borrow.
            // NLL ends the borrow at the last use of `dense` (the write),
            // allowing cache.mark_written afterward.
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

            match channel {
                Channel::Sky => dense.sky_light[nidx] = prop_level,
                Channel::BlockRgb(c) => {
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[nidx]);
                    let mut chans = [r, g, b];
                    chans[c] = prop_level;
                    dense.block_rgb[nidx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                }
            }
            // dense last used above; NLL ends its borrow here.
            cache.mark_written(nchunk_coord);

            ch_engine.increase.push(crate::lighting::queue::QueueEntry {
                pos: npos,
                from_level: prop_level,
                propagation_mask: 1u8 << opposite_face[face_i],
            });
        }
    }

    budget
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
        // Torch emission is [13, 13, 13]. Center holds 13; an adjacent
        // air cell receives one step of attenuation (cost 1), so >= 12.
        assert!(cr >= 13, "torch cell R should hold its own emission: got {}", cr);
        assert!(ar >= 12, "adjacent cell R should be lit at least 12: got {}", ar);
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
        {
            let d = cache.dense(&chunks, coord).expect("loaded chunk");
            let _read = d.sky_light[0];
        }
        // Note: NO mark_written call.
        cache.flush(&mut chunks);

        let ChunkSlot::Stored { meta, .. } = chunks.get(&coord).unwrap() else { panic!() };
        assert!(!meta.light_gpu_dirty, "flush must not mark untouched chunks dirty");
    }
}
