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
}

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
                    // TODO(T6): meta.light_gpu_dirty = true;
                    let _ = meta;
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
        // Budget enforcement is intentionally relaxed here — the recompute
        // is atomic per chunk and we already drained pending_block_changes
        // for this tick, so a partial bail would lose work. PR4 will
        // revisit budgeting under heavy edit load.
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
    // TODO(T6): meta.light_gpu_dirty = true;
    let _ = meta;
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
    use crate::voxel::coords::BlockPos;
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

        // Ensure the source cell itself holds at least `from_level`.
        // Callers (recompute_chunk_light_from_scratch, drain_pending_block_changes)
        // pre-write the cell before enqueueing, but manual enqueues in tests
        // (and future callers) may skip that. Writing here is idempotent if
        // the cell already holds a higher value.
        {
            // TODO(perf): this unconditionally decompresses the chunk to check
            // whether the source cell needs writing, even though most callers
            // (recompute_chunk_light_from_scratch, drain_pending_block_changes)
            // pre-write before enqueuing — `needs_write` then evaluates to false.
            // At a 50k-op budget that's 50k extra 128 KB decompress allocs/frame.
            // Mitigation: add PalettedChunk::sky_light_at(idx) and
            // block_rgb_at(idx) accessors that read directly from Packed4Bit
            // without decompressing the whole chunk, then check `needs_write`
            // before decompressing. Out of scope for PR3 (engine correctness
            // priority); revisit in PR4 or a perf-pass PR.
            let src_chunk_coord = entry.pos.to_chunk();
            let src_local = entry.pos.to_local();
            let src_idx = src_local.to_index();
            if let Some(ChunkSlot::Stored { data, meta }) = chunks.get_mut(&src_chunk_coord) {
                let mut dense = data.decompress();
                let needs_write = match channel {
                    Channel::Sky => dense.sky_light[src_idx] < entry.from_level,
                    Channel::BlockRgb(c) => {
                        let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[src_idx]);
                        [r, g, b][c] < entry.from_level
                    }
                };
                if needs_write {
                    match channel {
                        Channel::Sky => {
                            dense.sky_light[src_idx] = entry.from_level;
                        }
                        Channel::BlockRgb(c) => {
                            let (r, g, b) = crate::voxel::chunk::unpack_rgb(dense.block_rgb[src_idx]);
                            let mut chans = [r, g, b];
                            chans[c] = entry.from_level;
                            dense.block_rgb[src_idx] = crate::voxel::chunk::pack_rgb(chans[0], chans[1], chans[2]);
                        }
                    }
                    *data = std::sync::Arc::new(crate::voxel::chunk::PalettedChunk::compress(&dense));
                    // TODO(T6): meta.light_gpu_dirty = true;
                    let _ = meta;
                }
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
            // TODO(T6): meta.light_gpu_dirty = true;
            let _ = meta;

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
}
