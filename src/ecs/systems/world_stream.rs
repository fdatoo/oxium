//! Chunk streaming: keep the player surrounded by a horizontal disk + a
//! vertical column of loaded chunks, generating new ones on demand and
//! evicting ones that fall out of range.
//!
//! Streaming has two halves:
//!
//! - [`world_stream`] runs *before* job draining and *enqueues* generation
//!   jobs for any chunk inside the load radius that isn't already known.
//! - [`world_unload`] runs *after* job draining and *removes* chunks that
//!   have moved outside the (larger) unload radius. Doing the removal *after*
//!   the drain prevents a window where a chunk's mesh job completes for a
//!   chunk we just evicted.

use crate::ecs::components::Position;
use crate::ecs::GameEcs;
use crate::jobs::Jobs;
use crate::persistence::thread::{PersistRequest, Persistence};
use crate::persistence::SaveIndex;
use crate::voxel::block::BlockRegistry;
use crate::voxel::coords::ChunkCoord;
use crate::voxel::world::{ChunkSlot, World};
use crate::worldgen::Generator;
use glam::IVec3;
use std::path::Path;
use std::sync::Arc;

/// Horizontal load radius in chunks. M8 raises this from 6 → 12 once
/// LODs are in: the far chunks render as L1/L2 with far fewer triangles.
pub const RENDER_RADIUS: i32 = 12;
/// Vertical load radius in chunks. Bumped from 4 → 8 (= ±256 blocks)
/// so a player digging or flying deep underground still has every
/// chunk loaded between them and the surface — otherwise the world
/// has unloaded gaps above their head and the sky shader leaks
/// through into the cave view.
pub const VERTICAL_RADIUS: i32 = 8;
/// Unload radius slightly larger than the load radius so a tiny step doesn't
/// cause a re-load — hysteresis.
pub const UNLOAD_RADIUS: i32 = RENDER_RADIUS + 3;
pub const UNLOAD_VERT: i32 = VERTICAL_RADIUS + 1;

/// Compute the chunk the player currently stands in.
fn player_chunk(pos: glam::Vec3) -> ChunkCoord {
    ChunkCoord(IVec3::new(
        (pos.x / 32.0).floor() as i32,
        (pos.y / 32.0).floor() as i32,
        (pos.z / 32.0).floor() as i32,
    ))
}

/// Cached, distance-sorted candidate list for `world_stream`.
///
/// Rebuilding + sorting the ~10 625 chunks in the load radius every
/// frame showed up as the largest non-idle CPU hotspot in profiling
/// (quicksort + the Wang-hash tie-breaker dominated). The candidate
/// set is a pure function of `player_chunk(pos)`, so we cache it and
/// only rebuild when the player crosses a chunk boundary. With a
/// stationary player this is zero allocation and zero sort per frame.
#[derive(Default)]
pub struct WorldStreamCache {
    last_player_chunk: Option<ChunkCoord>,
    targets: Vec<ChunkCoord>,
}

/// Per-frame: enqueue generation jobs for every chunk inside the render
/// radius that the World doesn't already know about (either as `Pending` or
/// `Stored`). Visits chunks closest-first so nearby terrain appears before
/// the horizon fills in.
#[allow(clippy::too_many_arguments)]
pub fn world_stream(
    ecs: &GameEcs,
    world: &mut World,
    jobs: &Jobs,
    generator: &Arc<Generator>,
    registry: &Arc<BlockRegistry>,
    persistence: &Persistence,
    save_index: &mut SaveIndex,
    saves_dir: &Path,
    cache: &mut WorldStreamCache,
) {
    let mut q = ecs.world.query_one::<&Position>(ecs.player).unwrap();
    let pos = q.get().unwrap();
    let pc = player_chunk(pos.0);

    // Build the candidate list and sort. Two-level priority:
    //
    //   1. Horizontal distance² (dx² + dz²) — primary.
    //   2. Vertical distance² (dy²) — tiebreak within a horizontal
    //      ring.
    //
    // Pure 3-D Euclidean (`dx² + dy² + dz²`) put mountain peaks at
    // dy=1 ahead of their bases at dy=3+ when both were at the same
    // horizontal distance, so the player saw floating mountain tops
    // hanging in midair while the columns beneath them streamed in.
    // Horizontal-first dispatches every Y in a column at the same
    // primary priority, so a mountain at distance D arrives as a
    // coherent column instead of top-first.
    //
    // The forward-bonus experiment from an even earlier design is
    // still ruled out — direction-asymmetric priority made every
    // camera turn reveal unloaded voids. Horizontal-first preserves
    // direction symmetry: only horizontal *distance* matters, not
    // bearing.
    //
    // The candidate set + sort order is a pure function of `pc`, so
    // we cache it and only rebuild when the player crosses a chunk
    // boundary. Profiling showed the per-frame sort (over ~10 625
    // entries) was the single biggest non-idle CPU cost; with a
    // stationary player this branch never runs.
    if cache.last_player_chunk != Some(pc) {
        cache.targets.clear();
        for dy in -VERTICAL_RADIUS..=VERTICAL_RADIUS {
            for dz in -RENDER_RADIUS..=RENDER_RADIUS {
                for dx in -RENDER_RADIUS..=RENDER_RADIUS {
                    cache.targets.push(ChunkCoord(pc.0 + IVec3::new(dx, dy, dz)));
                }
            }
        }
        cache.targets.sort_by_key(|c| {
            let d = c.0 - pc.0;
            // Bounds at the current radii: h_dist_sq ∈ [0, 288]
            // (RENDER_RADIUS=12 → 144 each axis), v_dist_sq ∈ [0, 64]
            // (VERTICAL_RADIUS=8). The 100 000 multiplier on h_dist_sq
            // dwarfs everything below it so no horizontal ring ever
            // swaps with another — vertical and the hash only order
            // within a ring.
            let h_dist_sq = (d.x as i64).pow(2) + (d.z as i64).pow(2);
            let v_dist_sq = (d.y as i64).pow(2);
            // Symmetric tie-breaker for chunks that share both
            // horizontal and vertical distance. Without it,
            // equidistant chunks resolve in iteration order
            // (dy → dz → dx), which puts the +X+Z corner of every
            // ring at the very tail of the rayon queue. The
            // Wang-style coord hash spreads ties evenly across all
            // 8 spatial octants. Only the low 10 bits of the hash
            // contribute (max 1023), so v_dist_sq's 1024 multiplier
            // and h_dist_sq's 100 000 multiplier both keep the hash
            // strictly sub-ordinal — two chunks at different
            // distances can never swap, only ties do.
            let hash = c
                .0
                .x
                .wrapping_mul(73856093)
                .wrapping_add(c.0.y.wrapping_mul(19349663))
                .wrapping_add(c.0.z.wrapping_mul(83492791));
            h_dist_sq * 100_000 + v_dist_sq * 1024 + ((hash & 1023) as i64)
        });
        cache.last_player_chunk = Some(pc);
    }

    // Cap in-flight dispatches so the rayon queue's priority order
    // can't go stale. Without this, the first frame at world spawn
    // dispatches all ~10 625 candidates into rayon's injection queue
    // (sorted by distance from spawn). The queue is FIFO; workers
    // chew through it in that order regardless of where the player
    // walks next. After the player moves, new edge chunks get
    // appended to the *back* of a queue that's still holding
    // thousands of jobs prioritised around where the player *used to
    // be*. From the player's POV that reads as a directional wipe:
    // their current surroundings stay empty while workers fill in
    // old territory. The visible symptom — half the screen stuck on
    // sky while you wait for chunks behind you to finish — disappears
    // once the queue can never grow more than `DISPATCH_CAP` deep:
    // each new dispatch picks the closest currently-vacant chunk, so
    // workers are always servicing the player's actual current
    // priority order.
    //
    // 64 ≈ a frame of worker output at 7 gen workers × ~10 ms/chunk
    // = ~90 ms of buffered work — enough that workers never idle
    // between frames, small enough that the front of the queue
    // stays within ~90 ms of current player priority. Started at
    // 32 but that left high-core-count machines noticeably
    // under-fed; 64 keeps everyone busy without re-introducing the
    // stale-priority symptom.
    const DISPATCH_CAP: usize = 64;
    let pending_count = world
        .chunks
        .values()
        .filter(|s| matches!(s, ChunkSlot::Pending))
        .count();
    let mut budget = DISPATCH_CAP.saturating_sub(pending_count);

    for &c in &cache.targets {
        if budget == 0 {
            break;
        }
        // Mark Pending only when the slot is currently absent. Using
        // `entry` avoids the double-hash of contains_key + insert.
        if let std::collections::hash_map::Entry::Vacant(slot) = world.chunks.entry(c) {
            slot.insert(ChunkSlot::Pending);
            // Prefer loading from disk when a saved chunk exists for
            // this *specific* coord — persisted edits should reappear
            // next session.
            //
            // The presence check goes through `SaveIndex`, which
            // reads the region file's 16 KB header *once* per region
            // per session and answers in O(1) thereafter. Before
            // adding this cache, the check was `path.exists()` —
            // true for every chunk in a region as soon as the
            // player saved one of its 4096 slots, so every empty
            // sibling slot paid a round-trip through the
            // single-threaded persistence worker just to be told
            // NotPresent. With ~10 000 chunks in the load radius
            // that turned a single edit into seconds of useless
            // serial I/O at the next launch ("everything streams in
            // fast, but as soon as I edit anything, the next launch
            // takes ages to render"). The bitmap lookup turns the
            // empty-slot case back into a one-frame procedural gen.
            //
            // Loads still go through the persistence I/O thread
            // (single-threaded, but only N requests now where N =
            // chunks the player actually edited — not N = chunks in
            // the load radius). A previous attempt to run Loads on
            // the rayon pool (`spawn_load`) introduced a separate
            // out-of-order arrival bug; with the empty-slot case
            // pruned away here, parallel reads aren't urgent.
            if save_index.has(saves_dir, c) {
                let _ = persistence.req_tx.send(PersistRequest::Load { coord: c });
            } else {
                jobs.spawn_gen(c, generator.clone(), registry.clone());
            }
            budget -= 1;
        }
    }
}

/// Per-frame: evict chunks that have moved outside the unload radius.
/// Modified chunks are queued for save *before* eviction so we don't lose
/// the player's work. Runs *after* `drain_jobs` (see module docs).
pub fn world_unload(
    ecs: &GameEcs,
    world: &mut World,
    renderer: &mut crate::render::Renderer,
    persistence: &Persistence,
    save_index: &mut SaveIndex,
) {
    let mut q = ecs.world.query_one::<&Position>(ecs.player).unwrap();
    let pos = q.get().unwrap();
    let pc = player_chunk(pos.0).0;

    let to_remove: Vec<ChunkCoord> = world
        .chunks
        .keys()
        .filter(|c| {
            let d = c.0 - pc;
            d.x.abs() > UNLOAD_RADIUS || d.z.abs() > UNLOAD_RADIUS || d.y.abs() > UNLOAD_VERT
        })
        .copied()
        .collect();
    for c in to_remove {
        // Save before evict if the player modified this chunk. Pure-
        // generated chunks regenerate from seed on next visit, so we
        // don't waste disk on them.
        if let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get(&c)
            && meta.modified
        {
            let _ = persistence.req_tx.send(PersistRequest::Save {
                coord: c,
                data: data.clone(),
            });
            // Keep the in-memory presence cache in lockstep with the
            // queued write — the chunk hasn't reached disk yet but
            // it's about to, and a subsequent `world_stream` pass
            // for this coord (or a sibling in the same region) must
            // see "yes, there's saved data here" so it routes the
            // Load through the persistence thread rather than
            // re-generating from seed.
            save_index.mark(c);
        }
        world.chunks.remove(&c);
        renderer.remove_chunk_mesh(c);
    }
}
