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
use crate::persistence::region::region_path;
use crate::persistence::thread::{PersistRequest, Persistence};
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
    saves_dir: &Path,
) {
    let mut q = ecs.world.query_one::<&Position>(ecs.player).unwrap();
    let pos = q.get().unwrap();
    let pc = player_chunk(pos.0);

    // Build the candidate list and sort by Euclidean (squared)
    // distance from the player. Pure radial — every direction at
    // the same distance gets the same dispatch priority.
    //
    // A previous version added a "forward bonus" that subtracted a
    // chunk's projection along the camera look vector from the
    // priority key, on the theory that chunks the player is looking
    // at should load first. In practice that made the load
    // asymmetric: perpendicular and behind chunks consistently
    // arrived seconds after forward chunks, so any small camera
    // movement revealed unloaded voids. Pure radial is more
    // forgiving when the world is still streaming in.
    let mut targets: Vec<ChunkCoord> = Vec::new();
    for dy in -VERTICAL_RADIUS..=VERTICAL_RADIUS {
        for dz in -RENDER_RADIUS..=RENDER_RADIUS {
            for dx in -RENDER_RADIUS..=RENDER_RADIUS {
                targets.push(ChunkCoord(pc.0 + IVec3::new(dx, dy, dz)));
            }
        }
    }
    targets.sort_by_key(|c| {
        let d = c.0 - pc.0;
        // Euclidean squared as the primary key.
        let dist_sq =
            (d.x as i64).pow(2) + (d.y as i64).pow(2) + (d.z as i64).pow(2);
        // Symmetric tie-breaker. Without it, equidistant chunks
        // resolve in iteration order (dy → dz → dx), which puts the
        // +X+Z corner of every distance ring at the very tail of
        // the rayon queue. With ~10 000 chunks to dispatch on
        // spawn, those tail chunks waited multiple seconds to even
        // *start* gen — visible as a whole quadrant of the load
        // radius staying blank long after the others filled in.
        // A small Wang-style coord hash spreads ties evenly across
        // all 8 spatial octants. `dist_sq * 1024` keeps the
        // distance term dominant; only the low 10 bits of the hash
        // contribute, so two chunks at different distances never
        // swap order — only ties.
        let hash = c
            .0
            .x
            .wrapping_mul(73856093)
            .wrapping_add(c.0.y.wrapping_mul(19349663))
            .wrapping_add(c.0.z.wrapping_mul(83492791));
        dist_sq * 1024 + ((hash & 1023) as i64)
    });

    for c in targets {
        // Mark Pending only when the slot is currently absent. Using
        // `entry` avoids the double-hash of contains_key + insert.
        if let std::collections::hash_map::Entry::Vacant(slot) = world.chunks.entry(c) {
            slot.insert(ChunkSlot::Pending);
            // Prefer loading from disk when a region file exists —
            // persisted edits should reappear next session. Reads
            // go through the persistence I/O thread (single-threaded
            // but safe to interleave with the persistence thread's
            // own concurrent writes). A previous attempt to run
            // Loads on the rayon pool (`spawn_load`) introduced a
            // bug where chunks the player had previously edited
            // came back showing a flat fog-coloured plain — the
            // root cause is somewhere in concurrent-read vs the
            // chunk's saved light/block arrays, and reverting the
            // parallel path until we identify it.
            let path = region_path(saves_dir, c);
            if path.exists() {
                let _ = persistence.req_tx.send(PersistRequest::Load { coord: c });
            } else {
                jobs.spawn_gen(c, generator.clone(), registry.clone());
            }
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
        }
        world.chunks.remove(&c);
        renderer.remove_chunk_mesh(c);
    }
}
