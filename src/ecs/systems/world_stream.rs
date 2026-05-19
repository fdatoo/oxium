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

use crate::ecs::components::{Camera, Position};
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
    let mut q = ecs.world.query_one::<(&Position, &Camera)>(ecs.player).unwrap();
    let (pos, cam) = q.get().unwrap();
    let pc = player_chunk(pos.0);

    // Forward unit vector in world space, derived the same way the
    // renderer's view matrix does. Used to weight the chunk-load
    // sort: chunks the player is looking at load before chunks
    // behind them.
    let (sy, cy) = cam.yaw.sin_cos();
    let (sp, cp) = cam.pitch.sin_cos();
    let forward = glam::Vec3::new(cy * cp, sp, sy * cp);

    // Build the candidate list. Sort key combines:
    //   - Euclidean (squared) distance from the player — radial, so
    //     each load-distance ring fills in symmetrically instead of
    //     biasing toward whichever corner happened to come first in
    //     the iteration order (the old Manhattan-distance sort had
    //     thousands of ties that Rust's stable sort broke by
    //     `dy → dz → dx` iteration — one specific world-coord
    //     quadrant always loaded last).
    //   - A forward-direction bonus: subtract a chunk's projection
    //     along the forward vector from its priority key, so a
    //     chunk in the look direction outranks an equidistant
    //     chunk behind. The 0.5 weight is gentle enough not to
    //     leave behind-chunks too far adrift but strong enough
    //     that initial spawn sees the front of the world fill
    //     first.
    let mut targets: Vec<ChunkCoord> = Vec::new();
    for dy in -VERTICAL_RADIUS..=VERTICAL_RADIUS {
        for dz in -RENDER_RADIUS..=RENDER_RADIUS {
            for dx in -RENDER_RADIUS..=RENDER_RADIUS {
                targets.push(ChunkCoord(pc.0 + IVec3::new(dx, dy, dz)));
            }
        }
    }
    targets.sort_by_key(|c| {
        let d = (c.0 - pc.0).as_vec3();
        let dist_sq = d.length_squared();
        let forward_bonus = forward.dot(d).max(0.0) * 0.5;
        // Multiply by 1000 to keep ~3 decimal digits of precision
        // when collapsing to integer for the sort key.
        ((dist_sq - forward_bonus) * 1000.0) as i64
    });

    for c in targets {
        // Mark Pending only when the slot is currently absent. Using
        // `entry` avoids the double-hash of contains_key + insert.
        if let std::collections::hash_map::Entry::Vacant(slot) = world.chunks.entry(c) {
            slot.insert(ChunkSlot::Pending);
            // Prefer loading from disk when a region file exists —
            // persisted edits should reappear next session.
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
