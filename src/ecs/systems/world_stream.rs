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
use crate::voxel::block::BlockRegistry;
use crate::voxel::coords::ChunkCoord;
use crate::voxel::world::{ChunkSlot, World};
use crate::worldgen::Generator;
use glam::IVec3;
use std::sync::Arc;

/// Horizontal load radius in chunks. M8 raises this from 6 → 12 once
/// LODs are in: the far chunks render as L1/L2 with far fewer triangles.
pub const RENDER_RADIUS: i32 = 12;
/// Vertical load radius in chunks.
pub const VERTICAL_RADIUS: i32 = 4;
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
pub fn world_stream(
    ecs: &GameEcs,
    world: &mut World,
    jobs: &Jobs,
    generator: &Arc<Generator>,
    registry: &Arc<BlockRegistry>,
) {
    let mut q = ecs.world.query_one::<&Position>(ecs.player).unwrap();
    let pos = q.get().unwrap();
    let pc = player_chunk(pos.0);

    // Build the candidate list then sort by Manhattan distance so the
    // closest gen-jobs go to the rayon pool first.
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
        d.x.abs() + d.y.abs() + d.z.abs()
    });

    for c in targets {
        if !world.chunks.contains_key(&c) {
            // Mark Pending so we don't re-spawn the same job next frame.
            world.chunks.insert(c, ChunkSlot::Pending);
            jobs.spawn_gen(c, generator.clone(), registry.clone());
        }
    }
}

/// Per-frame: evict chunks that have moved outside the unload radius.
/// Runs *after* `drain_jobs` (see module docs).
pub fn world_unload(ecs: &GameEcs, world: &mut World, renderer: &mut crate::render::Renderer) {
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
        world.chunks.remove(&c);
        renderer.remove_chunk_mesh(c);
    }
}
