//! Bridge from completed background jobs to the GPU and world state.
//!
//! Every frame `drain_jobs` pulls a bounded number of results from the
//! channel (so a flood of completions doesn't blow the frame budget), and
//! for each one:
//!
//! - **Generated** → install the new chunk in the World, then enqueue mesh
//!   jobs both for the new chunk and for any of its already-loaded
//!   neighbours (a new chunk reveals previously-hidden face boundaries on
//!   those neighbours).
//! - **Meshed** → upload the new mesh into the GPU, replacing any previous
//!   one for that coordinate.

use crate::jobs::{JobResult, Jobs};
use crate::persistence::thread::{PersistResult, Persistence};
use crate::render::Renderer;
use crate::voxel::block::BlockRegistry;
use crate::voxel::chunk::{ChunkState, LightState, PalettedChunk};
use crate::voxel::coords::ChunkCoord;
use crate::voxel::world::{ChunkSlot, World};
use crate::worldgen::Generator;
use glam::IVec3;
use std::sync::Arc;

/// Upper bound on how many jobs to drain per frame. Sized to comfortably
/// outpace a 15-worker pool's burst rate during the initial fly-in: when
/// the player teleports or spawns into a new area, the pool can produce
/// dozens of completions before the first frame ends, and we don't want
/// the FIFO channel to fill with gen results so far ahead of mesh results
/// that meshing never gets a turn at the receiver.
const MAX_PER_FRAME: usize = 64;

/// Drain up to [`MAX_PER_FRAME`] job results and act on them.
///
/// We deliberately do NOT cap `Meshed` results separately. A previous
/// attempt to cap at 8/frame fixed a 61 ms spike during cascade-driven
/// mesh bursts but broke a more important property: while the world
/// is streaming in, `JobResult::Generated` handler spawns up to 9
/// downstream mesh jobs per generated chunk (self + 6 neighbours +
/// LOD1 + LOD2), so mesh-result production rate can easily exceed
/// any small per-frame cap. Capping meant a player-edit's mesh
/// queued behind streaming traffic effectively never reached the
/// GPU — the user reported breaks where collision flipped (world
/// data correct) but the block stayed visible. The 61 ms spike
/// happens once during initial cascade; the broken-edit symptom
/// happens constantly. We pick the spike.
pub fn drain_jobs(
    world: &mut World,
    jobs: &Jobs,
    renderer: &mut Renderer,
    registry: &Arc<BlockRegistry>,
) {
    for _ in 0..MAX_PER_FRAME {
        let result = match jobs.rx.try_recv() {
            Ok(r) => r,
            Err(_) => return,
        };
        match result {
            JobResult::Generated {
                coord,
                data,
                light_inputs,
            } => {
                // Install the new chunk first so neighbour mesh jobs can
                // see it.
                world.insert_with_light_inputs(coord, data, light_inputs);

                dispatch_relight(world, jobs, registry, coord);
                mark_loaded_neighbors_for_relight(world, coord);

                // Spawn LOD0 mesh for this chunk AND any already-
                // loaded neighbours — when an out-of-order arrival
                // (common with parallel `spawn_load`) plugs a gap
                // between two existing chunks, the neighbours need
                // to re-mesh so their boundary faces toward us get
                // hidden by face culling. Without this, a surface
                // chunk that meshed before its `Y-1` neighbour
                // loaded keeps a giant bottom quad at the chunk
                // floor — which back-face culling can't hide
                // because there's no other side to draw, and shows
                // as a flat fog-coloured plain.
                //
                // The `if let Some(Stored)` gate means we don't
                // re-mesh chunks that are still `Pending`; during
                // initial stream-in most neighbours are Pending so
                // the actual fan-out is small (~1-3 jobs per
                // arrival instead of always 7).
                //
                // LOD1/2 still skipped — the renderer falls back
                // to LOD0 via `slots.iter().flatten().next()`.
                for c in std::iter::once(coord).chain(neighbor_coords(coord)) {
                    spawn_mesh_lod0_if_ready(world, jobs, registry, c);
                }
            }
            JobResult::Meshed {
                coord,
                lod,
                mesh,
                version,
                light_volume,
            } => {
                // Drop the upload if the chunk has been re-edited since
                // this mesh job was spawned. Without this check, a
                // slow streaming mesh job can complete after a fast
                // edit-triggered mesh and overwrite the GPU buffer
                // with stale geometry — the visible "block flickers
                // back for a moment" artefact the user reported.
                let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get(&coord) else {
                    continue;
                };
                if version != meta.mesh_version {
                    continue;
                }

                if light_volume.is_none() && !renderer.has_chunk_light_volume(coord) {
                    if let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get_mut(&coord) {
                        meta.state = ChunkState::Generated;
                        meta.dirty.mesh = true;
                    }
                    log::debug!(
                        "deferring mesh upload for {coord:?}: real light volume not uploaded yet"
                    );
                    continue;
                }

                if let Some(blob) = light_volume {
                    renderer.upload_chunk_light_volume(coord, blob.as_ref());
                }
                if let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get_mut(&coord) {
                    meta.state = ChunkState::Ready;
                    meta.dirty.mesh = false;
                }
                if mesh.indices.is_empty() && renderer.has_chunk_mesh(coord) {
                    continue;
                }
                renderer.upload_chunk_mesh(coord, lod, &mesh);
            }
            JobResult::LightBlobReady { coord, blob } => {
                // The mesh-pool worker already did the expensive decompress +
                // 33³ sample loop; just hand the finished blob to the GPU.
                renderer.upload_chunk_light_volume(coord, blob.as_ref());
            }
            JobResult::Relit {
                coord,
                version,
                data,
                changed_faces,
                unresolved_faces,
                light_volume,
            } => {
                use crate::voxel::chunk::{ChunkDirty, ChunkState};
                let data_arc = Arc::new(data);
                let mut accepted = false;
                if let Some(ChunkSlot::Stored { data: cur, meta }) = world.chunks.get_mut(&coord) {
                    if meta.light_version != version {
                        continue;
                    }
                    *cur = data_arc.clone();
                    meta.dirty = ChunkDirty {
                        mesh: true,
                        light: false,
                    };
                    meta.state = ChunkState::Generated;
                    meta.light_state = LightState::Lit { version };
                    meta.unresolved_borders = unresolved_faces;
                    meta.light_gpu_dirty = true;
                    accepted = true;
                }
                if !accepted {
                    continue;
                }
                renderer.upload_chunk_light_volume(coord, light_volume.as_ref());
                // Bounded cascade: only mark the face neighbours whose
                // boundary actually changed as unlit. Most
                // relights downstream of a cascade produce no further
                // change, so propagation naturally terminates in
                // `O(diameter)` iterations rather than blowing up
                // exponentially.
                let nbrs = neighbor_coords(coord);
                for (i, changed) in changed_faces.iter().enumerate() {
                    if !changed {
                        continue;
                    }
                    mark_chunk_unlit_for_relight(world, nbrs[i], "relight boundary changed");
                }
                spawn_mesh_lod0_if_ready(world, jobs, registry, coord);
            }
            JobResult::LoadedFromDisk { coord, data } => match data {
                Some(data) => {
                    world.insert(coord, data);
                    dispatch_relight(world, jobs, registry, coord);
                    mark_loaded_neighbors_for_relight(world, coord);
                    // Same neighbour-remesh strategy as Generated.
                    // The parallel `spawn_load` was the trigger for
                    // the out-of-order arrival bug, and Loaded is
                    // where most of those arrivals come from.
                    for c in std::iter::once(coord).chain(neighbor_coords(coord)) {
                        spawn_mesh_lod0_if_ready(world, jobs, registry, c);
                    }
                }
                None => {
                    // Region file existed but the slot was empty —
                    // fall back to procedural gen.
                    // generator and persistence are passed to
                    // drain_persistence; we need them here too.
                    // (See the parameter additions below.)
                    log::trace!("loaded chunk slot empty at {coord:?}, falling back to gen");
                    // Re-mark as Vacant so world_stream picks it up
                    // next frame and dispatches gen.
                    world.chunks.remove(&coord);
                }
            },
        }
    }
}

/// Drain the `dirty.light` flag set across the loaded world: queue
/// relight jobs for up to [`RELIGHT_BUDGET`] chunks per frame and clear
/// each chunk's `dirty.light` so the next frame's scan doesn't
/// double-queue.
///
/// The bound matters: a fresh world stream-in or a player-edit cascade
/// can dirty hundreds of chunks at once. Without a cap, the rayon pool
/// gets buried and the main thread stalls waiting for results — the
/// same shape of regression as the cancelled v0.1.19 per-edit cascade.
/// At 128 chunks per frame the dedicated relight pool can stay fed during
/// stream-in without burying mesh work.
pub fn relight_pump(
    world: &mut World,
    jobs: &Jobs,
    registry: &Arc<BlockRegistry>,
    priority: Option<ChunkCoord>,
) -> usize {
    const RELIGHT_BUDGET: usize = 128;
    // Snapshot the candidate coords up-front so we don't hold an
    // immutable borrow over the loop body's `get_mut` + `spawn`.
    // Also count the *total* dirty set for the HUD perf readout.
    let mut total_dirty = 0usize;
    let mut candidates: Vec<ChunkCoord> = Vec::new();
    for (c, slot) in world.chunks.iter() {
        if let ChunkSlot::Stored { meta, .. } = slot
            && (meta.dirty.light
                || matches!(
                    meta.light_state,
                    LightState::Unlit | LightState::NeedsBorderReconcile
                ))
        {
            total_dirty += 1;
            candidates.push(*c);
        }
    }
    if let Some(center) = priority {
        candidates.sort_unstable_by_key(|c| priority_key(*c, center));
    }
    for c in candidates.into_iter().take(RELIGHT_BUDGET) {
        dispatch_relight(world, jobs, registry, c);
    }
    return total_dirty;
}

fn priority_key(coord: ChunkCoord, center: ChunkCoord) -> i64 {
    let d = coord.0 - center.0;
    let h_dist_sq = (d.x as i64).pow(2) + (d.z as i64).pow(2);
    let v_dist_sq = (d.y as i64).pow(2);
    h_dist_sq * 100_000 + v_dist_sq
}

/// The six face-adjacent chunk coordinates, in [`crate::mesher::Face`]
/// order (PosX, NegX, PosY, NegY, PosZ, NegZ).
pub fn neighbor_coords(c: ChunkCoord) -> [ChunkCoord; 6] {
    [
        ChunkCoord(c.0 + IVec3::new(1, 0, 0)),
        ChunkCoord(c.0 + IVec3::new(-1, 0, 0)),
        ChunkCoord(c.0 + IVec3::new(0, 1, 0)),
        ChunkCoord(c.0 + IVec3::new(0, -1, 0)),
        ChunkCoord(c.0 + IVec3::new(0, 0, 1)),
        ChunkCoord(c.0 + IVec3::new(0, 0, -1)),
    ]
}

fn mark_loaded_neighbors_for_relight(world: &mut World, coord: ChunkCoord) {
    for nc in neighbor_coords(coord) {
        if world.is_loaded(nc) {
            mark_chunk_unlit_for_relight(world, nc, "neighbor loaded");
        }
    }
}

fn mark_chunk_unlit_for_relight(world: &mut World, coord: ChunkCoord, reason: &'static str) {
    world.mark_chunk_unlit(coord, reason);
}

fn dispatch_relight(
    world: &mut World,
    jobs: &Jobs,
    registry: &Arc<BlockRegistry>,
    coord: ChunkCoord,
) {
    if !light_vertical_context_ready(world, coord) {
        return;
    }
    let neighbors = gather_neighbors(world, coord);
    let neighbor_lit = gather_neighbor_lit(world, coord);
    let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get(&coord) else {
        return;
    };
    let data_arc = data.clone();
    let inputs = meta.light_inputs.clone();
    let Some(version) = world.queue_relight(coord) else {
        return;
    };
    jobs.spawn_relight(
        coord,
        version,
        data_arc,
        inputs,
        neighbors,
        neighbor_lit,
        registry.clone(),
    );
}

fn light_vertical_context_ready(world: &World, coord: ChunkCoord) -> bool {
    let above = ChunkCoord(coord.0 + IVec3::new(0, 1, 0));
    if chunk_light_usable(world, above) {
        return true;
    }
    let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get(&coord) else {
        return false;
    };
    meta.light_inputs.top_sky.iter().all(|&v| v > 0)
}

fn spawn_mesh_lod0_if_ready(
    world: &mut World,
    jobs: &Jobs,
    registry: &Arc<BlockRegistry>,
    coord: ChunkCoord,
) {
    if !can_spawn_mesh_lod0(world, coord) {
        return;
    }
    let Some((data_arc, version)) = (match world.chunks.get_mut(&coord) {
        Some(ChunkSlot::Stored { data, meta }) => {
            let data_arc = data.clone();
            meta.state = ChunkState::Meshing;
            meta.dirty.mesh = false;
            meta.mesh_version = meta.mesh_version.wrapping_add(1);
            Some((data_arc, meta.mesh_version))
        }
        _ => None,
    }) else {
        return;
    };
    let neighbors = gather_neighbors(world, coord);
    jobs.spawn_mesh_lod0(coord, data_arc, neighbors, registry.clone(), version, false);
}

fn can_spawn_mesh_lod0(world: &World, coord: ChunkCoord) -> bool {
    let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get(&coord) else {
        return false;
    };
    if !matches!(meta.light_state, LightState::Lit { .. }) {
        return false;
    }
    if meta.state == ChunkState::Ready {
        return true;
    }
    let neighbors = neighbor_coords(coord);
    // Initial opaque geometry should not be built against absent horizontal
    // neighbours. Otherwise a stale mesh can expose chunk-edge side faces
    // until a later neighbour-triggered rebuild hides them.
    [0, 1, 4, 5]
        .into_iter()
        .all(|face| world.is_loaded(neighbors[face]))
}

pub(crate) fn chunk_light_usable(world: &World, coord: ChunkCoord) -> bool {
    matches!(
        world.chunks.get(&coord),
        Some(ChunkSlot::Stored {
            meta: crate::voxel::chunk::ChunkMeta {
                light_state: LightState::Lit { .. } | LightState::NeedsBorderReconcile,
                ..
            },
            ..
        })
    )
}

fn gather_neighbor_lit(world: &World, c: ChunkCoord) -> [bool; 6] {
    let coords = neighbor_coords(c);
    std::array::from_fn(|i| chunk_light_usable(world, coords[i]))
}

/// Drain pending persistence results: install loaded chunks, mark saved
/// chunks as no-longer-modified. Bounded per frame so a burst of saves
/// (e.g. autosave fanout) doesn't blow the frame budget.
pub fn drain_persistence(
    world: &mut World,
    jobs: &Jobs,
    persistence: &Persistence,
    generator: &Arc<Generator>,
    registry: &Arc<BlockRegistry>,
) {
    const MAX_PER_FRAME: usize = 16;
    for _ in 0..MAX_PER_FRAME {
        let Ok(res) = persistence.result_rx.try_recv() else {
            return;
        };
        match res {
            PersistResult::Loaded { coord, data } => match data {
                Some(data) => {
                    let dense = data.decompress();
                    let light_inputs = generator.light_inputs_for_chunk(coord, &dense, registry);
                    world.insert_with_light_inputs(coord, data, light_inputs);
                    dispatch_relight(world, jobs, registry, coord);
                    mark_loaded_neighbors_for_relight(world, coord);
                    // Re-mesh self + already-Stored neighbours, so
                    // out-of-order arrivals don't leave boundary
                    // faces conservatively-emitted.
                    for c in std::iter::once(coord).chain(neighbor_coords(coord)) {
                        spawn_mesh_lod0_if_ready(world, jobs, registry, c);
                    }
                }
                None => {
                    // Region file existed but the slot was empty —
                    // fall back to procedural gen. Snapshot neighbours
                    // so the gen worker's initial BFS uses real
                    // boundary data instead of all-`None` sentinels.
                    let neighbors = gather_neighbors(world, coord);
                    jobs.spawn_gen(coord, generator.clone(), registry.clone(), neighbors);
                }
            },
            PersistResult::Saved { coord } => {
                if let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get_mut(&coord) {
                    meta.modified = false;
                }
            }
        }
    }
}

/// Gather Arc-shared snapshots of each loaded neighbour's `PalettedChunk`
/// in Face order. Slots that aren't `Stored` come back as `None`.
pub fn gather_neighbors(world: &World, c: ChunkCoord) -> [Option<Arc<PalettedChunk>>; 6] {
    let coords = neighbor_coords(c);
    let mut out: [Option<Arc<PalettedChunk>>; 6] = Default::default();
    for (i, nc) in coords.iter().enumerate() {
        if let Some(ChunkSlot::Stored { data, .. }) = world.chunks.get(nc) {
            // `data` is already `Arc<PalettedChunk>` — cheap atomic
            // refcount bump, no 50 KB deep copy. This was the
            // single biggest hot path in the v0.1.31 samply profile
            // (~49 chunk clones per generated chunk).
            out[i] = Some(data.clone());
        }
    }
    out
}
