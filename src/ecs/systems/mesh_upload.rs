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
use crate::voxel::chunk::PalettedChunk;
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
            JobResult::Generated { coord, data } => {
                // Install the new chunk first so neighbour mesh jobs can
                // see it.
                world.insert(coord, data);

                // Hand the new chunk to the graph engine. on_chunk_loaded
                // enqueues sky sources, emissives, and neighbour boundary
                // cells; the engine's next tick spreads them. Replaces the
                // old "mark_below_dirty + self+6 cascade" bandaids that
                // sat here before the graph-engine cutover — those tried
                // to compensate for the per-chunk BFS not knowing about
                // out-of-order neighbour arrivals.
                world.on_chunk_loaded(coord);

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
                    if let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get(&c) {
                        let data_arc = data.clone();
                        let version = meta.mesh_version;
                        let neighbors = gather_neighbors(world, c);
                        jobs.spawn_mesh_lod0(c, data_arc, neighbors, registry.clone(), version);
                    }
                }
            }
            JobResult::Meshed { coord, lod, mesh, version, light_volume } => {
                // Drop the upload if the chunk has been re-edited since
                // this mesh job was spawned. Without this check, a
                // slow streaming mesh job can complete after a fast
                // edit-triggered mesh and overwrite the GPU buffer
                // with stale geometry — the visible "block flickers
                // back for a moment" artefact the user reported.
                let current = match world.chunks.get(&coord) {
                    Some(ChunkSlot::Stored { meta, .. }) => meta.mesh_version,
                    _ => 0,
                };
                if version >= current {
                    renderer.upload_chunk_mesh(coord, lod, &mesh);
                    if let Some(blob) = light_volume {
                        renderer.upload_chunk_light_volume(coord, blob.as_ref());
                    }
                }
            }
            #[cfg(feature = "legacy-lighting")]
            JobResult::Relit { coord, data, changed_faces, light_volume } => {
                // Push the light volume to the GPU FIRST so the chunk's bind
                // group picks up the new lighting on the next draw — even if
                // the mesh re-spawn lags.
                renderer.upload_chunk_light_volume(coord, light_volume.as_ref());
                // Swap the freshly-relit chunk into the World and reset
                // its dirty flags. Re-mesh **all three** LOD levels so
                // distant terrain reflects the new light values too —
                // without re-spawning LOD1/LOD2 here, far-away
                // chunks would keep showing whatever brightness was
                // baked in at initial gen even after their lighting
                // converged.
                use crate::voxel::chunk::{ChunkDirty, ChunkState};
                let data_arc = Arc::new(data);
                if let Some(ChunkSlot::Stored {
                    data: cur,
                    meta,
                }) = world.chunks.get_mut(&coord)
                {
                    // Preserve any `dirty.light` mark added *during the
                    // in-flight window* — between the pump clearing the
                    // flag (`relight_pump`) and this result landing.
                    // Such marks come from a cascade chain or a
                    // `+Y`-arrival mark that reached this chunk after
                    // its relight was dispatched but before it
                    // returned, and they want a fresh relight against
                    // even-newer neighbour state. Unconditionally
                    // resetting to `false` here used to drop those
                    // marks, leaving stale lighting in cells the
                    // cascade had already passed (visible as the
                    // "lit/dark patchwork" reported on adjacent
                    // chunks under deep water).
                    let still_dirty = meta.dirty.light;
                    *cur = data_arc.clone();
                    meta.dirty = ChunkDirty {
                        mesh: true,
                        light: still_dirty,
                    };
                    meta.state = ChunkState::Generated;
                    // Deliberately NOT bumping `mesh_version` here.
                    // Relight only rewrites the per-cell light bytes —
                    // geometry stays the same — so a pre-relight mesh
                    // (with slightly stale lighting baked into vertex
                    // colours) still renders correctly enough. Bumping
                    // the version meant the original `Generated`
                    // handler's mesh job got *rejected* on completion
                    // and the chunk had no mesh on the GPU at all
                    // until the cascade's follow-up mesh arrived —
                    // visible as huge sky-shader-coloured holes where
                    // streamed-in chunks should be. The version tag
                    // still fires on `set_block` (geometry change),
                    // which is the case that actually needs it.
                }
                // Bounded cascade: only mark the face neighbours whose
                // boundary actually changed as `dirty.light`. Most
                // relights downstream of a cascade produce no further
                // change, so propagation naturally terminates in
                // `O(diameter)` iterations rather than blowing up
                // exponentially.
                let nbrs = neighbor_coords(coord);
                for (i, changed) in changed_faces.iter().enumerate() {
                    if !changed {
                        continue;
                    }
                    if let Some(ChunkSlot::Stored { meta, .. }) =
                        world.chunks.get_mut(&nbrs[i])
                    {
                        meta.dirty.light = true;
                    }
                }
                // Only re-mesh LOD0 — distant LOD chunks sample one
                // light value per column (from the topmost air cell),
                // which almost never changes meaningfully when a deep
                // chunk re-lights itself. Re-meshing LOD1/LOD2 on
                // every relight was ~3× the mesh work per cascade
                // step with no visible benefit at distance.
                let neighbors = gather_neighbors(world, coord);
                let version = world
                    .chunks
                    .get(&coord)
                    .and_then(|s| match s {
                        ChunkSlot::Stored { meta, .. } => Some(meta.mesh_version),
                        _ => None,
                    })
                    .unwrap_or(0);
                jobs.spawn_mesh_lod0(coord, data_arc, neighbors, registry.clone(), version);
            }
            JobResult::LoadedFromDisk { coord, data } => match data {
                Some(data) => {
                    world.insert(coord, data);
                    world.on_chunk_loaded(coord);
                    // Same neighbour-remesh strategy as Generated.
                    // The parallel `spawn_load` was the trigger for
                    // the out-of-order arrival bug, and Loaded is
                    // where most of those arrivals come from.
                    for c in std::iter::once(coord).chain(neighbor_coords(coord)) {
                        if let Some(ChunkSlot::Stored { data, meta }) =
                            world.chunks.get(&c)
                        {
                            let data_arc = data.clone();
                            let version = meta.mesh_version;
                            let neighbors = gather_neighbors(world, c);
                            jobs.spawn_mesh_lod0(
                                c,
                                data_arc,
                                neighbors,
                                registry.clone(),
                                version,
                            );
                        }
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
/// At 4 chunks per frame a couple thousand pending chunks converge in
/// ~5 seconds at 120 fps without dropping frames.
#[cfg(feature = "legacy-lighting")]
pub fn relight_pump(
    world: &mut World,
    jobs: &Jobs,
    registry: &Arc<BlockRegistry>,
) -> usize {
    const RELIGHT_BUDGET: usize = 16;
    // Snapshot the candidate coords up-front so we don't hold an
    // immutable borrow over the loop body's `get_mut` + `spawn`.
    // Also count the *total* dirty set for the HUD perf readout.
    let mut total_dirty = 0usize;
    let mut candidates: Vec<ChunkCoord> = Vec::new();
    for (c, slot) in world.chunks.iter() {
        if let ChunkSlot::Stored { meta, .. } = slot
            && meta.dirty.light
        {
            total_dirty += 1;
            if candidates.len() < RELIGHT_BUDGET {
                candidates.push(*c);
            }
        }
    }
    for c in candidates {
        // Clone the chunk data + gather neighbour snapshots, then
        // clear the flag so the next frame's scan doesn't re-queue
        // this chunk while the worker is still busy.
        let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get_mut(&c) else {
            continue;
        };
        meta.dirty.light = false;
        let data_arc = data.clone();
        let neighbors = gather_neighbors(world, c);
        jobs.spawn_relight(c, data_arc, neighbors, registry.clone());
    }
    return total_dirty;
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
                    world.insert(coord, data);
                    world.on_chunk_loaded(coord);
                    // Mark `dirty.light = true` on the just-loaded
                    // chunk. Saved chunks can carry stale
                    // sky_light / block_light values when an
                    // autosave races a player edit: `set_block`
                    // bumps the data but leaves the light arrays
                    // for the BFS to recompute later, and the
                    // autosave/Drop flush writes whichever state is
                    // current. On reload nothing re-runs the BFS,
                    // so the chunk renders with stale (often zero)
                    // light around the edit — which fades to the
                    // cave-fog gray at distance and shows as the
                    // "flat fog plane where I modified terrain"
                    // bug. The relight pump picks this up and
                    // converges over a few frames.
                    #[cfg(feature = "legacy-lighting")]
                    if let Some(ChunkSlot::Stored { meta, .. }) =
                        world.chunks.get_mut(&coord)
                    {
                        meta.dirty.light = true;
                    }
                    // Re-mesh self + already-Stored neighbours, so
                    // out-of-order arrivals don't leave boundary
                    // faces conservatively-emitted.
                    for c in std::iter::once(coord).chain(neighbor_coords(coord)) {
                        if let Some(ChunkSlot::Stored { data, meta }) =
                            world.chunks.get(&c)
                        {
                            let data_arc = data.clone();
                            let version = meta.mesh_version;
                            let neighbors = gather_neighbors(world, c);
                            jobs.spawn_mesh_lod0(
                                c,
                                data_arc,
                                neighbors,
                                registry.clone(),
                                version,
                            );
                        }
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

/// Scan loaded chunks for `light_gpu_dirty` and re-upload their 3D
/// light textures. Bounded to `UPLOAD_BUDGET` chunks per frame so a
/// big convergence wave doesn't saturate PCIe bandwidth.
pub fn upload_dirty_light_volumes(
    world: &mut crate::voxel::world::World,
    renderer: &mut crate::render::Renderer,
) {
    use crate::voxel::world::ChunkSlot;
    use crate::voxel::chunk::Neighbors;
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
        // Gather neighbour Arc refs then decompress them — Neighbors<'_>
        // holds &DenseChunk refs so we need the decompressed values to
        // outlive the borrow. This mirrors the pattern in app.rs's edit path.
        let neighbor_arcs = gather_neighbors(world, coord);
        let neighbor_dense: Vec<Option<crate::voxel::chunk::DenseChunk>> = neighbor_arcs
            .iter()
            .map(|opt| opt.as_ref().map(|p| p.decompress()))
            .collect();
        let neighbor_refs: [Option<&crate::voxel::chunk::DenseChunk>; 6] = [
            neighbor_dense[0].as_ref(),
            neighbor_dense[1].as_ref(),
            neighbor_dense[2].as_ref(),
            neighbor_dense[3].as_ref(),
            neighbor_dense[4].as_ref(),
            neighbor_dense[5].as_ref(),
        ];
        let ns = Neighbors { chunks: neighbor_refs };
        let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get_mut(&coord) else { continue };
        let dense = data.decompress();
        let blob = crate::voxel::chunk::build_light_volume_blob(&dense, &ns);
        renderer.upload_chunk_light_volume(coord, blob.as_ref());
        meta.light_gpu_dirty = false;
        uploaded += 1;
    }
    let _ = uploaded;
}
