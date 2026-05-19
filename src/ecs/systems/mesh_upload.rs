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

                // Cascade `dirty.light` to the six face neighbours that
                // already exist, AND to this chunk itself. The chunk
                // was initially lit with `[None; 6]` neighbours by
                // `spawn_gen` — fine for surface chunks, but for an
                // underground chunk the "no above-neighbour ⇒ assume
                // full sky" fallback bakes in `sky_light = 15` on
                // every column. The relight pump downstream picks up
                // these dirty flags and rebuilds lighting with the
                // real neighbour boundaries, which the BFS's
                // `seed_from_neighbors` consumes correctly.
                if let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get_mut(&coord) {
                    meta.dirty.light = true;
                }
                for nc in neighbor_coords(coord) {
                    if let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get_mut(&nc) {
                        meta.dirty.light = true;
                    }
                }

                // Spawn a LOD0 mesh job for this chunk plus any neighbour
                // that's already loaded — generating a new chunk can
                // reveal previously-hidden faces on its neighbours.
                for c in std::iter::once(coord).chain(neighbor_coords(coord)) {
                    if let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get(&c) {
                        let data_arc = data.clone();
                        let version = meta.mesh_version;
                        let neighbors = gather_neighbors(world, c);
                        jobs.spawn_mesh_lod0(c, data_arc, neighbors, registry.clone(), version);
                    }
                }
                // Also spawn LOD1 and LOD2 jobs for the new chunk so the
                // far-distance render has something to draw. These don't
                // need neighbours (boundary precision is invisible at
                // distance), so they run independently.
                if let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get(&coord) {
                    let data_arc = data.clone();
                    let version = meta.mesh_version;
                    jobs.spawn_mesh_lod(coord, 1, data_arc.clone(), registry.clone(), version);
                    jobs.spawn_mesh_lod(coord, 2, data_arc, registry.clone(), version);
                }
            }
            JobResult::Meshed { coord, lod, mesh, version } => {
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
                }
            }
            JobResult::Relit { coord, data, changed_faces } => {
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
                    *cur = data_arc.clone();
                    meta.dirty = ChunkDirty {
                        mesh: true,
                        light: false,
                    };
                    meta.state = ChunkState::Generated;
                    // Relight rewrote the chunk's light bytes, so the
                    // mesh data has effectively changed — bump
                    // version so any in-flight pre-relight mesh job
                    // for this chunk gets discarded on completion.
                    meta.mesh_version = meta.mesh_version.wrapping_add(1);
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
                    // Install the loaded chunk and queue all three LODs.
                    world.insert(coord, data.clone());
                    // Cascade `dirty.light` the same way as Generated:
                    // the freshly-loaded chunk's neighbours may have been
                    // lit before this chunk existed and have stale
                    // boundary values, and this chunk's saved lighting
                    // may itself be stale relative to current neighbours.
                    if let Some(ChunkSlot::Stored { meta, .. }) =
                        world.chunks.get_mut(&coord)
                    {
                        meta.dirty.light = true;
                    }
                    for nc in neighbor_coords(coord) {
                        if let Some(ChunkSlot::Stored { meta, .. }) =
                            world.chunks.get_mut(&nc)
                        {
                            meta.dirty.light = true;
                        }
                    }
                    let data_arc = Arc::new(data);
                    let neighbors = gather_neighbors(world, coord);
                    let version = world
                        .chunks
                        .get(&coord)
                        .and_then(|s| match s {
                            ChunkSlot::Stored { meta, .. } => Some(meta.mesh_version),
                            _ => None,
                        })
                        .unwrap_or(0);
                    jobs.spawn_mesh_lod0(
                        coord,
                        data_arc.clone(),
                        neighbors,
                        registry.clone(),
                        version,
                    );
                    jobs.spawn_mesh_lod(coord, 1, data_arc.clone(), registry.clone(), version);
                    jobs.spawn_mesh_lod(coord, 2, data_arc, registry.clone(), version);
                }
                None => {
                    // Region file existed but the slot was empty —
                    // fall back to procedural gen.
                    jobs.spawn_gen(coord, generator.clone(), registry.clone());
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
