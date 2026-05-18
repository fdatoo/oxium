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
use crate::render::Renderer;
use crate::voxel::block::BlockRegistry;
use crate::voxel::chunk::PalettedChunk;
use crate::voxel::coords::ChunkCoord;
use crate::voxel::world::{ChunkSlot, World};
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

                // Spawn a mesh job for this chunk plus any neighbour that's
                // already loaded — generating a new chunk can reveal
                // previously-hidden faces on its already-meshed neighbours.
                for c in std::iter::once(coord).chain(neighbor_coords(coord)) {
                    if let Some(ChunkSlot::Stored { data, .. }) = world.chunks.get(&c) {
                        let data_arc = Arc::new(data.clone());
                        let neighbors = gather_neighbors(world, c);
                        jobs.spawn_mesh_lod0(c, data_arc, neighbors, registry.clone());
                    }
                }
            }
            JobResult::Meshed {
                coord,
                lod: _,
                mesh,
            } => {
                renderer.upload_chunk_mesh(coord, &mesh);
            }
            JobResult::Relit { coord, data } => {
                // Swap the freshly-relit chunk into the World and reset
                // its dirty flags. Then queue a mesh job so the new
                // light bytes reach the GPU.
                use crate::voxel::chunk::{ChunkDirty, ChunkState};
                if let Some(ChunkSlot::Stored {
                    data: cur,
                    meta,
                }) = world.chunks.get_mut(&coord)
                {
                    *cur = data.clone();
                    meta.dirty = ChunkDirty {
                        mesh: true,
                        light: false,
                    };
                    meta.state = ChunkState::Generated;
                }
                let data_arc = Arc::new(data);
                let neighbors = gather_neighbors(world, coord);
                jobs.spawn_mesh_lod0(coord, data_arc, neighbors, registry.clone());
            }
        }
    }
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

/// Gather Arc-shared snapshots of each loaded neighbour's `PalettedChunk`
/// in Face order. Slots that aren't `Stored` come back as `None`.
pub fn gather_neighbors(world: &World, c: ChunkCoord) -> [Option<Arc<PalettedChunk>>; 6] {
    let coords = neighbor_coords(c);
    let mut out: [Option<Arc<PalettedChunk>>; 6] = Default::default();
    for (i, nc) in coords.iter().enumerate() {
        if let Some(ChunkSlot::Stored { data, .. }) = world.chunks.get(nc) {
            out[i] = Some(Arc::new(data.clone()));
        }
    }
    out
}
