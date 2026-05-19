//! The active set of loaded chunks — a sparse, infinite grid keyed by
//! [`ChunkCoord`].
//!
//! The map is *not* an ECS resource because chunk data is too cache-hot and
//! touched by worker threads. Keeping it in a plain `HashMap` here lets us
//! pass it to systems by mutable reference without fighting `hecs`.

use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::{ChunkDirty, ChunkMeta, ChunkState, PalettedChunk};
use crate::voxel::coords::{BlockPos, ChunkCoord, CHUNK_DIM_U};
use glam::IVec3;
use std::collections::HashMap;

/// State of a single slot in the chunk map.
///
/// `Pending` means a gen job has been spawned but the result hasn't arrived
/// yet — the streaming system uses this to avoid re-spawning the same job.
pub enum ChunkSlot {
    Pending,
    Stored { data: PalettedChunk, meta: ChunkMeta },
}

/// The world: every currently-loaded chunk, the seed used to regenerate
/// from disk, and the shared block registry. The registry is co-located
/// because every chunk operation (meshing, lighting, physics) needs it.
pub struct World {
    pub chunks: HashMap<ChunkCoord, ChunkSlot>,
    pub registry: BlockRegistry,
    pub seed: u64,
}

impl World {
    /// Build a fresh, empty world. No chunks are pre-loaded; the streaming
    /// system will populate `chunks` on demand.
    pub fn new(seed: u64) -> Self {
        Self {
            chunks: HashMap::new(),
            registry: BlockRegistry::new(),
            seed,
        }
    }

    /// Look up the block at `pos`. Returns `None` if the containing chunk
    /// is not yet loaded (either absent from the map or still `Pending`).
    pub fn get_block(&self, pos: BlockPos) -> Option<Block> {
        let c = pos.to_chunk();
        match self.chunks.get(&c)? {
            ChunkSlot::Pending => None,
            ChunkSlot::Stored { data, .. } => Some(data.get(pos.to_local())),
        }
    }

    /// `true` if the chunk has data ready to read (i.e. is `Stored`).
    pub fn is_loaded(&self, c: ChunkCoord) -> bool {
        matches!(self.chunks.get(&c), Some(ChunkSlot::Stored { .. }))
    }

    /// Install a freshly-generated chunk at `c`. Marks the chunk as
    /// `Generated` + `mesh-dirty` so the next pass through the schedule
    /// will spawn a mesh job.
    pub fn insert(&mut self, c: ChunkCoord, data: PalettedChunk) {
        let meta = ChunkMeta {
            state: ChunkState::Generated,
            dirty: ChunkDirty {
                mesh: true,
                light: false,
            },
            ..Default::default()
        };
        self.chunks.insert(c, ChunkSlot::Stored { data, meta });
    }

    /// Overwrite the block at `pos` with `new_block` and mark every chunk
    /// that needs to be re-meshed as a consequence.
    ///
    /// The returned vector contains:
    ///
    /// 1. The chunk containing `pos`, always.
    /// 2. Each face-adjacent neighbour chunk *if* `pos` sits on the
    ///    corresponding edge of its chunk — those neighbours' boundary
    ///    faces are affected by the edit.
    ///
    /// Chunks that aren't loaded are silently skipped (you can't edit
    /// what hasn't streamed in yet). The chunk's dirty flags are toggled
    /// so the scheduler will queue a relight + remesh on the next pass.
    pub fn set_block(&mut self, pos: BlockPos, new_block: Block) -> Vec<ChunkCoord> {
        let chunk_coord = pos.to_chunk();
        let local = pos.to_local();
        let mut dirty = Vec::new();

        let Some(ChunkSlot::Stored { data, meta }) = self.chunks.get_mut(&chunk_coord) else {
            return dirty;
        };

        // Decompress, edit, re-compress. Cheap at ~50 KB; for bulk edits
        // we'd batch and decompress once, but the player can only edit
        // one block per click so it's fine.
        let mut dense = data.decompress();
        dense.set(local, new_block);
        *data = PalettedChunk::compress(&dense);

        meta.dirty.mesh = true;
        meta.dirty.light = true;
        meta.modified = true;
        meta.state = ChunkState::Generated;
        dirty.push(chunk_coord);

        // Border edits propagate to the neighbour on that side: its
        // boundary face may have changed visibility, so it needs a
        // remesh. We deliberately do NOT cascade `dirty.light` here:
        // every breaking-edit at a boundary would queue up to four
        // extra relights (chunk + face-adjacent neighbours), each
        // decompressing 7 chunks and running a full BFS. That blew
        // the job pool and the wgpu upload path far enough to dip
        // FPS in half on every click.
        //
        // The lighting BFS instead consumes the neighbour's *current*
        // boundary values when it next runs (see
        // `lighting::seed_from_neighbors`). So the chunk we're
        // editing relights correctly using the neighbour's old
        // boundary; the neighbour will catch up the next time it's
        // touched for any reason (new chunk gen, a later edit, etc.).
        // The visible cost is a single-cell-off light discontinuity
        // at the seam that fades on the next relight pass.
        let (lx, ly, lz) = (local.0.x, local.0.y, local.0.z);
        let dim = CHUNK_DIM_U;
        let mut maybe_mark = |this: &mut World, dc: IVec3| {
            let nc = ChunkCoord(chunk_coord.0 + dc);
            if let Some(ChunkSlot::Stored { meta, .. }) = this.chunks.get_mut(&nc) {
                meta.dirty.mesh = true;
                dirty.push(nc);
            }
        };
        if lx == 0 {
            maybe_mark(self, IVec3::new(-1, 0, 0));
        }
        if lx == dim - 1 {
            maybe_mark(self, IVec3::new(1, 0, 0));
        }
        if ly == 0 {
            maybe_mark(self, IVec3::new(0, -1, 0));
        }
        if ly == dim - 1 {
            maybe_mark(self, IVec3::new(0, 1, 0));
        }
        if lz == 0 {
            maybe_mark(self, IVec3::new(0, 0, -1));
        }
        if lz == dim - 1 {
            maybe_mark(self, IVec3::new(0, 0, 1));
        }

        dirty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::PalettedChunk;
    use glam::IVec3;

    #[test]
    fn empty_world_returns_none() {
        let w = World::new(42);
        assert_eq!(w.get_block(BlockPos(IVec3::new(0, 0, 0))), None);
    }

    #[test]
    fn inserted_air_chunk_returns_air() {
        let mut w = World::new(42);
        w.insert(ChunkCoord(IVec3::ZERO), PalettedChunk::all_air());
        assert_eq!(
            w.get_block(BlockPos(IVec3::new(5, 5, 5))),
            Some(Block::Air)
        );
    }
}
