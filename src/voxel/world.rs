//! The active set of loaded chunks — a sparse, infinite grid keyed by
//! [`ChunkCoord`].
//!
//! The map is *not* an ECS resource because chunk data is too cache-hot and
//! touched by worker threads. Keeping it in a plain `HashMap` here lets us
//! pass it to systems by mutable reference without fighting `hecs`.

use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::{ChunkDirty, ChunkMeta, ChunkState, PalettedChunk};
use crate::voxel::coords::{BlockPos, ChunkCoord};
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
