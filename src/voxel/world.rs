//! The active set of loaded chunks — a sparse, infinite grid keyed by
//! [`ChunkCoord`].
//!
//! The map is *not* an ECS resource because chunk data is too cache-hot and
//! touched by worker threads. Keeping it in a plain `HashMap` here lets us
//! pass it to systems by mutable reference without fighting `hecs`.

use crate::mesher::Face;
use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::{
    ChunkDirty, ChunkLightInputs, ChunkMeta, ChunkState, FaceMask, LightState, PalettedChunk,
};
use crate::voxel::coords::{BlockPos, CHUNK_DIM_U, ChunkCoord};
use glam::IVec3;
use std::collections::HashMap;

/// State of a single slot in the chunk map.
///
/// `Pending` means a gen job has been spawned but the result hasn't arrived
/// yet — the streaming system uses this to avoid re-spawning the same job.
pub enum ChunkSlot {
    Pending,
    /// `data` is `Arc<PalettedChunk>` so every consumer (mesh job
    /// neighbour gather, persistence save, edit dispatcher) can
    /// `Arc::clone` instead of deep-copying ~50 KB of palette + bits.
    /// The pre-Arc layout's `data.clone()` showed up as a top hot
    /// stack in the samply profile of the user's first-break
    /// regression: each `gather_neighbors` cloned 6 chunks, and the
    /// Generated handler called `gather_neighbors` seven times per
    /// new chunk → ~49 × 50 KB allocs per generated chunk every
    /// frame. Switching to `Arc` makes those clones O(1) atomic
    /// refcount bumps.
    Stored {
        data: std::sync::Arc<PalettedChunk>,
        meta: ChunkMeta,
    },
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
        let dense = data.decompress();
        let light_inputs = ChunkLightInputs::from_dense(&dense, c, &self.registry);
        self.insert_with_light_inputs(c, data, light_inputs);
    }

    pub fn insert_with_light_inputs(
        &mut self,
        c: ChunkCoord,
        data: PalettedChunk,
        light_inputs: ChunkLightInputs,
    ) {
        // Build the sky-source heightmap once at install time. Decompressing
        // the chunk is ~50 µs (palette + 4-bit unpack) and the scan is
        // 32×32×32 opacity lookups (~50 µs more) — small enough to do
        // inline; PR2/PR3 will move heightmap updates onto the per-edit
        // path so this only runs on initial install.
        let dense = data.decompress();
        let sky_sources =
            crate::lighting::ChunkSkyLightSources::build_from_dense(&dense, c, &self.registry);
        let meta = ChunkMeta {
            state: ChunkState::Generated,
            dirty: ChunkDirty {
                mesh: true,
                light: true,
            },
            light_state: LightState::Unlit,
            sky_sources,
            light_inputs,
            ..Default::default()
        };
        self.chunks.insert(
            c,
            ChunkSlot::Stored {
                data: std::sync::Arc::new(data),
                meta,
            },
        );
    }

    pub fn mark_chunk_unlit(&mut self, coord: ChunkCoord, _reason: &'static str) {
        let Some(ChunkSlot::Stored { meta, .. }) = self.chunks.get_mut(&coord) else {
            return;
        };
        let had_committed_light = matches!(
            meta.light_state,
            LightState::Lit { .. } | LightState::NeedsBorderReconcile
        );
        meta.light_version = meta.light_version.wrapping_add(1);
        meta.light_state = if had_committed_light {
            LightState::NeedsBorderReconcile
        } else {
            LightState::Unlit
        };
        meta.dirty.light = true;
    }

    pub fn queue_relight(&mut self, coord: ChunkCoord) -> Option<u64> {
        let Some(ChunkSlot::Stored { meta, .. }) = self.chunks.get_mut(&coord) else {
            return None;
        };
        if matches!(
            meta.light_state,
            LightState::Lighting { .. } | LightState::Queued
        ) {
            return None;
        }
        meta.light_state = LightState::Lighting {
            version: meta.light_version,
        };
        meta.dirty.light = false;
        Some(meta.light_version)
    }

    pub fn on_chunk_neighbor_available(&mut self, coord: ChunkCoord, face: Face) {
        let neighbor = ChunkCoord(coord.0 + IVec3::from(face.normal()));
        if self.is_loaded(coord) {
            self.mark_chunk_unlit(coord, "neighbor available");
        }
        if self.is_loaded(neighbor) {
            self.mark_chunk_unlit(neighbor, "neighbor available");
        }
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
        let old_block = dense.get(local);
        if old_block == new_block {
            return vec![];
        }
        dense.set(local, new_block);
        *data = std::sync::Arc::new(PalettedChunk::compress(&dense));

        meta.dirty.mesh = true;
        meta.dirty.light = true;
        meta.modified = true;
        meta.state = ChunkState::Generated;
        meta.light_version = meta.light_version.wrapping_add(1);
        meta.light_state = LightState::Unlit;
        meta.unresolved_borders = FaceMask::NONE;
        meta.sky_sources = crate::lighting::ChunkSkyLightSources::build_from_dense(
            &dense,
            chunk_coord,
            &self.registry,
        );
        meta.light_inputs =
            meta.light_inputs
                .rebuild_preserving_surface(&dense, chunk_coord, &self.registry);
        // Bump version so any in-flight mesh job using the pre-edit
        // data gets discarded when it eventually completes (the
        // "block flickers back after editing" bug). The newly-spawned
        // post-edit mesh job will carry this incremented version and
        // be the one that lands.
        meta.mesh_version = meta.mesh_version.wrapping_add(1);
        dirty.push(chunk_coord);

        // Border edits propagate to the neighbour on that side: its
        // boundary face may have changed visibility, so it needs a
        // remesh and a border-aware relight.
        let (lx, ly, lz) = (local.0.x, local.0.y, local.0.z);
        let dim = CHUNK_DIM_U;
        let face_dirs = [
            IVec3::new(-1, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, 0, -1),
            IVec3::new(0, 0, 1),
        ];
        for dc in face_dirs {
            let nc = ChunkCoord(chunk_coord.0 + dc);
            if let Some(ChunkSlot::Stored { meta, .. }) = self.chunks.get_mut(&nc) {
                meta.dirty.mesh = true;
                meta.dirty.light = true;
                meta.light_version = meta.light_version.wrapping_add(1);
                meta.light_state = LightState::NeedsBorderReconcile;
                if !dirty.contains(&nc) {
                    dirty.push(nc);
                }
            }
        }

        // Four-bit block light reaches at most 15 cells. If an emitter
        // is placed/removed near a chunk edge or corner, stale light can
        // live in diagonal neighbours even though their geometry did not
        // change. Mark the loaded chunks intersecting that radius so the
        // inline edit relight solves the whole possible stale-light set.
        const BLOCK_LIGHT_REACH_BLOCKS: i32 = 15;
        let min_light_chunk = BlockPos(pos.0 - IVec3::splat(BLOCK_LIGHT_REACH_BLOCKS)).to_chunk();
        let max_light_chunk = BlockPos(pos.0 + IVec3::splat(BLOCK_LIGHT_REACH_BLOCKS)).to_chunk();
        for cy in min_light_chunk.0.y..=max_light_chunk.0.y {
            for cz in min_light_chunk.0.z..=max_light_chunk.0.z {
                for cx in min_light_chunk.0.x..=max_light_chunk.0.x {
                    let nc = ChunkCoord(IVec3::new(cx, cy, cz));
                    if dirty.contains(&nc) {
                        continue;
                    }
                    if let Some(ChunkSlot::Stored { meta, .. }) = self.chunks.get_mut(&nc) {
                        meta.dirty.mesh = true;
                        meta.dirty.light = true;
                        meta.light_version = meta.light_version.wrapping_add(1);
                        meta.light_state = LightState::NeedsBorderReconcile;
                        dirty.push(nc);
                    }
                }
            }
        }

        let mut maybe_mark_boundary_mesh = |this: &mut World, dc: IVec3| {
            let nc = ChunkCoord(chunk_coord.0 + dc);
            if let Some(ChunkSlot::Stored { meta, .. }) = this.chunks.get_mut(&nc) {
                meta.dirty.mesh = true;
                meta.dirty.light = true;
                meta.light_state = LightState::NeedsBorderReconcile;
                if !dirty.contains(&nc) {
                    dirty.push(nc);
                }
            }
        };
        if lx == 0 {
            maybe_mark_boundary_mesh(self, IVec3::new(-1, 0, 0));
        }
        if lx == dim - 1 {
            maybe_mark_boundary_mesh(self, IVec3::new(1, 0, 0));
        }
        if ly == 0 {
            maybe_mark_boundary_mesh(self, IVec3::new(0, -1, 0));
        }
        if ly == dim - 1 {
            maybe_mark_boundary_mesh(self, IVec3::new(0, 1, 0));
        }
        if lz == 0 {
            maybe_mark_boundary_mesh(self, IVec3::new(0, 0, -1));
        }
        if lz == dim - 1 {
            maybe_mark_boundary_mesh(self, IVec3::new(0, 0, 1));
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
        assert_eq!(w.get_block(BlockPos(IVec3::new(5, 5, 5))), Some(Block::Air));
    }

    /// World::insert must populate the chunk's sky_sources heightmap
    /// so PR2/PR3 can consume it without further plumbing.
    #[test]
    fn insert_populates_sky_sources_from_chunk_data() {
        use crate::voxel::chunk::DenseChunk;
        use crate::voxel::coords::LocalPos;
        use glam::UVec3;

        let mut w = World::new(42);

        // Build a chunk with a single stone layer at local y=10
        // across the whole footprint; everything else is air.
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 10, lz)), Block::Stone);
            }
        }
        let chunk = PalettedChunk::compress(&dense);

        let coord = ChunkCoord(IVec3::ZERO);
        w.insert(coord, chunk);

        let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
            panic!("chunk should be Stored after insert");
        };

        // Every column should report world-y=11 (one above the stone).
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(
                    meta.sky_sources.lowest_source_y(lx, lz),
                    11,
                    "column ({lx},{lz}) should have floor at world y=11",
                );
            }
        }
    }

    /// All-air chunk: heightmap should be entirely NO_SOURCE_FLOOR
    /// (matches the default), but populated rather than default-stub.
    #[test]
    fn insert_populates_sky_sources_even_for_all_air() {
        use crate::lighting::NO_SOURCE_FLOOR;
        use crate::voxel::chunk::DenseChunk;

        let mut w = World::new(42);
        let chunk = PalettedChunk::compress(&DenseChunk::empty());
        let coord = ChunkCoord(IVec3::new(2, 1, -3));

        w.insert(coord, chunk);

        let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
            panic!("chunk should be Stored after insert");
        };

        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(meta.sky_sources.lowest_source_y(lx, lz), NO_SOURCE_FLOOR);
            }
        }
    }

    #[test]
    fn inserted_chunk_starts_unlit() {
        use crate::voxel::chunk::DenseChunk;

        let mut w = World::new(42);
        let coord = ChunkCoord(IVec3::ZERO);
        w.insert(coord, PalettedChunk::compress(&DenseChunk::empty()));
        let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
            panic!("chunk should be Stored after insert");
        };
        assert!(
            matches!(meta.light_state, LightState::Unlit),
            "new chunks must relight before their light is authoritative",
        );
        assert!(meta.dirty.light);
    }

    #[test]
    fn set_block_marks_chunk_unlit_and_bumps_light_version() {
        use crate::voxel::block::Block;
        use crate::voxel::coords::BlockPos;

        let mut w = World::new(42);
        let coord = ChunkCoord(IVec3::ZERO);
        w.insert(coord, PalettedChunk::all_air());
        let before = match w.chunks.get(&coord).unwrap() {
            ChunkSlot::Stored { meta, .. } => meta.light_version,
            ChunkSlot::Pending => unreachable!(),
        };
        w.set_block(BlockPos(IVec3::new(1, 1, 1)), Block::Stone);
        let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
            panic!("chunk should be Stored after edit");
        };
        assert_eq!(meta.light_version, before.wrapping_add(1));
        assert!(meta.dirty.light);
        assert!(matches!(meta.light_state, LightState::Unlit));
    }

    #[test]
    fn set_block_marks_face_neighbors_for_light_reconcile() {
        use crate::voxel::block::Block;
        use crate::voxel::coords::BlockPos;

        let mut w = World::new(42);
        let center = ChunkCoord(IVec3::ZERO);
        w.insert(center, PalettedChunk::all_air());
        for dc in [
            IVec3::new(-1, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, 0, -1),
            IVec3::new(0, 0, 1),
        ] {
            w.insert(ChunkCoord(center.0 + dc), PalettedChunk::all_air());
        }

        let returned = w.set_block(BlockPos(IVec3::new(8, 8, 8)), Block::Stone);
        assert_eq!(returned.len(), 7);
        assert_eq!(returned[0], center);

        for dc in [
            IVec3::new(-1, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, 0, -1),
            IVec3::new(0, 0, 1),
        ] {
            let coord = ChunkCoord(center.0 + dc);
            let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
                panic!("neighbor should be Stored");
            };
            assert!(meta.dirty.light, "{coord:?} was not queued for relight");
            assert!(
                matches!(meta.light_state, LightState::NeedsBorderReconcile),
                "{coord:?} did not enter border reconcile"
            );
        }
    }

    #[test]
    fn set_block_marks_diagonal_chunks_inside_block_light_radius() {
        use crate::voxel::block::Block;
        use crate::voxel::coords::BlockPos;

        let mut w = World::new(42);
        let center = ChunkCoord(IVec3::ZERO);
        let diagonal = ChunkCoord(IVec3::new(-1, -1, -1));
        w.insert(center, PalettedChunk::all_air());
        w.insert(diagonal, PalettedChunk::all_air());

        let returned = w.set_block(BlockPos(IVec3::ZERO), Block::Torch);

        assert!(
            returned.contains(&diagonal),
            "diagonal chunk in block-light radius should be relit"
        );
        let ChunkSlot::Stored { meta, .. } = w.chunks.get(&diagonal).unwrap() else {
            panic!("diagonal should be Stored");
        };
        assert!(meta.dirty.light);
        assert!(meta.dirty.mesh);
        assert!(matches!(meta.light_state, LightState::NeedsBorderReconcile));
    }

    #[test]
    fn mark_chunk_unlit_preserves_committed_light_as_reconcile_input() {
        use crate::voxel::chunk::DenseChunk;

        let mut w = World::new(42);
        let coord = ChunkCoord(IVec3::ZERO);
        w.insert(coord, PalettedChunk::compress(&DenseChunk::empty()));
        let ChunkSlot::Stored { meta, .. } = w.chunks.get_mut(&coord).unwrap() else {
            panic!("chunk should be Stored");
        };
        meta.light_state = LightState::Lit { version: 9 };

        w.mark_chunk_unlit(coord, "test");

        let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
            panic!("chunk should be Stored");
        };
        assert!(
            matches!(meta.light_state, LightState::NeedsBorderReconcile),
            "committed light should remain usable while queued for reconcile"
        );
    }
}
