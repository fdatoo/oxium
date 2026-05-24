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
#[path = "world_tests.rs"]
mod tests;
