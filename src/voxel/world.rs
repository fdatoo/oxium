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
    Stored { data: std::sync::Arc<PalettedChunk>, meta: ChunkMeta },
}

/// The world: every currently-loaded chunk, the seed used to regenerate
/// from disk, and the shared block registry. The registry is co-located
/// because every chunk operation (meshing, lighting, physics) needs it.
pub struct World {
    pub chunks: HashMap<ChunkCoord, ChunkSlot>,
    pub registry: BlockRegistry,
    pub seed: u64,
    /// Per-voxel graph light propagator. PR2 ships this as an idle
    /// skeleton; nothing calls `light_engine.tick` yet. PR3 wires
    /// `set_block` and chunk-load to enqueue work here, and the
    /// frame loop ticks the engine each frame.
    pub light_engine: crate::lighting::LightEngine,
}

impl World {
    /// Build a fresh, empty world. No chunks are pre-loaded; the streaming
    /// system will populate `chunks` on demand.
    pub fn new(seed: u64) -> Self {
        Self {
            chunks: HashMap::new(),
            registry: BlockRegistry::new(),
            seed,
            light_engine: crate::lighting::LightEngine::default(),
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
        // Build the sky-source heightmap once at install time. Decompressing
        // the chunk is ~50 µs (palette + 4-bit unpack) and the scan is
        // 32×32×32 opacity lookups (~50 µs more) — small enough to do
        // inline; PR2/PR3 will move heightmap updates onto the per-edit
        // path so this only runs on initial install.
        let dense = data.decompress();
        let sky_sources = crate::lighting::ChunkSkyLightSources::build_from_dense(
            &dense, c, &self.registry,
        );
        let meta = ChunkMeta {
            state: ChunkState::Generated,
            dirty: ChunkDirty {
                mesh: true,
                ..Default::default()
            },
            sky_sources,
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

    /// Engine entry point for "this block just changed". Called from
    /// `set_block`. Records the (old, new) tuple in the engine's
    /// pending-changes side-table and queues the position on every
    /// channel; the next `light_engine_tick` consumes it.
    pub fn on_block_changed(
        &mut self,
        pos: crate::voxel::coords::BlockPos,
        old_block: crate::voxel::block::Block,
        new_block: crate::voxel::block::Block,
    ) {
        self.light_engine.enqueue_block_change(pos, old_block, new_block);
    }

    /// Engine entry point for "a chunk just installed". Called from
    /// the `JobResult::Generated` and `JobResult::LoadedFromDisk`
    /// handlers in `mesh_upload`. Enqueues every sky-source cell and
    /// every emissive block in the chunk as increase ops, plus each
    /// neighbour's boundary cells (so the engine can spread our
    /// freshly-loaded chunk's light across the seam without a
    /// special seed pass).
    pub fn on_chunk_loaded(&mut self, coord: crate::voxel::coords::ChunkCoord) {
        use crate::voxel::block::Block;
        use crate::voxel::coords::{BlockPos, LocalPos, CHUNK_DIM, CHUNK_DIM_U};
        use glam::{IVec3, UVec3};

        // Destructure to split the borrow: chunks + registry are read,
        // light_engine is written, and Rust can't prove they don't alias
        // through &mut self if we use method calls.
        let Self { chunks, registry, light_engine, .. } = self;

        // Snapshot the chunk's blocks + heightmap for enqueuing.
        let Some(ChunkSlot::Stored { data, meta }) = chunks.get(&coord) else { return };
        let dense = data.decompress();
        let sky_sources = meta.sky_sources.clone();
        let chunk_bottom_y = coord.0.y * CHUNK_DIM;

        // Enqueue every sky-source cell at level 15.
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                let lsy = sky_sources.lowest_source_y(lx, lz);
                let start_ly: i32 = if lsy == crate::lighting::NO_SOURCE_FLOOR {
                    0
                } else {
                    (lsy - chunk_bottom_y).max(0)
                };
                for ly in (start_ly as u32)..CHUNK_DIM_U {
                    let pos = BlockPos(IVec3::new(
                        coord.0.x * CHUNK_DIM + lx as i32,
                        chunk_bottom_y + ly as i32,
                        coord.0.z * CHUNK_DIM + lz as i32,
                    ));
                    light_engine.sky.increase.push(crate::lighting::queue::QueueEntry {
                        pos, from_level: 15, propagation_mask: 0,
                    });
                }
            }
        }

        // Enqueue every emissive cell at its emission level.
        for lz in 0..CHUNK_DIM_U {
            for ly in 0..CHUNK_DIM_U {
                for lx in 0..CHUNK_DIM_U {
                    let idx = LocalPos(UVec3::new(lx, ly, lz)).to_index();
                    let info = registry.info(dense.blocks[idx]);
                    if info.emission[0] > 0 || info.emission[1] > 0 || info.emission[2] > 0 {
                        let pos = BlockPos(IVec3::new(
                            coord.0.x * CHUNK_DIM + lx as i32,
                            chunk_bottom_y + ly as i32,
                            coord.0.z * CHUNK_DIM + lz as i32,
                        ));
                        for ch_i in 0..3 {
                            if info.emission[ch_i] > 0 {
                                light_engine.block_rgb[ch_i].increase.push(
                                    crate::lighting::queue::QueueEntry {
                                        pos,
                                        from_level: info.emission[ch_i],
                                        propagation_mask: 0,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }

        // Re-enqueue face-adjacent neighbour boundary cells so the engine
        // spreads their existing light across the new seam. For each face,
        // walk the 32×32 boundary slice on the neighbour's side and push
        // its current sky_light + block_rgb levels as increase ops.
        let face_offsets: [IVec3; 6] = [
            IVec3::new( 1, 0, 0), IVec3::new(-1, 0, 0),
            IVec3::new( 0, 1, 0), IVec3::new( 0,-1, 0),
            IVec3::new( 0, 0, 1), IVec3::new( 0, 0,-1),
        ];
        for face_off in face_offsets {
            let nc = crate::voxel::coords::ChunkCoord(coord.0 + face_off);
            let Some(ChunkSlot::Stored { data: ndata, .. }) = chunks.get(&nc) else { continue };
            let ndense = ndata.decompress();
            // Walk the slice of the neighbour adjacent to the seam.
            // For face = +X, neighbour's slice is at lx=0; for -X, lx=31; etc.
            // We push EVERY cell in the neighbour's slice — the queue's
            // bucket sort ensures redundant pushes coalesce in the right order.
            for u in 0..CHUNK_DIM_U {
                for v in 0..CHUNK_DIM_U {
                    let (nlx, nly, nlz) = match (face_off.x, face_off.y, face_off.z) {
                        ( 1,  0,  0) => (0,                 v, u),
                        (-1,  0,  0) => (CHUNK_DIM_U - 1,   v, u),
                        ( 0,  1,  0) => (u,                 0,                 v),
                        ( 0, -1,  0) => (u,                 CHUNK_DIM_U - 1,   v),
                        ( 0,  0,  1) => (u, v, 0),
                        ( 0,  0, -1) => (u, v, CHUNK_DIM_U - 1),
                        _ => unreachable!(),
                    };
                    let nidx = LocalPos(UVec3::new(nlx, nly, nlz)).to_index();
                    let sky = ndense.sky_light[nidx];
                    let (r, g, b) = crate::voxel::chunk::unpack_rgb(ndense.block_rgb[nidx]);
                    let npos = BlockPos(IVec3::new(
                        nc.0.x * CHUNK_DIM + nlx as i32,
                        nc.0.y * CHUNK_DIM + nly as i32,
                        nc.0.z * CHUNK_DIM + nlz as i32,
                    ));
                    if sky > 0 {
                        light_engine.sky.increase.push(crate::lighting::queue::QueueEntry {
                            pos: npos, from_level: sky, propagation_mask: 0,
                        });
                    }
                    for (ch_i, level) in [r, g, b].iter().enumerate() {
                        if *level > 0 {
                            light_engine.block_rgb[ch_i].increase.push(
                                crate::lighting::queue::QueueEntry {
                                    pos: npos, from_level: *level, propagation_mask: 0,
                                },
                            );
                        }
                    }
                }
            }
            let _ = Block::Air;  // silence unused warning
        }
    }

    /// Engine entry point for "drain up to `budget` ops this frame".
    /// Called from `App::update` once per frame. Destructures `&mut self`
    /// to split the field-aliasing problem (`light_engine` is a field
    /// of `World`, so a method on `LightEngine` can't take `&mut World`).
    pub fn light_engine_tick(&mut self, budget: usize) {
        let Self { chunks, registry, light_engine, .. } = self;
        light_engine.tick_with(chunks, registry, budget);
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
        #[cfg(feature = "legacy-lighting")]
        { meta.dirty.light = true; }
        meta.modified = true;
        meta.state = ChunkState::Generated;
        // Bump version so any in-flight mesh job using the pre-edit
        // data gets discarded when it eventually completes (the
        // "block flickers back after editing" bug). The newly-spawned
        // post-edit mesh job will carry this incremented version and
        // be the one that lands.
        meta.mesh_version = meta.mesh_version.wrapping_add(1);
        dirty.push(chunk_coord);

        // Notify the graph engine. Engine tick (next frame) processes
        // the change. Today's BFS path (dirty.light above) continues
        // to run too — it's still authoritative until Task 9 flips
        // the GPU upload source. After the cutover, the engine is the
        // sole writer.
        self.light_engine.enqueue_block_change(pos, old_block, new_block);

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
                    meta.sky_sources.lowest_source_y(lx, lz), 11,
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

    /// World::new must construct a default LightEngine alongside the
    /// chunks map and registry. The engine is idle (no queued work)
    /// on a fresh world — PR3 will start feeding it.
    #[test]
    fn new_world_has_idle_light_engine() {
        let w = World::new(42);
        assert!(
            w.light_engine.is_idle(),
            "freshly-constructed World must have an idle LightEngine",
        );
    }

    #[test]
    fn on_block_changed_queues_work_in_engine() {
        use crate::voxel::block::Block;
        use crate::voxel::coords::BlockPos;
        let mut w = World::new(42);
        assert!(w.light_engine.is_idle());
        w.on_block_changed(BlockPos(IVec3::new(0, 0, 0)), Block::Air, Block::Torch);
        assert!(!w.light_engine.is_idle());
        assert!(w.light_engine.pending_block_changes.contains_key(&BlockPos(IVec3::new(0, 0, 0))));
    }

    #[test]
    fn on_chunk_loaded_enqueues_sky_sources_for_air_chunk() {
        use crate::voxel::chunk::DenseChunk;
        let mut w = World::new(42);
        let chunk = PalettedChunk::compress(&DenseChunk::empty());
        let coord = ChunkCoord(IVec3::ZERO);
        w.insert(coord, chunk);
        w.on_chunk_loaded(coord);
        // All-air chunk: every cell is a sky source within the chunk;
        // 32^3 = 32768 cells should land in the sky channel's queue.
        assert!(!w.light_engine.sky.increase.is_empty());
    }

    #[test]
    fn light_engine_tick_drains_queued_work() {
        use crate::voxel::block::Block;
        use crate::voxel::chunk::DenseChunk;
        use crate::voxel::coords::{BlockPos, LocalPos};
        use glam::UVec3;
        let mut w = World::new(42);
        // Build a chunk with a torch at center.
        let mut dense = DenseChunk::empty();
        dense.set(LocalPos(UVec3::new(16, 16, 16)), Block::Torch);
        let chunk = PalettedChunk::compress(&dense);
        let coord = ChunkCoord(IVec3::ZERO);
        w.insert(coord, chunk);
        w.on_chunk_loaded(coord);
        assert!(!w.light_engine.is_idle());
        // Drain in 50k-op slices (matching the planned production budget)
        // until the engine is idle. on_chunk_loaded seeds 32^3 sky entries
        // which can produce O(32^3 × 15) propagation ops; 20 ticks is
        // sufficient in practice but we cap at 100 to catch infinite loops.
        let mut iters = 0usize;
        while !w.light_engine.is_idle() && iters < 100 {
            w.light_engine_tick(50_000);
            iters += 1;
        }
        assert!(w.light_engine.is_idle(), "tick should drain all queued work given big enough budget");
        if let Some(ChunkSlot::Stored { meta, .. }) = w.chunks.get(&coord) {
            assert!(meta.light_gpu_dirty, "engine writes should set light_gpu_dirty");
        }
        let _ = BlockPos(IVec3::ZERO);  // silence unused import warning
    }
}
