//! Background job system: chunk generation, lighting, and meshing run on
//! a `rayon` worker pool; results return via a `crossbeam-channel`.
//!
//! Why decouple producer (worker) from consumer (main thread)?
//!
//! - Workers are CPU-bound and can run in parallel without touching the
//!   GPU. Main thread renders.
//! - Channels are lock-free for the common case (one producer, one
//!   consumer) — fine for our drain-each-frame pattern.
//!
//! The pool deliberately uses N-1 worker threads (where N = available
//! parallelism) so the main rendering thread keeps a CPU core to itself.

use crate::mesher::ChunkMesh;
use crate::voxel::block::BlockRegistry;
use crate::voxel::chunk::{DenseChunk, PalettedChunk};
use crate::voxel::coords::ChunkCoord;
use crate::worldgen::Generator;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::sync::Arc;

/// The output of a single completed job. The main thread matches on this
/// in `ecs::systems::mesh_upload::drain_jobs`.
pub enum JobResult {
    /// A worldgen job finished; `data` is the newly-generated chunk.
    Generated {
        coord: ChunkCoord,
        data: PalettedChunk,
    },
    /// A meshing job finished; `mesh` is ready for GPU upload. `lod` is the
    /// LOD level (0 = full resolution; 1 and 2 added in M8). `version`
    /// is the chunk's `mesh_version` at spawn time; the upload path
    /// drops the result if the chunk has since been re-edited so a
    /// slow streaming mesh job can't overwrite a fresh edit's mesh.
    Meshed {
        coord: ChunkCoord,
        lod: u8,
        mesh: ChunkMesh,
        version: u64,
    },
    /// A relight job finished; `data` is the re-illuminated paletted chunk
    /// to swap into the World. A follow-up mesh job runs as soon as the
    /// caller drains this — without that, the new sky/block light bytes
    /// never reach the GPU.
    ///
    /// `changed_faces[i]` is `true` when the chunk's boundary cells in
    /// the [`crate::mesher::Face`]`(i)` direction differ between the
    /// pre-BFS and post-BFS snapshots. The Relit handler uses this to
    /// cascade `dirty.light` only to the neighbours whose seed values
    /// would actually change — bounded propagation that converges in
    /// `O(loaded-chunk-diameter)` iterations without exploding.
    Relit {
        coord: ChunkCoord,
        data: PalettedChunk,
        changed_faces: [bool; 6],
    },
}

/// Owns the rayon pool plus the result channel. Held inside `AppState` and
/// shared by reference to all systems that spawn or drain jobs.
#[allow(clippy::too_many_arguments)] // mesh-spawn helpers naturally take many context args
pub struct Jobs {
    tx: Sender<JobResult>,
    /// Receivers drain `try_recv()` from this each frame.
    pub rx: Receiver<JobResult>,
    pool: rayon::ThreadPool,
}

impl Default for Jobs {
    fn default() -> Self {
        Self::new()
    }
}

impl Jobs {
    /// Build a fresh worker pool sized to *N - 1* logical CPUs (leaving one
    /// for the main thread). Result channel is unbounded — it can briefly
    /// queue up dozens of completions during a fast fly-around without
    /// stalling the workers.
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_thread_count())
            .thread_name(|i| format!("oxium-worker-{i}"))
            .build()
            .expect("rayon pool");
        Self { tx, rx, pool }
    }

    /// Spawn a worldgen job for the chunk at `coord`. The job:
    ///
    /// 1. Allocates a fresh `DenseChunk`.
    /// 2. Asks the `Generator` to fill it with terrain.
    /// 3. Runs the lighting BFS *locally* (no neighbours yet — cross-chunk
    ///    bleed gets reapplied later when the streaming system queues a
    ///    relight on the dirty neighbour).
    /// 4. Compresses it into a `PalettedChunk` (canonical form).
    /// 5. Sends the result through the channel.
    pub fn spawn_gen(
        &self,
        coord: ChunkCoord,
        generator: Arc<Generator>,
        registry: Arc<BlockRegistry>,
    ) {
        let tx = self.tx.clone();
        self.pool.spawn(move || {
            let mut dense = DenseChunk::empty();
            generator.fill_chunk(coord, &mut dense);
            // Local-only lighting; the streaming system will re-run light
            // jobs on neighbours later if light leaks across a boundary.
            let no_neighbors = crate::voxel::chunk::Neighbors { chunks: [None; 6] };
            crate::lighting::recompute_chunk(&mut dense, &no_neighbors, &registry);
            let data = PalettedChunk::compress(&dense);
            let _ = tx.send(JobResult::Generated { coord, data });
        });
    }

    /// Spawn a *relight* job for `coord`: decompress, run the lighting
    /// BFS with whatever neighbour data is available, recompress.
    ///
    /// Used by the interaction system whenever the player edits a block
    /// — that flips `meta.dirty.light` and requires the chunk's voxel
    /// light arrays to be regenerated before the next mesh job picks up
    /// fresh `light` bytes for the vertex format.
    pub fn spawn_relight(
        &self,
        coord: ChunkCoord,
        data: Arc<PalettedChunk>,
        neighbors: [Option<Arc<PalettedChunk>>; 6],
        registry: Arc<BlockRegistry>,
    ) {
        let tx = self.tx.clone();
        self.pool.spawn(move || {
            let mut dense = data.decompress();
            let neighbor_dense: Vec<Option<DenseChunk>> = neighbors
                .iter()
                .map(|opt| opt.as_ref().map(|p| p.decompress()))
                .collect();
            let n_refs: [Option<&DenseChunk>; 6] = [
                neighbor_dense[0].as_ref(),
                neighbor_dense[1].as_ref(),
                neighbor_dense[2].as_ref(),
                neighbor_dense[3].as_ref(),
                neighbor_dense[4].as_ref(),
                neighbor_dense[5].as_ref(),
            ];
            let ns = crate::voxel::chunk::Neighbors { chunks: n_refs };
            // Snapshot per-face boundary lighting before the BFS so we
            // can detect which faces actually changed and only cascade
            // dirty.light to those neighbours.
            let pre = crate::lighting::snapshot_face_boundaries(&dense);
            crate::lighting::recompute_chunk(&mut dense, &ns, &registry);
            let post = crate::lighting::snapshot_face_boundaries(&dense);
            let changed_faces: [bool; 6] = std::array::from_fn(|i| pre[i] != post[i]);
            let data = PalettedChunk::compress(&dense);
            let _ = tx.send(JobResult::Relit {
                coord,
                data,
                changed_faces,
            });
        });
    }

    /// Spawn a neighbour-free LOD1 or LOD2 mesh job for `coord`.
    ///
    /// LOD1 downsamples by 2 (16³ cells), LOD2 by 4 (8³ cells). Neither
    /// uses neighbour data — distant chunks don't need precise boundary
    /// face culling because the visual difference is invisible at range.
    /// LOD0 still goes through `spawn_mesh_lod0` because it does need
    /// neighbours for clean chunk boundaries.
    pub fn spawn_mesh_lod(
        &self,
        coord: ChunkCoord,
        lod: u8,
        data: Arc<PalettedChunk>,
        registry: Arc<BlockRegistry>,
        version: u64,
    ) {
        debug_assert!(lod == 1 || lod == 2, "use spawn_mesh_lod0 for LOD0");
        let tx = self.tx.clone();
        self.pool.spawn(move || {
            let dense = data.decompress();
            let factor: u32 = if lod == 1 { 2 } else { 4 };
            let lod_chunk = crate::mesher::lod::downsample(&dense, factor);
            let mesh = crate::mesher::lod::mesh_lod(&lod_chunk, factor, &registry);
            let _ = tx.send(JobResult::Meshed {
                coord,
                lod,
                mesh,
                version,
            });
        });
    }

    /// Spawn a neighbor-aware LOD0 mesh job for `coord`.
    ///
    /// The 6-element `neighbors` array is ordered by [`crate::mesher::Face`]
    /// discriminant (PosX, NegX, PosY, NegY, PosZ, NegZ). Where a neighbour
    /// is `None`, the boundary face on that side is conservatively emitted
    /// — the next mesh job for either chunk will fix things up once the
    /// neighbour exists.
    pub fn spawn_mesh_lod0(
        &self,
        coord: ChunkCoord,
        data: Arc<PalettedChunk>,
        neighbors: [Option<Arc<PalettedChunk>>; 6],
        registry: Arc<BlockRegistry>,
        version: u64,
    ) {
        let tx = self.tx.clone();
        self.pool.spawn(move || {
            // Catch worker panics so they surface in logs instead of being
            // silently swallowed by the rayon pool. (Cheap in steady state.)
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let dense = data.decompress();
                let neighbor_dense: Vec<Option<DenseChunk>> = neighbors
                    .iter()
                    .map(|opt| opt.as_ref().map(|p| p.decompress()))
                    .collect();
                let n_refs: [Option<&DenseChunk>; 6] = [
                    neighbor_dense[0].as_ref(),
                    neighbor_dense[1].as_ref(),
                    neighbor_dense[2].as_ref(),
                    neighbor_dense[3].as_ref(),
                    neighbor_dense[4].as_ref(),
                    neighbor_dense[5].as_ref(),
                ];
                // Greedy mesher (M4): same visual output as the naive
                // mesher but typically 5-10x fewer vertices per chunk.
                crate::mesher::greedy::mesh_greedy(&dense, &n_refs, &registry)
            }));
            match result {
                Ok(mesh) => {
                    let _ = tx.send(JobResult::Meshed {
                        coord,
                        lod: 0,
                        mesh,
                        version,
                    });
                }
                Err(payload) => {
                    log::error!("mesh job panic at {coord:?}: {}", panic_message(payload));
                }
            }
        });
    }
}

/// Extract a human-readable message from a `catch_unwind` panic payload.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        format!("non-string panic payload (type id: {:?})", payload.type_id())
    }
}

/// Pick a worker count: total threads minus one (for the main thread),
/// clamped to at least one. Falls back to a sensible default if querying
/// the platform fails.
fn worker_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .saturating_sub(1)
        .max(1)
}
