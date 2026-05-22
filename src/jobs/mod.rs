//! Background job system: chunk generation, lighting, and meshing run on
//! `rayon` worker pools; results return via a `crossbeam-channel`.
//!
//! Why decouple producer (worker) from consumer (main thread)?
//!
//! - Workers are CPU-bound and can run in parallel without touching the
//!   GPU. Main thread renders.
//! - Channels are lock-free for the common case (one producer, one
//!   consumer) — fine for our drain-each-frame pattern.
//!
//! The pools deliberately use N-1 worker threads total (where N =
//! available parallelism) so the main rendering thread keeps a CPU core
//! to itself.
//!
//! Why two pools instead of one? A single FIFO pool starves mesh jobs
//! during the initial worldgen burst — ~11,250 gen jobs queue up at
//! startup and every Generated result enqueues up to 7 mesh jobs behind
//! them. Splitting into a `gen_pool` (gen + disk load) and a `mesh_pool`
//! (meshing + relight) lets mesh jobs run the moment their source
//! chunk is Stored, in parallel with the gen pool draining its backlog.

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
        /// Light volume blob (33³ × 4 bytes), populated by LOD-0 mesher
        /// only — LODs 1 and 2 share the LOD-0 volume so they always set
        /// `None`. `None` on LOD-0 means the chunk's light wasn't ready
        /// yet (rare race; the next Relit will catch up).
        light_volume: Option<Box<[u8; 33 * 33 * 33 * 4]>>,
    },
    /// A chunk Load job finished — the persisted chunk has been read
    /// from disk on the worker pool (rather than the single-threaded
    /// persistence I/O thread, which was the bottleneck for chunks
    /// whose region file existed). `data = None` means the region
    /// file exists but the slot for this coord is empty — caller
    /// falls back to procedural gen.
    LoadedFromDisk {
        coord: ChunkCoord,
        data: Option<PalettedChunk>,
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
    #[cfg(feature = "legacy-lighting")]
    Relit {
        coord: ChunkCoord,
        data: PalettedChunk,
        changed_faces: [bool; 6],
        /// Always present — the relight worker built it from the same
        /// DenseChunk it just relit.
        light_volume: Box<[u8; 33 * 33 * 33 * 4]>,
    },
}

/// Owns the rayon pools plus the result channel. Held inside `AppState`
/// and shared by reference to all systems that spawn or drain jobs.
#[allow(clippy::too_many_arguments)] // mesh-spawn helpers naturally take many context args
pub struct Jobs {
    tx: Sender<JobResult>,
    /// Receivers drain `try_recv()` from this each frame.
    pub rx: Receiver<JobResult>,
    /// Pool that runs `spawn_gen` and `spawn_load` jobs (the producers of
    /// chunk data). Separated from `mesh_pool` so the initial gen burst
    /// can't starve mesh jobs.
    gen_pool: rayon::ThreadPool,
    /// Pool that runs `spawn_mesh_lod0`, `spawn_mesh_lod`, and
    /// `spawn_relight` jobs. Relight lives here because it's the
    /// precursor to a re-mesh and shares cost characteristics.
    mesh_pool: rayon::ThreadPool,
}

impl Default for Jobs {
    fn default() -> Self {
        Self::new()
    }
}

impl Jobs {
    /// Build two fresh worker pools whose combined size is *N - 1* logical
    /// CPUs (leaving one for the main thread). The total is split evenly
    /// between the gen pool (gen + disk load) and the mesh pool (meshing +
    /// relight), each clamped to at least one thread.
    ///
    /// Result channel is unbounded — it can briefly queue up dozens of
    /// completions during a fast fly-around without stalling the workers.
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        let (gen_threads, mesh_threads) = pool_thread_split();
        let gen_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(gen_threads)
            .thread_name(|i| format!("oxium-gen-{i}"))
            .build()
            .expect("rayon gen pool");
        let mesh_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(mesh_threads)
            .thread_name(|i| format!("oxium-mesh-{i}"))
            .build()
            .expect("rayon mesh pool");
        Self {
            tx,
            rx,
            gen_pool,
            mesh_pool,
        }
    }

    /// Spawn a chunk-load job: read the persisted chunk from disk on
    /// the rayon pool instead of queuing it on the single-threaded
    /// persistence I/O thread. Many small reads from the same region
    /// file are safe concurrently (reads don't mutate state); writes
    /// stay on the dedicated thread so the header-update sequence
    /// remains atomic.
    ///
    /// Returns `LoadedFromDisk { data: None }` when the region file
    /// exists but the slot for this coord is empty — the caller
    /// falls back to procedural gen for that case.
    pub fn spawn_load(&self, coord: ChunkCoord, path: std::path::PathBuf) {
        let tx = self.tx.clone();
        self.gen_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                use crate::persistence::region::{read_chunk, RegionError};
                match read_chunk(&path, coord) {
                    Ok(c) => Some(c),
                    Err(RegionError::NotPresent) => None,
                    Err(e) => {
                        log::warn!("load failed {coord:?}: {e:?}");
                        None
                    }
                }
            }));
            match result {
                Ok(data) => {
                    let _ = tx.send(JobResult::LoadedFromDisk { coord, data });
                }
                Err(payload) => {
                    log::error!("load job panic at {coord:?}: {}", panic_message(payload));
                }
            }
        });
    }

    /// Spawn a worldgen job for the chunk at `coord`. The job:
    ///
    /// 1. Allocates a fresh `DenseChunk`.
    /// 2. Asks the `Generator` to fill it with terrain.
    /// 3. Runs the lighting BFS using the `neighbours` snapshot supplied
    ///    by the caller. Chunks generated while their face-adjacent
    ///    neighbours are already loaded receive correct sky-light
    ///    column-drop inheritance from the +Y neighbour and lateral
    ///    block-light seeding from all six, on this single pass — no
    ///    follow-up relight needed. Chunks generated at the streaming
    ///    wavefront (neighbours mostly `None`) fall back to a
    ///    best-effort BFS, same as before.
    /// 4. Compresses it into a `PalettedChunk` (canonical form).
    /// 5. Sends the result through the channel.
    pub fn spawn_gen(
        &self,
        coord: ChunkCoord,
        generator: Arc<Generator>,
        registry: Arc<BlockRegistry>,
        neighbors: [Option<Arc<PalettedChunk>>; 6],
    ) {
        let tx = self.tx.clone();
        self.gen_pool.spawn(move || {
            // Catch worker panics so they surface in logs instead of
            // silently killing a worker thread. Without this, a single
            // deterministic panic in `fill_chunk` or `recompute_chunk`
            // for a specific coord would kill one worker per such
            // coord — and after ~11 panics the rayon pool has zero
            // live threads and every queued chunk waits forever
            // (visible as a quadrant of the load radius never
            // populating no matter how long the player waits).
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut dense = DenseChunk::empty();
                generator.fill_chunk(coord, &mut dense);
                // Decompress any neighbours the caller snapshotted so the
                // initial BFS does column-drop inheritance + lateral
                // seeding correctly. Matches the spawn_relight pattern at
                // jobs/mod.rs:232-245.
                #[cfg(feature = "legacy-lighting")]
                {
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
                    crate::lighting::recompute_chunk(&mut dense, &ns, &registry);
                }
                #[cfg(not(feature = "legacy-lighting"))]
                { let _ = neighbors; let _ = registry; }
                PalettedChunk::compress(&dense)
            }));
            match result {
                Ok(data) => {
                    let _ = tx.send(JobResult::Generated { coord, data });
                }
                Err(payload) => {
                    log::error!("gen job panic at {coord:?}: {}", panic_message(payload));
                }
            }
        });
    }

    /// Spawn a *relight* job for `coord`: decompress, run the lighting
    /// BFS with whatever neighbour data is available, recompress.
    ///
    /// Used by the interaction system whenever the player edits a block
    /// — that flips `meta.dirty.light` and requires the chunk's voxel
    /// light arrays to be regenerated before the next mesh job picks up
    /// fresh `light` bytes for the vertex format.
    #[cfg(feature = "legacy-lighting")]
    pub fn spawn_relight(
        &self,
        coord: ChunkCoord,
        data: Arc<PalettedChunk>,
        neighbors: [Option<Arc<PalettedChunk>>; 6],
        registry: Arc<BlockRegistry>,
    ) {
        let tx = self.tx.clone();
        self.mesh_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
                let pre = crate::lighting::snapshot_face_boundaries(&dense);
                crate::lighting::recompute_chunk(&mut dense, &ns, &registry);
                let post = crate::lighting::snapshot_face_boundaries(&dense);
                let changed_faces: [bool; 6] = std::array::from_fn(|i| pre[i] != post[i]);
                let light_volume = crate::voxel::chunk::build_light_volume_blob(&dense, &ns);
                let data = PalettedChunk::compress(&dense);
                (data, changed_faces, light_volume)
            }));
            match result {
                Ok((data, changed_faces, light_volume)) => {
                    let _ = tx.send(JobResult::Relit {
                        coord,
                        data,
                        changed_faces,
                        light_volume,
                    });
                }
                Err(payload) => {
                    log::error!("relight job panic at {coord:?}: {}", panic_message(payload));
                }
            }
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
        self.mesh_pool.spawn(move || {
            let dense = data.decompress();
            let factor: u32 = if lod == 1 { 2 } else { 4 };
            let lod_chunk = crate::mesher::lod::downsample(&dense, factor);
            let mesh = crate::mesher::lod::mesh_lod(&lod_chunk, factor, &registry);
            let _ = tx.send(JobResult::Meshed {
                coord,
                lod,
                mesh,
                version,
                light_volume: None,
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
        self.mesh_pool.spawn(move || {
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
                let mesh = crate::mesher::greedy::mesh_greedy(&dense, &n_refs, &registry);
                let ns = crate::voxel::chunk::Neighbors { chunks: n_refs };
                let light_volume = Some(crate::voxel::chunk::build_light_volume_blob(&dense, &ns));
                (mesh, light_volume)
            }));
            match result {
                Ok((mesh, light_volume)) => {
                    let _ = tx.send(JobResult::Meshed {
                        coord,
                        lod: 0,
                        mesh,
                        version,
                        light_volume,
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

/// Split the total worker budget between the gen pool and the mesh pool.
///
/// Gives the gen pool the ceiling half and the mesh pool the floor half
/// of `worker_thread_count()`, each clamped to at least one. On systems
/// with only 1-2 workers available this collapses to 1+1 — meaning the
/// budget grows by a thread, but only when there's truly nothing to
/// split. Better than starving either side.
fn pool_thread_split() -> (usize, usize) {
    let total = worker_thread_count();
    let gen_threads = total.div_ceil(2).max(1);
    let mesh_threads = total.saturating_sub(gen_threads).max(1);
    (gen_threads, mesh_threads)
}
