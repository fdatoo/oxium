//! Streaming world: cache + job pipeline.

pub mod cache;
pub mod invalidate;
pub mod mesher;
pub mod stream;

use crate::world::cache::{ChunkCache, ChunkMeshGpu};
use crate::world::mesher::mesh_chunk;
use crate::world::stream::{chunks_in_radius, StreamRadius};
use crossbeam_channel::{unbounded, Receiver, Sender};
use glam::Vec3;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::ChunkCoord;
use oxium::worldgen::Generator;
use rayon::ThreadPool;
use std::collections::HashSet;
use std::sync::Arc;

pub enum ChunkJobResult {
    Filled {
        coord: ChunkCoord,
        chunk: Arc<DenseChunk>,
        mesh: ChunkMeshGpu,
    },
}

/// Soft cap on the number of fill+mesh jobs queued at once. Without
/// this, `request_chunks` happily spawns hundreds of jobs per frame
/// while the rayon pool drains them at a fixed rate; the queue grows
/// unboundedly, and any config-change `wipe()` stalls behind a long
/// backlog of stale-config jobs the pool still needs to finish. Sized
/// at ~4× the CPU count: enough parallelism to keep all workers busy,
/// short enough that a fresh edit's requests reach the pool within a
/// frame or two.
fn max_in_flight() -> usize {
    rayon::current_num_threads().saturating_mul(4).max(8)
}

pub struct World {
    generator: Arc<Generator>,
    cache: ChunkCache,
    radius: StreamRadius,
    pool: Arc<ThreadPool>,
    tx: Sender<ChunkJobResult>,
    rx: Receiver<ChunkJobResult>,
    in_flight: HashSet<ChunkCoord>,
    max_in_flight: usize,
}

impl World {
    pub fn new(generator: Arc<Generator>, radius: StreamRadius, capacity: usize) -> Self {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(rayon::current_num_threads().saturating_sub(1).max(1))
                .thread_name(|i| format!("viz-{i}"))
                .build()
                .expect("rayon pool"),
        );
        let (tx, rx) = unbounded();
        Self {
            generator,
            cache: ChunkCache::new(capacity),
            radius,
            pool,
            tx,
            rx,
            in_flight: HashSet::new(),
            max_in_flight: max_in_flight(),
        }
    }

    pub fn radius(&self) -> StreamRadius {
        self.radius
    }

    /// Spawn fill+mesh jobs for any coord in `coords` that is neither
    /// already cached nor currently in flight. Order matters: callers
    /// pass coords already sorted closest-first.
    ///
    /// Caps the number of jobs in flight at `max_in_flight`. Coords
    /// past the cap are skipped this frame; the next call retries
    /// them once the pool has drained. Without this cap, every frame
    /// can pile on hundreds of new fills before earlier ones finish,
    /// starving any subsequent edit-time refill behind a long
    /// stale-config backlog.
    pub fn request_chunks(&mut self, coords: &[ChunkCoord]) {
        for &coord in coords {
            if self.in_flight.len() >= self.max_in_flight {
                break;
            }
            if self.cache.has_mesh(coord) || self.in_flight.contains(&coord) {
                continue;
            }
            self.in_flight.insert(coord);
            let tx = self.tx.clone();
            let generator = self.generator.clone();
            // FIXME(PR 7): wrap the body in std::panic::catch_unwind. Today a
            // panic inside fill_chunk or mesh_chunk leaves `coord` in `in_flight`
            // forever — silent data loss with no diagnostic. A failure variant
            // on ChunkJobResult plus a watchdog drain fixes it.
            self.pool.spawn(move || {
                let mut chunk = DenseChunk::empty();
                generator.fill_chunk(coord, &mut chunk);
                let mesh = mesh_chunk(coord, &chunk);
                let gpu = ChunkMeshGpu {
                    vertices: Arc::new(mesh.vertices),
                    indices: Arc::new(mesh.indices),
                };
                let _ = tx.send(ChunkJobResult::Filled {
                    coord,
                    chunk: Arc::new(chunk),
                    mesh: gpu,
                });
            });
        }
    }

    /// Drain any completed jobs into the cache. Returns the coords whose
    /// meshes just landed; the caller is expected to upload these to the
    /// GPU immediately. The LRU also keeps its own copy (cheap — Arc'd
    /// vertex+index buffers) for future cache-hit queries, but the LRU is
    /// NOT the source of truth for what's on the GPU. The `SceneRenderer`'s
    /// internal HashMap is. If the LRU evicts a coord whose GPU buffer is
    /// still live, the renderer keeps drawing it until the next request
    /// rebuilds the mesh.
    pub fn drain_results(&mut self) -> Vec<(ChunkCoord, ChunkMeshGpu)> {
        let mut out = Vec::new();
        while let Ok(r) = self.rx.try_recv() {
            match r {
                ChunkJobResult::Filled { coord, chunk, mesh } => {
                    // Drop stale completions: a job in flight at the time
                    // of a `wipe()` is no longer in `in_flight`, so its
                    // result references the old config and shouldn't
                    // reach the cache or the GPU. Without this gate,
                    // post-wipe frames briefly display stale-config geometry.
                    if !self.in_flight.remove(&coord) {
                        continue;
                    }
                    self.cache.put_chunk(coord, chunk);
                    self.cache.put_mesh(
                        coord,
                        ChunkMeshGpu {
                            vertices: mesh.vertices.clone(),
                            indices: mesh.indices.clone(),
                        },
                    );
                    out.push((coord, mesh));
                }
            }
        }
        out
    }

    /// Coords currently cached (meshed) — for the scene renderer to know
    /// which buffers to keep.
    pub fn cached_mesh_coords(&self) -> Vec<ChunkCoord> {
        self.cache.meshed_coords()
    }

    pub fn cache_mut(&mut self) -> &mut ChunkCache {
        &mut self.cache
    }

    /// Wipe both the chunk cache and the in-flight set, and drain any
    /// pre-wipe completions still sitting in the channel. Used by the
    /// `Invalidator` when the active `WorldgenConfig` changes.
    ///
    /// The channel drain matters because rayon spawn() can't be
    /// cancelled — workers still process pre-wipe jobs to completion
    /// and send their (stale-config) results. Without the drain,
    /// the next `drain_results` call would upload those stale meshes
    /// to the GPU before the fresh config's chunks reach the pool.
    ///
    /// Workers currently mid-fill still run to completion (we can't
    /// stop them); their results land in the channel and the next
    /// `drain_results` discards them by virtue of `in_flight` no
    /// longer containing the coord. With `max_in_flight` capped at
    /// ~4× CPU, that backlog is at most ~32 jobs deep, not thousands.
    pub fn wipe(&mut self) {
        self.cache.clear();
        self.in_flight.clear();
        while self.rx.try_recv().is_ok() {}
    }

    /// Convenience: request chunks within radius of camera position.
    pub fn request_around(&mut self, pos: Vec3) {
        let coords = chunks_in_radius(pos, self.radius);
        self.request_chunks(&coords);
    }

    pub fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxium::worldgen::config::{ConfigHolder, WorldgenConfig};

    #[test]
    fn world_eventually_meshes_requested_chunks() {
        let config = WorldgenConfig::bundled_default().unwrap();
        let holder = ConfigHolder::new(config);
        let generator = Arc::new(Generator::with_config(42, holder));
        let mut world = World::new(generator, StreamRadius { xz: 0, y: 0 }, 8);
        world.request_around(Vec3::new(0.0, 96.0, 0.0));

        // Spin-wait up to 10 seconds for the single center chunk to mesh.
        // Chunk-fill can be 100ms-1s for non-empty Y bands; first-call latency
        // includes worldgen region cache build, which is slow.
        let start = std::time::Instant::now();
        while world.drain_results().is_empty() {
            if start.elapsed() > std::time::Duration::from_secs(10) {
                panic!("center chunk did not mesh within 10s");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        assert!(world.cached_mesh_coords().len() >= 1);
        assert_eq!(world.in_flight_len(), 0);
    }
}
