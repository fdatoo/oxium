//! Fixed-region worldgen meshing.
//!
//! This binary used to stream chunks around the camera (LRU cache,
//! visible-set refill, in-flight queue cap, frustum-prioritised
//! request order). The streaming machinery accumulated a long tail
//! of subtle bugs — chunks not re-rendering when you didn't fly over
//! them, the visible-set refill missing chunks past the radius, the
//! empty-screen-after-edit at high camera Y, etc. The tool is a
//! tuning UI, not a flythrough: you pick a representative spot and
//! iterate. So `World` now owns a bounded **region** of chunks
//! centred on user-chosen XZ coords, regenerates the whole region
//! whenever the config bumps, and that's the entire model.

pub mod invalidate;
pub mod mesher;

use crate::paint::{PaintContext, PaintMode};
use crate::world::mesher::mesh_chunk;
use crossbeam_channel::{Receiver, Sender, unbounded};
use oxium::voxel::block::Block;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::{CHUNK_DIM_U, ChunkCoord, LocalPos};
use oxium::worldgen::Generator;
use rayon::ThreadPool;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::render::scene::Vertex;

/// One filled + meshed chunk in the region.
pub struct ChunkEntry {
    pub chunk: Arc<DenseChunk>,
    #[allow(dead_code)]
    pub vertices: Arc<Vec<Vertex>>,
    #[allow(dead_code)]
    pub indices: Arc<Vec<u32>>,
}

/// Bounded set of chunk coords. Centre is in CHUNK coords. The region
/// covers `[center_x - radius_xz, center_x + radius_xz]` chunks on
/// each XZ axis and `[-radius_y, radius_y]` chunks vertically
/// (anchored to world Y=0 — the vertical extent is fixed regardless
/// of camera height because terrain only lives in a known Y range).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub center_cx: i32,
    pub center_cz: i32,
    pub radius_xz: i32,
    pub radius_y: i32,
}

impl Region {
    pub const fn new(center_cx: i32, center_cz: i32, radius_xz: i32, radius_y: i32) -> Self {
        Self {
            center_cx,
            center_cz,
            radius_xz,
            radius_y,
        }
    }

    /// All chunk coords in the region, in row-major order (Y outer,
    /// then Z, then X). Used to enumerate spawn jobs after a regen.
    pub fn coords(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        let cx0 = self.center_cx - self.radius_xz;
        let cx1 = self.center_cx + self.radius_xz;
        let cz0 = self.center_cz - self.radius_xz;
        let cz1 = self.center_cz + self.radius_xz;
        let cy0 = -self.radius_y;
        let cy1 = self.radius_y;
        (cy0..=cy1).flat_map(move |cy| {
            (cz0..=cz1).flat_map(move |cz| {
                (cx0..=cx1).map(move |cx| ChunkCoord(glam::IVec3::new(cx, cy, cz)))
            })
        })
    }

    pub fn chunk_count(&self) -> usize {
        let xz = (2 * self.radius_xz + 1) as usize;
        let y = (2 * self.radius_y + 1) as usize;
        xz * xz * y
    }

    pub fn contains(&self, coord: ChunkCoord) -> bool {
        let dx = (coord.0.x - self.center_cx).abs();
        let dz = (coord.0.z - self.center_cz).abs();
        let dy = coord.0.y.abs();
        dx <= self.radius_xz && dz <= self.radius_xz && dy <= self.radius_y
    }
}

pub enum ChunkJobResult {
    Filled {
        coord: ChunkCoord,
        chunk: Arc<DenseChunk>,
        vertices: Arc<Vec<Vertex>>,
        indices: Arc<Vec<u32>>,
    },
}

pub struct World {
    generator: Arc<Generator>,
    pool: Arc<ThreadPool>,
    region: Region,
    entries: HashMap<ChunkCoord, ChunkEntry>,
    tx: Sender<ChunkJobResult>,
    rx: Receiver<ChunkJobResult>,
    /// Coords whose fill+mesh job is queued on the pool. Cleared on
    /// every full regen; drained as results arrive. Still useful as
    /// a dedup so simultaneous edits don't queue duplicate jobs.
    in_flight: HashSet<ChunkCoord>,
    paint_mode: PaintMode,
}

impl World {
    pub fn new(generator: Arc<Generator>, region: Region) -> Self {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(rayon::current_num_threads().saturating_sub(1).max(1))
                .thread_name(|i| format!("viz-{i}"))
                .build()
                .expect("rayon pool"),
        );
        let (tx, rx) = unbounded();
        let mut me = Self {
            generator,
            pool,
            region,
            entries: HashMap::with_capacity(region.chunk_count()),
            tx,
            rx,
            in_flight: HashSet::new(),
            paint_mode: PaintMode::default(),
        };
        // Spawn the initial fill so the user doesn't see an empty
        // world for the first edit.
        me.regen();
        me
    }

    pub fn region(&self) -> Region {
        self.region
    }

    pub fn set_paint_mode(&mut self, mode: PaintMode) {
        self.paint_mode = mode;
    }

    /// Replace the region. Chunks no longer inside are dropped (the
    /// returned vec gives the caller the coords to remove from any
    /// GPU buffer set). Then queues a full regen for the new region.
    pub fn set_region(&mut self, new_region: Region) -> Vec<ChunkCoord> {
        if new_region == self.region {
            return Vec::new();
        }
        self.region = new_region;
        let mut dropped = Vec::new();
        self.entries.retain(|coord, _| {
            if new_region.contains(*coord) {
                true
            } else {
                dropped.push(*coord);
                false
            }
        });
        self.in_flight.retain(|coord| new_region.contains(*coord));
        self.regen();
        dropped
    }

    /// Discard any pending channel messages, clear `in_flight`, then
    /// queue fill+mesh for every coord in the region. Existing
    /// entries are NOT cleared: new meshes replace old in place as
    /// they arrive, so the user sees a progressive refresh rather
    /// than a flash to empty.
    pub fn regen(&mut self) {
        while self.rx.try_recv().is_ok() {}
        self.in_flight.clear();
        // Collect into a Vec so the closure-borrow of `self.region`
        // ends before each `spawn_one` borrows `self` mutably.
        let coords: Vec<_> = self.region.coords().collect();
        for coord in coords {
            self.spawn_one(coord);
        }
    }

    fn spawn_one(&mut self, coord: ChunkCoord) {
        if !self.in_flight.insert(coord) {
            return;
        }
        let tx = self.tx.clone();
        let generator = self.generator.clone();
        let paint_mode = self.paint_mode;
        let dim = CHUNK_DIM_U as i32;
        let origin_x = coord.0.x * dim;
        let origin_z = coord.0.z * dim;
        self.pool.spawn(move || {
            let mut chunk = DenseChunk::empty();
            generator.fill_chunk(coord, &mut chunk);
            let paint = PaintContext::build(paint_mode, &generator, origin_x, origin_z);
            let mesh = mesh_chunk(coord, &chunk, &paint);
            let _ = tx.send(ChunkJobResult::Filled {
                coord,
                chunk: Arc::new(chunk),
                vertices: Arc::new(mesh.vertices),
                indices: Arc::new(mesh.indices),
            });
        });
    }

    /// Pull completed jobs from the channel into `entries`. Returns
    /// the coords whose meshes landed this frame (caller uploads to
    /// the SceneRenderer). Results whose coord is no longer in
    /// `in_flight` (e.g., another regen happened mid-flight) are
    /// dropped silently.
    #[allow(clippy::type_complexity)]
    pub fn drain_results(&mut self) -> Vec<(ChunkCoord, Arc<Vec<Vertex>>, Arc<Vec<u32>>)> {
        let mut out = Vec::new();
        while let Ok(r) = self.rx.try_recv() {
            match r {
                ChunkJobResult::Filled {
                    coord,
                    chunk,
                    vertices,
                    indices,
                } => {
                    if !self.in_flight.remove(&coord) {
                        continue;
                    }
                    self.entries.insert(
                        coord,
                        ChunkEntry {
                            chunk,
                            vertices: vertices.clone(),
                            indices: indices.clone(),
                        },
                    );
                    out.push((coord, vertices, indices));
                }
            }
        }
        out
    }

    #[allow(dead_code)]
    pub fn entries_len(&self) -> usize {
        self.entries.len()
    }

    pub fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    /// Step a ray through the region's filled chunks and return the
    /// `(wx, wz)` of the first column whose voxel along the ray is
    /// solid. Returns `None` if the ray traverses `max_distance`
    /// without hitting a solid voxel, or only touches chunks that
    /// aren't yet filled.
    pub fn raycast_column(
        &self,
        origin: glam::Vec3,
        dir: glam::Vec3,
        max_distance: f32,
    ) -> Option<(i32, i32)> {
        if dir.length_squared() < 1e-6 {
            return None;
        }
        let dir = dir.normalize();
        let step_size = 0.5_f32;
        let step = dir * step_size;
        let mut pos = origin;
        let mut t = 0.0_f32;
        let dim = CHUNK_DIM_U as i32;
        while t < max_distance {
            let voxel = pos.floor().as_ivec3();
            let chunk_coord = ChunkCoord(glam::IVec3::new(
                voxel.x.div_euclid(dim),
                voxel.y.div_euclid(dim),
                voxel.z.div_euclid(dim),
            ));
            if let Some(entry) = self.entries.get(&chunk_coord) {
                let local = LocalPos(glam::UVec3::new(
                    voxel.x.rem_euclid(dim) as u32,
                    voxel.y.rem_euclid(dim) as u32,
                    voxel.z.rem_euclid(dim) as u32,
                ));
                let block = entry.chunk.get(local);
                if !matches!(block, Block::Air | Block::Water) {
                    return Some((voxel.x, voxel.z));
                }
            }
            pos += step;
            t += step_size;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxium::worldgen::config::{ConfigHolder, WorldgenConfig};

    #[test]
    fn region_chunk_count_matches_box() {
        let r = Region::new(0, 0, 2, 1);
        let expected = (2 * 2 + 1) * (2 * 2 + 1) * (2 + 1);
        assert_eq!(r.chunk_count(), expected as usize);
        assert_eq!(r.coords().count(), expected as usize);
    }

    #[test]
    fn region_contains_center_chunk() {
        let r = Region::new(5, -3, 2, 1);
        assert!(r.contains(ChunkCoord(glam::IVec3::new(5, 0, -3))));
        assert!(r.contains(ChunkCoord(glam::IVec3::new(7, 1, -1))));
        assert!(!r.contains(ChunkCoord(glam::IVec3::new(8, 0, -3))));
        assert!(!r.contains(ChunkCoord(glam::IVec3::new(5, 2, -3))));
    }

    #[test]
    fn world_fills_region_within_timeout() {
        let config = WorldgenConfig::bundled_default().unwrap();
        let holder = ConfigHolder::new(config);
        let generator = Arc::new(Generator::with_config(42, holder));
        let region = Region::new(0, 0, 0, 0); // single chunk
        let mut world = World::new(generator, region);

        let start = std::time::Instant::now();
        while world.entries_len() == 0 {
            world.drain_results();
            if start.elapsed() > std::time::Duration::from_secs(10) {
                panic!("center chunk didn't fill within 10s");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(world.entries_len(), 1);
        assert_eq!(world.in_flight_len(), 0);
    }
}
