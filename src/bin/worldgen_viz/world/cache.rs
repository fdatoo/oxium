//! LRU caches for filled chunks + meshed chunks. Keyed by ChunkCoord.

use crate::render::scene::Vertex;
use lru::LruCache;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::ChunkCoord;
use std::num::NonZeroUsize;
use std::sync::Arc;

pub struct ChunkMeshGpu {
    pub vertices: Arc<Vec<Vertex>>,
    pub indices: Arc<Vec<u32>>,
}

impl ChunkMeshGpu {
    pub fn empty() -> Self {
        Self {
            vertices: Arc::new(Vec::new()),
            indices: Arc::new(Vec::new()),
        }
    }
}

pub struct ChunkCache {
    /// Filled (but not yet meshed) chunks. Keyed by coord; capacity is
    /// `radius_xz^2 * radius_y * 3` to give headroom for chunks that are
    /// in-flight or recently scrolled out.
    chunks: LruCache<ChunkCoord, Arc<DenseChunk>>,
    /// Meshed CPU-side vertex/index buffers, ready for GPU upload.
    meshes: LruCache<ChunkCoord, ChunkMeshGpu>,
}

impl ChunkCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            chunks: LruCache::new(cap),
            meshes: LruCache::new(cap),
        }
    }

    pub fn put_chunk(&mut self, coord: ChunkCoord, chunk: Arc<DenseChunk>) {
        self.chunks.put(coord, chunk);
    }

    pub fn put_mesh(&mut self, coord: ChunkCoord, mesh: ChunkMeshGpu) {
        self.meshes.put(coord, mesh);
    }

    pub fn get_chunk(&mut self, coord: ChunkCoord) -> Option<Arc<DenseChunk>> {
        self.chunks.get(&coord).cloned()
    }

    pub fn get_mesh(&mut self, coord: ChunkCoord) -> Option<&ChunkMeshGpu> {
        self.meshes.get(&coord)
    }

    pub fn has_chunk(&self, coord: ChunkCoord) -> bool {
        self.chunks.contains(&coord)
    }

    pub fn has_mesh(&self, coord: ChunkCoord) -> bool {
        self.meshes.contains(&coord)
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.meshes.clear();
    }

    pub fn len(&self) -> (usize, usize) {
        (self.chunks.len(), self.meshes.len())
    }

    pub fn meshed_coords(&self) -> Vec<ChunkCoord> {
        self.meshes.iter().map(|(k, _)| *k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    fn coord(x: i32, y: i32, z: i32) -> ChunkCoord {
        ChunkCoord(IVec3::new(x, y, z))
    }

    #[test]
    fn put_and_get_chunk() {
        let mut c = ChunkCache::new(4);
        c.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        assert!(c.has_chunk(coord(0, 0, 0)));
        assert!(c.get_chunk(coord(0, 0, 0)).is_some());
    }

    #[test]
    fn over_capacity_evicts_oldest() {
        let mut c = ChunkCache::new(2);
        c.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        c.put_chunk(coord(1, 0, 0), Arc::new(DenseChunk::empty()));
        c.put_chunk(coord(2, 0, 0), Arc::new(DenseChunk::empty()));
        // (0,0,0) was inserted first; on third insert it should evict.
        assert!(!c.has_chunk(coord(0, 0, 0)));
        assert!(c.has_chunk(coord(1, 0, 0)));
        assert!(c.has_chunk(coord(2, 0, 0)));
    }

    #[test]
    fn clear_drops_everything() {
        let mut c = ChunkCache::new(4);
        c.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        c.put_mesh(coord(0, 0, 0), ChunkMeshGpu::empty());
        c.clear();
        assert_eq!(c.len(), (0, 0));
    }
}
