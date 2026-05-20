//! Cache invalidation on config change. PR 1: wipe-all; PR 7 may add
//! field-aware partial invalidation if profiling shows it's needed.

use crate::world::cache::ChunkCache;

pub struct Invalidator {
    last_config_revision: u64,
    current: u64,
}

impl Invalidator {
    pub fn new() -> Self {
        Self { last_config_revision: 0, current: 0 }
    }

    /// Call when the active config changes (e.g., slider edit, preset
    /// load, file-watcher swap). Bumps the revision counter.
    pub fn bump(&mut self) {
        self.current = self.current.wrapping_add(1);
    }

    /// Call once per frame. If the revision has advanced since the last
    /// check, wipe the cache and synchronise. Returns `true` if it wiped.
    pub fn maybe_wipe(&mut self, cache: &mut ChunkCache) -> bool {
        if self.current != self.last_config_revision {
            cache.clear();
            self.last_config_revision = self.current;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::cache::ChunkMeshGpu;
    use glam::IVec3;
    use oxium::voxel::chunk::DenseChunk;
    use oxium::voxel::coords::ChunkCoord;
    use std::sync::Arc;

    fn coord(x: i32, y: i32, z: i32) -> ChunkCoord {
        ChunkCoord(IVec3::new(x, y, z))
    }

    #[test]
    fn no_bump_means_no_wipe() {
        let mut inv = Invalidator::new();
        let mut cache = ChunkCache::new(4);
        cache.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        assert!(!inv.maybe_wipe(&mut cache));
        assert!(cache.has_chunk(coord(0, 0, 0)));
    }

    #[test]
    fn bump_then_check_wipes_once() {
        let mut inv = Invalidator::new();
        let mut cache = ChunkCache::new(4);
        cache.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        cache.put_mesh(coord(0, 0, 0), ChunkMeshGpu::empty());
        inv.bump();
        assert!(inv.maybe_wipe(&mut cache));
        assert_eq!(cache.len(), (0, 0));
        // Second check after the same bump is a no-op.
        cache.put_chunk(coord(1, 0, 0), Arc::new(DenseChunk::empty()));
        assert!(!inv.maybe_wipe(&mut cache));
        assert!(cache.has_chunk(coord(1, 0, 0)));
    }
}
