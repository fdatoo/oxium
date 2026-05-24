//! Per-session cache of "which chunk slots have data on disk".
//!
//! The region file format packs 16³ = 4096 chunk slots into one file
//! with a 16 KB header that maps each slot to its blob offset. To answer
//! "does coord X exist on disk" we have to look at byte
//! `region_slot(X) * 4` of that header.
//!
//! Without a cache, the streaming system answers that question once per
//! chunk per session by sending a `Load` request through the
//! single-threaded persistence I/O worker — even when the region file
//! exists but the specific slot is empty. After a single block edit
//! anywhere in a region (which creates the region file), every other
//! chunk in that region's 4096-slot footprint pays the round-trip cost
//! just to be told "NotPresent" → fall back to procedural gen. With ~10
//! 000 chunks in the load radius and a region file or two in range,
//! that's seconds of useless serial I/O at world load — visible to the
//! player as "everything streams in fast normally, but as soon as I
//! edit anything, the next launch takes ages to render".
//!
//! `SaveIndex` reads each region's header *once* on first reference
//! and caches a 4096-element bitmap. Subsequent slot checks are O(1)
//! pointer-into-array, and the streaming system can short-circuit to
//! procedural gen for the empty-slot case without touching the
//! persistence thread at all.

use crate::persistence::region::{
    REGION_SLOTS, RegionCoord, read_presence_bitmap, region_coord, region_path, region_slot,
};
use crate::voxel::coords::ChunkCoord;
use std::collections::HashMap;
use std::path::Path;

/// Per-region presence bitmap. `None` means the region file doesn't
/// exist (so every slot is implicitly empty); `Some(bm)` is the
/// snapshot of the slot table as of the most recent header read.
type RegionEntry = Option<Box<[bool; REGION_SLOTS]>>;

/// In-memory mirror of which region-file slots have data on disk.
///
/// One `SaveIndex` lives in [`crate::app::AppState`] alongside
/// `saves_dir` and `persistence`. The streaming system asks it
/// `has(coord)` before deciding between a disk Load and a procedural
/// gen; the save side calls `mark(coord)` whenever it queues a Save
/// so the cache stays consistent with the disk state the persistence
/// thread is about to write.
#[derive(Default)]
pub struct SaveIndex {
    /// Lazily-populated, keyed by the chunk's owning region grid
    /// coordinate. Each entry is read once per
    /// session on its first reference; subsequent checks hit the
    /// in-memory bitmap.
    regions: HashMap<RegionCoord, RegionEntry>,
}

impl SaveIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// True if `coord` has a saved chunk on disk.
    ///
    /// First call for a given region pays an O(16 KB) header read on
    /// the main thread; subsequent calls for any chunk in the same
    /// region are O(1) bitmap lookups. The header read is sequential,
    /// hits the page cache after the first call, and runs in well
    /// under a millisecond even on cold disk — at startup we touch a
    /// handful of region files at most.
    pub fn has(&mut self, saves_dir: &Path, coord: ChunkCoord) -> bool {
        let rc = region_coord(coord);
        let entry = self.regions.entry(rc).or_insert_with(|| {
            let path = region_path(saves_dir, coord);
            match read_presence_bitmap(&path) {
                Ok(bm) => bm,
                Err(e) => {
                    // Treat read errors the same as "file doesn't
                    // exist" — the worst case is the chunk regenerates
                    // from seed instead of loading the saved version,
                    // which is far better than a hard crash. The log
                    // line gives us a breadcrumb if it ever fires.
                    log::warn!("save_index header read failed for region {:?}: {:?}", rc, e);
                    None
                }
            }
        });
        match entry {
            None => false,
            Some(bm) => bm[region_slot(coord).index()],
        }
    }

    /// Record that a Save for `coord` has been queued (or has just
    /// landed). Subsequent `has()` calls for `coord` return `true`
    /// even before the persistence thread updates the header on disk
    /// — the data hasn't reached the file yet but the *intent* is
    /// there, and any subsequent Load goes through the same serial
    /// persistence channel so it'll see the post-Save state by the
    /// time it runs.
    ///
    /// Used by both `world_unload` and `flush_modified` to keep the
    /// in-memory index in lockstep with what we've handed off to the
    /// I/O thread.
    pub fn mark(&mut self, coord: ChunkCoord) {
        let rc = region_coord(coord);
        let entry = self
            .regions
            .entry(rc)
            .or_insert_with(|| Some(Box::new([false; REGION_SLOTS])));
        // If we previously cached "this region file doesn't exist",
        // promote it to an empty bitmap — we're about to make the
        // file exist by saving into it.
        let bm = entry.get_or_insert_with(|| Box::new([false; REGION_SLOTS]));
        bm[region_slot(coord).index()] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::region::{region_path, write_chunk};
    use crate::voxel::chunk::PalettedChunk;
    use glam::IVec3;

    #[test]
    fn empty_saves_dir_reports_no_chunks() {
        let td = tempfile::tempdir().unwrap();
        let mut idx = SaveIndex::new();
        assert!(!idx.has(td.path(), ChunkCoord(IVec3::new(0, 0, 0))));
        assert!(!idx.has(td.path(), ChunkCoord(IVec3::new(15, 15, 15))));
    }

    #[test]
    fn detects_written_chunk_and_misses_its_siblings() {
        let td = tempfile::tempdir().unwrap();
        // Write one chunk; every other slot in that region file
        // should report `false`.
        let written = ChunkCoord(IVec3::new(3, 4, 5));
        let path = region_path(td.path(), written);
        write_chunk(&path, written, &PalettedChunk::all_air()).unwrap();

        let mut idx = SaveIndex::new();
        assert!(
            idx.has(td.path(), written),
            "the chunk we wrote should be present"
        );
        // Pick a sibling in the same region file (rx=0, ry=0, rz=0
        // because all components are < 16).
        let sibling = ChunkCoord(IVec3::new(0, 0, 0));
        assert!(
            !idx.has(td.path(), sibling),
            "a sibling slot in the same region must NOT report present"
        );
    }

    #[test]
    fn mark_makes_has_return_true() {
        let td = tempfile::tempdir().unwrap();
        let mut idx = SaveIndex::new();
        let c = ChunkCoord(IVec3::new(1, 2, 3));
        assert!(!idx.has(td.path(), c));
        idx.mark(c);
        assert!(
            idx.has(td.path(), c),
            "mark() must flip the cache for the saved slot"
        );
    }
}
