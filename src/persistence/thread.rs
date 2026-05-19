//! Dedicated I/O thread that owns disk reads/writes.
//!
//! Why a separate thread rather than the rayon pool? Disk I/O blocks. If
//! it ran on a rayon worker, that worker would stall for tens of
//! milliseconds, starving compute jobs. A separate thread takes the I/O
//! out of the compute critical path.
//!
//! The main thread enqueues `Save` / `Load` requests and drains
//! `PersistResult` each frame. Both channels are unbounded — the I/O
//! thread is generally much faster than human-scale edits, so backlog
//! is rare.

use crate::persistence::region::{read_chunk, region_path, write_chunk, RegionError};
use crate::voxel::chunk::PalettedChunk;
use crate::voxel::coords::ChunkCoord;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::path::PathBuf;
use std::thread;

/// A unit of work sent from the main thread.
pub enum PersistRequest {
    /// Write `data` to the appropriate region file. Carries `Arc` so
    /// the main thread doesn't have to deep-clone the chunk just to
    /// hand it off — the I/O thread reads through the Arc.
    Save {
        coord: ChunkCoord,
        data: std::sync::Arc<PalettedChunk>,
    },
    /// Read the chunk at `coord` if it exists on disk.
    Load {
        coord: ChunkCoord,
    },
    /// Politely tell the thread to exit. Used by `shutdown` to wait for
    /// in-flight writes to finish before the process exits.
    Shutdown,
}

/// Result of a completed request.
pub enum PersistResult {
    /// `data` is `None` when the slot was empty (or the region file didn't
    /// exist) — caller falls back to procedural gen.
    Loaded {
        coord: ChunkCoord,
        data: Option<PalettedChunk>,
    },
    /// A `Save` finished. The caller resets `meta.modified` so the same
    /// chunk doesn't get re-saved at the next autosave tick.
    Saved {
        coord: ChunkCoord,
    },
}

/// Owner of the I/O thread.
pub struct Persistence {
    pub req_tx: Sender<PersistRequest>,
    pub result_rx: Receiver<PersistResult>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Persistence {
    /// Launch the I/O thread. `saves_dir` becomes the root for all region
    /// files written/read by this instance.
    pub fn spawn(saves_dir: PathBuf) -> Self {
        let (req_tx, req_rx) = unbounded::<PersistRequest>();
        let (res_tx, res_rx) = unbounded::<PersistResult>();
        let handle = thread::Builder::new()
            .name("oxium-persist".into())
            .spawn(move || {
                for req in req_rx.iter() {
                    match req {
                        PersistRequest::Shutdown => break,
                        PersistRequest::Save { coord, data } => {
                            let path = region_path(&saves_dir, coord);
                            if let Err(e) = write_chunk(&path, coord, &*data) {
                                log::warn!("save failed {coord:?}: {e:?}");
                            }
                            let _ = res_tx.send(PersistResult::Saved { coord });
                        }
                        PersistRequest::Load { coord } => {
                            let path = region_path(&saves_dir, coord);
                            let data = if path.exists() {
                                match read_chunk(&path, coord) {
                                    Ok(c) => Some(c),
                                    Err(RegionError::NotPresent) => None,
                                    Err(e) => {
                                        log::warn!("load failed {coord:?}: {e:?}");
                                        None
                                    }
                                }
                            } else {
                                None
                            };
                            let _ = res_tx.send(PersistResult::Loaded { coord, data });
                        }
                    }
                }
            })
            .unwrap();
        Self {
            req_tx,
            result_rx: res_rx,
            handle: Some(handle),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::block::Block;
    use crate::voxel::chunk::{DenseChunk, PalettedChunk};
    use crate::voxel::coords::{ChunkCoord, LocalPos};
    use glam::{IVec3, UVec3};
    use std::sync::Arc;

    /// Regression test for the shutdown race: a Save enqueued just
    /// before drop *must* be on disk by the time drop returns. Before
    /// this fix, dropping `Persistence` killed the I/O thread without
    /// joining, leaving any in-flight or queued writes unwritten —
    /// which presented as "the chunks where I edited terrain render
    /// wrong on next launch".
    #[test]
    fn drop_flushes_queued_saves() {
        let td = tempfile::tempdir().unwrap();
        let coord = ChunkCoord(IVec3::new(0, 0, 0));
        let saves_dir = td.path().to_path_buf();

        // Build a one-off chunk with a sentinel block so we can tell
        // the save round-tripped (vs being silently re-read as the
        // empty default).
        let mut dense = DenseChunk::empty();
        dense.set(LocalPos(UVec3::new(7, 14, 21)), Block::Torch);
        let chunk = Arc::new(PalettedChunk::compress(&dense));

        {
            let p = Persistence::spawn(saves_dir.clone());
            p.req_tx
                .send(PersistRequest::Save {
                    coord,
                    data: chunk.clone(),
                })
                .unwrap();
            // Intentionally do NOT wait for the `Saved` ack: the
            // whole point is to verify Drop joins the thread so the
            // save reaches disk even if the main thread skipped
            // draining acknowledgements.
        }
        // `p` drops here — `Drop for Persistence` must join the
        // thread, which can only return after the queued Save has
        // finished its `write_all` calls.

        let path = crate::persistence::region::region_path(&saves_dir, coord);
        let read = crate::persistence::region::read_chunk(&path, coord).unwrap();
        assert_eq!(
            read.decompress().blocks[LocalPos(UVec3::new(7, 14, 21)).to_index()],
            Block::Torch,
            "save queued just before drop should still be on disk after drop returns"
        );
    }
}

impl Drop for Persistence {
    /// Wait for queued writes to complete before the thread is killed.
    ///
    /// When `main()` returns, the process exits and the OS reaps every
    /// non-main thread immediately — anything still queued on this
    /// channel (or mid-`write`) goes with it. That manifests in the
    /// next launch as "the chunks where I modified terrain render
    /// wrong":
    ///
    /// - Save request still in the queue → never on disk → next load
    ///   returns `None` → fallback to procedural gen, player sees
    ///   fresh terrain where their edit was.
    /// - Save mid-blob-write → blob bytes lost, header still points
    ///   at the old offset → next load returns OLD pre-edit data.
    /// - Save mid-header-write → header partial → next load reads
    ///   a garbage offset → zstd decode fails → fallback to gen.
    ///
    /// Sending `Shutdown` makes the worker loop break cleanly after
    /// its current request finishes, and `join()` blocks until every
    /// queued write has been handed back to the kernel. Field drop
    /// order in [`crate::app::AppState`] runs the user-facing `Drop`
    /// first (which calls `flush_modified` to enqueue any final
    /// edits), so by the time this `Drop` fires every save the
    /// session ever produced is in the channel.
    fn drop(&mut self) {
        let _ = self.req_tx.send(PersistRequest::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
