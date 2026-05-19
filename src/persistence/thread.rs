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

    /// Drain any in-flight requests then join the thread. Called from
    /// `AppState::drop` so a clean exit lets the last edits flush.
    pub fn shutdown(mut self) {
        let _ = self.req_tx.send(PersistRequest::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
