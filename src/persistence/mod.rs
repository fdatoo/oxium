//! Persistent world storage.
//!
//! Procedural chunks are reproducible from the save seed, so disk only stores
//! chunks the player has modified. The persistence layer is split into four
//! small pieces:
//!
//! | Module       | Responsibility                                             |
//! |--------------|------------------------------------------------------------|
//! | `manifest`   | `world.toml`: seed, worldgen version, creation metadata    |
//! | `region`     | 16 x 16 x 16 chunk region files, zstd-compressed payloads  |
//! | `save_index` | per-session cache of which region slots exist on disk      |
//! | `thread`     | single blocking I/O worker for async save/load requests    |
//!
//! Region files use a 16 KB header containing 4096 packed slot entries, then
//! append compressed `PalettedChunk` blobs at EOF. [`SaveIndex`] reads those
//! headers once per session so streaming can skip disk I/O for empty slots.

pub mod manifest;
pub mod region;
pub mod save_index;
pub mod thread;

pub use manifest::{CURRENT_VERSION as MANIFEST_CURRENT_VERSION, WorldManifest};
pub use save_index::SaveIndex;
