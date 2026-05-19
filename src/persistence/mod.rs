//! Disk save format: Minecraft-style region files (`.mca`-ish).
//!
//! Each region file holds a 16×16 grid of chunks with a 4 KB header at the
//! top. The header is 256 `u32` slot entries packed as `(offset_in_sectors:
//! u24, len_sectors: u8)` so the file can be opened, header-read, and a
//! single chunk located in two seeks. Chunk payload is `bincode(PalettedChunk)`
//! then `zstd`-compressed.

pub mod manifest;
pub mod region;
pub mod save_index;
pub mod thread;

pub use manifest::{WorldManifest, CURRENT_VERSION as MANIFEST_CURRENT_VERSION};
pub use save_index::SaveIndex;
