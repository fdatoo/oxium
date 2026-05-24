//! Minecraft-style region file format: one file per 16 x 16 x 16 chunk cube.
//!
//! A region file is the persistence unit for edited chunks. Procedural chunks
//! are regenerated from world seed; only chunks that differ from generation are
//! written here.
//!
//! ```text
//! +-------- header (16 KB, 4 sectors) --------+-----------------------+
//! | 4096 x u32 LE slot entries                | chunk blobs (zstd)    |
//! +-------------------------------------------+-----------------------+
//! ```
//!
//! Each slot entry packs `(offset_in_sectors: u24, len_sectors: u8)`. A zero
//! entry means "slot empty". Newly-saved chunks append at EOF; v0 accepts
//! fragmentation rather than doing in-place compaction.
//!
//! ### Submodule layout
//!
//! | Submodule  | Contents                                             |
//! |------------|------------------------------------------------------|
//! | `format`   | sector size, header size, version magic, errors      |
//! | `coords`   | region path, region coord, and slot-index math       |
//! | `io`       | chunk encode/compress/write and read/decompress      |
//! | `presence` | cheap header-only bitmap reads for [`SaveIndex`]     |
//! | `compat`   | legacy chunk blob decode helpers                     |
//!
//! The public surface is re-exported from this module so existing callers can
//! keep using `persistence::region::read_chunk`, `slot_index`, etc.
//!
//! [`SaveIndex`]: crate::persistence::SaveIndex

pub mod compat;
pub mod coords;
pub mod format;
pub mod io;
pub mod presence;

pub use coords::{
    REGION_SLOTS, RegionCoord, RegionSlot, region_coord, region_path, region_slot, slot_index,
};
pub use format::RegionError;
pub use io::{read_chunk, write_chunk};
pub use presence::read_presence_bitmap;

#[cfg(test)]
mod tests;
