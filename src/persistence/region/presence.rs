//! Header-only region presence scans for [`crate::persistence::SaveIndex`].

use std::fs::File;
use std::io::Read;

use super::coords::REGION_SLOTS;
use super::format::{HEADER_SECTORS, RegionError, SECTOR};

/// Read just the slot table of a region file and return a bitmap of which
/// slots are non-empty.
///
/// One 16 KB sequential read; skips the per-chunk seek + decompress that
/// [`super::read_chunk`] does. Returns `Ok(None)` when the file does not
/// exist, or when it is truncated before the full header.
pub fn read_presence_bitmap(
    path: &std::path::Path,
) -> Result<Option<Box<[bool; REGION_SLOTS]>>, RegionError> {
    if !path.exists() {
        return Ok(None);
    }
    let mut f = File::open(path)?;
    let mut header = vec![0u8; (HEADER_SECTORS * SECTOR) as usize];
    if let Err(e) = f.read_exact(&mut header) {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            return Ok(None);
        }
        return Err(e.into());
    }
    let mut bm: Box<[bool; REGION_SLOTS]> = Box::new([false; REGION_SLOTS]);
    for i in 0..REGION_SLOTS {
        let entry = u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
        bm[i] = entry != 0;
    }
    Ok(Some(bm))
}
