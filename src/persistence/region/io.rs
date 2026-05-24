//! Region file read/write operations.

use crate::voxel::chunk::PalettedChunk;
use crate::voxel::coords::ChunkCoord;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

use super::compat::decode_chunk_payload;
use super::coords::region_slot;
use super::format::{CHUNK_V2_MAGIC, HEADER_SECTORS, RegionError, SECTOR};

/// Encode, compress, append at EOF, and update the header. Creates the file
/// and parent `regions` directory if missing.
pub fn write_chunk(
    path: &std::path::Path,
    coord: ChunkCoord,
    data: &PalettedChunk,
) -> Result<(), RegionError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;

    let header_total = HEADER_SECTORS * SECTOR;
    let len = f.metadata()?.len();
    if len < header_total {
        f.set_len(header_total)?;
    }

    let mut header = vec![0u8; header_total as usize];
    f.seek(SeekFrom::Start(0))?;
    let _ = f.read(&mut header)?;

    let mut blob_bin = Vec::with_capacity(4 + 64 * 1024);
    blob_bin.extend_from_slice(CHUNK_V2_MAGIC);
    let payload = bincode::serialize(data).map_err(|e| RegionError::Decode(e.to_string()))?;
    blob_bin.extend_from_slice(&payload);
    let blob = zstd::stream::encode_all(blob_bin.as_slice(), 3)
        .map_err(|e| RegionError::Zstd(e.to_string()))?;
    let with_prefix_len = 4 + blob.len();
    let needed_sectors = (with_prefix_len as u64).div_ceil(SECTOR).max(1);

    let cur_end = f.metadata()?.len();
    let end_sector = (cur_end / SECTOR).max(HEADER_SECTORS);
    f.seek(SeekFrom::Start(end_sector * SECTOR))?;
    f.write_all(&(blob.len() as u32).to_le_bytes())?;
    f.write_all(&blob)?;
    let pad = (needed_sectors * SECTOR) as usize - with_prefix_len;
    if pad > 0 {
        f.write_all(&vec![0u8; pad])?;
    }

    let slot = region_slot(coord).index();
    let entry = ((end_sector as u32) << 8) | (needed_sectors as u32).min(0xFF);
    header[slot * 4..slot * 4 + 4].copy_from_slice(&entry.to_le_bytes());

    f.seek(SeekFrom::Start(0))?;
    f.write_all(&header)?;
    Ok(())
}

/// Read the chunk at `coord` out of the region file.
///
/// Returns [`RegionError::NotPresent`] if the slot is zero.
pub fn read_chunk(path: &std::path::Path, coord: ChunkCoord) -> Result<PalettedChunk, RegionError> {
    let mut f = File::open(path)?;
    let mut header = vec![0u8; (HEADER_SECTORS * SECTOR) as usize];
    f.read_exact(&mut header)?;

    let slot = region_slot(coord).index();
    let entry = u32::from_le_bytes(header[slot * 4..slot * 4 + 4].try_into().unwrap());
    if entry == 0 {
        return Err(RegionError::NotPresent);
    }
    let offset_sectors = (entry >> 8) as u64;
    f.seek(SeekFrom::Start(offset_sectors * SECTOR))?;

    let mut len_bytes = [0u8; 4];
    f.read_exact(&mut len_bytes)?;
    let blob_len = u32::from_le_bytes(len_bytes) as usize;
    let mut blob = vec![0u8; blob_len];
    f.read_exact(&mut blob)?;

    let blob_bin =
        zstd::stream::decode_all(blob.as_slice()).map_err(|e| RegionError::Zstd(e.to_string()))?;
    decode_chunk_payload(&blob_bin)
}
