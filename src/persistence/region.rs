//! Minecraft-style "region" file format: one file per 16 × 16 grid of
//! chunks.
//!
//! Layout (4 KB sector granularity):
//!
//! ```text
//! +-------- header (4 KB) --------+-----------------------------+
//! | 256 × u32 LE slot entries     | chunk blobs (zstd(bincode)) |
//! +-------------------------------+-----------------------------+
//! ```
//!
//! Each slot entry packs `(offset_in_sectors: u24, len_sectors: u8)`. A
//! zero entry means "slot empty — chunk hasn't been saved here yet".
//! Newly-saved chunks always go to EOF (we accept fragmentation in v0;
//! a future `compact` subcommand will rewrite).
//!
//! Slot index is `(local_cx << 4) | local_cz`, taking the `rem_euclid 16`
//! of the chunk coordinate. Y is *not* a region-axis: every chunk in a
//! given XZ column lives in the same region file.

use crate::voxel::chunk::PalettedChunk;
use crate::voxel::coords::ChunkCoord;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// On-disk sector size. Slot offsets and lengths are measured in sectors.
const SECTOR: u64 = 4096;
/// Edge length of the region grid, in chunks.
const REGION_DIM: i32 = 16;

/// Errors that can arise while reading/writing a region file.
#[derive(Debug, thiserror::Error)]
pub enum RegionError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(String),
    #[error("zstd: {0}")]
    Zstd(String),
    #[error("chunk not present in region")]
    NotPresent,
}

/// Path of the region file that owns `chunk_coord`. The file is in
/// `saves_dir/regions/r.{rx}.{rz}.bin`; `rx`/`rz` are the region grid
/// coordinates derived by floor-dividing the chunk coord by 16.
pub fn region_path(saves_dir: &std::path::Path, chunk_coord: ChunkCoord) -> PathBuf {
    let rx = chunk_coord.0.x.div_euclid(REGION_DIM);
    let rz = chunk_coord.0.z.div_euclid(REGION_DIM);
    saves_dir
        .join("regions")
        .join(format!("r.{rx}.{rz}.bin"))
}

/// Slot index for a given chunk coord (0..=255). Y is unused — see module
/// docs.
fn slot_index(chunk_coord: ChunkCoord) -> usize {
    let lx = chunk_coord.0.x.rem_euclid(REGION_DIM) as usize;
    let lz = chunk_coord.0.z.rem_euclid(REGION_DIM) as usize;
    (lx << 4) | lz
}

/// Encode → compress → append at EOF → update the header. Creates the
/// file (and the `regions` directory) if missing.
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

    // Ensure the file is at least one sector long so the header exists.
    let len = f.metadata()?.len();
    if len < SECTOR {
        f.set_len(SECTOR)?;
    }

    // Pull the current header. A brand-new file reads zeros, which means
    // every slot is "empty" — correct.
    let mut header = [0u8; SECTOR as usize];
    f.seek(SeekFrom::Start(0))?;
    let _ = f.read(&mut header)?;

    // Serialize + compress the chunk into a single sector-padded blob.
    // We prepend a 4-byte little-endian length so the reader knows how
    // much of the sector-padded space to hand to zstd (it can't ignore
    // trailing zeros itself).
    let blob_bin = bincode::serialize(data).map_err(|e| RegionError::Decode(e.to_string()))?;
    let blob = zstd::stream::encode_all(blob_bin.as_slice(), 3)
        .map_err(|e| RegionError::Zstd(e.to_string()))?;
    let with_prefix_len = 4 + blob.len();
    let needed_sectors = (with_prefix_len as u64).div_ceil(SECTOR).max(1);

    // Always append to EOF (v0 accepts fragmentation — a `compact`
    // subcommand can rewrite the file later).
    let end_sector = f.metadata()?.len().div_ceil(SECTOR).max(1);
    f.seek(SeekFrom::Start(end_sector * SECTOR))?;
    f.write_all(&(blob.len() as u32).to_le_bytes())?;
    f.write_all(&blob)?;
    let pad = (needed_sectors * SECTOR) as usize - with_prefix_len;
    if pad > 0 {
        f.write_all(&vec![0u8; pad])?;
    }

    // Pack `(offset_in_sectors, len_sectors)` into the slot's u32 entry.
    let slot = slot_index(coord);
    let entry = ((end_sector as u32) << 8) | (needed_sectors as u32).min(0xFF);
    header[slot * 4..slot * 4 + 4].copy_from_slice(&entry.to_le_bytes());

    // Write the updated header back.
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&header)?;
    Ok(())
}

/// Read the chunk at `coord` out of the region file. Returns
/// `RegionError::NotPresent` if the slot is zero (chunk was never saved).
pub fn read_chunk(
    path: &std::path::Path,
    coord: ChunkCoord,
) -> Result<PalettedChunk, RegionError> {
    let mut f = File::open(path)?;
    let mut header = [0u8; SECTOR as usize];
    f.read_exact(&mut header)?;

    let slot = slot_index(coord);
    let entry = u32::from_le_bytes(header[slot * 4..slot * 4 + 4].try_into().unwrap());
    if entry == 0 {
        return Err(RegionError::NotPresent);
    }
    let offset_sectors = (entry >> 8) as u64;
    f.seek(SeekFrom::Start(offset_sectors * SECTOR))?;

    // Read the 4-byte length prefix, then exactly that many compressed
    // bytes. Don't pass the sector-padding zeros to zstd — it treats them
    // as a malformed second frame.
    let mut len_bytes = [0u8; 4];
    f.read_exact(&mut len_bytes)?;
    let blob_len = u32::from_le_bytes(len_bytes) as usize;
    let mut blob = vec![0u8; blob_len];
    f.read_exact(&mut blob)?;

    let blob_bin = zstd::stream::decode_all(blob.as_slice())
        .map_err(|e| RegionError::Zstd(e.to_string()))?;
    let chunk: PalettedChunk =
        bincode::deserialize(&blob_bin).map_err(|e| RegionError::Decode(e.to_string()))?;
    Ok(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::block::Block;
    use crate::voxel::chunk::{DenseChunk, PalettedChunk};
    use crate::voxel::coords::LocalPos;
    use glam::{IVec3, UVec3};

    #[test]
    fn write_then_read_round_trip() {
        let td = tempfile::tempdir().unwrap();
        let coord = ChunkCoord(IVec3::new(2, 0, 3));
        let path = region_path(td.path(), coord);

        let mut d = DenseChunk::empty();
        d.set(LocalPos(UVec3::new(7, 14, 21)), Block::Torch);
        let chunk = PalettedChunk::compress(&d);

        write_chunk(&path, coord, &chunk).unwrap();
        let read = read_chunk(&path, coord).unwrap();
        let dr = read.decompress();
        assert_eq!(
            dr.blocks[LocalPos(UVec3::new(7, 14, 21)).to_index()],
            Block::Torch
        );
    }

    #[test]
    fn read_missing_slot_returns_not_present() {
        let td = tempfile::tempdir().unwrap();
        // Write *some* chunk so the file exists, but leave slot (0,0,0)
        // empty by writing into a different slot.
        let dummy = PalettedChunk::all_air();
        let other = ChunkCoord(IVec3::new(1, 0, 0));
        let path = region_path(td.path(), other);
        write_chunk(&path, other, &dummy).unwrap();
        let coord = ChunkCoord(IVec3::new(0, 0, 0));
        let r = read_chunk(&path, coord);
        assert!(matches!(r, Err(RegionError::NotPresent)));
    }
}
