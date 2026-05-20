//! Minecraft-style "region" file format: one file per 16 × 16 × 16 grid
//! of chunks.
//!
//! Layout (4 KB sector granularity):
//!
//! ```text
//! +-------- header (16 KB, 4 sectors) --------+-----------------------+
//! | 4096 × u32 LE slot entries                | chunk blobs (zstd)    |
//! +-------------------------------------------+-----------------------+
//! ```
//!
//! Each slot entry packs `(offset_in_sectors: u24, len_sectors: u8)`. A
//! zero entry means "slot empty — chunk hasn't been saved here yet".
//! Newly-saved chunks always go to EOF (we accept fragmentation in v0;
//! a future `compact` subcommand will rewrite).
//!
//! Slot index is `(local_cx, local_cy, local_cz)` packed as
//! `(lx * 256 + ly * 16 + lz)` where each component is the
//! `rem_euclid 16` of the chunk coordinate — Y is a real region axis,
//! so chunks stacked vertically (e.g., the surface chunk and the cave
//! chunk underneath) live in distinct slots instead of overwriting
//! each other. (v0 of this file had only the XZ axes in slot_index;
//! that bug meant any block break poisoned every Y-chunk in the same
//! XZ column on next load.)

use crate::voxel::chunk::PalettedChunk;
use crate::voxel::coords::ChunkCoord;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Magic prefix written before bincode payload to mark the v2 (RGB
/// block light) chunk format. Old saves have no prefix; the read path
/// falls back to deserialising a `PalettedChunkV1` when the prefix is
/// absent. See PR 2 of the lighting overhaul for context.
const CHUNK_V2_MAGIC: &[u8; 4] = b"OX2\0";

/// On-disk sector size. Slot offsets and lengths are measured in sectors.
const SECTOR: u64 = 4096;
/// Edge length of the region grid, in chunks. The region cube is
/// `REGION_DIM ³ = 4096` slots, which is exactly the size of the
/// 16 KB (= 4 sector) header.
const REGION_DIM: i32 = 16;
/// Header size in bytes: 4096 slots × 4-byte entries.
const HEADER_BYTES: u64 = (REGION_DIM as u64).pow(3) * 4;
/// Header size in sectors. Always >= 1, so the first chunk blob lives
/// at `HEADER_SECTORS * SECTOR`.
const HEADER_SECTORS: u64 = HEADER_BYTES.div_ceil(SECTOR);

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
/// `saves_dir/regions/r.{rx}.{ry}.{rz}.bin`; each component is the
/// region grid coordinate derived by floor-dividing the chunk coord
/// by [`REGION_DIM`]. A region file therefore covers exactly one
/// `16×16×16` cube of chunks (4096 slots).
pub fn region_path(saves_dir: &std::path::Path, chunk_coord: ChunkCoord) -> PathBuf {
    let rx = chunk_coord.0.x.div_euclid(REGION_DIM);
    let ry = chunk_coord.0.y.div_euclid(REGION_DIM);
    let rz = chunk_coord.0.z.div_euclid(REGION_DIM);
    saves_dir
        .join("regions")
        .join(format!("r.{rx}.{ry}.{rz}.bin"))
}

/// Slot index for a given chunk coord (0..4096). Includes Y so two
/// chunks at the same XZ but different Y don't collide on disk.
pub fn slot_index(chunk_coord: ChunkCoord) -> usize {
    let lx = chunk_coord.0.x.rem_euclid(REGION_DIM) as usize;
    let ly = chunk_coord.0.y.rem_euclid(REGION_DIM) as usize;
    let lz = chunk_coord.0.z.rem_euclid(REGION_DIM) as usize;
    (lx * (REGION_DIM as usize) + ly) * (REGION_DIM as usize) + lz
}

/// Number of slots in one region file (4096). Exposed so callers can
/// size their per-region bookkeeping without re-deriving it from
/// [`REGION_DIM`].
pub const REGION_SLOTS: usize = (REGION_DIM as usize).pow(3);

/// The (rx, ry, rz) region grid coordinate that owns `chunk_coord`.
/// Mirrors the math in [`region_path`] so callers can group chunks by
/// their region file without parsing the filename.
pub fn region_coord(chunk_coord: ChunkCoord) -> (i32, i32, i32) {
    (
        chunk_coord.0.x.div_euclid(REGION_DIM),
        chunk_coord.0.y.div_euclid(REGION_DIM),
        chunk_coord.0.z.div_euclid(REGION_DIM),
    )
}

/// Read just the slot table of a region file and return a
/// [`REGION_SLOTS`]-element bitmap of "is this slot non-empty".
///
/// One 16 KB sequential read; skips the per-chunk seek + decompress
/// that [`read_chunk`] does. Used by [`crate::persistence::SaveIndex`]
/// to answer "does any chunk in this region exist on disk" in O(1)
/// per chunk after a single per-region O(16 KB) header read — the
/// difference between "spawn 10 000 NotPresent Loads at startup" and
/// "spawn one Load per chunk the player actually edited".
///
/// Returns `Ok(None)` when the file doesn't exist; `Err` only for
/// real I/O errors (corrupt header, permission denied, …).
pub fn read_presence_bitmap(
    path: &std::path::Path,
) -> Result<Option<Box<[bool; REGION_SLOTS]>>, RegionError> {
    if !path.exists() {
        return Ok(None);
    }
    let mut f = File::open(path)?;
    let mut header = vec![0u8; (HEADER_SECTORS * SECTOR) as usize];
    // Short files (truncated mid-write) yield ErrorKind::UnexpectedEof
    // here; treat that as "no slots present" rather than bubbling — a
    // partial header from a long-ago crash shouldn't stall startup.
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

    // Ensure the file is at least one full header long so reads/writes
    // don't truncate the slot table.
    let header_total = HEADER_SECTORS * SECTOR;
    let len = f.metadata()?.len();
    if len < header_total {
        f.set_len(header_total)?;
    }

    // Pull the current header. A brand-new file reads zeros, which means
    // every slot is "empty" — correct.
    let mut header = vec![0u8; header_total as usize];
    f.seek(SeekFrom::Start(0))?;
    let _ = f.read(&mut header)?;

    // Serialize + compress the chunk into a single sector-padded blob.
    // We prepend a 4-byte little-endian length so the reader knows how
    // much of the sector-padded space to hand to zstd (it can't ignore
    // trailing zeros itself).
    let mut blob_bin = Vec::with_capacity(4 + 64 * 1024);
    blob_bin.extend_from_slice(CHUNK_V2_MAGIC);
    let payload = bincode::serialize(data).map_err(|e| RegionError::Decode(e.to_string()))?;
    blob_bin.extend_from_slice(&payload);
    let blob = zstd::stream::encode_all(blob_bin.as_slice(), 3)
        .map_err(|e| RegionError::Zstd(e.to_string()))?;
    let with_prefix_len = 4 + blob.len();
    let needed_sectors = (with_prefix_len as u64).div_ceil(SECTOR).max(1);

    // Always append to EOF (v0 accepts fragmentation — a `compact`
    // subcommand can rewrite the file later). Floor-divide here so the
    // first blob lands just after the header, not stranded a sector
    // beyond it.
    let cur_end = f.metadata()?.len();
    let end_sector = (cur_end / SECTOR).max(HEADER_SECTORS);
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
    let mut header = vec![0u8; (HEADER_SECTORS * SECTOR) as usize];
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
    let chunk: PalettedChunk = if blob_bin.starts_with(CHUNK_V2_MAGIC) {
        bincode::deserialize(&blob_bin[CHUNK_V2_MAGIC.len()..])
            .map_err(|e| RegionError::Decode(e.to_string()))?
    } else {
        // Legacy v1 path: no magic prefix, single-channel block_light.
        let v1: crate::voxel::chunk::PalettedChunkV1 =
            bincode::deserialize(&blob_bin).map_err(|e| RegionError::Decode(e.to_string()))?;
        v1.into()
    };
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

    /// Regression guard for the v0 region-format bug: two chunks at the
    /// same XZ but different Y *must* live in distinct slots. Without
    /// the Y axis in `slot_index`, writing the upper chunk overwrote
    /// the lower one, and on reload every Y chunk in that XZ column
    /// read back identical data — producing the "floating trees in
    /// midair" artefact across an entire vertical column.
    #[test]
    fn distinct_y_does_not_collide() {
        let td = tempfile::tempdir().unwrap();
        let coord_low = ChunkCoord(IVec3::new(3, 0, 5));
        let coord_high = ChunkCoord(IVec3::new(3, 4, 5));

        // Both coords map to the same XZ region file (rx = 0, rz = 0),
        // but different Y means a *different* region file too now.
        // Validate that anyway, then exercise the slot index by putting
        // two coords in the *same* region file with different ly.
        let path_low = region_path(td.path(), coord_low);
        let path_high = region_path(td.path(), coord_high);
        // Coords 0..15 share ry=0; 4 vs 0 are both ly < 16 so same file.
        assert_eq!(path_low, path_high);

        let mut low = DenseChunk::empty();
        low.set(LocalPos(UVec3::new(1, 1, 1)), Block::Stone);
        let mut high = DenseChunk::empty();
        high.set(LocalPos(UVec3::new(1, 1, 1)), Block::Wood);

        write_chunk(&path_low, coord_low, &PalettedChunk::compress(&low)).unwrap();
        write_chunk(&path_high, coord_high, &PalettedChunk::compress(&high)).unwrap();

        let read_low = read_chunk(&path_low, coord_low).unwrap().decompress();
        let read_high = read_chunk(&path_high, coord_high).unwrap().decompress();
        assert_eq!(read_low.blocks[LocalPos(UVec3::new(1, 1, 1)).to_index()], Block::Stone);
        assert_eq!(read_high.blocks[LocalPos(UVec3::new(1, 1, 1)).to_index()], Block::Wood);
    }

    #[test]
    fn legacy_v1_blob_upgrades_on_read() {
        use crate::voxel::chunk::{PalettedChunkV1, CHUNK_VOL};
        use crate::voxel::packed::Packed4Bit;
        let mut block_light = Packed4Bit::zeros(CHUNK_VOL);
        block_light.set(123, 0x9);
        let v1 = PalettedChunkV1 {
            palette: vec![Block::Air, Block::Torch],
            indices: Packed4Bit::zeros(CHUNK_VOL),
            sky_light: Packed4Bit::zeros(CHUNK_VOL),
            block_light,
        };

        // Write a v1-shaped blob to disk by hand (NO magic prefix).
        let tmp = tempfile::tempdir().unwrap();
        let region_path = tmp.path().join("r.0.0.0.bin");
        {
            let payload = bincode::serialize(&v1).unwrap();
            let zstd_data = zstd::stream::encode_all(payload.as_slice(), 3).unwrap();
            let header_total = (HEADER_SECTORS * SECTOR) as usize;
            let mut header = vec![0u8; header_total];
            // Sector layout: header is HEADER_SECTORS sectors; payload starts at sector HEADER_SECTORS.
            let end_sector = HEADER_SECTORS;
            let blob_len = zstd_data.len() as u32;
            let needed_sectors =
                ((4 + zstd_data.len()) as u64).div_ceil(SECTOR).max(1);
            let slot = slot_index(crate::voxel::coords::ChunkCoord(
                glam::IVec3::new(0, 0, 0),
            ));
            let entry = ((end_sector as u32) << 8) | (needed_sectors as u32).min(0xFF);
            header[slot * 4..slot * 4 + 4].copy_from_slice(&entry.to_le_bytes());
            let mut f = std::fs::File::create(&region_path).unwrap();
            f.write_all(&header).unwrap();
            f.write_all(&blob_len.to_le_bytes()).unwrap();
            f.write_all(&zstd_data).unwrap();
            let written = 4 + zstd_data.len();
            let pad = (needed_sectors * SECTOR) as usize - written;
            if pad > 0 {
                f.write_all(&vec![0u8; pad]).unwrap();
            }
        }

        // Now read it through the regular path and confirm v1→v2 upgrade.
        let chunk = read_chunk(
            &region_path,
            crate::voxel::coords::ChunkCoord(glam::IVec3::new(0, 0, 0)),
        )
        .unwrap();
        // All three channels should equal the original block_light.
        assert_eq!(chunk.block_red.get(123), 0x9);
        assert_eq!(chunk.block_green.get(123), 0x9);
        assert_eq!(chunk.block_blue.get(123), 0x9);
    }
}
