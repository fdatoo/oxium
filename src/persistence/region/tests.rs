use super::*;
use crate::voxel::block::Block;
use crate::voxel::chunk::{DenseChunk, PalettedChunk};
use crate::voxel::coords::{ChunkCoord, LocalPos};
use glam::{IVec3, UVec3};
use std::io::Write;

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
    let dummy = PalettedChunk::all_air();
    let other = ChunkCoord(IVec3::new(1, 0, 0));
    let path = region_path(td.path(), other);
    write_chunk(&path, other, &dummy).unwrap();
    let coord = ChunkCoord(IVec3::new(0, 0, 0));
    let r = read_chunk(&path, coord);
    assert!(matches!(r, Err(RegionError::NotPresent)));
}

#[test]
fn distinct_y_does_not_collide() {
    let td = tempfile::tempdir().unwrap();
    let coord_low = ChunkCoord(IVec3::new(3, 0, 5));
    let coord_high = ChunkCoord(IVec3::new(3, 4, 5));

    let path_low = region_path(td.path(), coord_low);
    let path_high = region_path(td.path(), coord_high);
    assert_eq!(path_low, path_high);

    let mut low = DenseChunk::empty();
    low.set(LocalPos(UVec3::new(1, 1, 1)), Block::Stone);
    let mut high = DenseChunk::empty();
    high.set(LocalPos(UVec3::new(1, 1, 1)), Block::Wood);

    write_chunk(&path_low, coord_low, &PalettedChunk::compress(&low)).unwrap();
    write_chunk(&path_high, coord_high, &PalettedChunk::compress(&high)).unwrap();

    let read_low = read_chunk(&path_low, coord_low).unwrap().decompress();
    let read_high = read_chunk(&path_high, coord_high).unwrap().decompress();
    assert_eq!(
        read_low.blocks[LocalPos(UVec3::new(1, 1, 1)).to_index()],
        Block::Stone
    );
    assert_eq!(
        read_high.blocks[LocalPos(UVec3::new(1, 1, 1)).to_index()],
        Block::Wood
    );
}

#[test]
fn region_coord_and_slot_wrap_negative_chunks() {
    let coord = ChunkCoord(IVec3::new(-1, -17, 16));
    let region = region_coord(coord);
    assert_eq!(region.x, -1);
    assert_eq!(region.y, -2);
    assert_eq!(region.z, 1);

    let slot = region_slot(coord);
    assert_eq!(slot.index(), slot_index(coord));
    assert!(slot.index() < REGION_SLOTS);
}

#[test]
fn legacy_v1_blob_upgrades_on_read() {
    use crate::voxel::chunk::{CHUNK_VOL, PalettedChunkV1};
    use crate::voxel::packed::Packed4Bit;

    let mut block_light = Packed4Bit::zeros(CHUNK_VOL);
    block_light.set(123, 0x9);
    let v1 = PalettedChunkV1 {
        palette: vec![Block::Air, Block::Torch],
        indices: Packed4Bit::zeros(CHUNK_VOL),
        sky_light: Packed4Bit::zeros(CHUNK_VOL),
        block_light,
    };

    let tmp = tempfile::tempdir().unwrap();
    let region_path = tmp.path().join("r.0.0.0.bin");
    {
        let payload = bincode::serialize(&v1).unwrap();
        let zstd_data = zstd::stream::encode_all(payload.as_slice(), 3).unwrap();
        let header_total = (format::HEADER_SECTORS * format::SECTOR) as usize;
        let mut header = vec![0u8; header_total];
        let end_sector = format::HEADER_SECTORS;
        let blob_len = zstd_data.len() as u32;
        let needed_sectors = ((4 + zstd_data.len()) as u64)
            .div_ceil(format::SECTOR)
            .max(1);
        let slot = slot_index(ChunkCoord(IVec3::new(0, 0, 0)));
        let entry = ((end_sector as u32) << 8) | (needed_sectors as u32).min(0xFF);
        header[slot * 4..slot * 4 + 4].copy_from_slice(&entry.to_le_bytes());
        let mut f = std::fs::File::create(&region_path).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&blob_len.to_le_bytes()).unwrap();
        f.write_all(&zstd_data).unwrap();
        let written = 4 + zstd_data.len();
        let pad = (needed_sectors * format::SECTOR) as usize - written;
        if pad > 0 {
            f.write_all(&vec![0u8; pad]).unwrap();
        }
    }

    let chunk = read_chunk(&region_path, ChunkCoord(IVec3::new(0, 0, 0))).unwrap();
    assert_eq!(chunk.block_red.get(123), 0x9);
    assert_eq!(chunk.block_green.get(123), 0x9);
    assert_eq!(chunk.block_blue.get(123), 0x9);
}
