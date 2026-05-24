use super::*;
use crate::voxel::block::Block;
use crate::voxel::coords::{CHUNK_DIM_U, LocalPos};
use glam::UVec3;

#[test]
fn new_filled_returns_all_block() {
    let c = DenseChunk::new_filled(Block::Stone);
    assert!(c.blocks.iter().all(|&b| b == Block::Stone));
}

#[test]
fn empty_is_all_air() {
    let c = DenseChunk::empty();
    assert!(c.blocks.iter().all(|&b| b == Block::Air));
}

#[test]
fn set_get_round_trip() {
    let mut c = DenseChunk::empty();
    let p = LocalPos(UVec3::new(5, 10, 20));
    c.set(p, Block::Dirt);
    assert_eq!(c.get(p), Block::Dirt);
}

#[test]
fn boundary_indices_in_range() {
    let p = LocalPos(UVec3::new(
        CHUNK_DIM_U - 1,
        CHUNK_DIM_U - 1,
        CHUNK_DIM_U - 1,
    ));
    assert_eq!(p.to_index(), CHUNK_VOL - 1);
}

#[test]
fn paletted_all_air_round_trip() {
    let p = PalettedChunk::all_air();
    let d = p.decompress();
    assert!(d.blocks.iter().all(|&b| b == Block::Air));
}

#[test]
fn paletted_compress_decompress_preserves_blocks() {
    let mut d = DenseChunk::empty();
    d.set(LocalPos(UVec3::new(0, 0, 0)), Block::Stone);
    d.set(LocalPos(UVec3::new(31, 31, 31)), Block::Grass);
    d.set(LocalPos(UVec3::new(5, 10, 20)), Block::Water);
    d.sky_light[100] = 0xA;
    d.block_rgb[200] = pack_rgb(0x7, 0x0, 0x0);

    let p = PalettedChunk::compress(&d);
    let d2 = p.decompress();

    assert_eq!(
        d2.blocks[LocalPos(UVec3::new(0, 0, 0)).to_index()],
        Block::Stone
    );
    assert_eq!(
        d2.blocks[LocalPos(UVec3::new(31, 31, 31)).to_index()],
        Block::Grass
    );
    assert_eq!(
        d2.blocks[LocalPos(UVec3::new(5, 10, 20)).to_index()],
        Block::Water
    );
    assert_eq!(d2.sky_light[100], 0xA);
    assert_eq!(d2.block_rgb[200], pack_rgb(0x7, 0x0, 0x0));
}

#[test]
fn paletted_compress_dedupes_palette() {
    let d = DenseChunk::new_filled(Block::Stone);
    let p = PalettedChunk::compress(&d);
    assert_eq!(p.palette.len(), 1);
}

#[test]
fn light_volume_blob_size_and_layout() {
    let mut d = DenseChunk::empty();
    d.sky_light[0] = 15;
    d.block_rgb[0] = pack_rgb(15, 0, 0);
    let n = Neighbors { chunks: [None; 6] };
    let blob = build_light_volume_blob(&d, &n);
    assert_eq!(blob.len(), 33 * 33 * 33 * 4);
    assert!(blob[0] >= 240, "R channel scaled wrong: {}", blob[0]);
    assert_eq!(blob[1], 0);
    assert_eq!(blob[2], 0);
    assert!(blob[3] >= 240, "A channel scaled wrong: {}", blob[3]);
}

#[test]
fn light_volume_halo_reads_negative_face_neighbor() {
    let mut d = DenseChunk::empty();
    let edge = LocalPos(UVec3::new(0, 10, 10));
    d.set(edge, Block::Stone);

    let mut neg_x = DenseChunk::empty();
    let neighbor_air = LocalPos(UVec3::new(31, 10, 10));
    neg_x.sky_light[neighbor_air.to_index()] = 15;

    let n = Neighbors {
        chunks: [None, Some(&neg_x), None, None, None, None],
    };
    let blob = build_light_volume_blob(&d, &n);
    let blob_idx = (10 * 33 * 33 + 10 * 33) * 4;
    assert!(
        blob[blob_idx + 3] >= 240,
        "negative boundary halo did not import sky light: {}",
        blob[blob_idx + 3]
    );
}

#[test]
fn paletted_block_at_matches_decompressed_get() {
    let mut d = DenseChunk::empty();
    d.set(LocalPos(UVec3::new(0, 0, 0)), Block::Stone);
    d.set(LocalPos(UVec3::new(31, 31, 31)), Block::Water);
    d.set(LocalPos(UVec3::new(5, 10, 20)), Block::Torch);
    let p = PalettedChunk::compress(&d);
    for i in 0..CHUNK_VOL {
        assert_eq!(p.block_at(i), d.blocks[i], "block_at mismatch at idx {i}");
    }
}

#[test]
fn paletted_sky_light_at_matches_decompressed() {
    let mut d = DenseChunk::empty();
    d.sky_light[0] = 15;
    d.sky_light[100] = 7;
    d.sky_light[CHUNK_VOL - 1] = 3;
    let p = PalettedChunk::compress(&d);
    assert_eq!(p.sky_light_at(0), 15);
    assert_eq!(p.sky_light_at(100), 7);
    assert_eq!(p.sky_light_at(CHUNK_VOL - 1), 3);
    assert_eq!(p.sky_light_at(50), 0);
}

#[test]
fn paletted_block_rgb_at_matches_decompressed() {
    let mut d = DenseChunk::empty();
    d.block_rgb[10] = pack_rgb(15, 7, 3);
    d.block_rgb[200] = pack_rgb(0, 8, 12);
    let p = PalettedChunk::compress(&d);
    assert_eq!(p.block_rgb_at(10), (15, 7, 3));
    assert_eq!(p.block_rgb_at(200), (0, 8, 12));
    assert_eq!(p.block_rgb_at(11), (0, 0, 0));
}

#[test]
fn paletted_set_sky_light_at_round_trip() {
    let mut p = PalettedChunk::all_air();
    p.set_sky_light_at(5, 12);
    p.set_sky_light_at(6, 8);
    assert_eq!(p.sky_light_at(5), 12);
    assert_eq!(p.sky_light_at(6), 8);
    assert_eq!(p.sky_light_at(7), 0);
}

#[test]
fn paletted_set_block_rgb_at_round_trip() {
    let mut p = PalettedChunk::all_air();
    p.set_block_rgb_at(42, 11, 9, 5);
    assert_eq!(p.block_rgb_at(42), (11, 9, 5));
    assert_eq!(p.block_rgb_at(43), (0, 0, 0));
    p.set_block_rgb_at(42, 0, 0, 0);
    assert_eq!(p.block_rgb_at(42), (0, 0, 0));
}

#[test]
fn packed_rgb_light_round_trips_channels() {
    let packed = PackedRgbLight::new(
        LightLevel::new(15).unwrap(),
        LightLevel::new(7).unwrap(),
        LightLevel::new(3).unwrap(),
    );

    assert_eq!(packed.raw(), pack_rgb(15, 7, 3));
    assert_eq!(packed.channels_u8(), (15, 7, 3));
    assert_eq!(packed.brightness().get(), 15);
    assert_eq!(LightLevel::new(16), None);
}
