use super::*;
use crate::voxel::chunk::{DenseChunk, PalettedChunk};
use crate::voxel::coords::LocalPos;
use glam::{IVec3, UVec3};

#[test]
fn empty_world_returns_none() {
    let w = World::new(42);
    assert_eq!(w.get_block(BlockPos(IVec3::new(0, 0, 0))), None);
}

#[test]
fn inserted_air_chunk_returns_air() {
    let mut w = World::new(42);
    w.insert(ChunkCoord(IVec3::ZERO), PalettedChunk::all_air());
    assert_eq!(w.get_block(BlockPos(IVec3::new(5, 5, 5))), Some(Block::Air));
}

#[test]
fn insert_populates_sky_sources_from_chunk_data() {
    let mut w = World::new(42);
    let mut dense = DenseChunk::empty();
    for lz in 0..CHUNK_DIM_U {
        for lx in 0..CHUNK_DIM_U {
            dense.set(LocalPos(UVec3::new(lx, 10, lz)), Block::Stone);
        }
    }
    let chunk = PalettedChunk::compress(&dense);

    let coord = ChunkCoord(IVec3::ZERO);
    w.insert(coord, chunk);

    let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
        panic!("chunk should be Stored after insert");
    };

    for lz in 0..CHUNK_DIM_U {
        for lx in 0..CHUNK_DIM_U {
            assert_eq!(
                meta.sky_sources.lowest_source_y(lx, lz),
                11,
                "column ({lx},{lz}) should have floor at world y=11",
            );
        }
    }
}

#[test]
fn insert_populates_sky_sources_even_for_all_air() {
    use crate::lighting::NO_SOURCE_FLOOR;

    let mut w = World::new(42);
    let chunk = PalettedChunk::compress(&DenseChunk::empty());
    let coord = ChunkCoord(IVec3::new(2, 1, -3));

    w.insert(coord, chunk);

    let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
        panic!("chunk should be Stored after insert");
    };

    for lz in 0..CHUNK_DIM_U {
        for lx in 0..CHUNK_DIM_U {
            assert_eq!(meta.sky_sources.lowest_source_y(lx, lz), NO_SOURCE_FLOOR);
        }
    }
}

#[test]
fn inserted_chunk_starts_unlit() {
    let mut w = World::new(42);
    let coord = ChunkCoord(IVec3::ZERO);
    w.insert(coord, PalettedChunk::compress(&DenseChunk::empty()));
    let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
        panic!("chunk should be Stored after insert");
    };
    assert!(
        matches!(meta.light_state, LightState::Unlit),
        "new chunks must relight before their light is authoritative",
    );
    assert!(meta.dirty.light);
}

#[test]
fn set_block_marks_chunk_unlit_and_bumps_light_version() {
    let mut w = World::new(42);
    let coord = ChunkCoord(IVec3::ZERO);
    w.insert(coord, PalettedChunk::all_air());
    let before = match w.chunks.get(&coord).unwrap() {
        ChunkSlot::Stored { meta, .. } => meta.light_version,
        ChunkSlot::Pending => unreachable!(),
    };
    w.set_block(BlockPos(IVec3::new(1, 1, 1)), Block::Stone);
    let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
        panic!("chunk should be Stored after edit");
    };
    assert_eq!(meta.light_version, before.wrapping_add(1));
    assert!(meta.dirty.light);
    assert!(matches!(meta.light_state, LightState::Unlit));
}

#[test]
fn set_block_marks_face_neighbors_for_light_reconcile() {
    let mut w = World::new(42);
    let center = ChunkCoord(IVec3::ZERO);
    w.insert(center, PalettedChunk::all_air());
    for dc in [
        IVec3::new(-1, 0, 0),
        IVec3::new(1, 0, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, 0, -1),
        IVec3::new(0, 0, 1),
    ] {
        w.insert(ChunkCoord(center.0 + dc), PalettedChunk::all_air());
    }

    let returned = w.set_block(BlockPos(IVec3::new(8, 8, 8)), Block::Stone);
    assert_eq!(returned.len(), 7);
    assert_eq!(returned[0], center);

    for dc in [
        IVec3::new(-1, 0, 0),
        IVec3::new(1, 0, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, 0, -1),
        IVec3::new(0, 0, 1),
    ] {
        let coord = ChunkCoord(center.0 + dc);
        let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
            panic!("neighbor should be Stored");
        };
        assert!(meta.dirty.light, "{coord:?} was not queued for relight");
        assert!(
            matches!(meta.light_state, LightState::NeedsBorderReconcile),
            "{coord:?} did not enter border reconcile"
        );
    }
}

#[test]
fn set_block_marks_diagonal_chunks_inside_block_light_radius() {
    let mut w = World::new(42);
    let center = ChunkCoord(IVec3::ZERO);
    let diagonal = ChunkCoord(IVec3::new(-1, -1, -1));
    w.insert(center, PalettedChunk::all_air());
    w.insert(diagonal, PalettedChunk::all_air());

    let returned = w.set_block(BlockPos(IVec3::ZERO), Block::Torch);

    assert!(
        returned.contains(&diagonal),
        "diagonal chunk in block-light radius should be relit"
    );
    let ChunkSlot::Stored { meta, .. } = w.chunks.get(&diagonal).unwrap() else {
        panic!("diagonal should be Stored");
    };
    assert!(meta.dirty.light);
    assert!(meta.dirty.mesh);
    assert!(matches!(meta.light_state, LightState::NeedsBorderReconcile));
}

#[test]
fn mark_chunk_unlit_preserves_committed_light_as_reconcile_input() {
    let mut w = World::new(42);
    let coord = ChunkCoord(IVec3::ZERO);
    w.insert(coord, PalettedChunk::compress(&DenseChunk::empty()));
    let ChunkSlot::Stored { meta, .. } = w.chunks.get_mut(&coord).unwrap() else {
        panic!("chunk should be Stored");
    };
    meta.light_state = LightState::Lit { version: 9 };

    w.mark_chunk_unlit(coord, "test");

    let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
        panic!("chunk should be Stored");
    };
    assert!(
        matches!(meta.light_state, LightState::NeedsBorderReconcile),
        "committed light should remain usable while queued for reconcile"
    );
}
