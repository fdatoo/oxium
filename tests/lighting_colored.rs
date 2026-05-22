//! Integration test for PR 2's colored block-light propagation. Uses a
//! custom `BlockRegistry` that overrides Torch to red-only emission so
//! we can assert per-channel falloff without needing a new block kind.

use glam::UVec3;
use oxium::lighting::recompute_chunk;
use oxium::voxel::block::{Block, BlockRegistry};
use oxium::voxel::chunk::{DenseChunk, Neighbors, unpack_rgb};
use oxium::voxel::coords::LocalPos;

fn empty_neighbors() -> Neighbors<'static> {
    Neighbors { chunks: [None; 6] }
}

#[test]
fn red_torch_propagates_only_red() {
    let mut reg = BlockRegistry::new();
    reg.set_emission_for_tests(Block::Torch, [15, 0, 0]);

    let mut c = DenseChunk::empty();
    c.set(LocalPos(UVec3::new(16, 16, 16)), Block::Torch);
    recompute_chunk(&mut c, &empty_neighbors(), &reg);

    let center = LocalPos(UVec3::new(16, 16, 16)).to_index();
    let adj = LocalPos(UVec3::new(17, 16, 16)).to_index();
    let (cr, cg, cb) = unpack_rgb(c.block_rgb[center]);
    let (ar, ag, ab) = unpack_rgb(c.block_rgb[adj]);

    assert_eq!(cr, 15);
    assert_eq!(cg, 0);
    assert_eq!(cb, 0);
    assert_eq!(ar, 14);
    assert_eq!(ag, 0, "green should not appear from a red-only source");
    assert_eq!(ab, 0, "blue should not appear from a red-only source");
}

#[test]
fn mixed_emissions_compose_per_channel() {
    // Two adjacent-but-not-touching torches, one red-only and one
    // green-only. The mid-point should pick up both R and G but no B.
    let mut reg = BlockRegistry::new();
    reg.set_emission_for_tests(Block::Torch, [15, 0, 0]);
    reg.set_emission_for_tests(Block::Lava, [0, 15, 0]);

    let mut c = DenseChunk::empty();
    c.set(LocalPos(UVec3::new(10, 16, 16)), Block::Torch);
    c.set(LocalPos(UVec3::new(20, 16, 16)), Block::Lava);
    recompute_chunk(&mut c, &empty_neighbors(), &reg);

    let mid = LocalPos(UVec3::new(15, 16, 16)).to_index();
    let (r, g, b) = unpack_rgb(c.block_rgb[mid]);
    assert!(r > 0, "red should reach midpoint from torch");
    assert!(g > 0, "green should reach midpoint from lava");
    assert_eq!(b, 0, "blue must remain zero");
    // Symmetric distance (5 cells from each), so r == g.
    assert_eq!(r, g, "channels are symmetric");
}
