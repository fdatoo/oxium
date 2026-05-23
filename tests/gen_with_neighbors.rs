//! Verifies that `Jobs::spawn_relight` plumbs lit neighbour data into
//! the chunk-lighting BFS.
//!
//! Strategy:
//!   1. Manually build a neighbour chunk `A` with a high-emission
//!      torch (15 R) sitting one cell back from its +X boundary, so
//!      A's x=31 column carries strong block-light into chunk B's
//!      x=0 boundary when seeded.
//!   2. Spawn a relight job for chunk `B` with A as the -X neighbour.
//!   3. Wait for the JobResult::Relit to arrive, decompress B,
//!      and assert that B's x=0 column carries non-zero red block-
//!      light — the seeded value attenuated from A's x=31 across
//!      the chunk seam.

use glam::{IVec3, UVec3};
use oxium::jobs::{JobResult, Jobs};
use oxium::voxel::block::{Block, BlockRegistry};
use oxium::voxel::chunk::{ChunkLightInputs, DenseChunk, Neighbors, PalettedChunk, unpack_rgb};
use oxium::voxel::coords::{ChunkCoord, LocalPos};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn spawn_relight_uses_lit_neighbour_block_light_at_boundary() {
    // ── Build neighbour chunk A with a red torch near its +X face.
    //
    // Place the torch at chunk-local (30, 16, 16): one cell inside
    // the +X boundary. A's BFS then propagates the torch's red light
    // outward; cell (31, 16, 16) gets emission - 1 = 14, which is
    // what gets seeded into B's x=0 boundary across the chunk seam.
    let mut reg = BlockRegistry::new();
    reg.set_emission_for_tests(Block::Torch, [15, 0, 0]);
    let reg = Arc::new(reg);

    let mut a_dense = DenseChunk::empty();
    a_dense.set(LocalPos(UVec3::new(30, 16, 16)), Block::Torch);
    let no_neighbors = Neighbors { chunks: [None; 6] };
    oxium::lighting::recompute_chunk(&mut a_dense, &no_neighbors, &reg);

    // Sanity: A's x=31 boundary row at y=16, z=16 should carry red
    // light (one step from the torch at x=30).
    let a_boundary_idx = LocalPos(UVec3::new(31, 16, 16)).to_index();
    let (a_r, _, _) = unpack_rgb(a_dense.block_rgb[a_boundary_idx]);
    assert!(
        a_r >= 13,
        "test setup wrong: A's +X boundary not strongly red ({a_r})"
    );

    let a_packed = Arc::new(PalettedChunk::compress(&a_dense));

    // ── Spawn relight for B with A as the -X neighbour.
    //
    // Coord B = (1, 0, 0), so A is at (0, 0, 0) i.e. B's -X.
    // Face order in Neighbors is [+X, -X, +Y, -Y, +Z, -Z], so A goes
    // into index 1.
    let jobs = Jobs::new();
    let b_coord = ChunkCoord(IVec3::new(1, 0, 0));
    let b_dense = DenseChunk::empty();
    let b_inputs = ChunkLightInputs::from_dense(&b_dense, b_coord, &reg);
    let b_packed = Arc::new(PalettedChunk::compress(&b_dense));
    let mut neighbours: [Option<Arc<PalettedChunk>>; 6] = Default::default();
    neighbours[1] = Some(a_packed);
    let mut neighbor_lit = [false; 6];
    neighbor_lit[1] = true;

    jobs.spawn_relight(
        b_coord,
        7,
        b_packed,
        b_inputs,
        neighbours,
        neighbor_lit,
        reg.clone(),
    );

    // ── Drain the channel until the Relit result for b_coord
    // arrives. The pool may produce unrelated results from
    // background work in worst case, but in this isolated test
    // there's only one job.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut b_packed: Option<PalettedChunk> = None;
    while Instant::now() < deadline {
        match jobs.rx.recv_timeout(Duration::from_millis(100)) {
            Ok(JobResult::Relit {
                coord,
                version,
                data,
                ..
            }) if coord == b_coord && version == 7 => {
                b_packed = Some(data);
                break;
            }
            Ok(_) => continue,
            Err(_) => continue,
        }
    }
    let b_packed = b_packed.expect("gen job timed out");

    // ── Assert: B's x=0 row at (y=16, z=16) carries seeded red light.
    //
    // The seed_from_neighbors pass takes A's mirror cell (x=31) value
    // and attenuates by 1 (cost of crossing the seam), so B's x=0
    // should read A's_x31 - 1. A's x=31 was ~14, so B's x=0 ≈ 13.
    let b_dense = b_packed.decompress();
    let b_boundary_idx = LocalPos(UVec3::new(0, 16, 16)).to_index();
    let (b_r, b_g, b_b) = unpack_rgb(b_dense.block_rgb[b_boundary_idx]);
    assert!(
        b_r >= 10,
        "B's -X boundary did not pick up neighbour seeding: red={b_r} (expected ≥ 10). \
         spawn_relight is not propagating the `neighbors` argument into recompute_chunk."
    );
    assert_eq!(
        b_g, 0,
        "green channel should not leak from a red-only source"
    );
    assert_eq!(
        b_b, 0,
        "blue channel should not leak from a red-only source"
    );
}
