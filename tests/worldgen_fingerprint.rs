//! Map fingerprint test: catches macro-scale drift in the worldgen
//! output that single-chunk hashes miss. The chunk-hash golden test
//! pins one cubic metre of stone; this test pins a 256 × 256 PNG of
//! the heightmap over a 2 km × 2 km world region.
//!
//! When this test fails, the worldgen output has drifted at the
//! kilometer scale. Re-generate the PNG locally (the test writes it
//! to `target/worldgen_fingerprint.png` on failure) for visual
//! review; if the change is intentional, update `EXPECTED_HASH`.

use oxium::worldgen::Generator;
use oxium::worldgen::heightmap::HeightmapNoise;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Pinned hash of the PNG bytes. Update whenever an intentional
/// generator change lands.
const EXPECTED_HASH: u64 = 0x121A_2133_D9E7_9848;

/// Render a 256 × 256 fingerprint PNG of `h_pre` at seed 42 sampled
/// every 8 blocks over `[-1024, 1024]²`. Returns the PNG bytes.
fn render_fingerprint() -> Vec<u8> {
    let seed = 42u64;
    let noise = HeightmapNoise::new(seed);
    let n = 256usize;
    let mut pixels = vec![0u8; n * n * 3];
    for iz in 0..n {
        for ix in 0..n {
            // Sample world coords from `[-1024, 1024)` at 8 m/pixel.
            let wx = (ix as i32 * 8) - 1024;
            let wz = (iz as i32 * 8) - 1024;
            let h = noise.h_pre(seed, wx as f32, wz as f32);
            // Map height to grayscale. Range roughly `[-50, 140]`;
            // we shift+scale to `[0, 255]` then clamp.
            let v = (((h + 60.0) / 200.0) * 255.0).clamp(0.0, 255.0) as u8;
            let i = (iz * n + ix) * 3;
            pixels[i] = v;
            pixels[i + 1] = v;
            pixels[i + 2] = v;
        }
    }
    let mut out = Vec::new();
    let enc = image::codecs::png::PngEncoder::new(&mut out);
    image::ImageEncoder::write_image(
        enc,
        &pixels,
        n as u32,
        n as u32,
        image::ExtendedColorType::Rgb8,
    )
    .expect("PNG encode failed");
    out
}

fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    b.hash(&mut h);
    h.finish()
}

#[test]
fn fingerprint_is_deterministic() {
    let a = render_fingerprint();
    let b = render_fingerprint();
    assert_eq!(hash_bytes(&a), hash_bytes(&b));
}

#[test]
fn fingerprint_hash_matches_pin() {
    let png = render_fingerprint();
    let actual = hash_bytes(&png);
    if actual != EXPECTED_HASH {
        // Persist the new PNG for visual review.
        let _ = std::fs::create_dir_all("target");
        let _ = std::fs::write("target/worldgen_fingerprint.png", &png);
        panic!(
            "worldgen map fingerprint changed.\n  expected: 0x{:016X}\n  actual:   0x{:016X}\n  \
             PNG written to target/worldgen_fingerprint.png — open to compare. If the \
             change is intentional, update EXPECTED_HASH in tests/worldgen_fingerprint.rs.",
            EXPECTED_HASH, actual
        );
    }
}

/// Cheap sanity: the generator full-pipeline (`fill_chunk`) should
/// still produce a valid chunk for the seed used by the fingerprint
/// — catches a complete generator failure that would also break the
/// fingerprint.
#[test]
fn generator_runs_for_fingerprint_seed() {
    use glam::IVec3;
    use oxium::voxel::chunk::DenseChunk;
    use oxium::voxel::coords::ChunkCoord;

    let g = Generator::new(42);
    let mut c = DenseChunk::empty();
    g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut c);
    let has_non_air = c
        .blocks
        .iter()
        .any(|b| !matches!(*b, oxium::voxel::block::Block::Air));
    assert!(has_non_air, "generator produced an empty chunk at (0, 2, 0)");
}
