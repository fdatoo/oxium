// Quick per-chunk gen timing harness.
// `cargo run --release --example bench_chunk_gen`.

use glam::IVec3;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::ChunkCoord;
use oxium::worldgen::Generator;
use std::time::Instant;

fn main() {
    let g = Generator::new(42);

    // Warm up the region/macro caches with a small disk near the origin.
    // The first chunk pays the full cold-region cost; subsequent
    // chunks reuse the cache and reveal the steady-state per-fill cost.
    let warmup: Vec<ChunkCoord> = (-1..=1)
        .flat_map(|x| (-1..=1).map(move |z| ChunkCoord(IVec3::new(x, 2, z))))
        .collect();
    println!("warming caches ({} chunks)...", warmup.len());
    let t0 = Instant::now();
    let mut d = DenseChunk::empty();
    for c in &warmup {
        g.fill_chunk(*c, &mut d);
    }
    println!(
        "  warm-up: {:?} ({:?}/chunk)",
        t0.elapsed(),
        t0.elapsed() / warmup.len() as u32
    );

    // Steady-state: measure a horizontal disk of radius 12 chunks at
    // the player's elevation (cy = 2). This matches the load radius.
    let mut chunks: Vec<ChunkCoord> = Vec::new();
    for cx in -12..=12 {
        for cz in -12..=12 {
            chunks.push(ChunkCoord(IVec3::new(cx, 2, cz)));
        }
    }
    println!("\nmeasuring {} chunks at cy=2 (RENDER_RADIUS disk)...", chunks.len());
    let mut times: Vec<u128> = Vec::with_capacity(chunks.len());
    let t0 = Instant::now();
    for c in &chunks {
        let t = Instant::now();
        g.fill_chunk(*c, &mut d);
        times.push(t.elapsed().as_micros());
    }
    let total = t0.elapsed();
    times.sort();
    let mean = times.iter().sum::<u128>() / times.len() as u128;
    let p50 = times[times.len() / 2];
    let p95 = times[times.len() * 95 / 100];
    let p99 = times[times.len() * 99 / 100];
    let max = *times.last().unwrap();
    println!(
        "  total: {:?}   per-chunk mean: {} µs   p50: {} µs   p95: {} µs   p99: {} µs   max: {} µs",
        total, mean, p50, p95, p99, max
    );

    // Compare a vertical column scan (the full y range 0..=4 — what
    // VERTICAL_RADIUS actually exercises) at a single column. Reveals
    // whether deep chunks (where caves dominate) cost the same as
    // surface chunks.
    println!("\nper-y cost at (0, cy, 0):");
    for cy in -2..=4 {
        let c = ChunkCoord(IVec3::new(0, cy, 0));
        let t = Instant::now();
        g.fill_chunk(c, &mut d);
        println!("  cy={:>2}  {:?}", cy, t.elapsed());
    }
}
