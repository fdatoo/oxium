// Per-chunk gen timing harness. Two modes:
//
//   single  — sequential timing, isolates per-chunk cost without
//             worker contention. Useful for catching algorithmic
//             regressions.
//   mt      — N workers pulling coords from a shared queue, mirroring
//             the production gen pool. Exposes contention-driven
//             variance (lock-held-during-build, duplicate cold-key
//             builds, etc.) that single-threaded benches hide.
//
// `cargo run --release --example bench_chunk_gen -- mt`
// `cargo run --release --example bench_chunk_gen -- single`  (default)

use glam::IVec3;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::ChunkCoord;
use oxium::worldgen::Generator;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "single".to_string());
    // Optional 2nd arg for MT mode: worker count override. Lets us
    // sweep variance vs workers to localize the contention point.
    let workers_override: Option<usize> = std::env::args().nth(2).and_then(|s| s.parse().ok());
    let g = Arc::new(Generator::new(42));

    // Same warmup as the original bench: a small disk near origin so
    // the region/macro caches aren't cold for the disk measurement
    // that follows.
    let warmup: Vec<ChunkCoord> = (-1..=1)
        .flat_map(|x| (-1..=1).map(move |z| ChunkCoord(IVec3::new(x, 2, z))))
        .collect();
    println!("warming caches ({} chunks)...", warmup.len());
    let t0 = Instant::now();
    let mut d = DenseChunk::empty();
    for c in &warmup {
        g.fill_chunk(*c, &mut d);
    }
    println!("  warm-up: {:?}", t0.elapsed());

    // Full RENDER_RADIUS disk at the player's chunk level, mirroring
    // the load shape the streaming system actually produces. Use the
    // y-range [0..=4] so the workload is a mix of underground / sea-
    // level / sky — the realistic variance the gen pool sees.
    let mut coords: Vec<ChunkCoord> = Vec::new();
    for cy in 0..=4 {
        for cx in -12..=12 {
            for cz in -12..=12 {
                coords.push(ChunkCoord(IVec3::new(cx, cy, cz)));
            }
        }
    }
    println!("\nmeasuring {} chunks across cy ∈ [0,4]...\n", coords.len());

    match mode.as_str() {
        "single" => bench_single(&g, &coords),
        "mt" => bench_mt(g, coords, workers_override),
        other => {
            eprintln!("unknown mode '{other}', use 'single' or 'mt'");
            std::process::exit(1);
        }
    }
}

fn bench_single(g: &Generator, coords: &[ChunkCoord]) {
    let mut times: Vec<u128> = Vec::with_capacity(coords.len());
    let t0 = Instant::now();
    let mut d = DenseChunk::empty();
    for c in coords {
        let t = Instant::now();
        g.fill_chunk(*c, &mut d);
        times.push(t.elapsed().as_micros());
    }
    print_stats("single-threaded", &times, t0.elapsed());
}

fn bench_mt(g: Arc<Generator>, coords: Vec<ChunkCoord>, workers_override: Option<usize>) {
    // Default to production: (N-1) total worker threads, ceil((N-1)/2)
    // of them in the gen pool. Caller can override via 2nd arg to
    // sweep variance vs worker count.
    let total_logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let gen_threads = workers_override
        .unwrap_or_else(|| ((total_logical - 1).div_ceil(2)).max(1));
    println!(
        "  workers: {gen_threads} ({} logical cores total)\n",
        total_logical
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(gen_threads)
        .build()
        .expect("rayon pool");

    let next = Arc::new(AtomicUsize::new(0));
    let coords = Arc::new(coords);
    // One Vec<u128> per worker so we don't need a mutex on the
    // recording side; merged into a global histogram after.
    let per_worker: Vec<std::sync::Mutex<Vec<u128>>> =
        (0..gen_threads).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let per_worker = Arc::new(per_worker);

    let t0 = Instant::now();
    pool.scope(|s| {
        for worker_id in 0..gen_threads {
            let next = Arc::clone(&next);
            let coords = Arc::clone(&coords);
            let g = Arc::clone(&g);
            let per_worker = Arc::clone(&per_worker);
            s.spawn(move |_| {
                let mut d = DenseChunk::empty();
                let mut local = Vec::with_capacity(coords.len() / gen_threads + 1);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= coords.len() {
                        break;
                    }
                    let t = Instant::now();
                    g.fill_chunk(coords[i], &mut d);
                    local.push(t.elapsed().as_micros());
                }
                *per_worker[worker_id].lock().unwrap() = local;
            });
        }
    });
    let wall = t0.elapsed();

    let mut times: Vec<u128> = Vec::new();
    for w in per_worker.iter() {
        times.extend(w.lock().unwrap().iter().copied());
    }
    print_stats(&format!("multi-threaded ({gen_threads} workers)"), &times, wall);

    // Per-worker spread reveals whether one worker is consistently
    // slower (suggests an asymmetric workload split) or all workers
    // share the same variance (suggests contention).
    println!("\nper-worker timing (mean / max):");
    for (i, w) in per_worker.iter().enumerate() {
        let v = w.lock().unwrap();
        if v.is_empty() {
            continue;
        }
        let mean = v.iter().sum::<u128>() / v.len() as u128;
        let max = *v.iter().max().unwrap();
        println!("  worker {i}: n={} mean={} µs max={} µs", v.len(), mean, max);
    }
}

fn print_stats(label: &str, times: &[u128], wall: std::time::Duration) {
    if times.is_empty() {
        return;
    }
    let mut sorted = times.to_vec();
    sorted.sort();
    let mean = sorted.iter().sum::<u128>() / sorted.len() as u128;
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[sorted.len() * 95 / 100];
    let p99 = sorted[sorted.len() * 99 / 100];
    let max = *sorted.last().unwrap();
    let min = *sorted.first().unwrap();
    println!(
        "{label}: wall={:?}   per-chunk min={} µs   p50={} µs   mean={} µs   p95={} µs   p99={} µs   max={} µs   spread={}×",
        wall,
        min,
        p50,
        mean,
        p95,
        p99,
        max,
        max as f64 / min.max(1) as f64,
    );

    // Histogram so we can SEE the distribution, not just summary
    // stats. Bins are 1 ms wide.
    let bin_w_us = 1000u128;
    let max_bin = (max / bin_w_us) as usize;
    let mut bins = vec![0usize; max_bin + 1];
    for t in &sorted {
        bins[(t / bin_w_us) as usize] += 1;
    }
    println!("histogram (1 ms bins, # chunks):");
    let max_count = *bins.iter().max().unwrap_or(&1);
    for (i, &count) in bins.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let bar_len = (count * 60) / max_count.max(1);
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        println!("  {:>3}-{:>3} ms  {:>4}  {bar}", i, i + 1, count);
    }
}
