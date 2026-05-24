//! Valley carving: per-column and per-chunk-grid depth computation.
//!
//! `valley_carve` and `valley_grid` subtract a U-shaped valley profile
//! from `h_pre` at chunk-fill time. `perpendicular_distance` computes
//! signed distance from a world column to a river segment's domain-warped
//! centerline. `for_each_segment` iterates over all segments in a region
//! and its 8 neighbours.
//!
//! See `docs/book/content/part-3-region-build/3.5-rivers-lakes.mdx`.

use crate::worldgen::region::{FineRegion, RiverSegment, bitset_get};
use crate::worldgen::tuning::*;

/// Return the depth (blocks) the river/valley pass should subtract
/// from `h_pre` at world `(wx, wz)`. Looks at all river segments in
/// the chunk's local fine region plus its 8 immediate neighbours
/// (so a river that exits one region carves the valley continuously
/// into the next).
pub fn valley_carve(
    wx: i32,
    wz: i32,
    region: &FineRegion,
    neighbours: &[Option<&FineRegion>; 8],
    seed: u64,
) -> f32 {
    let mut max_depth: f32 = 0.0;
    for_each_segment(region, neighbours, |seg| {
        let d = perpendicular_distance(wx, wz, seg, seed);
        let width = if seg.mouth {
            seg.width.0 * MOUTH_FLARE_MULT
        } else {
            seg.width.0
        };
        let half_w = width * 0.5;
        let half_valley = width * VALLEY_HALF_WIDTH_MULT;
        let depth = if d <= half_w {
            RIVER_BED_DEPTH as f32
        } else if d < half_valley {
            let t = (d - half_w) / (half_valley - half_w);
            // Smoothstep falloff.
            let s = 1.0 - t * t * (3.0 - 2.0 * t);
            s * RIVER_BED_DEPTH as f32
        } else {
            0.0
        };
        if depth > max_depth {
            max_depth = depth;
        }
    });
    max_depth
}

/// Precompute valley-carve depths for an entire 32×32 chunk column grid
/// in one pass over the segment list. Much faster than calling `valley_carve`
/// per-column because it AABB-culls each segment to only the columns it
/// can reach, then only computes `perpendicular_distance` for those columns.
///
/// `origin_wx` and `origin_wz` are the world-space X/Z of the chunk's
/// (0,0) column (i.e. `chunk_coord.x * 32` and `chunk_coord.z * 32`).
#[allow(clippy::needless_range_loop)]
pub fn valley_grid(
    origin_wx: i32,
    origin_wz: i32,
    region: &FineRegion,
    neighbours: &[Option<&FineRegion>; 8],
    seed: u64,
) -> [[f32; 32]; 32] {
    const DIM: i32 = 32;
    let mut grid = [[0.0f32; 32]; 32];

    for_each_segment(region, neighbours, |seg| {
        let width = if seg.mouth {
            seg.width.0 * MOUTH_FLARE_MULT
        } else {
            seg.width.0
        };
        let half_w = width * 0.5;
        let half_valley = width * VALLEY_HALF_WIDTH_MULT;

        // Compute segment AABB expanded by half_valley, then clip to chunk.
        let seg_min_wx = ((seg.from.0.min(seg.to.0) as f32) - half_valley).floor() as i32;
        let seg_max_wx = ((seg.from.0.max(seg.to.0) as f32) + half_valley).ceil() as i32;
        let seg_min_wz = ((seg.from.1.min(seg.to.1) as f32) - half_valley).floor() as i32;
        let seg_max_wz = ((seg.from.1.max(seg.to.1) as f32) + half_valley).ceil() as i32;

        // Clip to chunk world-space bounds; convert to local indices.
        let lx_min = ((seg_min_wx - origin_wx).max(0) as usize).min((DIM - 1) as usize);
        let lx_max = ((seg_max_wx - origin_wx).max(0) as usize).min((DIM - 1) as usize);
        let lz_min = ((seg_min_wz - origin_wz).max(0) as usize).min((DIM - 1) as usize);
        let lz_max = ((seg_max_wz - origin_wz).max(0) as usize).min((DIM - 1) as usize);

        // Guard: if the segment AABB doesn't intersect this chunk at all,
        // both ranges will be clamped to the same side and lx_min > lx_max
        // or lz_min > lz_max. Skip before entering the inner loop.
        if seg_min_wx > origin_wx + DIM - 1
            || seg_max_wx < origin_wx
            || seg_min_wz > origin_wz + DIM - 1
            || seg_max_wz < origin_wz
        {
            return;
        }

        for lz in lz_min..=lz_max {
            for lx in lx_min..=lx_max {
                let wx = origin_wx + lx as i32;
                let wz = origin_wz + lz as i32;
                let d = perpendicular_distance(wx, wz, seg, seed);
                let depth = if d <= half_w {
                    RIVER_BED_DEPTH as f32
                } else if d < half_valley {
                    let t = (d - half_w) / (half_valley - half_w);
                    // Smoothstep falloff — same formula as `valley_carve`.
                    let s = 1.0 - t * t * (3.0 - 2.0 * t);
                    s * RIVER_BED_DEPTH as f32
                } else {
                    0.0
                };
                if depth > grid[lz][lx] {
                    grid[lz][lx] = depth;
                }
            }
        }
    });

    // Second pass: lake bed carve. For every column already inside a lake
    // fine cell, apply its lake_bed_depth (guaranteed ≥ MIN_LAKE_BED_DROP).
    // This runs after the river pass so rivers inside lake basins keep their
    // bed depth where it exceeds the lake bed depth.
    let all_regions =
        std::iter::once(Some(region)).chain(neighbours.iter().map(|o| o.map(|r| r as &FineRegion)));
    for reg in all_regions.flatten() {
        let (rx, rz) = reg.coord.origin();
        for lz in 0..DIM as usize {
            for lx in 0..DIM as usize {
                let wx = origin_wx + lx as i32;
                let wz = origin_wz + lz as i32;
                let ix = (wx - rx).div_euclid(FINE_CELL);
                let iz = (wz - rz).div_euclid(FINE_CELL);
                if ix < 0 || iz < 0 || ix >= FINE_CELLS_PER_REGION || iz >= FINE_CELLS_PER_REGION {
                    continue;
                }
                let i = (iz * FINE_CELLS_PER_REGION + ix) as usize;
                if bitset_get(&reg.is_lake, i) {
                    let depth = reg.lake_bed_depth[i] as f32;
                    if depth > grid[lz][lx] {
                        grid[lz][lx] = depth;
                    }
                }
            }
        }
    }

    grid
}

/// Iterate over every river segment in `region` and `neighbours`.
pub(crate) fn for_each_segment<F: FnMut(&RiverSegment)>(
    region: &FineRegion,
    neighbours: &[Option<&FineRegion>; 8],
    mut f: F,
) {
    for s in &region.segments {
        f(s);
    }
    for r in neighbours.iter().flatten() {
        for s in &r.segments {
            f(s);
        }
    }
}

/// Perpendicular distance (blocks) from world `(wx, wz)` to the
/// segment's perturbed centerline. Adds a domain-warped offset
/// scaled by width so trunk rivers meander hard while small streams
/// stay nearly straight.
pub(crate) fn perpendicular_distance(wx: i32, wz: i32, seg: &RiverSegment, seed: u64) -> f32 {
    let p = (wx as f32, wz as f32);
    let a = (seg.from.0 as f32, seg.from.1 as f32);
    let b = (seg.to.0 as f32, seg.to.1 as f32);
    let ab = (b.0 - a.0, b.1 - a.1);
    let len_sq = ab.0 * ab.0 + ab.1 * ab.1;
    if len_sq < 1e-6 {
        // Degenerate segment — fall back to point distance.
        let dx = p.0 - a.0;
        let dz = p.1 - a.1;
        return (dx * dx + dz * dz).sqrt();
    }
    // Hoist the sqrt here so we can reuse `len` for both `along_world` and
    // the perpendicular unit vector, eliminating the redundant `hypot` call
    // that would otherwise perform a second sqrt on the same quantity.
    let len = len_sq.sqrt();
    let t = (((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / len_sq).clamp(0.0, 1.0);
    let proj = (a.0 + ab.0 * t, a.1 + ab.1 * t);
    // Perpendicular offset based on a quick deterministic hash of the
    // projected point — substitute for a real Simplex meander to
    // avoid pulling another noise field per query. Same shape /
    // smoothness as a hash-noise interpolation.
    let amp = (seg.width.0 * MEANDER_AMP_PER_WIDTH).min(MAX_MEANDER_AMP);
    // Sample a smooth 1D noise along the segment's parametric `t`:
    // hash adjacent buckets and linearly interpolate.
    //
    // `along_world` is the arc-length from `seg.from` to the projected point.
    // Since proj = a + t*ab and t is clamped to [0,1], this equals t * len —
    // one sqrt instead of the previous `hypot` (which hid an extra sqrt).
    let along_world = t * len;
    let bucket = (along_world / 8.0).floor() as i32;
    let frac = along_world / 8.0 - bucket as f32;
    let h0 = crate::worldgen::hash::mix_range(seed, &[seg.from.0, seg.from.1, bucket], -1.0, 1.0);
    let h1 =
        crate::worldgen::hash::mix_range(seed, &[seg.from.0, seg.from.1, bucket + 1], -1.0, 1.0);
    let offset = (h0 * (1.0 - frac) + h1 * frac) * amp;
    // Perpendicular unit vector.
    let perp = (-ab.1 / len, ab.0 / len);
    let perturbed = (proj.0 + perp.0 * offset, proj.1 + perp.1 * offset);
    let dx = p.0 - perturbed.0;
    let dz = p.1 - perturbed.1;
    (dx * dx + dz * dz).sqrt()
}
