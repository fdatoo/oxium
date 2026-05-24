//! River segment construction and classification.
//!
//! `build_segments_from_fine` converts the tagged interior cells of the
//! assembled fine grid into `RiverSegment` structs, classifying each cell
//! as `Waterfall`, `Rapid`, or `Channel` based on the terrain drop to its
//! downstream neighbour:
//!
//! - `Waterfall` — drop ≥ 12 blocks
//! - `Rapid`     — drop ≥ 5 blocks
//! - `Channel`   — all other flowing cells
//!
//! See `docs/book/content/part-3-region-build/3.5-rivers-lakes.mdx`.

use super::grid::{DIR_NONE, DIR_OFFSETS, Grid};
use crate::worldgen::region::{
    FineRegion, RegionCoord, RiverSegment, RiverSegmentKind, bitset_get,
};
use crate::worldgen::tuning::*;

/// Convert tagged river cells in `region` into `RiverSegment` structs.
///
/// Walks every interior river cell, computes `from`/`to` world-space
/// endpoints, classifies the drop kind, and derives the water surface
/// elevation from the Planchon-Darboux filled heights (rather than raw
/// terrain) so adjacent segments share a coherent, monotone water surface.
///
/// `halo` and `inner` define the window layout (same values used in
/// `build_fine_hydro`); `n` is the total window side length.
pub(super) fn build_segments_from_fine(
    grid: &Grid,
    coord: RegionCoord,
    region: &mut FineRegion,
    halo: i32,
    inner: i32,
    n: usize,
) {
    let halo_cells = (halo * inner) as usize;
    let inner_u = inner as usize;
    region.segments.clear();
    let region_origin_x = coord.x * FINE_REGION_SIZE;
    let region_origin_z = coord.z * FINE_REGION_SIZE;
    for iz in 0..inner_u {
        for ix in 0..inner_u {
            let dst = iz * inner_u + ix;
            if !bitset_get(&region.is_river, dst) {
                continue;
            }
            let dir = region.flow_dir[dst];
            if dir == DIR_NONE {
                continue;
            }
            let (dx, dz) = DIR_OFFSETS[dir as usize];
            let from = (
                region_origin_x + (ix as i32) * FINE_CELL + FINE_CELL / 2,
                region_origin_z + (iz as i32) * FINE_CELL + FINE_CELL / 2,
            );
            let to = (from.0 + dx * FINE_CELL, from.1 + dz * FINE_CELL);
            let width = region.width[dst];
            // Mouth: segment is at the downstream side of land — if
            // the downstream cell's h_pre is ≤ SEA_LEVEL.
            let nx = ix as i32 + dx;
            let nz = iz as i32 + dz;
            let mouth = nx >= 0
                && nz >= 0
                && nx < inner
                && nz < inner
                && region.h_pre[(nz as usize) * inner_u + (nx as usize)] <= SEA_LEVEL as i16;
            let src = (iz + halo_cells) * n + (ix + halo_cells);
            let here_h = grid.h[src] as i32;
            let (downstream_h, downstream_h_fill) =
                if nx >= 0 && nz >= 0 && nx < inner && nz < inner {
                    let ds = (nz as usize + halo_cells) * n + (nx as usize + halo_cells);
                    (grid.h[ds] as i32, grid.h_fill[ds] as i32)
                } else {
                    (here_h, grid.h_fill[src] as i32)
                };
            let drop = here_h - downstream_h;
            let kind = if drop >= 12 {
                RiverSegmentKind::Waterfall
            } else if drop >= 5 {
                RiverSegmentKind::Rapid
            } else {
                RiverSegmentKind::Channel
            };
            let water_y = if mouth {
                SEA_LEVEL
            } else {
                // Use sink-fill heights rather than raw terrain heights so
                // adjacent segments share a coherent, monotone water surface.
                // h_fill is non-decreasing along the upstream direction by
                // the Planchon-Darboux guarantee, eliminating the per-segment
                // stepping that produced visible water walls.
                (grid.h_fill[src] as i32)
                    .min(downstream_h_fill)
                    .max(SEA_LEVEL + 1)
            };
            let bed_y = water_y - RIVER_BED_DEPTH;
            region.segments.push(RiverSegment {
                from,
                to,
                width,
                water_y,
                bed_y,
                kind,
                mouth,
            });
        }
    }
}
