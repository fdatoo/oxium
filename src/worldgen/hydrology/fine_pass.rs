//! Fine-resolution hydrology pass (8 m / cell).
//!
//! `build_fine_hydro` runs the 8-step pipeline from the module header:
//! sample h_pre → sink fill → macro trunk injection → D8 flow direction →
//! flow accumulation → river/lake classification → segment extraction →
//! write into `region`.
//!
//! `NeighbourEdges` and `gather_neighbour_edges` provide read-only snapshots
//! of adjacent fine regions for seam stitching (PR 1).
//!
//! See `docs/book/content/part-3-region-build/3.4-hydrology.mdx` and
//! `docs/book/content/part-3-region-build/3.5-rivers-lakes.mdx`.

use super::grid::{DIR_NONE, DIR_OFFSETS, Grid};
use super::macro_pass::build_macro_region;
use crate::worldgen::heightmap::HeightmapNoise;
use crate::worldgen::region::{
    FineRegion, MacroCache, MacroRegionCoord, RegionCoord, RiverSegment, RiverSegmentKind,
    bitset_get, bitset_set,
};
use crate::worldgen::tuning::*;

/// Read-only snapshots of the four cardinal-neighbour fine regions.
///
/// Used by the fine hydrology builder to stitch river flow and
/// accumulation across region boundaries. Each entry is `Some` iff
/// the neighbour is already in the fine cache —
/// `gather_neighbour_edges` never triggers a build. Names denote the
/// direction *to* the neighbour. Corner neighbours are omitted
/// because diagonal contact is only one cell and not worth the added
/// bookkeeping.
///
/// When a neighbour is `Some`, `build_fine_hydro` reads the edge row
/// of that neighbour's flow field and injects it as `inbound_dir` /
/// `inbound_acc` on the corresponding edge of the new grid. This
/// breaks 2-cycles across the seam and keeps large rivers from
/// suddenly shrinking at region boundaries.
pub struct NeighbourEdges {
    pub west: Option<std::sync::Arc<crate::worldgen::region::FineRegion>>,
    pub east: Option<std::sync::Arc<crate::worldgen::region::FineRegion>>,
    pub north: Option<std::sync::Arc<crate::worldgen::region::FineRegion>>,
    pub south: Option<std::sync::Arc<crate::worldgen::region::FineRegion>>,
}

impl NeighbourEdges {
    /// All-`None` view. Equivalent to the pre-PR-1 free-edge behaviour.
    pub fn empty() -> Self {
        Self {
            west: None,
            east: None,
            north: None,
            south: None,
        }
    }
}

/// Peek the four cardinal neighbours in the fine cache without building.
///
/// Returns a [`NeighbourEdges`] where `None` entries indicate that the
/// neighbour has not yet been built. Cold cache entries stay `None`
/// rather than triggering a recursive build — the fine region builder
/// handles the missing-edge case by treating those edges as free-draining.
pub fn gather_neighbour_edges(
    coord: RegionCoord,
    fine_cache: &crate::worldgen::region::FineCache,
) -> NeighbourEdges {
    NeighbourEdges {
        west: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord {
                x: coord.x - 1,
                z: coord.z,
            },
        ),
        east: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord {
                x: coord.x + 1,
                z: coord.z,
            },
        ),
        north: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord {
                x: coord.x,
                z: coord.z - 1,
            },
        ),
        south: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord {
                x: coord.x,
                z: coord.z + 1,
            },
        ),
    }
}

/// Build the hydrology layer of a fine region.
///
/// Runs the 8-step pipeline described in the module header: sample
/// h_pre → sink fill → macro trunk injection → D8 flow direction →
/// flow accumulation → river/lake classification → segment extraction
/// → write into `region`. After this call, `region.flow_acc`,
/// `region.is_river`, `region.is_lake`, `region.lake_rim`,
/// `region.segments`, and `region.h_pre` are all populated.
///
/// Requires `macro_cache` so trunk drainage injected from outside the
/// fine window is properly accounted for. Requires `fine_cache` (read-only
/// peek) so existing cardinal-neighbour regions can stitch their edge
/// flow fields into the new region, preventing rivers from snapping to
/// new directions at region seams.
pub fn build_fine_hydro(
    seed: u64,
    coord: RegionCoord,
    heightmap: &HeightmapNoise,
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
    macro_cache: &MacroCache,
    fine_cache: &crate::worldgen::region::FineCache,
    region: &mut FineRegion,
) {
    // PR 1: gather neighbour edges (peek-only, no build).
    let neighbours = gather_neighbour_edges(coord, fine_cache);
    let halo = FINE_HALO_REGIONS;
    let inner = FINE_CELLS_PER_REGION;
    let n = ((1 + 2 * halo) * inner) as usize;
    let origin_x = (coord.x - halo) * FINE_REGION_SIZE;
    let origin_z = (coord.z - halo) * FINE_REGION_SIZE;

    let mut grid = Grid {
        n,
        h: vec![0i16; n * n],
        h_fill: vec![0i16; n * n],
        flow_dir: vec![DIR_NONE; n * n],
        flow_acc: vec![0u32; n * n],
        trunk_injection: vec![0u32; n * n],
        inbound_dir: vec![DIR_NONE; n * n],
        inbound_acc: vec![0u32; n * n],
    };

    // Sample h_pre at fine cell centers.
    for iz in 0..n {
        for ix in 0..n {
            let wx = origin_x + (ix as i32) * FINE_CELL + FINE_CELL / 2;
            let wz = origin_z + (iz as i32) * FINE_CELL + FINE_CELL / 2;
            grid.h[iz * n + ix] =
                heightmap.h_pre(seed, wx as f32, wz as f32, climate, density) as i16;
        }
    }

    grid.sink_fill();

    // Trunk injection: for each macro cell inside the window flagged
    // as trunk, find the fine cell at its center and add
    // `macro_acc * (FINE_PER_MACRO * FINE_PER_MACRO)` to the
    // injection (rescaling macro-cell-area units to fine-cell-area).
    //
    // The macro window only needs to cover what our fine window can
    // see. Compute which macro cells our window touches.
    let macro_unit = MACRO_CELL;
    let fine_per_macro_axis = FINE_PER_MACRO; // 8
    // Determine which macro regions overlap our fine window.
    let win_min_x = origin_x;
    let win_min_z = origin_z;
    let win_max_x = origin_x + (n as i32) * FINE_CELL;
    let win_max_z = origin_z + (n as i32) * FINE_CELL;
    let mr_min = MacroRegionCoord::containing(win_min_x, win_min_z);
    let mr_max = MacroRegionCoord::containing(win_max_x - 1, win_max_z - 1);
    for mrz in mr_min.z..=mr_max.z {
        for mrx in mr_min.x..=mr_max.x {
            let mr_coord = MacroRegionCoord { x: mrx, z: mrz };
            let mr = crate::worldgen::region::get_macro(macro_cache, mr_coord, || {
                build_macro_region(seed, mr_coord, heightmap, climate, density)
            });
            let mr_origin = mr_coord.origin();
            for miz in 0..(MACRO_CELLS_PER_REGION as usize) {
                for mix in 0..(MACRO_CELLS_PER_REGION as usize) {
                    let mi = miz * (MACRO_CELLS_PER_REGION as usize) + mix;
                    let mwx = mr_origin.0 + (mix as i32) * macro_unit + macro_unit / 2;
                    let mwz = mr_origin.1 + (miz as i32) * macro_unit + macro_unit / 2;
                    // Convert to fine-cell index inside the window.
                    let fix = (mwx - origin_x).div_euclid(FINE_CELL);
                    let fiz = (mwz - origin_z).div_euclid(FINE_CELL);
                    if fix < 0 || fiz < 0 || fix as usize >= n || fiz as usize >= n {
                        continue;
                    }
                    let fi = (fiz as usize) * n + (fix as usize);
                    if bitset_get(&mr.is_trunk, mi) {
                        let macro_acc = mr.flow_acc[mi];
                        let area_factor = (fine_per_macro_axis * fine_per_macro_axis) as u32;
                        grid.trunk_injection[fi] = grid.trunk_injection[fi]
                            .saturating_add(macro_acc.saturating_mul(area_factor));
                    }
                    // Also propagate macro lake rims into the fine
                    // grid: any fine cell that falls inside a macro
                    // lake gets its h_fill raised to the macro rim
                    // (if higher than the fine fill).
                    if bitset_get(&mr.is_lake, mi) {
                        let rim = mr.lake_rim[mi];
                        // Stamp the macro lake over the 8×8 block of
                        // fine cells covered by this macro cell.
                        for dz in 0..fine_per_macro_axis {
                            for dx in 0..fine_per_macro_axis {
                                let cx = fix as i32 + dx - fine_per_macro_axis / 2;
                                let cz = fiz as i32 + dz - fine_per_macro_axis / 2;
                                if cx < 0 || cz < 0 || cx as usize >= n || cz as usize >= n {
                                    continue;
                                }
                                let ci = (cz as usize) * n + (cx as usize);
                                if grid.h[ci] < rim {
                                    grid.h_fill[ci] = grid.h_fill[ci].max(rim);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // PR 1: stitch neighbour-edge inbound hints into this region's
    // interior-boundary cells. Window-grid layout: this region's
    // interior occupies indices in `[halo_cells, halo_cells + inner)`
    // on both axes, where `halo_cells = halo * inner`. The neighbour
    // regions' INTERIOR cells (size = inner * inner) are what we read.
    {
        let halo_cells = (halo * inner) as usize;
        let inner_u = inner as usize;

        // West neighbour: its east-most interior column flows into our
        // west-most interior column when its flow_dir == 2 (east).
        if let Some(west) = neighbours.west.as_ref() {
            for iz in 0..inner_u {
                let neigh_idx = iz * inner_u + (inner_u - 1);
                if west.flow_dir[neigh_idx] != 2 {
                    continue;
                }
                let grid_idx = (halo_cells + iz) * n + halo_cells;
                grid.inbound_dir[grid_idx] = 2;
                grid.inbound_acc[grid_idx] =
                    grid.inbound_acc[grid_idx].saturating_add(west.flow_acc[neigh_idx]);
            }
        }
        // East neighbour: its west-most interior column flows into our
        // east-most interior column when its flow_dir == 6 (west).
        if let Some(east) = neighbours.east.as_ref() {
            for iz in 0..inner_u {
                let neigh_idx = iz * inner_u;
                if east.flow_dir[neigh_idx] != 6 {
                    continue;
                }
                let grid_idx = (halo_cells + iz) * n + (halo_cells + inner_u - 1);
                grid.inbound_dir[grid_idx] = 6;
                grid.inbound_acc[grid_idx] =
                    grid.inbound_acc[grid_idx].saturating_add(east.flow_acc[neigh_idx]);
            }
        }
        // North neighbour: its south-most interior row flows into our
        // north-most interior row when its flow_dir == 4 (south).
        if let Some(north) = neighbours.north.as_ref() {
            for ix in 0..inner_u {
                let neigh_idx = (inner_u - 1) * inner_u + ix;
                if north.flow_dir[neigh_idx] != 4 {
                    continue;
                }
                let grid_idx = halo_cells * n + (halo_cells + ix);
                grid.inbound_dir[grid_idx] = 4;
                grid.inbound_acc[grid_idx] =
                    grid.inbound_acc[grid_idx].saturating_add(north.flow_acc[neigh_idx]);
            }
        }
        // South neighbour: its north-most interior row flows into our
        // south-most interior row when its flow_dir == 0 (north).
        if let Some(south) = neighbours.south.as_ref() {
            for ix in 0..inner_u {
                let neigh_idx = ix;
                if south.flow_dir[neigh_idx] != 0 {
                    continue;
                }
                let grid_idx = (halo_cells + inner_u - 1) * n + (halo_cells + ix);
                grid.inbound_dir[grid_idx] = 0;
                grid.inbound_acc[grid_idx] =
                    grid.inbound_acc[grid_idx].saturating_add(south.flow_acc[neigh_idx]);
            }
        }
    }

    grid.compute_flow();
    grid.compute_acc();

    // Extract the region's interior.
    let halo_cells = (halo * inner) as usize;
    let inner_u = inner as usize;
    region.flow_dir.fill(DIR_NONE);
    region.flow_acc.fill(0);
    for b in region.is_river.iter_mut() {
        *b = 0;
    }
    for b in region.is_lake.iter_mut() {
        *b = 0;
    }
    region.width.fill(0.0);
    region.lake_rim.fill(0);

    for iz in 0..inner_u {
        for ix in 0..inner_u {
            let src = (iz + halo_cells) * n + (ix + halo_cells);
            let dst = iz * inner_u + ix;
            region.flow_dir[dst] = grid.flow_dir[src];
            region.flow_acc[dst] = grid.flow_acc[src];
            region.h_pre[dst] = grid.h[src];
            if grid.flow_acc[src] >= RIVER_THRESH {
                bitset_set(&mut region.is_river, dst, true);
                let w = ((grid.flow_acc[src] as f32).sqrt() * RIVER_WIDTH_SCALE)
                    .clamp(MIN_RIVER_WIDTH, MAX_RIVER_WIDTH);
                region.width[dst] = w;
            }
            // Tag as lake only when the sink-fill raised the cell by at
            // least LAKE_MIN_NATURAL_DEPTH blocks AND the cell's natural
            // height is not deep ocean.
            //
            // The ≥ LAKE_MIN_NATURAL_DEPTH guard suppresses 1-block-deep
            // "scratch" basins that sink-fill creates at every slight
            // terrain depression. Without it, Step-5 carving
            // (MIN_LAKE_BED_DROP = 3) deepens those scratches into visible
            // ponds even though the basin is topographically insignificant.
            //
            // The ≥ SEA_LEVEL-6 guard prevents ocean cells from being
            // tagged as lakes; ocean is handled by the plate-driven
            // predicate in column_data_with, and tagging it here would
            // produce a non-flat "tilted ocean" rim.
            let natural_depth = grid.h_fill[src] - grid.h[src];
            if natural_depth >= LAKE_MIN_NATURAL_DEPTH as i16 && grid.h[src] >= SEA_LEVEL as i16 - 6
            {
                bitset_set(&mut region.is_lake, dst, true);
                region.lake_rim[dst] = grid.h_fill[src];
                // Guarantee at least MIN_LAKE_BED_DROP blocks of open water
                // above the terrain floor so lakes have visible depth.
                region.lake_bed_depth[dst] = natural_depth.max(MIN_LAKE_BED_DROP as i16);
            }
        }
    }

    // Build river segments inside the region.
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
