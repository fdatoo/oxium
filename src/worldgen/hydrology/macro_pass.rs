//! Macro-resolution hydrology pass (64 m / cell).
//!
//! `build_macro_region` runs the full D8 + sink-fill + flow-accumulation
//! pipeline on a coarse grid. The result is used by the fine-region builder
//! to inject trunk-river drainage from outside the fine window, producing
//! correctly-sized rivers that flow in from off-screen.
//!
//! See `docs/book/content/part-3-region-build/3.4-hydrology.mdx`.

use super::grid::{DIR_NONE, Grid};
use crate::worldgen::density::HeightmapNoise;
use crate::worldgen::region::{MacroRegion, MacroRegionCoord, bitset_set};
use crate::worldgen::tuning::*;

/// Build the macro region for `coord` from noise alone.
///
/// Runs the full D8 + sink-fill + flow-accumulation pipeline on the
/// coarse (64 m / cell) grid. The result is stored in the macro cache
/// and used by the fine-region builder to inject trunk-river drainage
/// from outside the fine window, producing correctly-sized rivers that
/// flow in from off-screen.
///
/// Pure in `(seed, coord)` — the same key always rebuilds to the same
/// result.
pub fn build_macro_region(
    seed: u64,
    coord: MacroRegionCoord,
    heightmap: &HeightmapNoise,
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
) -> MacroRegion {
    let halo = MACRO_HALO_REGIONS;
    let inner = MACRO_CELLS_PER_REGION;
    let n = ((1 + 2 * halo) * inner) as usize;
    let origin_x = (coord.x - halo) * MACRO_REGION_SIZE;
    let origin_z = (coord.z - halo) * MACRO_REGION_SIZE;

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

    // Sample h_pre at the cell centers of the macro window.
    for iz in 0..n {
        for ix in 0..n {
            let wx = origin_x + (ix as i32) * MACRO_CELL + MACRO_CELL / 2;
            let wz = origin_z + (iz as i32) * MACRO_CELL + MACRO_CELL / 2;
            grid.h[iz * n + ix] =
                heightmap.h_pre(seed, wx as f32, wz as f32, climate, density) as i16;
        }
    }

    grid.sink_fill();
    grid.compute_flow();
    grid.compute_acc();

    // Extract the macro region's interior (drop halo).
    let mut flow_dir = vec![DIR_NONE; (inner * inner) as usize];
    let mut flow_acc = vec![0u32; (inner * inner) as usize];
    let bs_bytes = (inner as usize * inner as usize).div_ceil(8);
    let mut is_trunk = vec![0u8; bs_bytes];
    let mut is_lake = vec![0u8; bs_bytes];
    let mut lake_rim = vec![0i16; (inner * inner) as usize];
    let halo_cells = (halo * inner) as usize;
    for iz in 0..(inner as usize) {
        for ix in 0..(inner as usize) {
            let src = (iz + halo_cells) * n + (ix + halo_cells);
            let dst = iz * (inner as usize) + ix;
            flow_dir[dst] = grid.flow_dir[src];
            flow_acc[dst] = grid.flow_acc[src];
            if grid.flow_acc[src] >= MACRO_RIVER_THRESH {
                bitset_set(&mut is_trunk, dst, true);
            }
            if grid.h_fill[src] > grid.h[src] {
                bitset_set(&mut is_lake, dst, true);
                lake_rim[dst] = grid.h_fill[src];
            }
        }
    }

    MacroRegion {
        coord,
        flow_dir: flow_dir.into_boxed_slice(),
        flow_acc: flow_acc.into_boxed_slice(),
        is_trunk: is_trunk.into_boxed_slice(),
        is_lake: is_lake.into_boxed_slice(),
        lake_rim: lake_rim.into_boxed_slice(),
    }
}
