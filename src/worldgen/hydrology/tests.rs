use super::*;
use super::grid::Grid;

fn make_grid(n: usize, h: Vec<i16>) -> Grid {
    Grid {
        n,
        h_fill: vec![0i16; n * n],
        flow_dir: vec![grid::DIR_NONE; n * n],
        flow_acc: vec![0u32; n * n],
        trunk_injection: vec![0u32; n * n],
        inbound_dir: vec![grid::DIR_NONE; n * n],
        inbound_acc: vec![0u32; n * n],
        h,
    }
}

#[test]
fn d8_picks_steepest_downhill() {
    // 3x3 with center higher than all neighbours. D8 picks the
    // *steepest slope* = drop / distance, not just the lowest
    // neighbour. With NW at value 1 (drop 8, slope 8/√2 ≈ 5.66)
    // and N at value 2 (drop 7, slope 7/1 = 7.0), N wins on
    // slope even though NW has the larger absolute drop.
    //   1 2 3
    //   4 9 5
    //   6 7 8
    let h = vec![1i16, 2, 3, 4, 9, 5, 6, 7, 8];
    let mut g = make_grid(3, h);
    g.h_fill = g.h.clone();
    g.compute_flow();
    assert_eq!(g.flow_dir[1 * 3 + 1], 0); // N
}

#[test]
fn sink_fill_raises_local_minimum() {
    // 3x3 with a pit in the middle:
    //   9 9 9
    //   9 0 9
    //   9 9 9
    let h = vec![9i16, 9, 9, 9, 0, 9, 9, 9, 9];
    let mut g = make_grid(3, h.clone());
    g.sink_fill();
    // Center cell should be raised to at least 9 (the rim).
    assert!(
        g.h_fill[1 * 3 + 1] >= 9,
        "center pit should fill to ≥ 9, got {}",
        g.h_fill[1 * 3 + 1]
    );
}

#[test]
fn flow_acc_concentrates_on_lowest_path() {
    // A 5x1 ramp dropping linearly: 5, 4, 3, 2, 1. (Make it 5x5
    // with all rows identical so D8 has a clear east-direction
    // flow.) Verify acc strictly increases from the high end to
    // the low end along the spine.
    let n = 5;
    let mut h = Vec::with_capacity(n * n);
    for iz in 0..n {
        for ix in 0..n {
            let _ = iz;
            h.push((n as i16 - 1 - ix as i16) * 10);
        }
    }
    let mut g = make_grid(n, h);
    g.sink_fill();
    g.compute_flow();
    g.compute_acc();
    // Cell at (0,2), (1,2), (2,2), (3,2): accumulation should
    // increase along the row toward x=n-1 (= the low end).
    let prev = g.flow_acc[2 * n + 0];
    for ix in 1..n {
        let here = g.flow_acc[2 * n + ix];
        assert!(
            here >= prev,
            "acc should be non-decreasing down the ramp: at ix={ix} got {here}, prev {prev}"
        );
    }
}

#[test]
fn build_macro_region_is_deterministic() {
    use super::macro_pass::build_macro_region;
    use crate::worldgen::heightmap::HeightmapNoise;
    use crate::worldgen::region::MacroRegionCoord;
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let coord = MacroRegionCoord { x: 0, z: 0 };
    let m1 = build_macro_region(42, coord, &hm, &cfg.climate, &cfg.density);
    let m2 = build_macro_region(42, coord, &hm, &cfg.climate, &cfg.density);
    // Compare a small sample of cells.
    assert_eq!(m1.flow_dir, m2.flow_dir);
    assert_eq!(m1.flow_acc, m2.flow_acc);
    assert_eq!(m1.is_trunk, m2.is_trunk);
}

#[test]
fn fine_hydro_produces_some_river_cells() {
    // Scan a 5 × 5 grid of regions around the origin. At least one
    // should produce river cells. Some regions are pure ocean and
    // won't have any; we just need one with land + drainage.
    use super::fine_pass::build_fine_hydro;
    use crate::worldgen::heightmap::HeightmapNoise;
    use crate::worldgen::region::{RegionCoord, bitset_get};
    use crate::worldgen::tuning::FINE_CELLS_PER_REGION;
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let macro_cache = crate::worldgen::region::fresh_macro_cache();
    let fine_cache = crate::worldgen::region::fresh_fine_cache();
    let mut total_river_cells = 0usize;
    for z in -2..=2 {
        for x in -2..=2 {
            let coord = RegionCoord { x, z };
            let mut region = crate::worldgen::region::build_fine_region_placeholder(coord);
            region.coord = coord;
            build_fine_hydro(
                42,
                coord,
                &hm,
                &cfg.climate,
                &cfg.density,
                &macro_cache,
                &fine_cache,
                &mut region,
            );
            let n = (FINE_CELLS_PER_REGION * FINE_CELLS_PER_REGION) as usize;
            total_river_cells += (0..n).filter(|&i| bitset_get(&region.is_river, i)).count();
        }
    }
    assert!(
        total_river_cells > 0,
        "no river cells across the 5×5 region scan at seed=42"
    );
}

#[test]
fn river_width_monotonic_downstream() {
    // Flow accumulation only ever increases downstream, so width
    // (which is monotone in acc) should too. Pick any river cell
    // in a built region and chase its flow_dir; widths must be
    // non-decreasing.
    use super::fine_pass::build_fine_hydro;
    use super::grid::{DIR_NONE, DIR_OFFSETS};
    use crate::worldgen::heightmap::HeightmapNoise;
    use crate::worldgen::region::{RegionCoord, bitset_get};
    use crate::worldgen::tuning::FINE_CELLS_PER_REGION;
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let macro_cache = crate::worldgen::region::fresh_macro_cache();
    let fine_cache = crate::worldgen::region::fresh_fine_cache();
    let coord = RegionCoord { x: 0, z: 0 };
    let mut region = crate::worldgen::region::build_fine_region_placeholder(coord);
    region.coord = coord;
    build_fine_hydro(
        42,
        coord,
        &hm,
        &cfg.climate,
        &cfg.density,
        &macro_cache,
        &fine_cache,
        &mut region,
    );
    let n = FINE_CELLS_PER_REGION as usize;
    for iz in 1..n - 1 {
        for ix in 1..n - 1 {
            let idx = iz * n + ix;
            if !bitset_get(&region.is_river, idx) {
                continue;
            }
            let dir = region.flow_dir[idx];
            if dir == DIR_NONE {
                continue;
            }
            let (dx, dz) = DIR_OFFSETS[dir as usize];
            let nx = ix as i32 + dx;
            let nz = iz as i32 + dz;
            if nx < 0 || nz < 0 || nx as usize >= n || nz as usize >= n {
                continue;
            }
            let ni = (nz as usize) * n + (nx as usize);
            if !bitset_get(&region.is_river, ni) {
                continue;
            }
            assert!(
                region.width[ni] >= region.width[idx] - 1e-3,
                "width decreased downstream at ({ix},{iz}) → ({nx},{nz}): {} → {}",
                region.width[idx],
                region.width[ni]
            );
        }
    }
}
