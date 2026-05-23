use super::noise_carvers::CARVER_CELL_COUNT;
use super::*;
use crate::worldgen::density::HeightmapNoise;
use crate::worldgen::region::{CaveSystem, FineRegion, RegionCoord};
use crate::worldgen::terrain_ref::TerrainRef;
use crate::worldgen::tuning::{
    CARVER_CELL_SIZE, CAVE_BAND_MIDDLE, CAVE_BAND_SHALLOW, CAVE_SYSTEMS_PER_REGION,
};
use glam::IVec3;
use noise::NoiseFn;

#[test]
fn smin_extremes() {
    // k=0 reduces to ordinary min
    assert_eq!(smin(0.5, 0.8, 0.0), 0.5);
    assert_eq!(smin(0.8, 0.5, 0.0), 0.5);
    // smin(0, 0, k) pulls below min by k/4 (polynomial peak)
    let v = smin(0.0, 0.0, 1.0);
    assert!(
        (v + 0.25).abs() < 1e-5,
        "smin(0,0,1) should be -0.25, got {v}"
    );
    // smin(a, b, k) <= min(a, b) for all k >= 0
    for k in [0.0, 0.5, 1.5, 3.0] {
        for a in [-0.5, 0.0, 0.5, 2.0] {
            for b in [-0.5, 0.0, 0.5, 2.0] {
                let s = smin(a, b, k);
                assert!(s <= a.min(b) + 1e-5, "smin({a},{b},{k}) = {s} > min");
            }
        }
    }
}

#[test]
fn vertical_connector_connects_adjacent_band_systems() {
    // Force prob = 1.0, scan 8×8 regions, verify at least one connector emitted.
    let mut cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    cfg.cave.vertical_connector_prob = 1.0;
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let terrain = TerrainRef {
        heightmap: &hm,
        climate: &cfg.climate,
        density: &cfg.density,
    };
    let mut found_connector = false;
    'outer: for rx in 0..8_i32 {
        for rz in 0..8_i32 {
            let coord = RegionCoord { x: rx, z: rz };
            let mut region = FineRegion::empty(coord);
            build_systems_for_region(42, coord, terrain, &mut region, &cfg.cave);
            let bands: Vec<DepthBand> = region
                .cave_systems
                .iter()
                .map(infer_band_for_test)
                .collect();
            if !pair_is_adjacent_for_test(&bands) {
                continue;
            }
            for sys in &region.cave_systems {
                if !sys.vertical_connectors.is_empty() {
                    found_connector = true;
                    break 'outer;
                }
            }
        }
    }
    assert!(
        found_connector,
        "no vertical connector emitted in any 2-band region with prob=1.0"
    );
}

fn infer_band_for_test(sys: &CaveSystem) -> DepthBand {
    let cy = (sys.bb_min.y + sys.bb_max.y) / 2;
    if cy >= CAVE_BAND_SHALLOW.0 {
        DepthBand::Shallow
    } else if cy >= CAVE_BAND_MIDDLE.0 {
        DepthBand::Middle
    } else {
        DepthBand::Deep
    }
}

fn pair_is_adjacent_for_test(bands: &[DepthBand]) -> bool {
    (bands.iter().any(|b| matches!(b, DepthBand::Shallow))
        && bands.iter().any(|b| matches!(b, DepthBand::Middle)))
        || (bands.iter().any(|b| matches!(b, DepthBand::Middle))
            && bands.iter().any(|b| matches!(b, DepthBand::Deep)))
}

#[test]
fn system_count_within_bounds() {
    // Roll systems for many regions; the count should always
    // sit in `CAVE_SYSTEMS_PER_REGION` inclusive.
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let terrain = TerrainRef {
        heightmap: &hm,
        climate: &cfg.climate,
        density: &cfg.density,
    };
    for z in -3..=3 {
        for x in -3..=3 {
            let coord = RegionCoord { x, z };
            let mut region = FineRegion::empty(coord);
            build_systems_for_region(42, coord, terrain, &mut region, &cfg.cave);
            let n = region.cave_systems.len();
            assert!(
                (CAVE_SYSTEMS_PER_REGION.0 as usize..=CAVE_SYSTEMS_PER_REGION.1 as usize)
                    .contains(&n),
                "region ({x},{z}) had {n} systems"
            );
        }
    }
}

#[test]
fn system_is_pure_in_seed_and_coord() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let terrain = TerrainRef {
        heightmap: &hm,
        climate: &cfg.climate,
        density: &cfg.density,
    };
    let coord = RegionCoord { x: 2, z: -3 };
    let mut r1 = FineRegion::empty(coord);
    let mut r2 = FineRegion::empty(coord);
    build_systems_for_region(42, coord, terrain, &mut r1, &cfg.cave);
    build_systems_for_region(42, coord, terrain, &mut r2, &cfg.cave);
    assert_eq!(r1.cave_systems.len(), r2.cave_systems.len());
    for (a, b) in r1.cave_systems.iter().zip(&r2.cave_systems) {
        assert_eq!(a.chambers.len(), b.chambers.len());
        assert_eq!(a.tunnels.len(), b.tunnels.len());
        assert_eq!(a.bb_min, b.bb_min);
    }
}

#[test]
fn mst_connects_all_chambers() {
    // Build a system with ≥2 chambers and confirm the tunnel set
    // makes them reachable via BFS over the chamber graph.
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let terrain = TerrainRef {
        heightmap: &hm,
        climate: &cfg.climate,
        density: &cfg.density,
    };
    let coord = RegionCoord { x: 0, z: 0 };
    let mut region = FineRegion::empty(coord);
    build_systems_for_region(42, coord, terrain, &mut region, &cfg.cave);
    for sys in &region.cave_systems {
        if sys.chambers.len() < 2 {
            continue;
        }
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); sys.chambers.len()];
        for t in &sys.tunnels {
            // Find which chamber pair this tunnel connects (by
            // matching control_points[0] and last to chamber
            // centers).
            let pa = t.control_points.first().copied().unwrap();
            let pb = t.control_points.last().copied().unwrap();
            let mut a_idx: Option<usize> = None;
            let mut b_idx: Option<usize> = None;
            for (ci, c) in sys.chambers.iter().enumerate() {
                if (c.center - pa).length() < 0.01 {
                    a_idx = Some(ci);
                }
                if (c.center - pb).length() < 0.01 {
                    b_idx = Some(ci);
                }
            }
            if let (Some(a), Some(b)) = (a_idx, b_idx) {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
        // BFS from chamber 0.
        let mut seen = vec![false; sys.chambers.len()];
        let mut q = std::collections::VecDeque::new();
        q.push_back(0);
        seen[0] = true;
        while let Some(x) = q.pop_front() {
            for &n in &adj[x] {
                if !seen[n] {
                    seen[n] = true;
                    q.push_back(n);
                }
            }
        }
        assert!(
            seen.iter().all(|&v| v),
            "MST didn't connect all chambers in system: {seen:?}"
        );
    }
}

#[test]
fn cave_air_returns_true_inside_chamber_center() {
    // Scan a 4×4 grid of regions until we find one with a chamber.
    // CAVE_SYSTEMS_PER_REGION = (0, 3), so most regions have one.
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let terrain = TerrainRef {
        heightmap: &hm,
        climate: &cfg.climate,
        density: &cfg.density,
    };
    for rx in 0..4 {
        for rz in 0..4 {
            let coord = RegionCoord { x: rx, z: rz };
            let mut region = FineRegion::empty(coord);
            build_systems_for_region(42, coord, terrain, &mut region, &cfg.cave);
            for sys in &region.cave_systems {
                if let Some(c) = sys.chambers.first() {
                    let wx = c.center.x as i32;
                    let wy = c.center.y as i32;
                    let wz = c.center.z as i32;
                    let arr: Vec<&CaveSystem> = vec![sys];
                    assert!(
                        cave_air(wx, wy, wz, &arr),
                        "chamber center should be air, was solid at ({wx},{wy},{wz})"
                    );
                    return;
                }
            }
        }
    }
    panic!("no chambers across 4×4 regions — graph caves disabled?");
}

// ── Noise carver tests ───────────────────────────────────────

#[test]
fn noise_carvers_builds_from_config_without_panicking() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let _ = NoiseCarvers::new(42, &cfg.cave);
}

#[test]
fn noise_carvers_is_deterministic_in_seed() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let a = NoiseCarvers::new(42, &cfg.cave);
    let b = NoiseCarvers::new(42, &cfg.cave);
    let p = [10.0, 5.0, -3.0];
    assert_eq!(a.cheese.get(p), b.cheese.get(p));
    assert_eq!(a.pillar.get(p), b.pillar.get(p));
}

#[test]
fn noise_carvers_different_seeds_differ() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let a = NoiseCarvers::new(42, &cfg.cave);
    let b = NoiseCarvers::new(43, &cfg.cave);
    assert_ne!(
        a.cheese.get([10.0, 5.0, -3.0]),
        b.cheese.get([10.0, 5.0, -3.0])
    );
}

#[test]
fn cheese_signed_density_is_finite_and_in_expected_range() {
    // term1 ∈ [-1, 1], supp ∈ [0, 0.5], layerized ∈ [0,
    // cave_layer_intensity] → result ∈ [-1, 1.5 + intensity].
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    let upper = 1.5 + cfg.cave.cave_layer_intensity + 0.01;
    for wy in (-100..=100).step_by(11) {
        for wx in (-100..100).step_by(17) {
            for wz in (-100..100).step_by(17) {
                for raw_density in [-1.0_f32, 0.0, 1.0, 4.0, 16.0] {
                    let v = cheese_contribution(wx, wy, wz, raw_density, &nc, &cfg.cave);
                    assert!(v.is_finite(), "cheese non-finite at ({wx},{wy},{wz})");
                    assert!(
                        (-1.001..=upper).contains(&v),
                        "out-of-range cheese: {v} (expected [-1, {upper}])"
                    );
                }
            }
        }
    }
}

#[test]
fn cheese_surface_suppression_pushes_density_solid() {
    // At raw_density ≈ 0 (the surface), the suppression term
    // saturates at supp_max (default 0.5). Adding 0.5 to a
    // [-1, 1]-clamped term1 means cheese can never go below
    // -0.5 at the surface — so cheese alone can't fully carve
    // through to air at the natural heightmap.
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    for wx in (-50..50).step_by(7) {
        for wz in (-50..50).step_by(7) {
            let v = cheese_contribution(wx, 60, wz, 0.0, &nc, &cfg.cave);
            assert!(
                v >= -0.5 - 1e-4,
                "surface cheese should be suppressed (>= -0.5), got {v}"
            );
        }
    }
}

#[test]
fn cheese_deep_can_go_negative() {
    // Cheese goes negative only where cave_layer ≈ 0 (cave-rich band)
    // AND cheese noise is sufficiently negative. Both noises share the
    // same XZ base frequency (first_octave=-8, xz_scale=1.0), so their
    // zero-crossings are spatially correlated; a ±256 scan (one
    // wavelength) can miss the combination. We assert the global minimum
    // cheese result over a ±512 × 8-Y-cycle scan is negative, confirming
    // at least one cave-rich voxel exists with seed 42.
    // raw_density=20 zeros the suppression term (deep underground).
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    let mut min_result = f32::MAX;
    for wy in (-256..=0).step_by(1) {
        for wx in (-512..512).step_by(2) {
            for wz in (-512..512).step_by(2) {
                let v = cheese_contribution(wx, wy, wz, 20.0, &nc, &cfg.cave);
                if v < min_result {
                    min_result = v;
                }
            }
        }
    }
    assert!(
        min_result < 0.0,
        "cheese never goes negative (min={min_result:.4}): cheese_offset too high or cave_layer blocks all carving"
    );
}

#[test]
fn pillar_contribution_is_non_negative_and_bounded() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    for wx in (-200..200).step_by(13) {
        for wz in (-200..200).step_by(13) {
            for wy in (-100..=80).step_by(7) {
                let v = pillar_contribution(wx, wy, wz, &nc, &cfg.cave);
                assert!(v >= 0.0);
                assert!(v <= cfg.cave.pillar_intensity + 1e-4);
            }
        }
    }
}

#[test]
fn pillar_cutoff_gate_makes_most_voxels_zero() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    let mut zero_count = 0;
    let mut total = 0;
    for wx in (-128..128).step_by(4) {
        for wz in (-128..128).step_by(4) {
            let v = pillar_contribution(wx, -40, wz, &nc, &cfg.cave);
            if v == 0.0 {
                zero_count += 1;
            }
            total += 1;
        }
    }
    let frac_zero = zero_count as f32 / total as f32;
    assert!(frac_zero >= 0.7, "expected ≥70% zero, got {frac_zero}");
}

// Carver evaluator parity: at corner positions (multiples of 4 from
// the chunk origin), trilerp degenerates to the corner sample, so
// `*_at` must reproduce the direct contribution function exactly.
// Non-corner positions tolerate small differences from the trilerp
// approximation; we verify they stay within a band that's
// comparable to the noise field's local variation.
#[test]
fn carver_evaluator_exact_at_corners() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    let origin = IVec3::new(0, -64, 32);
    let eval = CarverEvaluator::new(&nc, &cfg.cave, origin);

    // Sweep every cell-corner inside the chunk. CARVER_CELL_SIZE=4
    // → corner offsets 0,4,...,28 (the +32 boundary corner is the
    // start of the next chunk; trilerp can still reach corner
    // index 8, but the per-voxel call uses local positions
    // strictly inside the chunk).
    for cz in 0..CARVER_CELL_COUNT {
        for cy in 0..CARVER_CELL_COUNT {
            for cx in 0..CARVER_CELL_COUNT {
                let wx = origin.x + (cx as i32) * CARVER_CELL_SIZE;
                let wy = origin.y + (cy as i32) * CARVER_CELL_SIZE;
                let wz = origin.z + (cz as i32) * CARVER_CELL_SIZE;

                // Multiple raw_density values exercise the cheese
                // suppression band's full range.
                for raw_density in [-2.0_f32, 0.0, 4.0, 16.0] {
                    let direct = cheese_contribution(wx, wy, wz, raw_density, &nc, &cfg.cave);
                    let lerped = eval.cheese_at(wx, wy, wz, raw_density, &cfg.cave);
                    let d = (direct - lerped).abs();
                    assert!(
                        d < 1e-5,
                        "cheese mismatch at ({wx},{wy},{wz}) rd={raw_density}: direct={direct} lerp={lerped} d={d}"
                    );
                }

                let direct = pillar_contribution(wx, wy, wz, &nc, &cfg.cave);
                let lerped = eval.pillar_at(wx, wy, wz, &cfg.cave);
                assert!(
                    (direct - lerped).abs() < 1e-5,
                    "pillar mismatch at ({wx},{wy},{wz}): direct={direct} lerp={lerped}"
                );

                let direct = terasology_ambient(wx, wy, wz, &nc, &cfg.cave, 64.0);
                let lerped = eval.terasology_ambient_at(wx, wy, wz, &cfg.cave, 64.0);
                assert!(
                    (direct - lerped).abs() < 1e-5,
                    "tera mismatch at ({wx},{wy},{wz}): direct={direct} lerp={lerped}"
                );
            }
        }
    }
}

/// Build a test-local `CaveConfig` from `bundled_default` with tighter
/// spatial params so the algorithm's properties show up in a small
/// sampling window (~256 blocks).  The production defaults
/// (`tera_wave: 200`) have a 200-block wavelength; a 256-block window
/// fits only ~1.3 cycles — too few for reliable statistics.  At
/// `tera_wave: 32` (~8 cycles per 256 blocks) the shape and threshold
/// properties appear clearly at the grid sizes used in the tests below.
///
/// This separates "verify algorithm shape" (test concern) from
/// "verify production visual quality" (eyeball in PR1.7).
fn test_cave_cfg() -> crate::worldgen::config::CaveConfig {
    let mut cfg = crate::worldgen::config::WorldgenConfig::bundled_default()
        .expect("default.ron must load")
        .cave;
    cfg.tera_wave = 32.0;
    cfg.tera_supp = 0.55;
    cfg.tera_thresh_depth = 500.0;
    cfg
}

#[test]
fn terasology_ambient_depth_monotonicity() {
    // At deeper Y, more samples should hit "cave" (signed-value < 0).
    // Uses test-local config (tera_wave=32) so the 256-block window
    // captures enough noise cycles for reliable statistics.
    let cave_cfg = test_cave_cfg();
    let nc = NoiseCarvers::new(42, &cave_cfg);
    let surface_y = 64.0;
    let mut frac_by_depth = vec![];
    for depth in [10, 50, 100, 150] {
        let wy = surface_y as i32 - depth;
        let mut hit = 0usize;
        let mut total = 0usize;
        for wx in (-128..128).step_by(4) {
            for wz in (-128..128).step_by(4) {
                total += 1;
                let v = terasology_ambient(wx, wy, wz, &nc, &cave_cfg, surface_y);
                if v < 0.0 {
                    hit += 1;
                }
            }
        }
        frac_by_depth.push(hit as f32 / total as f32);
    }
    // Monotonic non-decreasing toward depth.
    for w in frac_by_depth.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-3,
            "cave fraction decreased with depth: {:?}",
            frac_by_depth
        );
    }
    // Surface band should be ~0%.
    assert!(
        frac_by_depth[0] < 0.05,
        "too many caves near surface: {:?}",
        frac_by_depth
    );
    // Deep band should be > shallow.
    assert!(
        frac_by_depth[3] > frac_by_depth[0] * 2.0,
        "deep band not vastly more cave-rich than shallow: {:?}",
        frac_by_depth
    );
}

#[test]
fn terasology_ambient_surface_suppression_complete_by_supp_depth() {
    // Uses test-local config so supp_depth (123 blocks) is reachable in the
    // sampling window and the suppression effect is strong enough to measure.
    let cave_cfg = test_cave_cfg();
    let nc = NoiseCarvers::new(42, &cave_cfg);
    let surface_y = 64.0;
    // At depth == tera_supp_depth, freq_reduction = max(0, tera_supp - 1.0)
    // which is 0 for any tera_supp <= 1.0. Suppression has fully faded.
    // Verify the cave fraction at that depth is non-trivial.
    let wy = (surface_y - cave_cfg.tera_supp_depth) as i32;
    let mut hit = 0usize;
    let mut total = 0usize;
    for wx in (-128..128).step_by(4) {
        for wz in (-128..128).step_by(4) {
            total += 1;
            if terasology_ambient(wx, wy, wz, &nc, &cave_cfg, surface_y) < 0.0 {
                hit += 1;
            }
        }
    }
    let frac = hit as f32 / total as f32;
    assert!(
        frac > 0.02,
        "expected some caves at supp_depth, got {frac:.3}"
    );
}

#[test]
fn terasology_ambient_horizontal_bias_at_high_y_factor() {
    // With high tera_y_factor, the cave footprint in any horizontal slice
    // should be wider XZ than tall Y for typical features. We approximate
    // this by counting how many cells have horizontal-only-cave vs
    // vertical-only-cave neighbours.
    // Uses test-local config (tera_wave=32) for sufficient statistics in the
    // 256-block sampling window.
    let cave_cfg = test_cave_cfg();
    let nc = NoiseCarvers::new(42, &cave_cfg);
    let surface_y = 64.0;
    let wy_center = -40i32;
    let mut horizontal_runs = 0usize;
    let mut vertical_runs = 0usize;
    for wx in (-128..128).step_by(2) {
        let mut h_run = 0;
        let mut v_run = 0;
        for wz in (-128..128).step_by(2) {
            if terasology_ambient(wx, wy_center, wz, &nc, &cave_cfg, surface_y) < 0.0 {
                h_run += 1;
            } else if h_run > 0 {
                horizontal_runs += h_run;
                h_run = 0;
            }
        }
        for dy in (-30..30).step_by(2) {
            if terasology_ambient(wx, wy_center + dy, 0, &nc, &cave_cfg, surface_y) < 0.0 {
                v_run += 1;
            } else if v_run > 0 {
                vertical_runs += v_run;
                v_run = 0;
            }
        }
    }
    assert!(
        horizontal_runs > vertical_runs,
        "expected horizontal cave extent > vertical: h={horizontal_runs} v={vertical_runs}"
    );
}

#[test]
fn carver_evaluator_off_corner_is_close_to_direct() {
    // Off-corner positions: the trilerp is an approximation. We
    // accept any difference that's bounded by the channel's
    // local variation. Tight enough to catch real bugs (e.g. a
    // wrong corner index) but loose enough to permit smoothing.
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    let origin = IVec3::new(0, -64, 32);
    let eval = CarverEvaluator::new(&nc, &cfg.cave, origin);
    // Off-corner sweep using prime strides so we hit non-multiples
    // of 4 across the whole chunk.
    let mut max_cheese: f32 = 0.0;
    let mut max_pillar: f32 = 0.0;
    for ly in (1..32).step_by(3) {
        for lz in (1..32).step_by(5) {
            for lx in (1..32).step_by(5) {
                let wx = origin.x + lx;
                let wy = origin.y + ly;
                let wz = origin.z + lz;
                let raw_density = 4.0;
                max_cheese = max_cheese.max(
                    (cheese_contribution(wx, wy, wz, raw_density, &nc, &cfg.cave)
                        - eval.cheese_at(wx, wy, wz, raw_density, &cfg.cave))
                    .abs(),
                );
                max_pillar = max_pillar.max(
                    (pillar_contribution(wx, wy, wz, &nc, &cfg.cave)
                        - eval.pillar_at(wx, wy, wz, &cfg.cave))
                    .abs(),
                );
            }
        }
    }
    // The channels live in clamped bands roughly [-1, 1+]. A
    // trilerp approximation of a noise function with frequency
    // ~1/(20-100) blocks over a 4-block span typically gives
    // differences in the 0.05-0.3 range. Pillar can spike when
    // the cutoff gate flips between corners.
    assert!(max_cheese < 0.5, "cheese off-corner max delta {max_cheese}");
    assert!(max_pillar < 1.0, "pillar off-corner max delta {max_pillar}");
}

#[test]
fn style_band_distribution_matches_weights() {
    // Roll 500 systems in each band; verify distributions match weights
    // within ±10%.
    use crate::worldgen::region::RegionCoord;
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let table = &cfg.cave.style_table;
    let bands = [
        ("shallow", DepthBand::Shallow, &table.style_weights_shallow),
        ("middle", DepthBand::Middle, &table.style_weights_middle),
        ("deep", DepthBand::Deep, &table.style_weights_deep),
    ];
    for (name, band, weights) in &bands {
        let mut counts = [0u32; 5];
        for i in 0..500 {
            let s = pick_style(42, RegionCoord { x: i, z: 0 }, 0, *band, &cfg.cave);
            let idx = match s {
                CaveStyle::Cathedral => 0,
                CaveStyle::Warren => 1,
                CaveStyle::Slot => 2,
                CaveStyle::Sump => 3,
                CaveStyle::Karst => 4,
            };
            counts[idx] += 1;
        }
        for (i, &expected_weight) in weights.iter().enumerate() {
            let actual = counts[i] as f32 / 500.0;
            let diff = (actual - expected_weight).abs();
            assert!(
                diff < 0.10,
                "band {name}, style index {i}: expected {expected_weight:.2}, got {actual:.2}"
            );
        }
    }
}

#[test]
fn trunk_links_nearest_neighbour_region_system() {
    let mut cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    cfg.cave.trunk_prob = 1.0;
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let terrain = TerrainRef {
        heightmap: &hm,
        climate: &cfg.climate,
        density: &cfg.density,
    };

    // Build all systems in a 3×3 region grid (owned FineRegions).
    let mut regions: Vec<(RegionCoord, FineRegion)> = vec![];
    for rx in 0..3_i32 {
        for rz in 0..3_i32 {
            let coord = RegionCoord { x: rx, z: rz };
            let mut region = FineRegion::empty(coord);
            build_systems_for_region(42, coord, terrain, &mut region, &cfg.cave);
            regions.push((coord, region));
        }
    }

    // Run build_trunks across the 3×3 grid.
    build_trunks(42, &cfg.cave, &mut regions);

    // After the trunk pass, find at least one system with .trunk set,
    // and verify its endpoint matches a chamber center of some
    // neighbour-region system.
    let mut found = false;
    // Snapshot (region_index, coord, system centers) for lookup.
    let snap: Vec<(usize, RegionCoord, Vec<glam::Vec3>)> = regions
        .iter()
        .enumerate()
        .map(|(i, (coord, region))| {
            let centers: Vec<glam::Vec3> = region
                .cave_systems
                .iter()
                .filter_map(|s| s.chambers.first().map(|c| c.center))
                .collect();
            (i, *coord, centers)
        })
        .collect();

    'outer: for (i, (coord, region)) in regions.iter().enumerate() {
        for sys in &region.cave_systems {
            let Some(trunk) = &sys.trunk else {
                continue;
            };
            let endpoint = *trunk.control_points.last().unwrap();
            // Check if endpoint matches any chamber-0 center in a
            // neighbouring region (8-connected, different region index).
            for (j, other_coord, centers) in &snap {
                if *j == i {
                    continue;
                }
                if (other_coord.x - coord.x).abs() > 1 || (other_coord.z - coord.z).abs() > 1 {
                    continue;
                }
                for &c in centers {
                    if (c - endpoint).length() < 0.5 {
                        found = true;
                        break 'outer;
                    }
                }
            }
        }
    }
    assert!(
        found,
        "no trunk linked to any neighbour-region chamber center"
    );
}
