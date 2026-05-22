//! Graph-based cave systems.
//!
//! Each fine region deterministically rolls `1..=4` cave systems.
//! A system is a 4–8 chamber graph wired by spline tunnels (MST +
//! 1–2 loops), placed inside a bounding box at a depth band
//! (shallow / middle / deep). Each chamber independently rolls
//! whether to expose itself to the surface via a sinkhole, cliff
//! mouth, or skylight.
//!
//! Carving runs at chunk fill time: for each cave system whose
//! bounding box intersects the chunk, ellipsoid + capsule SDFs
//! decide which voxels are air. The grass cap is preserved by
//! refusing to carve within `CAVE_SURFACE_BUFFER` blocks of `h_pre`,
//! except where an explicit entrance feature punches through.
//!
use crate::worldgen::hash::{mix_range, mix_u32, mix_unit};
use crate::worldgen::heightmap::HeightmapNoise;
use crate::worldgen::region::{
    CaveSystem, Chamber, Entrance, EntranceKind, FineRegion, RegionCoord, Tunnel,
};
use crate::worldgen::tuning::*;
use glam::{IVec3, Vec3};
use noise::{Fbm, NoiseFn, Simplex};

/// Distinct cave-system personalities, rolled per system from the
/// region cell id and the system's depth band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaveStyle {
    /// Few large chambers, wide tunnels. Deep-band-biased.
    Cathedral,
    /// Many small chambers, narrow tunnels. Shallow-band-biased.
    Warren,
    /// XZ-stretched chambers, narrow vertical sheets. Mid-band-biased.
    Slot,
    /// Low-clustered chambers (flooded look). Deep-band-biased.
    Sump,
    /// Default — medium chambers, medium tunnels.
    Karst,
}

/// Depth band a system belongs to. Drives bounding-box Y placement
/// and entrance-roll probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthBand {
    Shallow,
    Middle,
    Deep,
}

impl DepthBand {
    fn pick(seed: u64, system_id: i32, region: RegionCoord) -> Self {
        // Three-way roll: 35% Shallow, 35% Middle, 30% Deep.
        let r = mix_unit(seed, &[region.x, region.z, system_id, 100]);
        if r < 0.35 {
            DepthBand::Shallow
        } else if r < 0.70 {
            DepthBand::Middle
        } else {
            DepthBand::Deep
        }
    }

    /// `[y_min, y_max]` for chamber placement (inclusive).
    fn range(self) -> (i32, i32) {
        match self {
            DepthBand::Shallow => CAVE_BAND_SHALLOW,
            DepthBand::Middle => CAVE_BAND_MIDDLE,
            DepthBand::Deep => CAVE_BAND_DEEP,
        }
    }

    fn entrance_prob(self) -> f32 {
        match self {
            DepthBand::Shallow => ENTRANCE_PROB_SHALLOW,
            DepthBand::Middle => ENTRANCE_PROB_MIDDLE,
            DepthBand::Deep => ENTRANCE_PROB_DEEP,
        }
    }
}

/// Roll a `CaveStyle` deterministically from `(seed, region_coord,
/// system_idx, band)`. Band-weighted via `CaveStyleTable`.
pub fn pick_style(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    band: DepthBand,
    cfg: &crate::worldgen::config::CaveConfig,
) -> CaveStyle {
    let u = mix_unit(seed, &[coord.x, coord.z, system_idx, 7000]);
    let weights = match band {
        DepthBand::Shallow => &cfg.style_table.style_weights_shallow,
        DepthBand::Middle  => &cfg.style_table.style_weights_middle,
        DepthBand::Deep    => &cfg.style_table.style_weights_deep,
    };
    let mut acc = 0.0;
    let styles = [
        CaveStyle::Cathedral, CaveStyle::Warren, CaveStyle::Slot,
        CaveStyle::Sump,      CaveStyle::Karst,
    ];
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if u <= acc { return styles[i]; }
    }
    CaveStyle::Karst
}

/// Build all cave systems for the given fine region. Each system is
/// deterministically derived from `(seed, coord, system_idx)`.
pub fn build_systems_for_region(
    seed: u64,
    coord: RegionCoord,
    heightmap: &HeightmapNoise,
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
    region: &mut FineRegion,
    cave_cfg: &crate::worldgen::config::CaveConfig,
) {
    // Decide how many systems this region hosts.
    // CAVE_SYSTEMS_PER_REGION acts as a compile-time safety cap;
    // cave_cfg.systems_per_region_max is the hot-reloadable config value.
    let n_min = CAVE_SYSTEMS_PER_REGION.0;
    let n_max = cave_cfg.systems_per_region_max.min(CAVE_SYSTEMS_PER_REGION.1);
    let n = n_min
        + (mix_u32(seed, &[coord.x, coord.z, 1])
            % (n_max - n_min + 1));
    region.cave_systems.clear();
    for system_idx in 0..n as i32 {
        let sys = build_system(seed, coord, system_idx, heightmap, climate, density, cave_cfg);
        region.cave_systems.push(sys);
    }
    build_vertical_connectors(seed, coord, cave_cfg, region);
}

/// For each pair of systems in adjacent bands within `region`, roll
/// `vertical_connector_prob`. If passing, push a `Tunnel` from the upper
/// system's lowest chamber to the lower system's highest chamber.
pub fn build_vertical_connectors(
    seed: u64,
    coord: RegionCoord,
    cfg: &crate::worldgen::config::CaveConfig,
    region: &mut FineRegion,
) {
    if cfg.vertical_connector_prob <= 0.0 {
        return;
    }
    // Snapshot which band each system belongs to (avoids borrow conflict).
    let bands: Vec<DepthBand> = region.cave_systems.iter().map(|s| {
        let cy = (s.bb_min.y + s.bb_max.y) / 2;
        if cy >= CAVE_BAND_SHALLOW.0 {
            DepthBand::Shallow
        } else if cy >= CAVE_BAND_MIDDLE.0 {
            DepthBand::Middle
        } else {
            DepthBand::Deep
        }
    }).collect();
    let band_idx = |b: DepthBand| -> u8 {
        match b {
            DepthBand::Shallow => 0,
            DepthBand::Middle  => 1,
            DepthBand::Deep    => 2,
        }
    };
    for i in 0..region.cave_systems.len() {
        for j in 0..region.cave_systems.len() {
            if i == j {
                continue;
            }
            // Only Shallow→Middle or Middle→Deep (i is upper, j is lower).
            if band_idx(bands[j]) != band_idx(bands[i]) + 1 {
                continue;
            }
            let u = mix_unit(seed, &[coord.x, coord.z, i as i32, j as i32, 9500]);
            if u > cfg.vertical_connector_prob {
                continue;
            }
            // Upper system's lowest chamber (minimum center.y).
            let upper_lowest = region.cave_systems[i]
                .chambers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.center.y.partial_cmp(&b.center.y).unwrap())
                .map(|(idx, _)| idx);
            // Lower system's highest chamber (maximum center.y).
            let lower_highest = region.cave_systems[j]
                .chambers
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.center.y.partial_cmp(&b.center.y).unwrap())
                .map(|(idx, _)| idx);
            let (Some(a_idx), Some(b_idx)) = (upper_lowest, lower_highest) else {
                continue;
            };
            let pa = region.cave_systems[i].chambers[a_idx].center;
            let pb = region.cave_systems[j].chambers[b_idx].center;
            let connector = Tunnel {
                control_points: vec![pa, pb],
                radius: cfg.vertical_connector_r,
            };
            region.cave_systems[i].vertical_connectors.push(connector);
        }
    }
}

/// For each cave system in the centre region of `regions`, roll
/// `trunk_prob`. If passing, link to the nearest chamber-0 in any 8-
/// neighbour region. The trunk runs from the system's chamber 0 through
/// a 40-block perpendicular-offset midpoint to the other system's
/// chamber 0.
///
/// Architecture note: trunks are cross-region features — a single
/// `FineRegion` build cannot see neighbour regions (they may not exist
/// yet in the cache). `build_trunks` takes an owned `[(RegionCoord,
/// FineRegion)]` slice that covers all regions of interest and is
/// called:
///   * from tests (on owned data — `trunk` field verified directly), and
///   * from `cave_sdf` / `cave_air` on-the-fly (trunk geometry
///     recomputed from the flat `&[&CaveSystem]` slice already
///     aggregated across the 3×3 region halo).
///
/// Salt 9000 = trunk probability roll; 9001 = midpoint offset roll.
pub fn build_trunks(
    seed: u64,
    cfg: &crate::worldgen::config::CaveConfig,
    regions: &mut [(RegionCoord, FineRegion)],
) {
    if cfg.trunk_prob <= 0.0 {
        return;
    }
    // Snapshot (region_index, system_index, coord, chamber-0 center)
    // for lookup — avoids borrow conflict when writing .trunk later.
    let snap: Vec<(usize, usize, RegionCoord, glam::Vec3)> = regions
        .iter()
        .enumerate()
        .flat_map(|(ri, (coord, region))| {
            region
                .cave_systems
                .iter()
                .enumerate()
                .filter_map(|(si, sys)| {
                    sys.chambers.first().map(|c| (ri, si, *coord, c.center))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // For each system, find the nearest chamber-0 in an 8-neighbour region.
    for &(ri, si, coord, my_center) in &snap {
        // Stable salt: chamber-0 center coords (integer-rounded). Stable
        // across chunks because chamber centers are immutable; unique per
        // system within a region because Poisson sampling enforces
        // min_spacing > 0.
        let salt_x = my_center.x as i32;
        let salt_y = my_center.y as i32;
        let salt_z = my_center.z as i32;
        let u = mix_unit(seed, &[coord.x, coord.z, salt_x, salt_y, salt_z, 9000]);
        if u > cfg.trunk_prob {
            continue;
        }
        // Find nearest neighbour-region chamber-0 (different region index,
        // within 1 step in both x and z).
        let mut best: Option<(glam::Vec3, f32)> = None;
        for &(ori, _osi, other_coord, other_center) in &snap {
            if ori == ri {
                continue;
            }
            let dx = (other_coord.x - coord.x).abs();
            let dz = (other_coord.z - coord.z).abs();
            if dx > 1 || dz > 1 {
                continue;
            }
            let d = (other_center - my_center).length();
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((other_center, d));
            }
        }
        let Some((other_center, _)) = best else {
            continue;
        };
        // Mid-arc: midpoint with a 40-block perpendicular offset in XZ.
        let axis = other_center - my_center;
        let len = (axis.x * axis.x + axis.z * axis.z).sqrt().max(1.0);
        let perp = glam::Vec3::new(-axis.z / len, 0.0, axis.x / len);
        let o = (mix_unit(seed, &[coord.x, coord.z, salt_x, salt_y, salt_z, 9001]) * 2.0 - 1.0) * 40.0;
        let mid = glam::Vec3::new(
            (my_center.x + other_center.x) * 0.5 + perp.x * o,
            (my_center.y + other_center.y) * 0.5,
            (my_center.z + other_center.z) * 0.5 + perp.z * o,
        );
        regions[ri].1.cave_systems[si].trunk = Some(Tunnel {
            control_points: vec![my_center, mid, other_center],
            radius: cfg.trunk_r,
        });
    }
}

/// Per-style parameter set extracted from the style table.
struct StyleParams {
    chamber_count: (u32, u32),
    r_xz: (f32, f32),
    r_y: (f32, f32),
    tunnel_r: (f32, f32),
}

fn style_params(
    style: CaveStyle,
    table: &crate::worldgen::config::CaveStyleTable,
) -> StyleParams {
    match style {
        CaveStyle::Cathedral => StyleParams {
            chamber_count: table.cathedral_chamber_count,
            r_xz: table.cathedral_r_xz,
            r_y: table.cathedral_r_y,
            tunnel_r: table.cathedral_tunnel_r,
        },
        CaveStyle::Warren => StyleParams {
            chamber_count: table.warren_chamber_count,
            r_xz: table.warren_r_xz,
            r_y: table.warren_r_y,
            tunnel_r: table.warren_tunnel_r,
        },
        CaveStyle::Slot => StyleParams {
            chamber_count: table.slot_chamber_count,
            r_xz: table.slot_r_xz,
            r_y: table.slot_r_y,
            tunnel_r: table.slot_tunnel_r,
        },
        CaveStyle::Sump => StyleParams {
            chamber_count: table.sump_chamber_count,
            r_xz: table.sump_r_xz,
            r_y: table.sump_r_y,
            tunnel_r: table.sump_tunnel_r,
        },
        CaveStyle::Karst => StyleParams {
            chamber_count: table.karst_chamber_count,
            r_xz: table.karst_r_xz,
            r_y: table.karst_r_y,
            tunnel_r: table.karst_tunnel_r,
        },
    }
}

/// Build one cave system inside region `coord`.
fn build_system(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    heightmap: &HeightmapNoise,
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
    cave_cfg: &crate::worldgen::config::CaveConfig,
) -> CaveSystem {
    let band = DepthBand::pick(seed, system_idx, coord);
    let (y_min, y_max) = band.range();

    // Roll the style for this system.
    let style = pick_style(seed, coord, system_idx, band, cave_cfg);
    let sp = style_params(style, &cave_cfg.style_table);

    // Bounding-box footprint inside the region. The box is allowed to
    // straddle the region boundary — neighbouring regions consult our
    // systems via the 3 × 3 region neighbourhood at chunk fill time.
    let bb_size = IVec3::new(220, (y_max - y_min).min(64), 220);
    let region_origin = coord.origin();
    let bb_origin_x = region_origin.0
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, 10])
            % (FINE_REGION_SIZE - bb_size.x).max(1) as u32) as i32
        - (bb_size.x / 2);
    let bb_origin_z = region_origin.1
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, 11])
            % (FINE_REGION_SIZE - bb_size.z).max(1) as u32) as i32
        - (bb_size.z / 2);
    let bb_origin_y = y_min;
    let bb_min = IVec3::new(bb_origin_x, bb_origin_y, bb_origin_z);
    let bb_max = bb_min + bb_size;

    // Chamber count — from style table.
    let (cn_min, cn_max) = sp.chamber_count;
    let chamber_count = cn_min
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, 20])
            % (cn_max - cn_min + 1));

    // Pre-compute Sump bb_center_y / bb_half_y for the bias formula.
    let bb_center_y = (bb_min.y + bb_max.y) as f32 * 0.5;
    let bb_half_y = (bb_max.y - bb_min.y) as f32 * 0.5;

    // Poisson-disk-like rejection sampling for chamber centers.
    let mut chambers: Vec<Chamber> = Vec::with_capacity(chamber_count as usize);
    let mut tries = 0u32;
    let max_tries = 200u32;
    let mut attempt = 0i32;
    while chambers.len() < chamber_count as usize && tries < max_tries {
        let sx = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 30, attempt],
            0.0,
            (bb_max.x - bb_min.x) as f32,
        );
        let sy = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 31, attempt],
            0.0,
            (bb_max.y - bb_min.y) as f32,
        );
        let sz = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 32, attempt],
            0.0,
            (bb_max.z - bb_min.z) as f32,
        );

        // Depth multiplier: deeper = larger chambers.
        let cy_raw = bb_min.y as f32 + sy;
        let depth_mult = 1.0 + cave_cfg.depth_scale * ((40.0 - cy_raw).max(0.0) / 80.0);

        let rx = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 40, attempt],
            sp.r_xz.0,
            sp.r_xz.1,
        ) * depth_mult;
        let ry = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 41, attempt],
            sp.r_y.0,
            sp.r_y.1,
        ) * depth_mult;
        let rz = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 42, attempt],
            sp.r_xz.0,
            sp.r_xz.1,
        ) * depth_mult;

        // Slot deliberately drops rz and derives both XZ radii from rx for
        // the elongated-in-X look. Other styles use independent rx, rz.
        let (rx_final, rz_final) = if style == CaveStyle::Slot {
            (rx * 1.4, rx * 0.6)
        } else {
            (rx, rz)
        };

        // Sump style: bias chambers low in the bounding box (floor cluster).
        let cy_final = if style == CaveStyle::Sump {
            bb_center_y - bb_half_y * 0.4 + (cy_raw - bb_center_y).abs() * 0.5
        } else {
            cy_raw
        };

        let center = Vec3::new(
            bb_min.x as f32 + sx,
            cy_final,
            bb_min.z as f32 + sz,
        );
        let radii = Vec3::new(rx_final, ry, rz_final);
        let mean_r = (rx_final + ry + rz_final) / 3.0;
        let min_spacing = mean_r * POISSON_MIN_SPACING_MULT;
        // Reject if too close to any existing chamber.
        let too_close = chambers.iter().any(|c| {
            let d = c.center - center;
            d.length() < min_spacing
        });
        attempt += 1;
        tries += 1;
        if too_close {
            continue;
        }
        chambers.push(Chamber { center, radii });
    }

    // MST via Kruskal.
    let mut tunnels: Vec<Tunnel> = Vec::new();
    if chambers.len() >= 2 {
        let mut edges: Vec<(f32, usize, usize)> = Vec::new();
        for i in 0..chambers.len() {
            for j in (i + 1)..chambers.len() {
                let d = (chambers[i].center - chambers[j].center).length();
                edges.push((d, i, j));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut parent: Vec<usize> = (0..chambers.len()).collect();
        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        let mut mst_edges: Vec<(usize, usize)> = Vec::new();
        for &(_, a, b) in &edges {
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            if ra != rb {
                parent[ra] = rb;
                mst_edges.push((a, b));
            }
        }
        // Optional loop edges: pick the shortest edges not already in
        // the MST.
        let extra_loop_count = MST_EXTRA_LOOPS.0
            + (mix_u32(seed, &[coord.x, coord.z, system_idx, 50])
                % (MST_EXTRA_LOOPS.1 - MST_EXTRA_LOOPS.0 + 1));
        let mut added_extras = 0u32;
        for &(_, a, b) in &edges {
            if added_extras >= extra_loop_count {
                break;
            }
            if mst_edges.iter().any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
            {
                continue;
            }
            mst_edges.push((a, b));
            added_extras += 1;
        }
        // Build a tunnel for each edge — radius from per-style range.
        for (idx, &(a, b)) in mst_edges.iter().enumerate() {
            let radius = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 60, idx as i32],
                sp.tunnel_r.0,
                sp.tunnel_r.1,
            );
            let pa = chambers[a].center;
            let pb = chambers[b].center;
            // Catmull-Rom-friendly control polyline: pa, mid +
            // domain-warped perpendicular offset 1, mid +
            // perpendicular offset 2, pb. Offsets break the otherwise
            // straight line into a meander; the 4-point polyline is
            // sampled later by the SDF as N straight capsules.
            let axis = (pb - pa).normalize_or_zero();
            // Two arbitrary orthogonal axes for perpendicular warp.
            let world_up = Vec3::Y;
            let perp1 = axis.cross(world_up).try_normalize().unwrap_or(Vec3::X);
            let perp2 = axis.cross(perp1).try_normalize().unwrap_or(Vec3::Z);
            let amp = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 61, idx as i32],
                radius * 1.0,
                radius * 1.5,
            ) * 4.0;
            let off1_u = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 62, idx as i32],
                -amp,
                amp,
            );
            let off1_v = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 63, idx as i32],
                -amp,
                amp,
            );
            let off2_u = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 64, idx as i32],
                -amp,
                amp,
            );
            let off2_v = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 65, idx as i32],
                -amp,
                amp,
            );
            let p1 = pa.lerp(pb, 0.33) + perp1 * off1_u + perp2 * off1_v;
            let p2 = pa.lerp(pb, 0.66) + perp1 * off2_u + perp2 * off2_v;
            tunnels.push(Tunnel {
                control_points: vec![pa, p1, p2, pb],
                radius,
            });
        }
    }

    // Entrance rolls.
    let mut entrances: Vec<Entrance> = Vec::new();
    for (ci, chamber) in chambers.iter().enumerate() {
        let try_roll = mix_unit(
            seed,
            &[coord.x, coord.z, system_idx, 70, ci as i32],
        );
        if try_roll >= band.entrance_prob() {
            continue;
        }
        let cwx = chamber.center.x as i32;
        let cwy_top = (chamber.center.y + chamber.radii.y) as i32;
        let cwz = chamber.center.z as i32;
        let surface_h = heightmap.h_pre(
            seed,
            chamber.center.x,
            chamber.center.z,
            climate,
            density,
        ) as i32;
        // 1. Sinkhole.
        if surface_h - cwy_top <= SINKHOLE_DEPTH_MAX && surface_h - cwy_top >= -2 {
            // Extend the shaft top by SURFACE_BAND so it carves through any
            // 3D-density bumps above h_pre — otherwise those bumps become
            // floating terrain islands above the entrance opening.
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::Sinkhole,
                surface: IVec3::new(cwx, surface_h + SURFACE_BAND as i32, cwz),
            });
            continue;
        }
        // 2. Cliff mouth: look for a steep-slope column within
        //    CLIFF_ENTRANCE_DIST. Sample a few directions.
        let mut found_cliff: Option<IVec3> = None;
        for step in 0..16 {
            let theta = step as f32 * std::f32::consts::TAU / 16.0;
            let cwx_f = chamber.center.x + theta.cos() * CLIFF_ENTRANCE_DIST as f32;
            let cwz_f = chamber.center.z + theta.sin() * CLIFF_ENTRANCE_DIST as f32;
            if heightmap.is_cliff(seed, cwx_f, cwz_f, climate, density) {
                found_cliff = Some(IVec3::new(
                    cwx_f as i32,
                    heightmap.h_pre(seed, cwx_f, cwz_f, climate, density) as i32,
                    cwz_f as i32,
                ));
                break;
            }
        }
        if let Some(surface) = found_cliff {
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::CliffMouth,
                surface,
            });
            continue;
        }
        // 3. Skylight: chamber 30–60 below surface.
        let dy = surface_h - cwy_top;
        if (30..=60).contains(&dy) {
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::Skylight,
                surface: IVec3::new(cwx, surface_h + SURFACE_BAND as i32, cwz),
            });
        }
    }

    // Compute the system's actual bounding box including chambers,
    // tunnels, and any entrance shafts so the chunk-side AABB
    // intersection test catches everything we might carve.
    let mut bb_min = bb_min;
    let mut bb_max = bb_max;
    for c in &chambers {
        let cmin = (c.center - c.radii - Vec3::splat(1.0)).floor();
        let cmax = (c.center + c.radii + Vec3::splat(1.0)).ceil();
        bb_min.x = bb_min.x.min(cmin.x as i32);
        bb_min.y = bb_min.y.min(cmin.y as i32);
        bb_min.z = bb_min.z.min(cmin.z as i32);
        bb_max.x = bb_max.x.max(cmax.x as i32);
        bb_max.y = bb_max.y.max(cmax.y as i32);
        bb_max.z = bb_max.z.max(cmax.z as i32);
    }
    for t in &tunnels {
        for p in &t.control_points {
            bb_min.x = bb_min.x.min((p.x - t.radius - 1.0) as i32);
            bb_min.y = bb_min.y.min((p.y - t.radius - 1.0) as i32);
            bb_min.z = bb_min.z.min((p.z - t.radius - 1.0) as i32);
            bb_max.x = bb_max.x.max((p.x + t.radius + 1.0) as i32);
            bb_max.y = bb_max.y.max((p.y + t.radius + 1.0) as i32);
            bb_max.z = bb_max.z.max((p.z + t.radius + 1.0) as i32);
        }
    }
    for e in &entrances {
        // Vertical shaft extent — clip downward to chamber, upward to
        // surface.
        bb_min.y = bb_min.y.min(e.surface.y);
        bb_max.y = bb_max.y.max(e.surface.y + 1);
        bb_min.x = bb_min.x.min(e.surface.x - 4);
        bb_max.x = bb_max.x.max(e.surface.x + 4);
        bb_min.z = bb_min.z.min(e.surface.z - 4);
        bb_max.z = bb_max.z.max(e.surface.z + 4);
    }

    CaveSystem {
        bb_min,
        bb_max,
        chambers,
        tunnels,
        entrances,
        style,
        trunk: None,
        vertical_connectors: vec![],
    }
}

// ── Per-chunk carve ──────────────────────────────────────────────────

/// Soft SDF: positive inside chambers / tunnels, 0 outside.
/// Peak `CAVE_SDF_INTENSITY` deep inside; tapers smoothly to 0 at
/// the strict geometric boundary. The density-based fill chunk
/// subtracts this from the per-voxel density so cave walls have
/// soft, chamfered edges instead of the pixel-sharp ellipsoid /
/// capsule boundaries.
pub fn cave_sdf(wx: i32, wy: i32, wz: i32, systems: &[&CaveSystem]) -> f32 {
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    let mut max_sdf: f32 = 0.0;
    for sys in systems {
        if !system_bb_contains(sys, wx, wy, wz) {
            continue;
        }
        // Chamber ellipsoid: normalised squared distance. <= 1 inside,
        // > 1 outside. Soft falloff over the [0, 1] range.
        for c in &sys.chambers {
            let d = p - c.center;
            let ratio = (d.x / c.radii.x).powi(2)
                + (d.y / c.radii.y).powi(2)
                + (d.z / c.radii.z).powi(2);
            if ratio <= 1.0 {
                // 1.0 at center → 0.0 at boundary.
                let sdf = (1.0 - ratio) * CAVE_SDF_INTENSITY;
                if sdf > max_sdf {
                    max_sdf = sdf;
                }
            }
        }
        // Tunnel capsule along control polyline.
        for t in &sys.tunnels {
            if t.control_points.len() < 2 {
                continue;
            }
            for i in 0..t.control_points.len() - 1 {
                let a = t.control_points[i];
                let b = t.control_points[i + 1];
                let ab = b - a;
                let len_sq = ab.length_squared();
                if len_sq < 1e-6 {
                    continue;
                }
                let t_param = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                let dist = (p - closest).length();
                if dist <= t.radius {
                    let sdf = (1.0 - dist / t.radius) * CAVE_SDF_INTENSITY;
                    if sdf > max_sdf {
                        max_sdf = sdf;
                    }
                }
            }
        }
        // Vertical connectors between adjacent-band systems in the same
        // region. Same capsule SDF math as tunnels.
        for t in &sys.vertical_connectors {
            if t.control_points.len() < 2 {
                continue;
            }
            for i in 0..t.control_points.len() - 1 {
                let a = t.control_points[i];
                let b = t.control_points[i + 1];
                let ab = b - a;
                let len_sq = ab.length_squared();
                if len_sq < 1e-6 {
                    continue;
                }
                let t_param = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                let dist = (p - closest).length();
                if dist <= t.radius {
                    let sdf = (1.0 - dist / t.radius) * CAVE_SDF_INTENSITY;
                    if sdf > max_sdf {
                        max_sdf = sdf;
                    }
                }
            }
        }
        // Cross-region trunk (populated by build_trunks; None in the
        // production cache path, Some in test-built regions).
        if let Some(trunk) = &sys.trunk {
            for i in 0..trunk.control_points.len().saturating_sub(1) {
                let a = trunk.control_points[i];
                let b = trunk.control_points[i + 1];
                let ab = b - a;
                let len_sq = ab.length_squared();
                if len_sq < 1e-6 {
                    continue;
                }
                let t_param = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                let dist = (p - closest).length();
                if dist <= trunk.radius {
                    let sdf = (1.0 - dist / trunk.radius) * CAVE_SDF_INTENSITY;
                    if sdf > max_sdf {
                        max_sdf = sdf;
                    }
                }
            }
        }
    }
    max_sdf
}

/// Compute the cross-region trunk SDF on the fly from a flat `systems`
/// slice spanning multiple regions. Called at chunk-fill time where
/// `sys.trunk` is always `None` (regions are immutable in the Arc
/// cache). Returns the maximum trunk SDF value (positive inside,
/// 0 outside).
///
/// For each system, derives its region coord from its chamber-0
/// world position, finds the nearest chamber-0 in a different region,
/// and evaluates the deterministic 3-point capsule trunk.
pub fn trunks_sdf(
    wx: i32,
    wy: i32,
    wz: i32,
    systems: &[&CaveSystem],
    seed: u64,
    trunk_r: f32,
    trunk_prob: f32,
) -> f32 {
    if trunk_prob <= 0.0 || systems.len() < 2 {
        return 0.0;
    }
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    let mut max_sdf = 0.0_f32;

    // Snapshot chamber-0 center + derived region coord for each system.
    let snap: Vec<(RegionCoord, Vec3)> = systems
        .iter()
        .filter_map(|s| {
            s.chambers.first().map(|c| {
                (
                    RegionCoord::containing(c.center.x as i32, c.center.z as i32),
                    c.center,
                )
            })
        })
        .collect();

    for (si, sys) in systems.iter().enumerate() {
        let Some(c0) = sys.chambers.first() else { continue; };
        let my_coord = RegionCoord::containing(c0.center.x as i32, c0.center.z as i32);
        let my_center = c0.center;

        // Stable salt: chamber-0 center coords (integer-rounded). Stable
        // across chunks because chamber centers are immutable; unique per
        // system within a region because Poisson sampling enforces
        // min_spacing > 0.
        let salt_x = c0.center.x as i32;
        let salt_y = c0.center.y as i32;
        let salt_z = c0.center.z as i32;

        let u = mix_unit(seed, &[my_coord.x, my_coord.z, salt_x, salt_y, salt_z, 9000]);
        if u > trunk_prob {
            continue;
        }

        // Find nearest chamber-0 in a neighbouring region (different
        // region, within 1 step in both x and z).
        let mut best_center: Option<Vec3> = None;
        let mut best_dist = f32::MAX;
        for (j, (other_coord, other_center)) in snap.iter().enumerate() {
            if j == si { continue; }
            let dx = (other_coord.x - my_coord.x).abs();
            let dz = (other_coord.z - my_coord.z).abs();
            if dx > 1 || dz > 1 { continue; }
            // Require different region (not same region coord).
            if *other_coord == my_coord { continue; }
            let d = (*other_center - my_center).length();
            if d < best_dist {
                best_dist = d;
                best_center = Some(*other_center);
            }
        }
        let Some(other_center) = best_center else { continue; };

        // Build trunk geometry (mirrors build_trunks).
        let axis = other_center - my_center;
        let len = (axis.x * axis.x + axis.z * axis.z).sqrt().max(1.0);
        let perp = Vec3::new(-axis.z / len, 0.0, axis.x / len);
        let o = (mix_unit(seed, &[my_coord.x, my_coord.z, salt_x, salt_y, salt_z, 9001]) * 2.0 - 1.0) * 40.0;
        let mid = Vec3::new(
            (my_center.x + other_center.x) * 0.5 + perp.x * o,
            (my_center.y + other_center.y) * 0.5,
            (my_center.z + other_center.z) * 0.5 + perp.z * o,
        );
        let control = [my_center, mid, other_center];

        // Capsule SDF along the 3-point polyline.
        for i in 0..2 {
            let a = control[i];
            let b = control[i + 1];
            let ab = b - a;
            let len_sq = ab.length_squared();
            if len_sq < 1e-6 { continue; }
            let t_param = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
            let closest = a + ab * t_param;
            let dist = (p - closest).length();
            if dist <= trunk_r {
                let sdf = (1.0 - dist / trunk_r) * CAVE_SDF_INTENSITY;
                if sdf > max_sdf {
                    max_sdf = sdf;
                }
            }
        }
    }
    max_sdf
}

/// Soft SDF for entrance features (sinkholes, cliff mouths,
/// skylights). Always punches through regardless of surface buffer;
/// peak intensity twice the regular cave SDF so entrances reliably
/// breach the density even right at the surface.
pub fn entrance_sdf(wx: i32, wy: i32, wz: i32, systems: &[&CaveSystem]) -> f32 {
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    let mut max_sdf: f32 = 0.0;
    for sys in systems {
        if !system_bb_contains(sys, wx, wy, wz) {
            continue;
        }
        for e in &sys.entrances {
            let ch = match sys.chambers.get(e.chamber_idx as usize) {
                Some(c) => c,
                None => continue,
            };
            let intensity = CAVE_SDF_INTENSITY * 2.0;
            match e.kind {
                EntranceKind::Sinkhole => {
                    let chamber_top_y = ch.center.y + ch.radii.y;
                    if wy as f32 >= chamber_top_y - 1.0
                        && wy as f32 <= e.surface.y as f32
                    {
                        let dx = p.x - e.surface.x as f32;
                        let dz = p.z - e.surface.z as f32;
                        let depth = (e.surface.y as f32 - wy as f32).max(0.0);
                        let max_depth = (e.surface.y as f32 - chamber_top_y).max(1.0);
                        let r = 2.0 + (depth / max_depth) * 1.0;
                        let d = (dx * dx + dz * dz).sqrt();
                        if d <= r {
                            let sdf = (1.0 - d / r) * intensity;
                            if sdf > max_sdf {
                                max_sdf = sdf;
                            }
                        }
                    }
                }
                EntranceKind::Skylight => {
                    let chamber_top_y = ch.center.y + ch.radii.y;
                    if wy as f32 >= chamber_top_y - 1.0
                        && wy as f32 <= e.surface.y as f32
                    {
                        let dx = p.x - e.surface.x as f32;
                        let dz = p.z - e.surface.z as f32;
                        let d = (dx * dx + dz * dz).sqrt();
                        if d <= 1.5 {
                            let sdf = (1.0 - d / 1.5) * intensity;
                            if sdf > max_sdf {
                                max_sdf = sdf;
                            }
                        }
                    }
                }
                EntranceKind::CliffMouth => {
                    let chamber_p = ch.center;
                    let cliff_p =
                        Vec3::new(e.surface.x as f32, chamber_p.y, e.surface.z as f32);
                    let ab = cliff_p - chamber_p;
                    let len_sq = ab.length_squared();
                    if len_sq < 1e-6 {
                        continue;
                    }
                    let t_param =
                        ((p - chamber_p).dot(ab) / len_sq).clamp(0.0, 1.0);
                    let closest = chamber_p + ab * t_param;
                    let d = (p - closest).length();
                    if d <= 2.5 {
                        let sdf = (1.0 - d / 2.5) * intensity;
                        if sdf > max_sdf {
                            max_sdf = sdf;
                        }
                    }
                }
            }
        }
    }
    max_sdf
}

/// Does the cell at `(wx, wy, wz)` lie inside any chamber or tunnel
/// of any cave system whose bounding box covers it? Surface buffer is
/// enforced *outside* this function; this only answers the geometric
/// question.
pub fn cave_air(wx: i32, wy: i32, wz: i32, systems: &[&CaveSystem]) -> bool {
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    for sys in systems {
        if !system_bb_contains(sys, wx, wy, wz) {
            continue;
        }
        // Ellipsoid SDF: (dx/rx)² + (dy/ry)² + (dz/rz)² <= 1.
        for c in &sys.chambers {
            let d = p - c.center;
            let ratio =
                (d.x / c.radii.x).powi(2)
                    + (d.y / c.radii.y).powi(2)
                    + (d.z / c.radii.z).powi(2);
            if ratio <= 1.0 {
                return true;
            }
        }
        // Tunnel: distance to the polyline (sampling the 4-point
        // control polyline as 3 straight capsules — a piecewise
        // approximation of the Catmull-Rom).
        for t in &sys.tunnels {
            if t.control_points.len() < 2 {
                continue;
            }
            for i in 0..t.control_points.len() - 1 {
                let a = t.control_points[i];
                let b = t.control_points[i + 1];
                let ab = b - a;
                let t_param = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                if (p - closest).length() <= t.radius {
                    return true;
                }
            }
        }
        // Vertical connectors between adjacent-band systems in the same
        // region. Same capsule SDF math as tunnels.
        for t in &sys.vertical_connectors {
            if t.control_points.len() < 2 {
                continue;
            }
            for i in 0..t.control_points.len() - 1 {
                let a = t.control_points[i];
                let b = t.control_points[i + 1];
                let ab = b - a;
                let t_param = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                if (p - closest).length() <= t.radius {
                    return true;
                }
            }
        }
        // Cross-region trunk (populated by build_trunks; None in the
        // production cache path, Some in test-built regions).
        if let Some(trunk) = &sys.trunk {
            for i in 0..trunk.control_points.len().saturating_sub(1) {
                let a = trunk.control_points[i];
                let b = trunk.control_points[i + 1];
                let ab = b - a;
                let t_param = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                if (p - closest).length() <= trunk.radius {
                    return true;
                }
            }
        }
    }
    false
}

/// Does `(wx, wy, wz)` lie inside one of the system's entrance
/// features (sinkhole shaft, skylight, cliff mouth)? Entrances
/// override the surface buffer.
pub fn entrance_air(wx: i32, wy: i32, wz: i32, systems: &[&CaveSystem]) -> bool {
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    for sys in systems {
        if !system_bb_contains(sys, wx, wy, wz) {
            continue;
        }
        for e in &sys.entrances {
            let ch = match sys.chambers.get(e.chamber_idx as usize) {
                Some(c) => c,
                None => continue,
            };
            match e.kind {
                EntranceKind::Sinkhole => {
                    // Vertical shaft from chamber top up to surface,
                    // radius 2.5, slightly funnel-shaped at top.
                    let chamber_top_y = ch.center.y + ch.radii.y;
                    if wy as f32 >= chamber_top_y - 1.0 && wy as f32 <= e.surface.y as f32 {
                        let dx = p.x - e.surface.x as f32;
                        let dz = p.z - e.surface.z as f32;
                        // Funnel: radius grows from 2 at chamber top to
                        // 3 at surface.
                        let depth = (e.surface.y as f32 - wy as f32).max(0.0);
                        let max_depth = (e.surface.y as f32 - chamber_top_y).max(1.0);
                        let r = 2.0 + (depth / max_depth) * 1.0;
                        if (dx * dx + dz * dz).sqrt() <= r {
                            return true;
                        }
                    }
                }
                EntranceKind::Skylight => {
                    // Narrow 1.5-block radius shaft from chamber top
                    // to surface.
                    let chamber_top_y = ch.center.y + ch.radii.y;
                    if wy as f32 >= chamber_top_y - 1.0 && wy as f32 <= e.surface.y as f32 {
                        let dx = p.x - e.surface.x as f32;
                        let dz = p.z - e.surface.z as f32;
                        if (dx * dx + dz * dz).sqrt() <= 1.5 {
                            return true;
                        }
                    }
                }
                EntranceKind::CliffMouth => {
                    // Horizontal tunnel from chamber to the cliff
                    // anchor. Capsule with radius 2.5.
                    let chamber_p = ch.center;
                    let cliff_p = Vec3::new(
                        e.surface.x as f32,
                        chamber_p.y,
                        e.surface.z as f32,
                    );
                    let ab = cliff_p - chamber_p;
                    let len_sq = ab.length_squared();
                    if len_sq < 1e-6 {
                        continue;
                    }
                    let t_param = ((p - chamber_p).dot(ab) / len_sq).clamp(0.0, 1.0);
                    let closest = chamber_p + ab * t_param;
                    if (p - closest).length() <= 2.5 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Is `wy` inside any cave system's "carved range"? Used as a quick
/// rejection — only check the expensive ellipsoid / capsule SDFs if
/// at least one system's Y extent covers the column.
pub fn any_system_y_in_range(wy: i32, systems: &[&CaveSystem]) -> bool {
    systems.iter().any(|s| wy >= s.bb_min.y && wy <= s.bb_max.y)
}

#[inline]
fn system_bb_contains(sys: &CaveSystem, wx: i32, wy: i32, wz: i32) -> bool {
    wx >= sys.bb_min.x
        && wx <= sys.bb_max.x
        && wy >= sys.bb_min.y
        && wy <= sys.bb_max.y
        && wz >= sys.bb_min.z
        && wz <= sys.bb_max.z
}

// ── Noise carver layers ──────────────────────────────────────────────
//
// Cheese / pillars: MC-style ambient noise-based cave density.
// Live alongside the graph cave systems.
// The contributions wire into fill_chunk's `cave_contribution`
// composition (cheese subtracts from density; pillars add back).

use crate::worldgen::config::CaveConfig;
use crate::worldgen::noise_channel::build_channel;

/// All MC-derived noise channels needed for the cheese and pillar
/// carvers. Built once per Generator.
pub struct NoiseCarvers {
    // Cheese.
    pub cheese: Fbm<Simplex>,
    /// `cave_layer` — the regional gating noise added to cheese as
    /// `intensity * layer²`. See [`CaveConfig::cave_layer`].
    pub cave_layer: Fbm<Simplex>,
    // Pillars.
    pub pillar: Fbm<Simplex>,
    pub pillar_rareness: Fbm<Simplex>,
    pub pillar_thickness: Fbm<Simplex>,
    // Terasology ambient.
    pub tera_a: Fbm<Simplex>,
    pub tera_b: Fbm<Simplex>,
}

impl NoiseCarvers {
    /// Build all channels from their config descriptors. Each
    /// channel uses a different seed-salt so they're uncorrelated.
    pub fn new(seed: u64, cfg: &CaveConfig) -> Self {
        Self {
            cheese: build_channel(&cfg.cheese, seed, 1001),
            cave_layer: build_channel(&cfg.cave_layer, seed, 1011),
            pillar: build_channel(&cfg.pillar, seed, 1006),
            pillar_rareness: build_channel(&cfg.pillar_rareness, seed, 1007),
            pillar_thickness: build_channel(&cfg.pillar_thickness, seed, 1008),
            tera_a: build_channel(&cfg.tera_a, seed, 2000),
            tera_b: build_channel(&cfg.tera_b, seed, 2001),
        }
    }
}

/// Signed-density cheese contribution.
///
/// ```text
///   term1  = clamp(cheese_offset + cheese_noise, -1, 1)
///   term2  = clamp(supp_offset + supp_slope * raw_density,
///                  supp_min, supp_max)
///   result = term1 + term2
/// ```
///
/// `term1` is the raw cheese signal — negative values bias the
/// voxel toward air. `term2` is a surface-suppression term that
/// pushes the result strongly positive (solid) near the surface
/// (where `raw_density` is near zero) and dies off at depth, so
/// cheese carves freely underground but not just under the
/// heightmap.
///
/// The caller composes this via `min(other_caves, cheese)` — any
/// signed component going negative pulls the voxel to air.
pub fn cheese_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    raw_density: f32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    let xz_scale = cfg.cheese_xz_scale as f64;
    let y_scale = cfg.cheese_y_scale as f64;
    let cheese = carvers.cheese.get([
        wx as f64 * xz_scale,
        wy as f64 * y_scale,
        wz as f64 * xz_scale,
    ]) as f32;
    let term1 = (cfg.cheese_offset + cheese).clamp(-1.0, 1.0);
    let supp = (cfg.cheese_suppression_offset
        + cfg.cheese_suppression_slope * raw_density)
        .clamp(cfg.cheese_suppression_min, cfg.cheese_suppression_max);

    // MC-parity `layerizedCaverns`: add `intensity * layer²` so
    // cheese carving is regionally gated by horizontal strata.
    // Without this the cheese clamp at -1 makes ~50% of deep
    // voxels carve, producing uniform swiss cheese.
    let layer = carvers.cave_layer.get([
        wx as f64 * cfg.cave_layer_xz_scale as f64,
        wy as f64 * cfg.cave_layer_y_scale as f64,
        wz as f64 * cfg.cave_layer_xz_scale as f64,
    ]) as f32;
    let layerized = cfg.cave_layer_intensity * layer * layer;

    term1 + supp + layerized
}

/// Per-voxel pillar contribution. Returns a non-negative value in
/// `[0, pillar_intensity]` that is ADDED to density (not subtracted)
/// in `fill_chunk`'s composition. Composition order matters:
/// pillars apply AFTER all cave subtractions so they can refill
/// previously-carved voxels — the MC "columns inside open caves"
/// look.
///
/// MC formula (from `data/.../caves/pillars.json`):
///
/// ```text
///   pillar_raw  = 2 * noise(pillar, xz_scale, y_scale)
///   pillar_rare = -1 - noise(pillar_rareness)
///   thickness   = (0.55 + 0.55 * noise(pillar_thickness))^3
///   pillars     = (pillar_raw + pillar_rare) * thickness
///   range_choice: if pillars >= cutoff → pillars, else 0
/// ```
pub fn pillar_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    let p_x = wx as f64 * cfg.pillar_xz_scale as f64;
    let p_y = wy as f64 * cfg.pillar_y_scale as f64;
    let p_z = wz as f64 * cfg.pillar_xz_scale as f64;
    let pillar_raw = 2.0 * carvers.pillar.get([p_x, p_y, p_z]) as f32;
    let pillar_rare = -1.0
        - carvers.pillar_rareness.get([
            wx as f64,
            wy as f64,
            wz as f64,
        ]) as f32;
    let thickness_noise = carvers.pillar_thickness.get([
        wx as f64,
        wy as f64,
        wz as f64,
    ]) as f32;
    let thickness = (0.55 + 0.55 * thickness_noise).powi(3);
    let raw = (pillar_raw + pillar_rare) * thickness;
    if raw < cfg.pillar_cutoff {
        return 0.0;
    }
    let depth = (raw - cfg.pillar_cutoff).clamp(0.0, 1.0);
    cfg.pillar_intensity * depth
}

/// Polynomial smooth-min — pulls the result below `min(a, b)` by up to
/// `k/4` when `|a - b| < k`. Used to merge cave SDFs near layer boundaries
/// so close-but-not-touching pockets connect into one volume.
#[inline]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 { return a.min(b); }
    let h = ((k - (a - b).abs()).max(0.0)) / k;
    a.min(b) - h * h * k * 0.25
}

/// Terasology-style depth-driven 2-noise cave carver.
///
/// Inspired by `org.terasology.caves.CaveFacetProvider`. Two independent
/// 4-octave FBM-Simplex channels are intersected: voxels where both are
/// near zero are cave. The cave region in 2D noise space is a disk of
/// radius `freq_depth`, centered at `(0, -freq_reduction)`. The disk
/// grows with depth (more caves deeper) and shifts off-axis near the
/// surface (caves rare up top). Y is sampled at `tera_y_factor` × the
/// XZ frequency, which forces the resulting tubes to lean horizontal.
///
/// Returns signed density: negative = carve, positive = solid. Magnitude
/// scales by `* 5.0` so the output aligns with the cheese carver's range
/// for downstream `min`/`smin` composition. Typical values: `[-5.4, +6.7]`
/// (lower bound at deep + on-axis noise, upper bound at noise extrema with
/// no cave region).
pub fn terasology_ambient(
    wx: i32, wy: i32, wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
    surface_y: f32,
) -> f32 {
    let depth = (surface_y - wy as f32).max(0.0);
    let freq_reduction = (cfg.tera_supp - depth / cfg.tera_supp_depth).max(0.0);
    let freq_depth     = cfg.tera_thresh_base + depth / cfg.tera_thresh_depth;
    let freq           = 1.0 / cfg.tera_wave;
    let wy_scaled      = wy as f32 * cfg.tera_y_factor;
    let n0 = carvers.tera_a.get([
        (wx as f32 * freq) as f64,
        (wy_scaled * freq) as f64,
        (wz as f32 * freq) as f64,
    ]) as f32;
    let n1 = carvers.tera_b.get([
        (wx as f32 * freq) as f64,
        (wy_scaled * freq) as f64,
        (wz as f32 * freq) as f64,
    ]) as f32 + freq_reduction;
    ((n0 * n0 + n1 * n1).sqrt() - freq_depth) * 5.0
    // scale: align magnitude with cheese carver for downstream smin composition
}

// ── Carver evaluator (corner-lattice trilerp) ────────────────────────
//
// The per-voxel `cheese_contribution` and `pillar_contribution` each
// issue several FBM samples per voxel. With 32³ voxels per chunk and
// 2-octave FBM, that dominates the chunk-fill cost.
//
// `CarverEvaluator` mirrors `density_graph::CellEvaluator`: sample
// each underlying noise on a 9³ corner lattice (4-block spacing,
// 729 corners per chunk), then trilerp per voxel. The per-voxel
// formulas (clamps, `raw_density`-dependent suppression, gates) run
// unchanged on the lerped values, so cave shapes track the original
// formulas to within FBM's local smoothness — visually equivalent at
// 4-block resolution, the same precedent as the base density.

const CARVER_CELL_SIZE: i32 = 4;
const CARVER_CELL_COUNT: usize =
    (crate::voxel::coords::CHUNK_DIM_U as usize) / CARVER_CELL_SIZE as usize; // 8
const CARVER_CORNER_COUNT: usize = CARVER_CELL_COUNT + 1; // 9
const CARVER_CORNER_CUBE: usize =
    CARVER_CORNER_COUNT * CARVER_CORNER_COUNT * CARVER_CORNER_COUNT; // 729

/// One corner's worth of pre-sampled noise. Keeping the channels
/// AoS means each voxel touches 8 contiguous corner structs instead
/// of striding separate `Vec<f32>` arenas — better cache behavior
/// in the per-voxel inner loop.
#[derive(Default, Clone, Copy)]
struct CarverCorner {
    cheese: f32,
    cave_layer: f32,
    pillar: f32,
    pillar_rare: f32,
    pillar_thick: f32,
    tera_a: f32,
    tera_b: f32,
}

/// Pre-sampled noise lattice for the carver layers. Built once per
/// chunk fill; `*_at` accessors trilerp from the 8 surrounding
/// corners and apply the original per-voxel formula.
pub struct CarverEvaluator {
    corners: Box<[CarverCorner]>,
    origin: IVec3,
}

impl CarverEvaluator {
    pub fn new(
        carvers: &NoiseCarvers,
        cfg: &CaveConfig,
        chunk_origin: IVec3,
    ) -> Self {
        let mut corners = vec![CarverCorner::default(); CARVER_CORNER_CUBE].into_boxed_slice();
        for cz in 0..CARVER_CORNER_COUNT {
            for cy in 0..CARVER_CORNER_COUNT {
                for cx in 0..CARVER_CORNER_COUNT {
                    let wx = chunk_origin.x + (cx as i32) * CARVER_CELL_SIZE;
                    let wy = chunk_origin.y + (cy as i32) * CARVER_CELL_SIZE;
                    let wz = chunk_origin.z + (cz as i32) * CARVER_CELL_SIZE;
                    let idx = corner_index(cx, cy, cz);

                    let cheese = carvers.cheese.get([
                        wx as f64 * cfg.cheese_xz_scale as f64,
                        wy as f64 * cfg.cheese_y_scale as f64,
                        wz as f64 * cfg.cheese_xz_scale as f64,
                    ]) as f32;
                    let cave_layer = carvers.cave_layer.get([
                        wx as f64 * cfg.cave_layer_xz_scale as f64,
                        wy as f64 * cfg.cave_layer_y_scale as f64,
                        wz as f64 * cfg.cave_layer_xz_scale as f64,
                    ]) as f32;

                    let pillar = carvers.pillar.get([
                        wx as f64 * cfg.pillar_xz_scale as f64,
                        wy as f64 * cfg.pillar_y_scale as f64,
                        wz as f64 * cfg.pillar_xz_scale as f64,
                    ]) as f32;
                    let pillar_rare = carvers
                        .pillar_rareness
                        .get([wx as f64, wy as f64, wz as f64])
                        as f32;
                    let pillar_thick = carvers
                        .pillar_thickness
                        .get([wx as f64, wy as f64, wz as f64])
                        as f32;

                    let tera_freq = 1.0 / cfg.tera_wave;
                    let tera_wy = wy as f32 * cfg.tera_y_factor;
                    let tera_a = carvers.tera_a.get([
                        (wx as f32 * tera_freq) as f64,
                        (tera_wy * tera_freq) as f64,
                        (wz as f32 * tera_freq) as f64,
                    ]) as f32;
                    let tera_b = carvers.tera_b.get([
                        (wx as f32 * tera_freq) as f64,
                        (tera_wy * tera_freq) as f64,
                        (wz as f32 * tera_freq) as f64,
                    ]) as f32;

                    corners[idx] = CarverCorner {
                        cheese,
                        cave_layer,
                        pillar,
                        pillar_rare,
                        pillar_thick,
                        tera_a,
                        tera_b,
                    };
                }
            }
        }
        Self {
            corners,
            origin: chunk_origin,
        }
    }

    #[inline]
    fn lerp_coords(&self, wx: i32, wy: i32, wz: i32) -> LerpCoords {
        let lx = wx - self.origin.x;
        let ly = wy - self.origin.y;
        let lz = wz - self.origin.z;
        let cx = ((lx / CARVER_CELL_SIZE) as usize).min(CARVER_CELL_COUNT - 1);
        let cy = ((ly / CARVER_CELL_SIZE) as usize).min(CARVER_CELL_COUNT - 1);
        let cz = ((lz / CARVER_CELL_SIZE) as usize).min(CARVER_CELL_COUNT - 1);
        let tx = (lx - cx as i32 * CARVER_CELL_SIZE) as f32 / CARVER_CELL_SIZE as f32;
        let ty = (ly - cy as i32 * CARVER_CELL_SIZE) as f32 / CARVER_CELL_SIZE as f32;
        let tz = (lz - cz as i32 * CARVER_CELL_SIZE) as f32 / CARVER_CELL_SIZE as f32;
        LerpCoords { cx, cy, cz, tx, ty, tz }
    }

    /// Trilinear interp of one channel — `get` picks the channel from a
    /// `CarverCorner`. The Y→X→Z lerp order matches
    /// `density_graph::CellEvaluator::evaluate`.
    #[inline]
    fn trilerp<F: Fn(&CarverCorner) -> f32>(&self, c: &LerpCoords, get: F) -> f32 {
        let i = |x: usize, y: usize, z: usize| {
            &self.corners[corner_index(x, y, z)]
        };
        let c000 = get(i(c.cx, c.cy, c.cz));
        let c100 = get(i(c.cx + 1, c.cy, c.cz));
        let c010 = get(i(c.cx, c.cy + 1, c.cz));
        let c110 = get(i(c.cx + 1, c.cy + 1, c.cz));
        let c001 = get(i(c.cx, c.cy, c.cz + 1));
        let c101 = get(i(c.cx + 1, c.cy, c.cz + 1));
        let c011 = get(i(c.cx, c.cy + 1, c.cz + 1));
        let c111 = get(i(c.cx + 1, c.cy + 1, c.cz + 1));
        let xz00 = c000 + (c010 - c000) * c.ty;
        let xz10 = c100 + (c110 - c100) * c.ty;
        let xz01 = c001 + (c011 - c001) * c.ty;
        let xz11 = c101 + (c111 - c101) * c.ty;
        let z0 = xz00 + (xz10 - xz00) * c.tx;
        let z1 = xz01 + (xz11 - xz01) * c.tx;
        z0 + (z1 - z0) * c.tz
    }

    /// Same shape as `cheese_contribution`: `term1 + supp + layerized`.
    /// Trilerps the noise channels; the clamps and `raw_density`-driven
    /// suppression term run unchanged per voxel.
    pub fn cheese_at(&self, wx: i32, wy: i32, wz: i32, raw_density: f32, cfg: &CaveConfig) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let cheese = self.trilerp(&lc, |c| c.cheese);
        let cave_layer = self.trilerp(&lc, |c| c.cave_layer);
        let term1 = (cfg.cheese_offset + cheese).clamp(-1.0, 1.0);
        let supp = (cfg.cheese_suppression_offset + cfg.cheese_suppression_slope * raw_density)
            .clamp(cfg.cheese_suppression_min, cfg.cheese_suppression_max);
        let layerized = cfg.cave_layer_intensity * cave_layer * cave_layer;
        term1 + supp + layerized
    }

    /// Same shape as `pillar_contribution`. The cutoff gate runs per voxel.
    pub fn pillar_at(&self, wx: i32, wy: i32, wz: i32, cfg: &CaveConfig) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let pillar = self.trilerp(&lc, |c| c.pillar);
        let pillar_rare_n = self.trilerp(&lc, |c| c.pillar_rare);
        let thickness_noise = self.trilerp(&lc, |c| c.pillar_thick);
        let pillar_raw = 2.0 * pillar;
        let pillar_rare = -1.0 - pillar_rare_n;
        let thickness = (0.55 + 0.55 * thickness_noise).powi(3);
        let raw = (pillar_raw + pillar_rare) * thickness;
        if raw < cfg.pillar_cutoff {
            return 0.0;
        }
        let depth = (raw - cfg.pillar_cutoff).clamp(0.0, 1.0);
        cfg.pillar_intensity * depth
    }

    /// Trilerp the pre-sampled tera_a and tera_b noise values, then run the
    /// same `terasology_ambient` arithmetic on the lerped result. Exact at
    /// corners by construction.
    pub fn terasology_ambient_at(
        &self, wx: i32, wy: i32, wz: i32, cfg: &CaveConfig, surface_y: f32,
    ) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let n0_raw = self.trilerp(&lc, |c| c.tera_a);
        let n1_raw = self.trilerp(&lc, |c| c.tera_b);
        let depth = (surface_y - wy as f32).max(0.0);
        let freq_reduction = (cfg.tera_supp - depth / cfg.tera_supp_depth).max(0.0);
        let freq_depth     = cfg.tera_thresh_base + depth / cfg.tera_thresh_depth;
        let n1 = n1_raw + freq_reduction;
        ((n0_raw * n0_raw + n1 * n1).sqrt() - freq_depth) * 5.0
    }

}

#[derive(Clone, Copy)]
struct LerpCoords {
    cx: usize,
    cy: usize,
    cz: usize,
    tx: f32,
    ty: f32,
    tz: f32,
}

#[inline]
fn corner_index(cx: usize, cy: usize, cz: usize) -> usize {
    cx + cy * CARVER_CORNER_COUNT + cz * CARVER_CORNER_COUNT * CARVER_CORNER_COUNT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smin_extremes() {
        // k=0 reduces to ordinary min
        assert_eq!(smin(0.5, 0.8, 0.0), 0.5);
        assert_eq!(smin(0.8, 0.5, 0.0), 0.5);
        // smin(0, 0, k) pulls below min by k/4 (polynomial peak)
        let v = smin(0.0, 0.0, 1.0);
        assert!((v + 0.25).abs() < 1e-5, "smin(0,0,1) should be -0.25, got {v}");
        // smin(a, b, k) <= min(a, b) for all k >= 0
        for k in [0.0, 0.5, 1.5, 3.0] {
            for a in [-0.5, 0.0, 0.5, 2.0] {
                for b in [-0.5, 0.0, 0.5, 2.0] {
                    let s = smin(a, b, k);
                    assert!(s <= a.min(b) + 1e-5,
                        "smin({a},{b},{k}) = {s} > min");
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
        let mut found_connector = false;
        'outer: for rx in 0..8_i32 {
            for rz in 0..8_i32 {
                let coord = RegionCoord { x: rx, z: rz };
                let mut region = FineRegion::empty(coord);
                build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region, &cfg.cave);
                let bands: Vec<DepthBand> = region.cave_systems.iter().map(|s| {
                    infer_band_for_test(s)
                }).collect();
                if !pair_is_adjacent_for_test(&bands) { continue; }
                for sys in &region.cave_systems {
                    if !sys.vertical_connectors.is_empty() {
                        found_connector = true;
                        break 'outer;
                    }
                }
            }
        }
        assert!(found_connector, "no vertical connector emitted in any 2-band region with prob=1.0");
    }

    fn infer_band_for_test(sys: &CaveSystem) -> DepthBand {
        let cy = (sys.bb_min.y + sys.bb_max.y) / 2;
        if cy >= CAVE_BAND_SHALLOW.0 { DepthBand::Shallow }
        else if cy >= CAVE_BAND_MIDDLE.0 { DepthBand::Middle }
        else { DepthBand::Deep }
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
        for z in -3..=3 {
            for x in -3..=3 {
                let coord = RegionCoord { x, z };
                let mut region = FineRegion::empty(coord);
                build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region, &cfg.cave);
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
        let coord = RegionCoord { x: 2, z: -3 };
        let mut r1 = FineRegion::empty(coord);
        let mut r2 = FineRegion::empty(coord);
        build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut r1, &cfg.cave);
        build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut r2, &cfg.cave);
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
        let coord = RegionCoord { x: 0, z: 0 };
        let mut region = FineRegion::empty(coord);
        build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region, &cfg.cave);
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
        for rx in 0..4 {
            for rz in 0..4 {
                let coord = RegionCoord { x: rx, z: rz };
                let mut region = FineRegion::empty(coord);
                build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region, &cfg.cave);
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
        assert_ne!(a.cheese.get([10.0, 5.0, -3.0]), b.cheese.get([10.0, 5.0, -3.0]));
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
                    if v < min_result { min_result = v; }
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
                        assert!(d < 1e-5, "cheese mismatch at ({wx},{wy},{wz}) rd={raw_density}: direct={direct} lerp={lerped} d={d}");
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
                    if v < 0.0 { hit += 1; }
                }
            }
            frac_by_depth.push(hit as f32 / total as f32);
        }
        // Monotonic non-decreasing toward depth.
        for w in frac_by_depth.windows(2) {
            assert!(w[1] >= w[0] - 1e-3,
                "cave fraction decreased with depth: {:?}", frac_by_depth);
        }
        // Surface band should be ~0%.
        assert!(frac_by_depth[0] < 0.05, "too many caves near surface: {:?}", frac_by_depth);
        // Deep band should be > shallow.
        assert!(frac_by_depth[3] > frac_by_depth[0] * 2.0,
            "deep band not vastly more cave-rich than shallow: {:?}", frac_by_depth);
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
        assert!(frac > 0.02, "expected some caves at supp_depth, got {frac:.3}");
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
                    horizontal_runs += h_run; h_run = 0;
                }
            }
            for dy in (-30..30).step_by(2) {
                if terasology_ambient(wx, wy_center + dy, 0, &nc, &cave_cfg, surface_y) < 0.0 {
                    v_run += 1;
                } else if v_run > 0 {
                    vertical_runs += v_run; v_run = 0;
                }
            }
        }
        assert!(horizontal_runs > vertical_runs,
            "expected horizontal cave extent > vertical: h={horizontal_runs} v={vertical_runs}");
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
            ("middle",  DepthBand::Middle,  &table.style_weights_middle),
            ("deep",    DepthBand::Deep,    &table.style_weights_deep),
        ];
        for (name, band, weights) in &bands {
            let mut counts = [0u32; 5];
            for i in 0..500 {
                let s = pick_style(42, RegionCoord { x: i, z: 0 }, 0, *band, &cfg.cave);
                let idx = match s {
                    CaveStyle::Cathedral => 0,
                    CaveStyle::Warren    => 1,
                    CaveStyle::Slot      => 2,
                    CaveStyle::Sump      => 3,
                    CaveStyle::Karst     => 4,
                };
                counts[idx] += 1;
            }
            for (i, &expected_weight) in weights.iter().enumerate() {
                let actual = counts[i] as f32 / 500.0;
                let diff = (actual - expected_weight).abs();
                assert!(diff < 0.10,
                    "band {name}, style index {i}: expected {expected_weight:.2}, got {actual:.2}");
            }
        }
    }

    #[test]
    fn trunk_links_nearest_neighbour_region_system() {
        let mut cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        cfg.cave.trunk_prob = 1.0;
        let hm = HeightmapNoise::new(42, &cfg.climate);

        // Build all systems in a 3×3 region grid (owned FineRegions).
        let mut regions: Vec<(RegionCoord, FineRegion)> = vec![];
        for rx in 0..3_i32 {
            for rz in 0..3_i32 {
                let coord = RegionCoord { x: rx, z: rz };
                let mut region = FineRegion::empty(coord);
                build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region, &cfg.cave);
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
        let snap: Vec<(usize, RegionCoord, Vec<glam::Vec3>)> = regions.iter().enumerate().map(|(i, (coord, region))| {
            let centers: Vec<glam::Vec3> = region.cave_systems.iter()
                .filter_map(|s| s.chambers.first().map(|c| c.center))
                .collect();
            (i, *coord, centers)
        }).collect();

        'outer: for (i, (coord, region)) in regions.iter().enumerate() {
            for sys in &region.cave_systems {
                let Some(trunk) = &sys.trunk else { continue; };
                let endpoint = *trunk.control_points.last().unwrap();
                // Check if endpoint matches any chamber-0 center in a
                // neighbouring region (8-connected, different region index).
                for (j, other_coord, centers) in &snap {
                    if *j == i { continue; }
                    if (other_coord.x - coord.x).abs() > 1 || (other_coord.z - coord.z).abs() > 1 { continue; }
                    for &c in centers {
                        if (c - endpoint).length() < 0.5 {
                            found = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        assert!(found, "no trunk linked to any neighbour-region chamber center");
    }

}
