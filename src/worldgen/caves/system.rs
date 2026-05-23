//! Cave system builder — Poisson-disk chamber placement, Kruskal MST tunnel
//! graph, entrance rolling, and the public `build_systems_for_region` entry
//! point.
use super::connectors::build_vertical_connectors;
use super::pools::derive_cave_pools;
use super::style::{
    CaveStyle, DepthBand, SALT_BB_ORIGIN_X, SALT_BB_ORIGIN_Z, SALT_CHAMBER_COUNT, SALT_EXTRA_LOOPS,
    SALT_SYSTEM_COUNT, pick_style,
};
use crate::worldgen::density::HeightmapNoise;
use crate::worldgen::hash::{mix_range, mix_u32, mix_unit};
use crate::worldgen::region::{
    CaveSystem, Chamber, Entrance, EntranceKind, FineRegion, RegionCoord, Tunnel,
};
use crate::worldgen::tuning::*;
use glam::{IVec3, Vec3};

/// Per-style parameter set extracted from the style table.
pub(super) struct StyleParams {
    pub chamber_count: (u32, u32),
    pub r_xz: (f32, f32),
    pub r_y: (f32, f32),
    pub tunnel_r: (f32, f32),
}

pub(super) fn style_params(
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
    let n_max = cave_cfg
        .systems_per_region_max
        .min(CAVE_SYSTEMS_PER_REGION.1);
    let n = n_min + (mix_u32(seed, &[coord.x, coord.z, SALT_SYSTEM_COUNT]) % (n_max - n_min + 1));
    region.cave_systems.clear();
    for system_idx in 0..n as i32 {
        let sys = build_system(
            seed, coord, system_idx, heightmap, climate, density, cave_cfg,
        );
        region.cave_systems.push(sys);
    }
    build_vertical_connectors(seed, coord, cave_cfg, region);
    derive_cave_pools(seed, region);
}

/// Roll the system bounding box X/Z/Y within the region.
///
/// The box is allowed to straddle the region boundary — neighbouring regions
/// consult systems via the 3×3 region neighbourhood at chunk fill time.
///
/// Returns `(bb_min, bb_max)` as world-space `IVec3` corners (inclusive).
fn roll_bounding_box(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    y_min: i32,
    y_max: i32,
) -> (IVec3, IVec3) {
    let bb_size = IVec3::new(
        CAVE_SYSTEM_BB_HALF_EXTENT,
        (y_max - y_min).min(64),
        CAVE_SYSTEM_BB_HALF_EXTENT,
    );
    let region_origin = coord.origin();
    let bb_origin_x = region_origin.0
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, SALT_BB_ORIGIN_X])
            % (FINE_REGION_SIZE - bb_size.x).max(1) as u32) as i32
        - (bb_size.x / 2);
    let bb_origin_z = region_origin.1
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, SALT_BB_ORIGIN_Z])
            % (FINE_REGION_SIZE - bb_size.z).max(1) as u32) as i32
        - (bb_size.z / 2);
    let bb_origin_y = y_min;
    let bb_min = IVec3::new(bb_origin_x, bb_origin_y, bb_origin_z);
    let bb_max = bb_min + bb_size;
    (bb_min, bb_max)
}

/// Poisson-disk rejection sampling for chamber centers within the bounding box.
///
/// Candidate centers are drawn uniformly at random; a candidate is rejected
/// if it lies within `mean_radius × POISSON_MIN_SPACING_MULT` of any
/// already-placed chamber. Sampling stops after `max_tries = 200` attempts
/// regardless of whether the target count was reached — the system uses
/// however many fit, down to a minimum of 1.
#[allow(clippy::too_many_arguments)]
fn sample_chambers(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    bb_min: IVec3,
    bb_max: IVec3,
    style: CaveStyle,
    sp: &StyleParams,
    cave_cfg: &crate::worldgen::config::CaveConfig,
) -> Vec<Chamber> {
    // Chamber count — from style table.
    let (cn_min, cn_max) = sp.chamber_count;
    let chamber_count = cn_min
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, SALT_CHAMBER_COUNT])
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
        let depth_mult = 1.0
            + cave_cfg.depth_scale * ((DEPTH_SCALE_PIVOT_Y - cy_raw).max(0.0) / DEPTH_SCALE_RANGE);

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

        let center = Vec3::new(bb_min.x as f32 + sx, cy_final, bb_min.z as f32 + sz);
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
    chambers
}

/// Build a tunnel graph connecting all chambers via Kruskal's MST, then
/// add `extra_loop_count` short non-MST edges to introduce cycles.
///
/// Each edge becomes a 4-point Catmull-Rom-friendly control polyline with
/// two domain-warped perpendicular offsets that break the otherwise straight
/// line into a meander.
fn connect_chambers_mst(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    chambers: &[Chamber],
    sp: &StyleParams,
) -> Vec<Tunnel> {
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
            + (mix_u32(seed, &[coord.x, coord.z, system_idx, SALT_EXTRA_LOOPS])
                % (MST_EXTRA_LOOPS.1 - MST_EXTRA_LOOPS.0 + 1));
        let mut added_extras = 0u32;
        for &(_, a, b) in &edges {
            if added_extras >= extra_loop_count {
                break;
            }
            if mst_edges
                .iter()
                .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
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
            ) * TUNNEL_WARP_AMP;
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
    tunnels
}

/// Roll Sinkhole / CliffMouth / Skylight entrances for each chamber that
/// passes the band-weighted probability gate.
///
/// Sinkhole wins if the chamber top is within `SINKHOLE_DEPTH_MAX` of the
/// surface. CliffMouth wins if a cliff column is reachable within
/// `CLIFF_ENTRANCE_DIST` in any of 16 radial directions. Skylight wins
/// if the chamber top is `SKYLIGHT_DEPTH_MIN–SKYLIGHT_DEPTH_MAX` below
/// the surface. Exactly one entrance per chamber; priority Sinkhole →
/// CliffMouth → Skylight.
#[allow(clippy::too_many_arguments)]
fn roll_entrances(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    chambers: &[Chamber],
    band: DepthBand,
    heightmap: &HeightmapNoise,
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
) -> Vec<Entrance> {
    // Entrance rolls.
    let mut entrances: Vec<Entrance> = Vec::new();
    for (ci, chamber) in chambers.iter().enumerate() {
        let try_roll = mix_unit(seed, &[coord.x, coord.z, system_idx, 70, ci as i32]);
        if try_roll >= band.entrance_prob() {
            continue;
        }
        let cwx = chamber.center.x as i32;
        let cwy_top = (chamber.center.y + chamber.radii.y) as i32;
        let cwz = chamber.center.z as i32;
        let surface_h =
            heightmap.h_pre(seed, chamber.center.x, chamber.center.z, climate, density) as i32;
        // 1. Sinkhole.
        if surface_h - cwy_top <= SINKHOLE_DEPTH_MAX && surface_h - cwy_top >= -2 {
            // Extend the shaft top by SURFACE_BAND so it carves through any
            // 3D-density bumps above h_pre — otherwise those bumps become
            // floating terrain islands above the entrance opening.
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::Sinkhole,
                surface: IVec3::new(cwx, surface_h + SURFACE_BAND, cwz),
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
        // 3. Skylight: chamber SKYLIGHT_DEPTH_MIN–SKYLIGHT_DEPTH_MAX below surface.
        let dy = surface_h - cwy_top;
        if (SKYLIGHT_DEPTH_MIN..=SKYLIGHT_DEPTH_MAX).contains(&dy) {
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::Skylight,
                surface: IVec3::new(cwx, surface_h + SURFACE_BAND, cwz),
            });
        }
    }
    entrances
}

/// Expand the initial bounding box to include all chamber ellipsoids, tunnel
/// capsule control points, and entrance shaft footprints.
///
/// The AABB must conservatively cover everything the SDF functions might
/// carve, so the chunk-side intersection test never misses a carve.
fn finalize_aabb(
    bb_min: IVec3,
    bb_max: IVec3,
    chambers: &[Chamber],
    tunnels: &[Tunnel],
    entrances: &[Entrance],
) -> (IVec3, IVec3) {
    // Compute the system's actual bounding box including chambers,
    // tunnels, and any entrance shafts so the chunk-side AABB
    // intersection test catches everything we might carve.
    let mut bb_min = bb_min;
    let mut bb_max = bb_max;
    for c in chambers {
        let cmin = (c.center - c.radii - Vec3::splat(1.0)).floor();
        let cmax = (c.center + c.radii + Vec3::splat(1.0)).ceil();
        bb_min.x = bb_min.x.min(cmin.x as i32);
        bb_min.y = bb_min.y.min(cmin.y as i32);
        bb_min.z = bb_min.z.min(cmin.z as i32);
        bb_max.x = bb_max.x.max(cmax.x as i32);
        bb_max.y = bb_max.y.max(cmax.y as i32);
        bb_max.z = bb_max.z.max(cmax.z as i32);
    }
    for t in tunnels {
        for p in &t.control_points {
            bb_min.x = bb_min.x.min((p.x - t.radius - 1.0) as i32);
            bb_min.y = bb_min.y.min((p.y - t.radius - 1.0) as i32);
            bb_min.z = bb_min.z.min((p.z - t.radius - 1.0) as i32);
            bb_max.x = bb_max.x.max((p.x + t.radius + 1.0) as i32);
            bb_max.y = bb_max.y.max((p.y + t.radius + 1.0) as i32);
            bb_max.z = bb_max.z.max((p.z + t.radius + 1.0) as i32);
        }
    }
    for e in entrances {
        // Vertical shaft extent — clip downward to chamber, upward to
        // surface.
        bb_min.y = bb_min.y.min(e.surface.y);
        bb_max.y = bb_max.y.max(e.surface.y + 1);
        bb_min.x = bb_min.x.min(e.surface.x - ENTRANCE_BB_EXPAND);
        bb_max.x = bb_max.x.max(e.surface.x + ENTRANCE_BB_EXPAND);
        bb_min.z = bb_min.z.min(e.surface.z - ENTRANCE_BB_EXPAND);
        bb_max.z = bb_max.z.max(e.surface.z + ENTRANCE_BB_EXPAND);
    }
    (bb_min, bb_max)
}

/// Build one cave system inside region `coord`.
///
/// ### Chamber placement (Poisson-disk rejection sampling)
///
/// Candidate chamber centers are drawn uniformly at random inside the
/// system bounding box. A candidate is rejected if it lies within
/// `mean_radius × POISSON_MIN_SPACING_MULT` of any already-placed
/// chamber. This minimum-distance constraint prevents rooms from
/// overlapping or crowding into impenetrable clusters. Each attempt
/// increments an independent counter; after `max_tries` attempts the
/// sampling stops — if a target chamber count can't be reached, the
/// system uses however many fit, down to a minimum of 1.
///
/// ### Tunnel graph (Kruskal's MST)
///
/// After all chambers are placed, every pair of chambers becomes a
/// candidate edge weighted by their 3D Euclidean distance. Kruskal's
/// minimum spanning tree algorithm selects the subset of edges that
/// connects all chambers with the smallest total tunnel length. Path-
/// compression union-find gives near-O(α(n)) per edge, amortised over
/// the whole system. Additionally `MST_EXTRA_LOOPS.0..=1` short non-MST
/// edges are added back to introduce cycles — without loops the cave
/// graph is a tree and every room has exactly one entrance/exit, which
/// feels unnatural.
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

    let (bb_min, bb_max) = roll_bounding_box(seed, coord, system_idx, y_min, y_max);
    let chambers = sample_chambers(
        seed, coord, system_idx, bb_min, bb_max, style, &sp, cave_cfg,
    );
    let tunnels = connect_chambers_mst(seed, coord, system_idx, &chambers, &sp);
    let entrances = roll_entrances(
        seed, coord, system_idx, &chambers, band, heightmap, climate, density,
    );
    let (bb_min, bb_max) = finalize_aabb(bb_min, bb_max, &chambers, &tunnels, &entrances);

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
