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
//! Below `WORMHOLE_BAND_Y` a thin 3D-noise band carves scattered
//! "wormhole" passages independent of any system — the classic
//! "dig deep, occasionally hit a passage" reward loop, restricted
//! to the deep band so shallow caves stay coherent.

use crate::worldgen::hash::{mix_range, mix_u32, mix_unit};
use crate::worldgen::heightmap::HeightmapNoise;
use crate::worldgen::region::{
    CaveSystem, Chamber, Entrance, EntranceKind, FineRegion, RegionCoord, Tunnel,
};
use crate::worldgen::tuning::*;
use glam::{IVec3, Vec3};
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

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

/// Build all cave systems for the given fine region. Each system is
/// deterministically derived from `(seed, coord, system_idx)`.
pub fn build_systems_for_region(
    seed: u64,
    coord: RegionCoord,
    heightmap: &HeightmapNoise,
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
    region: &mut FineRegion,
) {
    // Decide how many systems this region hosts.
    let n_min = CAVE_SYSTEMS_PER_REGION.0;
    let n_max = CAVE_SYSTEMS_PER_REGION.1;
    let n = n_min
        + (mix_u32(seed, &[coord.x, coord.z, 1])
            % (n_max - n_min + 1));
    region.cave_systems.clear();
    for system_idx in 0..n as i32 {
        let sys = build_system(seed, coord, system_idx, heightmap, climate, density);
        region.cave_systems.push(sys);
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
) -> CaveSystem {
    let band = DepthBand::pick(seed, system_idx, coord);
    let (y_min, y_max) = band.range();
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

    // Chamber count.
    let cn_min = CHAMBERS_PER_SYSTEM.0;
    let cn_max = CHAMBERS_PER_SYSTEM.1;
    let chamber_count = cn_min
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, 20])
            % (cn_max - cn_min + 1));

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
        let rx = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 40, attempt],
            CHAMBER_RADIUS_RANGE.0,
            CHAMBER_RADIUS_RANGE.1,
        );
        let ry = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 41, attempt],
            CHAMBER_RADIUS_RANGE.0,
            CHAMBER_RADIUS_RANGE.1,
        );
        let rz = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 42, attempt],
            CHAMBER_RADIUS_RANGE.0,
            CHAMBER_RADIUS_RANGE.1,
        );
        let center = Vec3::new(
            bb_min.x as f32 + sx,
            bb_min.y as f32 + sy,
            bb_min.z as f32 + sz,
        );
        let radii = Vec3::new(rx, ry, rz);
        let mean_r = (rx + ry + rz) / 3.0;
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
        // Build a tunnel for each edge.
        for (idx, &(a, b)) in mst_edges.iter().enumerate() {
            let radius = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 60, idx as i32],
                TUNNEL_RADIUS.0,
                TUNNEL_RADIUS.1,
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
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::Sinkhole,
                surface: IVec3::new(cwx, surface_h, cwz),
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
                surface: IVec3::new(cwx, surface_h, cwz),
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

// ── Deep-band wormhole filler ────────────────────────────────────────

/// Sparse 3D-noise wormholes layered only in the deep band. Built
/// once per `Generator` (low cost).
pub struct WormholeNoise {
    a: Fbm<Simplex>,
    b: Fbm<Simplex>,
}

impl WormholeNoise {
    pub fn new(seed: u64) -> Self {
        let a = Fbm::<Simplex>::new(seed.wrapping_add(301) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 40.0)
            .set_persistence(0.5);
        let b = Fbm::<Simplex>::new(seed.wrapping_add(302) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 40.0)
            .set_persistence(0.5);
        Self { a, b }
    }

    /// True if the cell at `(wx, wy, wz)` lies inside a deep-band
    /// wormhole.
    pub fn carve(&self, wx: i32, wy: i32, wz: i32) -> bool {
        if wy >= WORMHOLE_BAND_Y {
            return false;
        }
        let p = [wx as f64, wy as f64, wz as f64];
        // Two zero-crossing sheets that intersect: classic tunnel
        // geometry. `WORMHOLE_BAND` controls width.
        self.a.get(p).abs() < WORMHOLE_BAND && self.b.get(p).abs() < WORMHOLE_BAND
    }
}

// ── Noise carver layers (PR 8) ───────────────────────────────────────
//
// Cheese / spaghetti / pillars: MC-style ambient noise-based cave
// density. Live alongside the graph cave systems and wormholes above.
// The three contributions wire into fill_chunk's `cave_contribution`
// composition (cheese + spaghetti subtract from density; pillars add
// back).

use crate::worldgen::config::CaveConfig;
use crate::worldgen::noise_channel::{
    build_channel, map_from_unit_to, weird_scaled_sample, y_clamped_gradient,
};

/// All MC-derived noise channels needed for the cheese, spaghetti,
/// and pillar carvers. Built once per Generator.
pub struct NoiseCarvers {
    // Cheese.
    pub cheese: Fbm<Simplex>,
    // Spaghetti.
    pub spaghetti_2d: Fbm<Simplex>,
    pub spaghetti_2d_modulator: Fbm<Simplex>,
    pub spaghetti_2d_elevation: Fbm<Simplex>,
    pub spaghetti_2d_thickness: Fbm<Simplex>,
    pub spaghetti_roughness: Fbm<Simplex>,
    // Pillars.
    pub pillar: Fbm<Simplex>,
    pub pillar_rareness: Fbm<Simplex>,
    pub pillar_thickness: Fbm<Simplex>,
    // Surface entrance noise.
    pub surface_entrance: Fbm<Simplex>,
}

impl NoiseCarvers {
    /// Build all channels from their config descriptors. Each
    /// channel uses a different seed-salt so they're uncorrelated.
    pub fn new(seed: u64, cfg: &CaveConfig) -> Self {
        Self {
            cheese: build_channel(&cfg.cheese, seed, 1001),
            spaghetti_2d: build_channel(&cfg.spaghetti_2d, seed, 1002),
            spaghetti_2d_modulator: build_channel(&cfg.spaghetti_2d_modulator, seed, 1003),
            spaghetti_2d_elevation: build_channel(&cfg.spaghetti_2d_elevation, seed, 1009),
            spaghetti_2d_thickness: build_channel(&cfg.spaghetti_2d_thickness, seed, 1004),
            spaghetti_roughness: build_channel(&cfg.spaghetti_roughness, seed, 1005),
            pillar: build_channel(&cfg.pillar, seed, 1006),
            pillar_rareness: build_channel(&cfg.pillar_rareness, seed, 1007),
            pillar_thickness: build_channel(&cfg.pillar_thickness, seed, 1008),
            surface_entrance: build_channel(&cfg.surface_entrance, seed, 1010),
        }
    }
}

/// Signed-density surface entrance contribution. Returns a strongly
/// negative value where the noise crosses threshold inside the
/// Y-band, ramping smoothly to zero outside; composed via
/// `min(other_caves, surface_entrance)` like the rest.
///
/// Unlike cheese and spaghetti, this carver is NOT gated by the
/// underground density threshold — that's the whole point: it
/// fires in the surface band specifically, punching small holes
/// through the heightmap.
pub fn surface_entrance_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    let fade = surface_entrance_y_fade(wy, cfg);
    if fade <= 0.0 {
        return 1.0; // sentinel positive (no cave)
    }
    let v = carvers.surface_entrance.get([
        wx as f64 * cfg.surface_entrance_xz_scale as f64,
        wy as f64 * cfg.surface_entrance_y_scale as f64,
        wz as f64 * cfg.surface_entrance_xz_scale as f64,
    ]) as f32;
    // Smooth ramp above threshold so the entrance edges aren't a
    // hard cliff.
    let above = v - cfg.surface_entrance_threshold;
    if above <= 0.0 {
        return 1.0;
    }
    let t = (above / 0.05).clamp(0.0, 1.0);
    let depth = t * t * (3.0 - 2.0 * t); // smoothstep
    // Return a strongly negative signed value; magnitude scaled by
    // both the smoothstep and the Y-band fade so edges are soft.
    -cfg.surface_entrance_intensity * depth * fade
}

fn surface_entrance_y_fade(wy: i32, cfg: &CaveConfig) -> f32 {
    if wy < cfg.surface_entrance_y_min || wy > cfg.surface_entrance_y_max {
        return 0.0;
    }
    let from_bottom = (wy - cfg.surface_entrance_y_min) as f32;
    let from_top = (cfg.surface_entrance_y_max - wy) as f32;
    let fade_w = cfg.surface_entrance_fade_blocks.max(1) as f32;
    (from_bottom.min(from_top) / fade_w).clamp(0.0, 1.0)
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
    let scale = cfg.cheese_xz_scale as f64;
    let cheese = carvers.cheese.get([
        wx as f64 * scale,
        wy as f64,
        wz as f64 * scale,
    ]) as f32;
    let term1 = (cfg.cheese_offset + cheese).clamp(-1.0, 1.0);
    let supp = (cfg.cheese_suppression_offset
        + cfg.cheese_suppression_slope * raw_density)
        .clamp(cfg.cheese_suppression_min, cfg.cheese_suppression_max);
    term1 + supp
}

/// Signed-density spaghetti tube contribution.
///
/// ```text
///   elev_mod        = map_from_unit_to(elev_noise, elev_min, elev_max)
///   sloped_spag     = abs(elev_mod + y_clamped_gradient(...))
///   thickness_mod   = thickness_offset
///                     + thickness_slope * thickness_noise
///   layer_ridged    = (sloped_spag + thickness_mod) ^ 3
///   cave_noise      = weird_scaled(modulator, sp2d)
///                     + cave_noise_offset * thickness_mod
///   spaghetti       = clamp(max(cave_noise, layer_ridged),
///                           clamp_min, clamp_max)
/// ```
///
/// The cube term `layer_ridged` is the secret to thin meandering
/// tubes: it's hugely positive (= solid) almost everywhere except
/// along a thin curve where `sloped_spag ≈ 0` — i.e., where the
/// elevation noise happens to cross `-y_gradient`. Inside that
/// curve the cube goes negative, defining the tube path. The cave
/// noise (region-modulated via [`weird_scaled_sample`]) further
/// gates the carving so different regions get different tube
/// scales.
///
/// Returns a signed density in roughly `[-1, 1]`. The caller
/// composes via `min(other_caves, spaghetti)`.
pub fn spaghetti_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    // 1. Elevation modulator: noise[-1,1] → [elev_min, elev_max].
    let elev_unit = carvers.spaghetti_2d_elevation.get([
        wx as f64,
        0.0,             // MC: y_scale=0 — sample independent of Y
        wz as f64,
    ]) as f32;
    let elev = map_from_unit_to(
        elev_unit,
        cfg.spaghetti_elevation_min,
        cfg.spaghetti_elevation_max,
    );

    // 2. Y-clamped gradient.
    let y_grad = y_clamped_gradient(
        wy,
        cfg.spaghetti_gradient_from_y,
        cfg.spaghetti_gradient_from_value,
        cfg.spaghetti_gradient_to_y,
        cfg.spaghetti_gradient_to_value,
    );

    // 3. slopedSpaghetti = abs(elev + ygrad).
    let sloped = (elev + y_grad).abs();

    // 4. Thickness modulator: linear remap of noise[-1,1].
    let thickness_noise = carvers.spaghetti_2d_thickness.get([
        wx as f64 * 2.0, // MC: xz_scale=2.0 on thickness
        wy as f64,
        wz as f64 * 2.0,
    ]) as f32;
    let thickness_mod =
        cfg.spaghetti_thickness_offset + cfg.spaghetti_thickness_slope * thickness_noise;

    // 5. layerRidged = (sloped + thickness_mod) ^ 3.
    let inner = sloped + thickness_mod;
    let layer_ridged = inner * inner * inner;

    // 6. caveNoise = weird_scaled(modulator, sp2d) + offset * thickness_mod.
    let modulator = carvers.spaghetti_2d_modulator.get([
        wx as f64 * 2.0, // MC: xz_scale=2.0 on modulator
        wy as f64,
        wz as f64 * 2.0,
    ]) as f32;
    let ws = weird_scaled_sample(
        &carvers.spaghetti_2d,
        modulator,
        wx as f64,
        wy as f64,
        wz as f64,
    );
    let cave_noise = ws + cfg.spaghetti_cave_noise_offset * thickness_mod;

    // 7. clamp(max(caveNoise, layerRidged), clamp_min, clamp_max).
    cave_noise
        .max(layer_ridged)
        .clamp(cfg.spaghetti_clamp_min, cfg.spaghetti_clamp_max)
}

/// Per-voxel spaghetti-roughness perturbation. Added to
/// [`spaghetti_contribution`] before composing with the rest of the
/// cave components — a tiny signed value that gives tube walls a
/// chiseled feel instead of mathematically-smooth boundaries.
pub fn spaghetti_roughness(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
) -> f32 {
    // -0.05 * abs(noise): negative sign means roughness pushes
    // density *down*, slightly enlarging carved volumes along
    // their edges.
    let v = carvers.spaghetti_roughness.get([
        wx as f64,
        wy as f64,
        wz as f64,
    ]) as f32;
    -0.05 * v.abs()
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

#[cfg(test)]
mod tests {
    use super::*;

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
                build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region);
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
        build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut r1);
        build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut r2);
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
        build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region);
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
        // Graph systems are now rare (CAVE_SYSTEMS_PER_REGION =
        // (0, 1)) — many regions have none. Scan a 4×4 grid of
        // regions until we find one with a chamber.
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let hm = HeightmapNoise::new(42, &cfg.climate);
        for rx in 0..4 {
            for rz in 0..4 {
                let coord = RegionCoord { x: rx, z: rz };
                let mut region = FineRegion::empty(coord);
                build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region);
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

    // ── PR 8: noise carver tests ─────────────────────────────────

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
        assert_eq!(a.spaghetti_2d.get(p), b.spaghetti_2d.get(p));
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
        // Signed-density: result is roughly in [-1, 1.5]. Verify
        // no NaN/Inf and that the band is respected across many
        // raw_density values.
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        for wy in (-100..=100).step_by(11) {
            for wx in (-100..100).step_by(17) {
                for wz in (-100..100).step_by(17) {
                    for raw_density in [-1.0_f32, 0.0, 1.0, 4.0, 16.0] {
                        let v = cheese_contribution(wx, wy, wz, raw_density, &nc, &cfg.cave);
                        assert!(v.is_finite(), "cheese non-finite at ({wx},{wy},{wz})");
                        // term1 ∈ [-1, 1], term2 ∈ [0, 0.5] → result ∈ [-1, 1.5]
                        assert!((-1.001..=1.501).contains(&v), "out-of-range cheese: {v}");
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
        // At raw_density large (deep underground), the suppression
        // term clamps to 0 — so cheese is just `clamp(offset +
        // noise, -1, 1)` and can freely go negative.
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        let mut found_negative = false;
        'outer: for wx in (-256..256).step_by(4) {
            for wz in (-256..256).step_by(4) {
                let v = cheese_contribution(wx, -40, wz, 20.0, &nc, &cfg.cave);
                if v < -0.1 {
                    found_negative = true;
                    break 'outer;
                }
            }
        }
        assert!(
            found_negative,
            "expected cheese to go strongly negative somewhere deep underground"
        );
    }

    #[test]
    fn spaghetti_signed_density_is_finite_and_clamped() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        for wx in (-200..200).step_by(13) {
            for wz in (-200..200).step_by(13) {
                for wy in (-100..=80).step_by(7) {
                    let v = spaghetti_contribution(wx, wy, wz, &nc, &cfg.cave);
                    assert!(v.is_finite(), "spaghetti non-finite at ({wx},{wy},{wz})");
                    assert!(
                        (cfg.cave.spaghetti_clamp_min - 1e-4..=cfg.cave.spaghetti_clamp_max + 1e-4)
                            .contains(&v),
                        "spaghetti out of clamp band: {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn spaghetti_carves_negative_somewhere_underground() {
        // Sanity check: signed-density spaghetti should hit
        // negative values somewhere in a wide deep-band sweep.
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        let mut any_carve = false;
        'outer: for y in (-100..=80).step_by(10) {
            for wx in (-256..256).step_by(4) {
                for wz in (-256..256).step_by(4) {
                    if spaghetti_contribution(wx, y, wz, &nc, &cfg.cave) < 0.0 {
                        any_carve = true;
                        break 'outer;
                    }
                }
            }
        }
        assert!(any_carve, "no spaghetti carving across a wide underground sweep");
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

    #[test]
    fn y_clamped_gradient_endpoint_values() {
        // Helper accepts either (lo,hi) ordering. Probe both orientations.
        assert!((y_clamped_gradient(-120, 140, 8.0, -120, -40.0) - (-40.0)).abs() < 1e-4);
        assert!((y_clamped_gradient(140, 140, 8.0, -120, -40.0) - 8.0).abs() < 1e-4);
        // Above the higher end → clamped to the higher Y's value.
        assert!((y_clamped_gradient(200, 140, 8.0, -120, -40.0) - 8.0).abs() < 1e-4);
        assert!((y_clamped_gradient(-200, 140, 8.0, -120, -40.0) - (-40.0)).abs() < 1e-4);
    }

    #[test]
    fn wormhole_noise_does_not_carve_above_band() {
        let w = WormholeNoise::new(42);
        // Above WORMHOLE_BAND_Y, never carve.
        for wx in (-100..=100).step_by(7) {
            for wy in (WORMHOLE_BAND_Y..=80).step_by(5) {
                for wz in (-100..=100).step_by(7) {
                    assert!(
                        !w.carve(wx, wy, wz),
                        "unexpected wormhole at ({wx},{wy},{wz}) — above the band"
                    );
                }
            }
        }
    }

}
