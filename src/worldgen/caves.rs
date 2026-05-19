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
        let sys = build_system(seed, coord, system_idx, heightmap);
        region.cave_systems.push(sys);
    }
}

/// Build one cave system inside region `coord`.
fn build_system(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    heightmap: &HeightmapNoise,
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
        let surface_h = heightmap.h_pre(seed, chamber.center.x, chamber.center.z) as i32;
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
            if heightmap.is_cliff(seed, cwx_f, cwz_f) {
                found_cliff = Some(IVec3::new(
                    cwx_f as i32,
                    heightmap.h_pre(seed, cwx_f, cwz_f) as i32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_count_within_bounds() {
        // Roll systems for many regions; the count should always
        // sit in `CAVE_SYSTEMS_PER_REGION` inclusive.
        let hm = HeightmapNoise::new(42);
        for z in -3..=3 {
            for x in -3..=3 {
                let coord = RegionCoord { x, z };
                let mut region = FineRegion::empty(coord);
                build_systems_for_region(42, coord, &hm, &mut region);
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
        let hm = HeightmapNoise::new(42);
        let coord = RegionCoord { x: 2, z: -3 };
        let mut r1 = FineRegion::empty(coord);
        let mut r2 = FineRegion::empty(coord);
        build_systems_for_region(42, coord, &hm, &mut r1);
        build_systems_for_region(42, coord, &hm, &mut r2);
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
        let hm = HeightmapNoise::new(42);
        let coord = RegionCoord { x: 0, z: 0 };
        let mut region = FineRegion::empty(coord);
        build_systems_for_region(42, coord, &hm, &mut region);
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
        let hm = HeightmapNoise::new(42);
        let coord = RegionCoord { x: 0, z: 0 };
        let mut region = FineRegion::empty(coord);
        build_systems_for_region(42, coord, &hm, &mut region);
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
        panic!("no chambers across region (0,0) — adjust test seed or region count");
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
