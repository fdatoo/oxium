//! Signed-distance functions for cave carving at chunk fill time.
//!
//! The SDF functions return a positive intensity inside cave volumes and
//! zero outside. The fill loop subtracts these from the per-voxel density
//! via `smin` so cave walls have soft, chamfered edges instead of the
//! pixel-sharp ellipsoid / capsule boundaries.
use super::style::{SALT_TRUNK_MID_OFFSET, SALT_TRUNK_PROB};
use crate::worldgen::hash::mix_unit;
use crate::worldgen::region::{CaveSystem, EntranceKind, RegionCoord};
use crate::worldgen::tuning::*;
use glam::Vec3;

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
            let ratio =
                (d.x / c.radii.x).powi(2) + (d.y / c.radii.y).powi(2) + (d.z / c.radii.z).powi(2);
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
        let Some(c0) = sys.chambers.first() else {
            continue;
        };
        let my_coord = RegionCoord::containing(c0.center.x as i32, c0.center.z as i32);
        let my_center = c0.center;

        // Stable salt: chamber-0 center coords (integer-rounded). Stable
        // across chunks because chamber centers are immutable; unique per
        // system within a region because Poisson sampling enforces
        // min_spacing > 0.
        let salt_x = c0.center.x as i32;
        let salt_y = c0.center.y as i32;
        let salt_z = c0.center.z as i32;

        let u = mix_unit(
            seed,
            &[
                my_coord.x,
                my_coord.z,
                salt_x,
                salt_y,
                salt_z,
                SALT_TRUNK_PROB,
            ],
        );
        if u > trunk_prob {
            continue;
        }

        // Find nearest chamber-0 in a neighbouring region (different
        // region, within 1 step in both x and z).
        let mut best_center: Option<Vec3> = None;
        let mut best_dist = f32::MAX;
        for (j, (other_coord, other_center)) in snap.iter().enumerate() {
            if j == si {
                continue;
            }
            let dx = (other_coord.x - my_coord.x).abs();
            let dz = (other_coord.z - my_coord.z).abs();
            if dx > 1 || dz > 1 {
                continue;
            }
            // Require different region (not same region coord).
            if *other_coord == my_coord {
                continue;
            }
            let d = (*other_center - my_center).length();
            if d < best_dist {
                best_dist = d;
                best_center = Some(*other_center);
            }
        }
        let Some(other_center) = best_center else {
            continue;
        };

        // Build trunk geometry (mirrors build_trunks).
        let axis = other_center - my_center;
        let len = (axis.x * axis.x + axis.z * axis.z).sqrt().max(1.0);
        let perp = Vec3::new(-axis.z / len, 0.0, axis.x / len);
        let o = (mix_unit(
            seed,
            &[
                my_coord.x,
                my_coord.z,
                salt_x,
                salt_y,
                salt_z,
                SALT_TRUNK_MID_OFFSET,
            ],
        ) * 2.0
            - 1.0)
            * 40.0;
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
            if len_sq < 1e-6 {
                continue;
            }
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
                    if wy as f32 >= chamber_top_y - 1.0 && wy as f32 <= e.surface.y as f32 {
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
                    if wy as f32 >= chamber_top_y - 1.0 && wy as f32 <= e.surface.y as f32 {
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
                    let cliff_p = Vec3::new(e.surface.x as f32, chamber_p.y, e.surface.z as f32);
                    let ab = cliff_p - chamber_p;
                    let len_sq = ab.length_squared();
                    if len_sq < 1e-6 {
                        continue;
                    }
                    let t_param = ((p - chamber_p).dot(ab) / len_sq).clamp(0.0, 1.0);
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
                (d.x / c.radii.x).powi(2) + (d.y / c.radii.y).powi(2) + (d.z / c.radii.z).powi(2);
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
                    let cliff_p = Vec3::new(e.surface.x as f32, chamber_p.y, e.surface.z as f32);
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
    systems
        .iter()
        .any(|s| wy >= s.bbox.min.y && wy <= s.bbox.max.y)
}

/// Polynomial smooth-min (Inigo Quilez's C1 smooth-min).
///
/// Produces a soft blend between two SDF surfaces within a blending radius
/// `k`. When `k = 0`, this is ordinary `min(a, b)`. Increasing `k` merges
/// nearby cave chambers and tunnels into one organic-looking connected
/// volume rather than leaving hard intersections where SDFs meet.
///
/// The formula: `min(a, b) - h²·k/4` where `h = max(0, k - |a-b|) / k`.
/// This is continuous and has a continuous first derivative at the blend
/// boundary; the maximum "pull below min" is exactly `k/4`.
///
/// Used to merge cave SDFs near layer boundaries so close-but-not-touching
/// pockets connect into one volume instead of staying as isolated bubbles.
#[inline]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = ((k - (a - b).abs()).max(0.0)) / k;
    a.min(b) - h * h * k * 0.25
}

/// Thin wrapper over [`CaveSystem::contains_point`]. Kept for call-site
/// symmetry with `any_system_y_in_range` inside this module.
#[inline]
pub(super) fn system_bb_contains(sys: &CaveSystem, wx: i32, wy: i32, wz: i32) -> bool {
    sys.contains_point(wx, wy, wz)
}
