//! Vertical connectors between adjacent-band systems within a region, and
//! cross-region trunk tunnels linking neighbouring region systems.
use super::style::{DepthBand, SALT_TRUNK_MID_OFFSET, SALT_TRUNK_PROB, SALT_VERTICAL_CONNECTOR};
use crate::worldgen::hash::mix_unit;
use crate::worldgen::region::{FineRegion, RegionCoord, Tunnel};
use crate::worldgen::tuning::CAVE_BAND_MIDDLE;
use crate::worldgen::tuning::CAVE_BAND_SHALLOW;

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
    let bands: Vec<DepthBand> = region
        .cave_systems
        .iter()
        .map(|s| {
            let cy = (s.bbox.min.y + s.bbox.max.y) / 2;
            if cy >= CAVE_BAND_SHALLOW.0 {
                DepthBand::Shallow
            } else if cy >= CAVE_BAND_MIDDLE.0 {
                DepthBand::Middle
            } else {
                DepthBand::Deep
            }
        })
        .collect();
    let band_idx = |b: DepthBand| -> u8 {
        match b {
            DepthBand::Shallow => 0,
            DepthBand::Middle => 1,
            DepthBand::Deep => 2,
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
            let u = mix_unit(
                seed,
                &[
                    coord.x,
                    coord.z,
                    i as i32,
                    j as i32,
                    SALT_VERTICAL_CONNECTOR,
                ],
            );
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
                .filter_map(|(si, sys)| sys.chambers.first().map(|c| (ri, si, *coord, c.center)))
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
        let u = mix_unit(
            seed,
            &[coord.x, coord.z, salt_x, salt_y, salt_z, SALT_TRUNK_PROB],
        );
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
            if best.is_none_or(|(_, bd)| d < bd) {
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
        let o = (mix_unit(
            seed,
            &[
                coord.x,
                coord.z,
                salt_x,
                salt_y,
                salt_z,
                SALT_TRUNK_MID_OFFSET,
            ],
        ) * 2.0
            - 1.0)
            * 40.0;
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
