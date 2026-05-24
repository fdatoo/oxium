//! Poisson-disk chamber placement for one cave system.
//!
//! [`sample_chambers`] draws candidate chamber centers uniformly at random
//! inside the system bounding box and rejects any candidate that lies within
//! `mean_radius × POISSON_MIN_SPACING_MULT` of an already-placed chamber.
//! This minimum-distance constraint prevents rooms from overlapping or
//! crowding into impenetrable clusters.
//!
//! ## Algorithm
//!
//! Standard rejection sampling (not bridson-fast-poisson, which would require
//! an active-cell list). Simple and good enough because chamber counts are
//! small (2–12 per system) so the expected number of retries is low.
//!
//! Sampling stops after `max_tries = 200` attempts regardless of target count
//! — the system uses however many chambers fit. The `POISSON_MIN_SPACING_MULT`
//! constant controls the tightness tradeoff: larger → fewer chambers per
//! bounding box; smaller → more risk of overlap bleed.
//!
//! ## Style adjustments
//!
//! - **Slot** style: both XZ radii derived from a single `rx` sample with a
//!   1.4/0.6 elongation factor, giving corridor-like rooms stretched in X.
//! - **Sump** style: `cy` is biased toward the bottom of the bounding box,
//!   clustering chambers near the floor for pit-like topology.
//!
//! See `docs/book/content/part-3-region-build/3.5-caves.mdx`.

use super::ctx::{CaveCtx, StyleParams};
use super::style::{CaveStyle, SALT_CHAMBER_COUNT};
use crate::worldgen::hash::{mix_range, mix_u32};
use crate::worldgen::region::{Chamber, ChamberRadius, SystemBoundingBox};
use crate::worldgen::tuning::*;
use glam::Vec3;

/// Place chambers inside `bb` using Poisson-disk rejection sampling.
///
/// Returns a `Vec<Chamber>` with at least 1 and at most `sp.chamber_count.1`
/// entries. The exact count depends on how many candidates pass the
/// minimum-spacing gate within `max_tries` attempts.
pub(super) fn sample_chambers(
    ctx: CaveCtx,
    bb: SystemBoundingBox,
    style: CaveStyle,
    sp: &StyleParams,
    cave_cfg: &crate::worldgen::config::CaveConfig,
) -> Vec<Chamber> {
    let CaveCtx {
        seed,
        coord,
        system_idx,
    } = ctx;
    // Chamber count from the style table.
    let (cn_min, cn_max) = sp.chamber_count;
    let chamber_count = cn_min
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, SALT_CHAMBER_COUNT])
            % (cn_max - cn_min + 1));

    // Pre-compute Sump bounding-box mid and half-extent for the bias formula.
    let bb_center_y = (bb.min.y + bb.max.y) as f32 * 0.5;
    let bb_half_y = (bb.max.y - bb.min.y) as f32 * 0.5;

    let mut chambers: Vec<Chamber> = Vec::with_capacity(chamber_count as usize);
    let mut tries = 0u32;
    let max_tries = 200u32;
    let mut attempt = 0i32;
    while chambers.len() < chamber_count as usize && tries < max_tries {
        let sx = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 30, attempt],
            0.0,
            (bb.max.x - bb.min.x) as f32,
        );
        let sy = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 31, attempt],
            0.0,
            (bb.max.y - bb.min.y) as f32,
        );
        let sz = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 32, attempt],
            0.0,
            (bb.max.z - bb.min.z) as f32,
        );

        // Depth multiplier: deeper chambers are larger.
        // `DEPTH_SCALE_PIVOT_Y` is the Y above which depth scaling is neutral;
        // below it the multiplier grows linearly with distance from the pivot.
        let cy_raw = bb.min.y as f32 + sy;
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

        // Slot: derive both XZ radii from rx with a 1.4/0.6 stretch so rooms
        // look corridor-like along X rather than roughly spherical.
        let (rx_final, rz_final) = if style == CaveStyle::Slot {
            (rx * 1.4, rx * 0.6)
        } else {
            (rx, rz)
        };

        // Sump: bias cy toward the bottom half of the bounding box so chambers
        // cluster near the floor, producing a pit-like connected layout.
        let cy_final = if style == CaveStyle::Sump {
            bb_center_y - bb_half_y * 0.4 + (cy_raw - bb_center_y).abs() * 0.5
        } else {
            cy_raw
        };

        let center = Vec3::new(bb.min.x as f32 + sx, cy_final, bb.min.z as f32 + sz);
        let radii = ChamberRadius(Vec3::new(rx_final, ry, rz_final));
        let mean_r = (rx_final + ry + rz_final) / 3.0;
        let min_spacing = mean_r * POISSON_MIN_SPACING_MULT;
        // Reject if too close to any existing chamber — the Poisson constraint.
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
