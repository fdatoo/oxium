//! Voronoi plate decomposition driving continental geography.
//!
//! The world is partitioned into **plates** — irregular convex-ish
//! regions, each with a `kind` (continental or oceanic), a base
//! elevation, and a base-relief roughness multiplier. Every world
//! column belongs to exactly one plate; mountain ranges form along
//! the boundaries between plates.
//!
//! Geometry: each `PLATE_CELL_SIZE × PLATE_CELL_SIZE` world cell rolls
//! one Voronoi seed point at a hashed jitter offset inside the cell.
//! To find the plate at `world_xz`, scan the 3×3 grid of cells
//! surrounding it, pick the nearest seed → `plate_a`, second-nearest
//! → `plate_b`. The continuous **boundary intensity**
//! `t = (d_b − d_a) / (d_b + d_a)` is 0 exactly on a boundary and 1
//! deep inside a plate; downstream modules use `t` to interpolate
//! between adjacent plates' base elevations (continental shelf taper)
//! and to gate the mountain-ridge lift.

use crate::worldgen::hash::{mix_range, mix_u32, mix_unit};
use crate::worldgen::tuning::*;
use glam::Vec2;

/// Identifier for one plate. Unique per `(cell_x, cell_z)`; the cell
/// coordinates double as the ID payload so the plate's parameters can
/// be recomputed from the ID alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlateId {
    pub cell_x: i32,
    pub cell_z: i32,
}

/// What kind of plate this is. Drives base elevation and whether the
/// plate contributes to continental-side ridge formation along its
/// boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlateKind {
    Continental,
    Oceanic,
}

/// Static properties of a plate, derived deterministically from
/// `(seed, plate_id)`.
#[derive(Debug, Clone, Copy)]
pub struct Plate {
    pub id: PlateId,
    pub kind: PlateKind,
    /// World-space seed point inside the cell. Distances are measured
    /// to this point when classifying a column's plate.
    pub seed_xz: Vec2,
    /// Sea-level-frame base elevation contribution. Positive for
    /// continental plates, negative for oceanic.
    pub base_elevation: f32,
    /// Multiplier applied to the warped-FBM base relief. Higher →
    /// more dramatic hills inside this plate.
    pub roughness: f32,
}

impl Plate {
    /// Build the plate that lives in cell `(cell_x, cell_z)` of the
    /// jittered Voronoi grid for world `seed`. Pure in `(seed, id)` —
    /// the result is byte-stable regardless of how the function is
    /// invoked.
    pub fn of(seed: u64, cell_x: i32, cell_z: i32) -> Self {
        let id = PlateId { cell_x, cell_z };
        // Seed jitter: uniform random offset inside the cell. Salt 0
        // for X-jitter, 1 for Z-jitter so they're independent.
        let jx = mix_unit(seed, &[cell_x, cell_z, 0]);
        let jz = mix_unit(seed, &[cell_x, cell_z, 1]);
        let seed_xz = Vec2::new(
            (cell_x as f32 + jx) * PLATE_CELL_SIZE as f32,
            (cell_z as f32 + jz) * PLATE_CELL_SIZE as f32,
        );
        // Kind: roll against CONTINENTAL_RATIO. Salt 2.
        let kind = if mix_unit(seed, &[cell_x, cell_z, 2]) < CONTINENTAL_RATIO {
            PlateKind::Continental
        } else {
            PlateKind::Oceanic
        };
        // Base elevation: per-kind range, salt 3.
        let base_elevation = match kind {
            PlateKind::Continental => mix_range(
                seed,
                &[cell_x, cell_z, 3],
                CONTINENTAL_BASE_RANGE.0,
                CONTINENTAL_BASE_RANGE.1,
            ),
            PlateKind::Oceanic => mix_range(
                seed,
                &[cell_x, cell_z, 3],
                OCEANIC_BASE_RANGE.0,
                OCEANIC_BASE_RANGE.1,
            ),
        };
        // Roughness, salt 4.
        let roughness = mix_range(
            seed,
            &[cell_x, cell_z, 4],
            ROUGHNESS_RANGE.0,
            ROUGHNESS_RANGE.1,
        );
        Plate {
            id,
            kind,
            seed_xz,
            base_elevation,
            roughness,
        }
    }
}

/// Per-column plate classification: nearest plate, second-nearest
/// plate, and the smooth boundary intensity between them.
#[derive(Debug, Clone, Copy)]
pub struct PlateLookup {
    pub a: Plate,
    pub b: Plate,
    /// Distance to `a`'s seed (blocks).
    pub d_a: f32,
    /// Distance to `b`'s seed (blocks).
    pub d_b: f32,
    /// Boundary intensity: `0` exactly on a plate boundary, `1` deep
    /// inside `a`'s plate. Smooth.
    pub t: f32,
}

/// Find the plate(s) at world position `(wx, wz)`.
///
/// Scans the 3×3 grid of plate cells around the query point and
/// computes the two nearest seed points. With `PLATE_CELL_SIZE` = 1024
/// and jitter ∈ [0, 1), the second-nearest is guaranteed to live in
/// the 3×3 window (worst case: the query sits at a cell corner; the
/// candidate seeds are in the four cells touching that corner, all
/// within the 3×3 window).
pub fn plate_at(seed: u64, wx: i32, wz: i32) -> PlateLookup {
    let q = Vec2::new(wx as f32, wz as f32);
    let qcx = wx.div_euclid(PLATE_CELL_SIZE);
    let qcz = wz.div_euclid(PLATE_CELL_SIZE);
    // Find two closest seeds in a 3×3 window of plate cells.
    let mut best = (f32::INFINITY, None::<Plate>);
    let mut second = (f32::INFINITY, None::<Plate>);
    for dz in -1..=1 {
        for dx in -1..=1 {
            let cell_x = qcx + dx;
            let cell_z = qcz + dz;
            let p = Plate::of(seed, cell_x, cell_z);
            let d = (p.seed_xz - q).length();
            if d < best.0 {
                second = best;
                best = (d, Some(p));
            } else if d < second.0 {
                second = (d, Some(p));
            }
        }
    }
    let a = best.1.expect("3×3 window always yields a nearest plate");
    let b = second
        .1
        .expect("3×3 window has 9 candidates, at least 2 exist");
    let d_a = best.0;
    let d_b = second.0;
    // Boundary intensity. Guard the denominator: when query sits exactly
    // on a plate seed point both distances are zero — return `t = 1` in
    // that case (deep inside the plate).
    let denom = d_a + d_b;
    let t = if denom < f32::EPSILON {
        1.0
    } else {
        (d_b - d_a) / denom
    };
    PlateLookup { a, b, d_a, d_b, t }
}

/// Per-pair maximum mountain-ridge peak height. Driven by the two
/// plates' kinds. See spec Section "Plate-edge ridges".
pub fn ridge_peak_for_pair(a: PlateKind, b: PlateKind) -> f32 {
    use PlateKind::*;
    match (a, b) {
        (Continental, Continental) => RIDGE_PEAK_CC,
        (Continental, Oceanic) | (Oceanic, Continental) => RIDGE_PEAK_CO,
        (Oceanic, Oceanic) => RIDGE_PEAK_OO,
    }
}

/// Mountain-ridge contribution (blocks) added on top of the base
/// continental shelf at world `(wx, wz)`.
///
/// Geometry: triangular falloff with boundary intensity `t`. Inside
/// `t < BOUNDARY_RIDGE_WIDTH`, lift ramps from `peak` at `t = 0` to 0
/// at `t = BOUNDARY_RIDGE_WIDTH`. Outside that window, zero.
///
/// The `peak` itself is modulated along the boundary curve by a 1D
/// noise (parameterised by `mix`-hash of the midpoint between the two
/// seed points) so the chain has saddles and crests, not a uniform
/// wall. Range: peak * [0.45, 1.0].
pub fn ridge_lift(look: &PlateLookup, seed: u64) -> f32 {
    // Outside the ridge window — no contribution.
    if look.t >= BOUNDARY_RIDGE_WIDTH {
        return 0.0;
    }
    let peak_max = ridge_peak_for_pair(look.a.kind, look.b.kind);
    // 1D noise along the boundary: hash the midpoint of the two seed
    // points, quantised to ~32-block buckets so the noise has spatial
    // continuity (neighbouring columns hash the same bucket).
    let mid = (look.a.seed_xz + look.b.seed_xz) * 0.5;
    // Project the query onto the line between the two seed points to
    // get a 1D position along the boundary.
    let axis = (look.b.seed_xz - look.a.seed_xz).normalize_or_zero();
    // Use the query distance along that axis as the noise parameter.
    let along = mid.dot(axis) as i32 / 32; // 32-block bucket
    // Two adjacent buckets, linearly interpolated by the fractional
    // remainder of `along` so the lift varies smoothly along the
    // chain instead of stair-stepping.
    let along_frac = ((mid.dot(axis) / 32.0) - along as f32).clamp(0.0, 1.0);
    let pair_salt = mix_u32(seed, &[look.a.id.cell_x, look.a.id.cell_z, look.b.id.cell_x, look.b.id.cell_z]);
    let n0 = mix_unit(seed, &[pair_salt as i32, along, 0]);
    let n1 = mix_unit(seed, &[pair_salt as i32, along + 1, 0]);
    let n = n0 * (1.0 - along_frac) + n1 * along_frac;
    // Map noise to [0.45, 1.0] so saddles aren't full zero.
    let mod_factor = 0.45 + 0.55 * n;
    let falloff = 1.0 - look.t / BOUNDARY_RIDGE_WIDTH;
    peak_max * falloff * mod_factor
}

/// Continental-shelf-tapered base elevation at world `(wx, wz)`.
/// Lerps between plate `a`'s and plate `b`'s base elevations by a
/// weight derived from boundary intensity `t`. Inside the ridge band
/// the blend is sharp; deep inside a plate the blend pulls toward
/// `a`'s base.
pub fn shelf_base(look: &PlateLookup) -> f32 {
    // Weight: how much of `a`'s base elevation to use. `t = 1` (deep
    // inside `a`) → fully `a`. `t = 0` (on the boundary) → 50/50.
    let w_a = 0.5 + 0.5 * look.t.clamp(0.0, 1.0);
    look.a.base_elevation * w_a + look.b.base_elevation * (1.0 - w_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plate_at_is_pure() {
        let p1 = plate_at(42, 100, 200);
        let p2 = plate_at(42, 100, 200);
        assert_eq!(p1.a.id, p2.a.id);
        assert_eq!(p1.b.id, p2.b.id);
        assert!((p1.t - p2.t).abs() < 1e-6);
    }

    #[test]
    fn different_seeds_give_different_plates() {
        // Plate IDs come from cell coordinates, which DON'T depend on
        // the seed — what changes between seeds is the jitter inside
        // each cell (which seed wins the nearest-neighbour race for
        // any given query). So we check the *winning seed_xz* coords,
        // not the cell id.
        let mut differ = 0;
        for (wx, wz) in [(0, 0), (400, 400), (1000, 1000), (-500, 700), (2000, -800), (-1500, -1200)] {
            let a = plate_at(0xAA00_AA00_AA00_AA00, wx, wz);
            let b = plate_at(0x55FF_55FF_55FF_55FF, wx, wz);
            if a.a.seed_xz != b.a.seed_xz {
                differ += 1;
            }
        }
        assert!(
            differ >= 4,
            "expected ≥4 of 6 sample plates' seed points to differ between seeds; saw {differ}"
        );
    }

    #[test]
    fn continental_ratio_in_5pct() {
        // Scan a wide area, accumulate continental fraction.
        let mut cont = 0;
        let mut total = 0;
        for cz in -16..16 {
            for cx in -16..16 {
                let p = Plate::of(42, cx, cz);
                total += 1;
                if matches!(p.kind, PlateKind::Continental) {
                    cont += 1;
                }
            }
        }
        let frac = cont as f32 / total as f32;
        let diff = (frac - CONTINENTAL_RATIO).abs();
        assert!(
            diff < 0.05,
            "continental ratio drifted: got {frac}, expected {CONTINENTAL_RATIO} ± 0.05"
        );
    }

    #[test]
    fn boundary_intensity_zero_to_one() {
        // Spot-check several queries: t should be in [0, 1].
        for (wx, wz) in [(0, 0), (500, 500), (2000, -1000), (-3500, 2200)] {
            let look = plate_at(42, wx, wz);
            assert!(
                (0.0..=1.0).contains(&look.t),
                "t {} out of range at ({wx}, {wz})",
                look.t
            );
        }
    }

    #[test]
    fn far_from_boundary_t_is_close_to_one() {
        // At a plate seed point, t should be near 1.
        let p = Plate::of(42, 0, 0);
        let wx = p.seed_xz.x as i32;
        let wz = p.seed_xz.y as i32;
        let look = plate_at(42, wx, wz);
        assert!(look.t > 0.5, "expected t > 0.5 at plate center, got {}", look.t);
    }

    #[test]
    fn plate_id_round_trips_through_of() {
        let p = Plate::of(42, 7, -3);
        assert_eq!(p.id, PlateId { cell_x: 7, cell_z: -3 });
    }

    #[test]
    fn ridge_lift_zero_outside_band() {
        let mut look = plate_at(42, 0, 0);
        look.t = 0.5; // way outside BOUNDARY_RIDGE_WIDTH
        assert_eq!(ridge_lift(&look, 42), 0.0);
    }

    #[test]
    fn ridge_lift_positive_at_boundary() {
        // Find a column on a CC boundary by scanning until t is small
        // and both plates are continental.
        for wx in (-2000..2000).step_by(20) {
            for wz in (-2000..2000).step_by(20) {
                let look = plate_at(42, wx, wz);
                if look.t < 0.05
                    && matches!(look.a.kind, PlateKind::Continental)
                    && matches!(look.b.kind, PlateKind::Continental)
                {
                    let lift = ridge_lift(&look, 42);
                    assert!(
                        lift > 0.0,
                        "expected positive ridge lift on CC boundary, got {lift}"
                    );
                    return;
                }
            }
        }
        panic!("no CC boundary found in scan — adjust CONTINENTAL_RATIO?");
    }

}
