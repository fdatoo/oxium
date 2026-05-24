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
//!
//! See `docs/book/content/part-3-region-build/3.1-plates.mdx` and
//! `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`.

use crate::worldgen::hash::{mix_range, mix_unit};
use crate::worldgen::tuning::*;
use glam::Vec2;

/// Identifier for one plate. Unique per `(cell_x, cell_z)`; the cell
/// coordinates double as the ID payload so the plate's parameters can
/// be recomputed from the ID alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlateId {
    /// Plate grid column (cell units, not block units).
    pub cell_x: i32,
    /// Plate grid row (cell units, not block units).
    pub cell_z: i32,
}

/// What kind of plate this is. Drives base elevation and whether the
/// plate contributes to continental-side ridge formation along its
/// boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlateKind {
    /// Thick, buoyant crust; sits above sea level in the spline output.
    Continental,
    /// Thin, dense crust; biased toward ocean depths in the spline output.
    Oceanic,
}

/// Static properties of a plate, derived deterministically from
/// `(seed, plate_id)`.
#[derive(Debug, Clone, Copy)]
pub struct Plate {
    /// Unique identifier; encodes the grid cell this plate belongs to.
    pub id: PlateId,
    /// Continental or oceanic; determines elevation bias in the splines.
    pub kind: PlateKind,
    /// World-space seed point inside the cell. Distances are measured
    /// to this point when classifying a column's plate.
    pub seed_xz: Vec2,
    /// Per-plate variation used as an additive bias on the climate
    /// `terrain_shape` channel. Higher value → more mountainous
    /// continent (post-PR-3; pre-PR-3 this was a multiplicative
    /// scale on the now-removed warped-FBM relief). Still raw
    /// `ROUGHNESS_RANGE` so other consumers (visualizer, debug)
    /// can interpret it; the bias map lives in heightmap.rs.
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
            roughness,
        }
    }
}

/// Per-column plate classification: nearest plate, second-nearest
/// plate, and the smooth boundary intensity between them.
#[derive(Debug, Clone, Copy)]
pub struct PlateLookup {
    /// The plate whose seed is nearest to the query point.
    pub a: Plate,
    /// The plate whose seed is second-nearest to the query point.
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

// `ridge_peak_for_pair`, `ridge_lift`, and `shelf_base` were the
// pre-PR-3 plate-mosaic heightmap primitives. Removed: the spline
// pipeline in `heightmap.rs` replaces them. Plate Voronoi geometry
// is still used to derive `signed_continentalness` for the spline.

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
        for (wx, wz) in [
            (0, 0),
            (400, 400),
            (1000, 1000),
            (-500, 700),
            (2000, -800),
            (-1500, -1200),
        ] {
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
        assert!(
            look.t > 0.5,
            "expected t > 0.5 at plate center, got {}",
            look.t
        );
    }

    #[test]
    fn plate_id_round_trips_through_of() {
        let p = Plate::of(42, 7, -3);
        assert_eq!(
            p.id,
            PlateId {
                cell_x: 7,
                cell_z: -3
            }
        );
    }
}
