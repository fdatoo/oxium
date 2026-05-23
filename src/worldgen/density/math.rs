//! Standalone mathematical helpers for the density pipeline.
//!
//! These are pure functions — no struct state — used by
//! [`super::heightmap`] and [`super::cell_evaluator`] to implement
//! Minecraft 1.18-style terrain shape arithmetic.
//!
//! See `docs/superpowers/specs/2026-05-19-worldgen-3d-design.md` and
//! `docs/book/content/part-3-region-build/3.3-heightmap.mdx`.

use crate::worldgen::config::{ClimateConfig, DensityConfig};
use crate::worldgen::plates::{Plate, PlateKind, PlateLookup};
use crate::worldgen::tuning::*;
use glam::Vec2;

/// Derive a signed continentalness in `[-1.1, 1.1]` from the plate
/// Voronoi lookup. Both plate kinds contribute, weighted by `plate_t`:
///
/// * Deep inside a continental plate (t=1): c ≈ +1
/// * Deep inside an oceanic plate (t=1): c ≈ -1
/// * At a *continental-continental* boundary (t=0): c ≈ +1
///   (still inland — no spurious "ocean at C-C boundary")
/// * At an *oceanic-oceanic* boundary (t=0): c ≈ -1
/// * At a *continental-oceanic* boundary (t=0): c ≈ 0
///   (the only place the spline produces beaches / ocean shore)
///
/// The slight ×1.1 saturation lets the spline's outermost
/// mushroom-cap knot (loc=-1.10 / +1.10) catch the deepest cores.
pub fn signed_continentalness(look: &PlateLookup) -> f32 {
    let sign_of = |k: PlateKind| -> f32 {
        match k {
            PlateKind::Continental => 1.0,
            PlateKind::Oceanic => -1.0,
        }
    };
    let a_sign = sign_of(look.a.kind);
    let b_sign = sign_of(look.b.kind);
    // Weights: 0.5/0.5 at boundary, 1.0/0.0 deep inside `a`.
    let wa = 0.5 + 0.5 * look.t;
    let wb = 1.0 - wa;
    let raw = a_sign * wa + b_sign * wb;
    (raw * 1.1).clamp(-1.1, 1.1)
}

/// Per-plate roughness bias on the terrain_shape input. Maps the
/// plate's [`Plate::roughness`] (in [`ROUGHNESS_RANGE`]) to the
/// configured `plate_roughness_bias_range`. Continental "rocky"
/// plates have higher roughness → more negative bias (more
/// mountainous, since the offset spline's mountains live at low
/// terrain_shape).
pub fn plate_roughness_bias(plate: &Plate, cfg: &ClimateConfig) -> f32 {
    let (rmin, rmax) = ROUGHNESS_RANGE;
    let t = ((plate.roughness - rmin) / (rmax - rmin)).clamp(0.0, 1.0);
    let (bmin, bmax) = cfg.plate_roughness_bias_range;
    // Invert the mapping so high roughness → negative bias (mountains).
    bmax + (bmin - bmax) * t
}

/// Smooth N-plate blend over the 3×3 Voronoi cell window. Returns
/// `(signed_continentalness, terrain_shape_bias)` computed by
/// weighting every candidate plate with a quadratic falloff in
/// `d_i - d_min`.
///
/// Why this exists: the 2-nearest scheme used by [`PlateLookup`]
/// embeds a hidden step discontinuity — `look.b` is the
/// second-nearest plate, and as the query point moves the
/// second-nearest *identity* can flip from one plate to another at a
/// line locus. When the two competing b candidates have different
/// `kind` or different `roughness`, [`signed_continentalness`] and
/// [`plate_roughness_bias`] step at that locus, which the offset
/// spline at c≈±1 turns into a 30-50 block vertical cliff (see the
/// `probe_cliff` example).
///
/// The smooth blend below weights every plate seed in the 3×3 window
/// by `(1 - (d_i - d_min)/scale)^2`. A plate that's about to take over
/// the second slot enters the blend continuously from weight 0;
/// likewise a plate falling out leaves continuously to 0. `scale =
/// max(d_min, 64)` keeps the blend width finite at plate centres
/// (where `d_min → 0`) and proportional to local plate spacing
/// elsewhere. Plates with `d_i > d_min + scale` contribute exactly
/// zero, so in plate interiors a single plate dominates and behaviour
/// matches the old `look.t = 1` case.
pub fn smooth_plate_contribution(seed: u64, wx: i32, wz: i32, cfg: &ClimateConfig) -> (f32, f32) {
    let q = Vec2::new(wx as f32, wz as f32);
    let qcx = wx.div_euclid(PLATE_CELL_SIZE);
    let qcz = wz.div_euclid(PLATE_CELL_SIZE);

    let mut samples: [Option<(Plate, f32)>; 9] = [None; 9];
    let mut d_min = f32::INFINITY;
    let mut k = 0;
    for dz in -1..=1 {
        for dx in -1..=1 {
            let p = Plate::of(seed, qcx + dx, qcz + dz);
            let d = (p.seed_xz - q).length();
            if d < d_min {
                d_min = d;
            }
            samples[k] = Some((p, d));
            k += 1;
        }
    }
    let scale = d_min.max(64.0);

    let mut weighted_sign = 0.0_f32;
    let mut weighted_bias = 0.0_f32;
    let mut total = 0.0_f32;
    for slot in samples.iter() {
        let (p, d) = slot.expect("3×3 window always fills all 9 slots");
        let t = ((d - d_min) / scale).clamp(0.0, 1.0);
        let w = (1.0 - t).powi(2);
        if w <= 0.0 {
            continue;
        }
        let sign = match p.kind {
            PlateKind::Continental => 1.0_f32,
            PlateKind::Oceanic => -1.0_f32,
        };
        weighted_sign += sign * w;
        weighted_bias += plate_roughness_bias(&p, cfg) * w;
        total += w;
    }
    let cont = (weighted_sign / total * 1.1).clamp(-1.1, 1.1);
    let bias = weighted_bias / total;
    (cont, bias)
}

/// Peaks-and-valleys triangle fold on a raw ridge value `w ∈ [-1, 1]`.
/// Formula from MC's `NoiseRouterData.peaksAndValleys`:
///   `-(||w| - 2/3| - 1/3) * 3`
/// Output is a triangle wave on `|w|` with peaks at `|w| = 2/3`
/// (value `+1`) and valleys at `|w| = 0` and `|w| = 1` (value `-1`).
pub fn peaks_and_valleys(w: f32) -> f32 {
    -((w.abs() - 2.0 / 3.0).abs() - 1.0 / 3.0) * 3.0
}

/// Convert a spline offset value to a world-Y surface height. The
/// surface is where `y_gradient(y) + offset = 0`:
///   `y_grad(y) = amp * (1 - 2*(y - y_min) / (y_max - y_min))`
/// Solve for `y` when `y_grad = -offset`:
///   `y = y_min + (y_max - y_min) * (1 + offset/amp) * 0.5`
pub fn offset_to_world_y(offset: f32, density: &DensityConfig) -> f32 {
    let span = (density.y_max - density.y_min) as f32;
    density.y_min as f32 + span * (1.0 + offset / density.y_gradient_amplitude) * 0.5
}

/// Apply top and bottom slides at world Y. Within `slide_top_blocks`
/// of `y_max`, lerps density toward `slide_top_target` (negative →
/// pull to air). Within `slide_bottom_blocks` of `y_min`, lerps
/// toward `slide_bottom_target` (positive → pull to solid). Below
/// `y_min` (the world's designed range), returns a hard-positive
/// floor that no cave carver or aquifer can override — the void
/// stays solid.
pub fn slide(density: f32, wy: i32, cfg: &DensityConfig) -> f32 {
    // Hard void floor. Without this, a player who falls below
    // `y_min` lands inside the aquifer's per-cell water table at
    // extreme depth (slide_bottom_target ≈ 0.12 is smaller than
    // aquifer pressure ≈ 1, so the aquifer wins every voxel and
    // the entire deep void becomes water). 100 is well above any
    // single carver / aquifer contribution we ever produce.
    if wy < cfg.y_min {
        return 100.0;
    }
    let top_start = cfg.y_max - cfg.slide_top_blocks;
    let top_f = ((wy - top_start) as f32 / cfg.slide_top_blocks.max(1) as f32).clamp(0.0, 1.0);
    let after_top = density + (cfg.slide_top_target - density) * top_f;

    let bot_end = cfg.y_min + cfg.slide_bottom_blocks;
    let bot_f = ((bot_end - wy) as f32 / cfg.slide_bottom_blocks.max(1) as f32).clamp(0.0, 1.0);
    after_top + (cfg.slide_bottom_target - after_top) * bot_f
}
