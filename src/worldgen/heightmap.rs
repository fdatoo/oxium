//! Pre-river heightmap composition.
//!
//! `h_pre` = (continental shelf from plates) + (mountain ridge lift on
//! plate boundaries) + (domain-warped FBM relief, scaled per-plate).
//! `h_final` (computed elsewhere) subtracts river-valley carve from
//! `h_pre`.
//!
//! Three things live here:
//!
//! 1. `HeightmapNoise` — the three noise fields needed: one base FBM
//!    for relief, two scalar fields for the X/Z components of the
//!    domain-warp vector.
//! 2. `h_pre` — the actual heightmap evaluator.
//! 3. `slope_at` — sampled gradient magnitude used by `surface.rs` to
//!    decide whether a column is exposed as bare rock (cliff).

use crate::worldgen::plates::{plate_at, ridge_lift, shelf_base};
use crate::worldgen::tuning::*;
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

/// Peak-to-peak amplitude of the base FBM relief contribution. Per-
/// plate `roughness` scales this up or down per region of the world.
pub const BASE_FBM_AMPLITUDE: f32 = 24.0;
/// Spatial period of the inner FBM. ~96 blocks gives hills you can
/// walk over comfortably; smaller → busier terrain.
pub const BASE_FBM_PERIOD: f32 = 96.0;

/// Bundle of all noise fields needed for `h_pre`. Built once per
/// `Generator` and shared across chunk fills (the `NoiseFn::get` calls
/// are immutable so concurrent reads are fine).
pub struct HeightmapNoise {
    /// Base relief — 4-octave FBM with reduced persistence so finer
    /// octaves contribute less amplitude.
    base: Fbm<Simplex>,
    /// X component of the domain-warp vector field.
    warp_x: Fbm<Simplex>,
    /// Z component of the domain-warp vector field. Independently
    /// seeded so it doesn't correlate with warp_x.
    warp_z: Fbm<Simplex>,
}

impl HeightmapNoise {
    /// Build the noise fields for a given world `seed`. Per-field
    /// seed offsets keep the fields uncorrelated.
    pub fn new(seed: u64) -> Self {
        let base = Fbm::<Simplex>::new(seed as u32)
            .set_octaves(4)
            .set_frequency(1.0 / BASE_FBM_PERIOD as f64)
            .set_persistence(0.5);
        let warp_x = Fbm::<Simplex>::new(seed.wrapping_add(101) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / WARP_PERIOD as f64)
            .set_persistence(0.5);
        let warp_z = Fbm::<Simplex>::new(seed.wrapping_add(202) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / WARP_PERIOD as f64)
            .set_persistence(0.5);
        Self {
            base,
            warp_x,
            warp_z,
        }
    }

    /// Pre-river heightmap (blocks, world-Y).
    ///
    /// ```text
    /// h_pre = SEA_LEVEL
    ///       + shelf_base(plates)              // continental contribution, plate-lerped
    ///       + ridge_lift(plates, seed)        // mountain spine on plate boundaries
    ///       + base_fbm(warp(wx, wz)) * BASE_FBM_AMPLITUDE * plate_roughness
    /// ```
    ///
    /// Final clamp keeps everything inside the vertical chunk-load
    /// radius even on pathological plate-base + ridge combinations.
    pub fn h_pre(&self, seed: u64, wx: f32, wz: f32) -> f32 {
        let look = plate_at(seed, wx as i32, wz as i32);
        let shelf = shelf_base(&look);
        let ridge = ridge_lift(&look, glam::Vec2::new(wx, wz), seed);
        // Domain warp: two independently seeded scalar fields produce
        // a 2D offset that perturbs the FBM input. Breaks the
        // rounded-blob signature of plain FBM.
        let p_warp = [wx as f64, wz as f64];
        let wxp = wx + (self.warp_x.get(p_warp) as f32) * WARP_AMPLITUDE;
        let wzp = wz + (self.warp_z.get(p_warp) as f32) * WARP_AMPLITUDE;
        let relief = (self.base.get([wxp as f64, wzp as f64]) as f32) * BASE_FBM_AMPLITUDE;
        let h = SEA_LEVEL as f32 + shelf + ridge + relief * look.a.roughness;
        h.clamp((CAVE_FLOOR_Y + 8) as f32, MAX_TERRAIN_Y as f32)
    }

    /// Magnitude of the horizontal gradient of `h_pre` at `(wx, wz)`,
    /// in blocks per block. Cheap finite-difference over an 8-block
    /// window (±4 blocks each axis). Wider than the obvious ±1 stencil
    /// because we want cliff detection to fire only on genuinely
    /// large-scale steepness — small-scale FBM jitter and natural
    /// land-to-ocean shelf transitions should *not* register as
    /// cliffs (they were producing straight cliff strips along
    /// coastlines and plate boundaries).
    pub fn slope_at(&self, seed: u64, wx: f32, wz: f32) -> f32 {
        let step = 4.0;
        let hxp = self.h_pre(seed, wx + step, wz);
        let hxn = self.h_pre(seed, wx - step, wz);
        let hzp = self.h_pre(seed, wx, wz + step);
        let hzn = self.h_pre(seed, wx, wz - step);
        let gx = (hxp - hxn).abs() / (2.0 * step);
        let gz = (hzp - hzn).abs() / (2.0 * step);
        gx.max(gz)
    }

    /// True if any of a wider stencil's samples around `(wx, wz)`
    /// dips below sea level. Used by the subsurface block selector
    /// to extend the dirt cap of coastal columns down to sea level
    /// so their water-facing sides don't reveal the underlying
    /// stone bedrock.
    ///
    /// Samples 8 directions at distances of 8 and 20 blocks so we
    /// catch columns whose water is up to ~20 blocks away — wide
    /// enough for low coastal hills but narrow enough that inland
    /// mountains far from any shoreline still get the normal
    /// thin-dirt + stone subsurface (no giant dirt walls on
    /// mountain sides).
    pub fn is_coastal(&self, seed: u64, wx: f32, wz: f32) -> bool {
        let sea = SEA_LEVEL as f32;
        for step in [8.0_f32, 20.0] {
            for (dx, dz) in [
                (step, 0.0),
                (-step, 0.0),
                (0.0, step),
                (0.0, -step),
                (step * 0.71, step * 0.71),
                (-step * 0.71, step * 0.71),
                (step * 0.71, -step * 0.71),
                (-step * 0.71, -step * 0.71),
            ] {
                if self.h_pre(seed, wx + dx, wz + dz) < sea {
                    return true;
                }
            }
        }
        false
    }

    /// True if the column is steep enough to expose bare rock.
    /// Two gates before the slope test:
    ///
    /// 1. The column's own `h_pre` must be at or above
    ///    `CLIFF_MIN_HEIGHT` — low-elevation columns never cliff
    ///    regardless of slope.
    /// 2. **No stencil sample may dip below sea level.** If any of
    ///    the ±4-block stencil sample positions falls below
    ///    `SEA_LEVEL`, the column is *coastal* and we leave its
    ///    surface to the biome rules (grass / sand / etc) so the
    ///    shoreline doesn't read as a continuous stone wall.
    ///
    /// Only inland cliffs (mountain faces, canyon walls) where the
    /// terrain stays above sea level across the whole stencil pass
    /// both gates and then face the slope test.
    pub fn is_cliff(&self, seed: u64, wx: f32, wz: f32) -> bool {
        let h = self.h_pre(seed, wx, wz);
        if h < CLIFF_MIN_HEIGHT as f32 {
            return false;
        }
        let step = 4.0;
        let hxp = self.h_pre(seed, wx + step, wz);
        let hxn = self.h_pre(seed, wx - step, wz);
        let hzp = self.h_pre(seed, wx, wz + step);
        let hzn = self.h_pre(seed, wx, wz - step);
        let sea = SEA_LEVEL as f32;
        if hxp < sea || hxn < sea || hzp < sea || hzn < sea {
            return false;
        }
        let gx = (hxp - hxn).abs() / (2.0 * step);
        let gz = (hzp - hzn).abs() / (2.0 * step);
        gx.max(gz) > CLIFF_SLOPE_THRESH
    }
}

/// 3D density evaluator: combines a height-bias term (positive below
/// `h_target`, negative above) with a 3D relief FBM. A voxel is solid
/// iff `density > 0`. Inside a narrow `SURFACE_BAND` around `h_target`
/// the relief noise jitters the surface position, so the resulting
/// terrain doesn't read as a clean integer-rounded staircase.
///
/// Composed independently of `HeightmapNoise` so chunk fill can
/// evaluate the bias from the existing 2D height target while the
/// 3D noise field lives here.
pub struct DensityNoise {
    relief: Fbm<Simplex>,
}

impl DensityNoise {
    pub fn new(seed: u64) -> Self {
        // 3 octaves of 3D simplex at `RELIEF_PERIOD` base period;
        // persistence 0.5 keeps the fine octave subtle but present.
        let relief = Fbm::<Simplex>::new(seed.wrapping_add(701) as u32)
            .set_octaves(3)
            .set_frequency(1.0 / RELIEF_PERIOD as f64)
            .set_persistence(0.5);
        Self { relief }
    }

    /// New MC-style composition. Replaces the old `evaluate` once all call
    /// sites migrate (task 10). Until then, both coexist.
    ///
    /// Formula:
    ///   y_grad     = lerp from +amp at y_min → −amp at y_max
    ///   offset     = (h_target placement) shifts surface to h_target
    ///   depth      = y_grad + offset
    ///   shaped     = (depth + jagged) * factor
    ///   shaped'    = quarter_negative(shaped)   // soften above surface
    ///   density    = scale * shaped' + base_3d_noise
    ///   density    = slide(density, y)          // top→air, bottom→solid
    pub fn evaluate_v2(
        &self,
        h_target: f32,
        wx: i32,
        wy: i32,
        wz: i32,
        cfg: &crate::worldgen::config::DensityConfig,
    ) -> f32 {
        // y_gradient: +amplitude at y_min, -amplitude at y_max,
        // linear in between.
        let t = (wy - cfg.y_min) as f32 / (cfg.y_max - cfg.y_min) as f32;
        let y_gradient = cfg.y_gradient_amplitude * (1.0 - 2.0 * t);

        // Offset: shifts y_gradient so depth=0 at h_target.
        // PR 2: derive from h_target (preserves current terrain shape).
        // PR 3 will replace with a spline of (continentalness, erosion, ridges).
        let t_at_target = (h_target - cfg.y_min as f32) / (cfg.y_max - cfg.y_min) as f32;
        let offset = cfg.y_gradient_amplitude * (2.0 * t_at_target - 1.0);

        let depth = y_gradient + offset;
        // PR 3 introduces real jaggedness; for PR 2 it's zero.
        let jagged = 0.0;
        let factor = cfg.factor;

        // Quarter-negative softening: positive (below surface) keeps
        // full magnitude; negative (above surface) scales by
        // `above_surface_softening`. Net effect: solid below grows fast,
        // air above grows slowly, so 3D noise can carve overhangs
        // without piercing solid ground.
        let shaped_raw = (depth + jagged) * factor;
        let shaped = if shaped_raw > 0.0 {
            shaped_raw
        } else {
            shaped_raw * cfg.above_surface_softening
        };

        let base_3d = self.evaluate_base_3d(h_target, wx, wy, wz, cfg);
        let pre_slide = cfg.composition_scale * shaped + base_3d;
        slide(pre_slide, wy, cfg)
    }

    /// Sample the anisotropic base 3D noise. Y is scaled by
    /// `cfg.base_3d_y_scale` before sampling — values < 1 stretch
    /// vertical features (make them taller than wide). Used by the
    /// new MC-style composition in [`Self::evaluate_v2`].
    pub fn evaluate_base_3d(
        &self,
        _h_target: f32, // ignored; kept for symmetry with evaluate()
        wx: i32,
        wy: i32,
        wz: i32,
        cfg: &crate::worldgen::config::DensityConfig,
    ) -> f32 {
        let scaled_y = wy as f64 * cfg.base_3d_y_scale as f64;
        self.relief.get([wx as f64, scaled_y, wz as f64]) as f32 * cfg.base_3d_amplitude
    }

    /// Walk `(wx, wz)` top-down through the density function and
    /// return the first voxel `wy` where `density > 0` (the topmost
    /// solid block). Searches from `top` downward to a hard floor
    /// at `h_target - SURFACE_BAND - 1` (below that, everything is
    /// definitely solid, so the first solid is at most that far
    /// below the target). Returns `None` if nothing solid found
    /// within the search range (shouldn't happen for normal
    /// terrain).
    pub fn topmost_solid(
        &self,
        h_target: f32,
        wx: i32,
        wz: i32,
        search_top: i32,
        cfg: &crate::worldgen::config::DensityConfig,
    ) -> Option<i32> {
        // Don't bother searching above `h_target + SURFACE_BAND`:
        // that region is unconditionally air.
        let top = search_top.min(h_target as i32 + SURFACE_BAND);
        let bottom = (h_target as i32 - SURFACE_BAND).max(CAVE_FLOOR_Y);
        for wy in (bottom..=top).rev() {
            if self.evaluate_v2(h_target, wx, wy, wz, cfg) > 0.0 {
                return Some(wy);
            }
        }
        // Below the band, terrain is unconditionally solid → topmost
        // solid is the bottom of the search range.
        Some(bottom)
    }
}

/// Apply top and bottom slides to a density value at world Y.
/// Within `slide_top_blocks` of `y_max`, lerps toward
/// `slide_top_target`. Within `slide_bottom_blocks` of `y_min`,
/// lerps toward `slide_bottom_target`. Free helper, no state.
pub fn slide(density: f32, wy: i32, cfg: &crate::worldgen::config::DensityConfig) -> f32 {
    // Top: factor goes 0 (no pull) → 1 (full pull) over
    // (y_max - slide_top_blocks .. y_max).
    let top_start = cfg.y_max - cfg.slide_top_blocks;
    let top_f = ((wy - top_start) as f32 / cfg.slide_top_blocks.max(1) as f32).clamp(0.0, 1.0);
    let after_top = density + (cfg.slide_top_target - density) * top_f;

    // Bottom: factor goes 1 → 0 over (y_min .. y_min + slide_bottom_blocks).
    let bot_end = cfg.y_min + cfg.slide_bottom_blocks;
    let bot_f =
        ((bot_end - wy) as f32 / cfg.slide_bottom_blocks.max(1) as f32).clamp(0.0, 1.0);
    after_top + (cfg.slide_bottom_target - after_top) * bot_f
}

#[cfg(test)]
mod tests_anisotropy {
    use super::*;

    /// At y = h_target the density should be close to 0 (the surface).
    #[test]
    fn new_evaluator_density_at_surface_is_near_zero() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // Average over a small grid to wash out noise.
        let mut sum = 0.0_f32;
        let mut n = 0;
        for wx in 0..16 {
            for wz in 0..16 {
                sum += d.evaluate_v2(80.0, wx, 80, wz, &cfg.density);
                n += 1;
            }
        }
        let avg = sum / n as f32;
        assert!(
            avg.abs() < 1.5,
            "average density at surface should be near 0 (got {avg})"
        );
    }

    #[test]
    fn new_evaluator_density_well_below_is_strongly_positive() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        let v = d.evaluate_v2(80.0, 0, 30, 0, &cfg.density);
        assert!(v > 1.0, "density 50 below surface should be > 1, got {v}");
    }

    #[test]
    fn new_evaluator_density_well_above_is_negative() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        let v = d.evaluate_v2(80.0, 0, 130, 0, &cfg.density);
        assert!(v < -0.05, "density 50 above surface should be < -0.05, got {v}");
    }

    #[test]
    fn new_evaluator_density_at_world_top_pulled_toward_air() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        let v = d.evaluate_v2(80.0, 0, 140, 0, &cfg.density);
        assert!((v - (-0.078125)).abs() < 0.5, "at y_max density should be near slide_top_target, got {v}");
    }

    #[test]
    fn new_evaluator_density_at_world_bottom_pulled_toward_solid() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        let v = d.evaluate_v2(80.0, 0, -120, 0, &cfg.density);
        assert!((v - 0.1171875).abs() < 0.5, "at y_min density should be near slide_bottom_target, got {v}");
    }

    /// Anisotropic noise: a Y-step should change the noise value
    /// less than an equivalent XZ-step (vertical features are
    /// 2× taller than wide).
    #[test]
    fn base_3d_noise_y_scale_is_half_xz() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // Average step magnitude across multiple sample points to
        // wash out single-point noise idiosyncrasies.
        let mut dx_sum = 0.0_f32;
        let mut dy_sum = 0.0_f32;
        let mut n = 0;
        for wx in (0..32).step_by(4) {
            for wz in (0..32).step_by(4) {
                let dx = (d.evaluate_base_3d(0.0, wx, 0, wz, &cfg.density)
                    - d.evaluate_base_3d(0.0, wx + 8, 0, wz, &cfg.density))
                .abs();
                let dy = (d.evaluate_base_3d(0.0, wx, 0, wz, &cfg.density)
                    - d.evaluate_base_3d(0.0, wx, 8, wz, &cfg.density))
                .abs();
                dx_sum += dx;
                dy_sum += dy;
                n += 1;
            }
        }
        let dx_avg = dx_sum / n as f32;
        let dy_avg = dy_sum / n as f32;
        assert!(
            dy_avg < dx_avg,
            "y-step avg ({dy_avg}) should be smaller than x-step avg ({dx_avg})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> crate::worldgen::config::WorldgenConfig {
        crate::worldgen::config::WorldgenConfig::bundled_default().unwrap()
    }

    #[test]
    fn density_is_pure_in_seed_and_coord() {
        let d = DensityNoise::new(42);
        let cfg = test_cfg();
        let a = d.evaluate_v2(70.0, 100, 65, 200, &cfg.density);
        let b = d.evaluate_v2(70.0, 100, 65, 200, &cfg.density);
        assert_eq!(a, b);
    }

    #[test]
    fn density_below_target_mostly_solid() {
        let d = DensityNoise::new(42);
        let cfg = test_cfg();
        let mut solid = 0;
        let mut total = 0;
        // Sample voxels 4 blocks below h_target = 70: should be
        // mostly (≥80%) solid (positive density).
        for wx in (-200..200).step_by(7) {
            for wz in (-200..200).step_by(7) {
                total += 1;
                if d.evaluate_v2(70.0, wx, 66, wz, &cfg.density) > 0.0 {
                    solid += 1;
                }
            }
        }
        let frac = solid as f32 / total as f32;
        assert!(
            frac > 0.80,
            "4 blocks below h_target should be ≥80% solid; got {frac:.2}"
        );
    }

    #[test]
    fn density_above_target_mostly_air() {
        let d = DensityNoise::new(42);
        let cfg = test_cfg();
        let mut air = 0;
        let mut total = 0;
        // Sample voxels 4 blocks above h_target = 70: should be
        // mostly air.
        for wx in (-200..200).step_by(7) {
            for wz in (-200..200).step_by(7) {
                total += 1;
                if d.evaluate_v2(70.0, wx, 74, wz, &cfg.density) <= 0.0 {
                    air += 1;
                }
            }
        }
        let frac = air as f32 / total as f32;
        assert!(
            frac > 0.80,
            "4 blocks above h_target should be ≥80% air; got {frac:.2}"
        );
    }

    #[test]
    fn density_at_target_is_balanced() {
        let d = DensityNoise::new(42);
        let cfg = test_cfg();
        let mut solid = 0;
        let mut total = 0;
        for wx in (-200..200).step_by(7) {
            for wz in (-200..200).step_by(7) {
                total += 1;
                if d.evaluate_v2(70.0, wx, 70, wz, &cfg.density) > 0.0 {
                    solid += 1;
                }
            }
        }
        let frac = solid as f32 / total as f32;
        // Exactly at the target height, ~50% of voxels should be
        // solid (bias is zero; noise determines).
        assert!(
            (0.30..0.70).contains(&frac),
            "at h_target the solid fraction should be near 50%; got {frac:.2}"
        );
    }

    #[test]
    fn topmost_solid_within_band() {
        let d = DensityNoise::new(42);
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        // For a few sample columns, the topmost solid should be
        // within ±SURFACE_BAND of h_target.
        for (wx, wz) in [(0, 0), (50, 100), (-200, 150), (300, -200)] {
            let h = 70.0;
            let top = d.topmost_solid(h, wx, wz, 200, &cfg.density).unwrap();
            assert!(
                (h as i32 - SURFACE_BAND..=h as i32 + SURFACE_BAND).contains(&top),
                "topmost solid at ({wx},{wz}) was y={top}, expected in [{}..{}]",
                h as i32 - SURFACE_BAND,
                h as i32 + SURFACE_BAND
            );
        }
    }

    #[test]
    fn h_pre_is_deterministic() {
        let n = HeightmapNoise::new(42);
        let a = n.h_pre(42, 100.0, 200.0);
        let b = n.h_pre(42, 100.0, 200.0);
        assert_eq!(a, b);
    }

    #[test]
    fn h_pre_respects_cap() {
        let n = HeightmapNoise::new(42);
        for wx in (-2000..=2000).step_by(40) {
            for wz in (-2000..=2000).step_by(40) {
                let h = n.h_pre(42, wx as f32, wz as f32);
                assert!(
                    h <= MAX_TERRAIN_Y as f32,
                    "h_pre {h} above cap {MAX_TERRAIN_Y} at ({wx}, {wz})"
                );
                assert!(
                    h >= (CAVE_FLOOR_Y + 8) as f32,
                    "h_pre {h} below floor at ({wx}, {wz})"
                );
            }
        }
    }

    #[test]
    fn warped_fbm_breaks_axis_symmetry() {
        // With domain warp, h_pre(x, z) should differ from h_pre(z, x)
        // at most query points. Without warp, FBM is radially
        // symmetric so the two would be identical. We scan a generous
        // grid and assert at least 95% of points differ — leaves room
        // for the rare coincidence at axis-symmetric noise zeros.
        let n = HeightmapNoise::new(42);
        let mut differ = 0;
        let mut total = 0;
        for wx in (-500..=500).step_by(25) {
            for wz in (-500..=500).step_by(25) {
                if wx == wz {
                    continue;
                }
                let a = n.h_pre(42, wx as f32, wz as f32);
                let b = n.h_pre(42, wz as f32, wx as f32);
                total += 1;
                if (a - b).abs() > 0.5 {
                    differ += 1;
                }
            }
        }
        let frac = differ as f32 / total as f32;
        assert!(
            frac > 0.95,
            "expected >95% of swapped queries to differ; got {frac:.3}"
        );
    }

    #[test]
    fn slope_is_low_on_flat_oceanic_plate() {
        // Find an oceanic-plate interior column (t > 0.5 deep inside
        // the plate, kind = Oceanic). Slope should be modest there.
        let n = HeightmapNoise::new(42);
        let mut found = None;
        'outer: for wx in (-2000..2000).step_by(50) {
            for wz in (-2000..2000).step_by(50) {
                let look = crate::worldgen::plates::plate_at(42, wx, wz);
                if look.t > 0.6
                    && matches!(look.a.kind, crate::worldgen::plates::PlateKind::Oceanic)
                {
                    found = Some((wx, wz));
                    break 'outer;
                }
            }
        }
        let (wx, wz) = found.expect("no deep-oceanic column found in scan");
        let slope = n.slope_at(42, wx as f32, wz as f32);
        // Flat ocean basin should have nowhere near cliff-grade slope.
        assert!(
            slope < CLIFF_SLOPE_THRESH,
            "deep-oceanic slope {slope} unexpectedly above cliff threshold {CLIFF_SLOPE_THRESH}"
        );
    }

    #[test]
    fn slope_can_be_high_on_plate_boundary() {
        // Find a near-CC boundary column. Slope should be substantial
        // (ridge crest falls off perpendicular to the boundary).
        let n = HeightmapNoise::new(42);
        let mut high_slopes_seen = 0;
        for wx in (-2000..2000).step_by(20) {
            for wz in (-2000..2000).step_by(20) {
                let look = crate::worldgen::plates::plate_at(42, wx, wz);
                if look.t < 0.03
                    && matches!(look.a.kind, crate::worldgen::plates::PlateKind::Continental)
                    && matches!(look.b.kind, crate::worldgen::plates::PlateKind::Continental)
                {
                    let s = n.slope_at(42, wx as f32, wz as f32);
                    if s > CLIFF_SLOPE_THRESH {
                        high_slopes_seen += 1;
                    }
                    if high_slopes_seen >= 3 {
                        return;
                    }
                }
            }
        }
        panic!("expected ≥3 cliff-grade slopes on CC boundaries; saw {high_slopes_seen}");
    }
}
