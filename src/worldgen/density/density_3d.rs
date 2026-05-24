//! Per-voxel 3D density evaluator.
//!
//! [`DensityNoise`] combines a height-based y-gradient with 3D FBM noise
//! to produce a scalar density at each voxel. A voxel is **solid iff
//! density > 0**.
//!
//! ## The composition formula (MC `sloped_cheese`)
//!
//! ```text
//!   depth     = y_gradient(wy) + offset     // offset from climate spline
//!   shaped    = (depth + jagged * J_noise) * factor
//!   shaped'   = quarter_negative(shaped)    // soften overhangs above surface
//!   density   = composition_scale * shaped' + base_3d_noise(anisotropic)
//!   density   = slide(density, wy)          // clamp top to air, bottom to solid
//! ```
//!
//! The three spline scalars (`offset`, `factor`, `jagged`) are sampled once
//! per column and bundled into [`DensityComposition`] so they travel as a
//! unit to every voxel evaluation in that column.
//!
//! ## Hot path
//!
//! Per-voxel evaluation via [`DensityNoise::evaluate`] is used only for
//! tree placement (`topmost_solid`). The chunk-fill hot path uses
//! [`super::cell_evaluator::CellEvaluator`], which samples a 9³ corner
//! lattice and trilinearly interpolates — ~134× fewer evaluations per chunk
//! at the cost of a smooth-density approximation near the iso-surface.
//!
//! See `docs/book/content/part-3-region-build/3.3-heightmap.mdx` and
//! `docs/superpowers/specs/2026-05-19-worldgen-3d-design.md`.

use super::math::slide;
use crate::worldgen::config::DensityConfig;
use crate::worldgen::tuning::{CAVE_FLOOR_Y, SURFACE_BAND};
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

/// Density-composition scalars derived from the climate splines at one column.
///
/// These three values are sampled once per column from [`crate::worldgen::config::ClimateConfig`]'s
/// nested splines and broadcast across the entire vertical voxel loop:
///
/// - `offset` shifts the height-based y-gradient (positive → surface higher,
///   negative → surface lower). Comes from `ClimateConfig::offset_spline`.
/// - `factor` scales the shaped density term — controls how quickly density
///   rises with depth and how pronounced overhangs are. Comes from
///   `ClimateConfig::factor_spline`.
/// - `jagged` blends 3D FBM noise into the density near the surface,
///   producing overhangs and ledges. `0.0` = smooth surface; higher values
///   = more 3D structure. Comes from `ClimateConfig::jaggedness_spline`.
///
/// All three always travel together from the climate-sample site to
/// [`DensityNoise::evaluate`] / [`DensityNoise::topmost_solid`], so
/// grouping them avoids repeating the triplet at every call site.
///
/// `DensityComposition` is `Copy` (three `f32`s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityComposition {
    /// Additive height-offset term from the climate offset spline.
    pub offset: f32,
    /// Multiplicative scale on the shaped density term.
    pub factor: f32,
    /// Amplitude of the per-voxel 3D FBM rider near the surface.
    pub jagged: f32,
}

/// 3D density evaluator: combines the column's y-gradient (positive
/// below the surface, negative above) with the climate-spline outputs
/// to produce a per-voxel density value. Solid iff `> 0`.
pub struct DensityNoise {
    relief: Fbm<Simplex>,
}

impl DensityNoise {
    /// Build the per-voxel base 3D noise field. Frequency comes from
    /// `DensityConfig::base_3d_period`; this is baked at construction,
    /// so hot-reloading the period requires a Generator restart.
    pub fn new(seed: u64, density: &DensityConfig) -> Self {
        let relief = Fbm::<Simplex>::new(seed.wrapping_add(701) as u32)
            .set_octaves(3)
            .set_frequency(1.0 / density.base_3d_period as f64)
            .set_persistence(0.5);
        Self { relief }
    }

    /// Sample the anisotropic base 3D noise at world-space
    /// `(wx, wy, wz)`. Y is scaled by `cfg.base_3d_y_scale` (< 1 →
    /// vertical features taller than wide; MC's BlendedNoise uses 0.5).
    pub fn evaluate_base_3d(&self, wx: i32, wy: i32, wz: i32, cfg: &DensityConfig) -> f32 {
        let scaled_y = wy as f64 * cfg.base_3d_y_scale as f64;
        self.relief.get([wx as f64, scaled_y, wz as f64]) as f32 * cfg.base_3d_amplitude
    }

    /// Evaluate density at `(wx, wy, wz)` using the spline outputs
    /// for this column.
    ///
    /// Composition (matches MC's `sloped_cheese`):
    /// ```text
    ///   depth     = y_gradient(y) + offset
    ///   shaped    = (depth + jagged * J_noise) * factor
    ///   shaped'   = quarter_negative(shaped)    // soften above surface
    ///   density   = composition_scale * shaped' + base_3d_noise(anisotropic)
    ///   density   = slide(density, y)            // top→air, bottom→solid
    /// ```
    ///
    /// `J_noise` reuses the base 3D noise sampled at the voxel position
    /// without y-scale. Above the surface (negative `depth`), the
    /// `quarter_negative` softening reduces magnitude so 3D noise can carve
    /// overhangs without piercing solid ground.
    ///
    /// Single-voxel evaluation. Used by [`Self::topmost_solid`] for tree
    /// placement; the chunk-fill hot path uses
    /// [`super::cell_evaluator::CellEvaluator`] which samples a 9×9×9
    /// corner lattice and trilerps.
    pub fn evaluate(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        comp: DensityComposition,
        density: &DensityConfig,
    ) -> f32 {
        let t = (wy - density.y_min) as f32 / (density.y_max - density.y_min) as f32;
        let y_gradient = density.y_gradient_amplitude * (1.0 - 2.0 * t);
        let depth = y_gradient + comp.offset;
        let j_noise = if comp.jagged.abs() > 1e-6 {
            self.evaluate_base_3d(wx, wy, wz, density)
        } else {
            0.0
        };
        let shaped_raw = (depth + comp.jagged * j_noise) * comp.factor;
        let shaped = if shaped_raw > 0.0 {
            shaped_raw
        } else {
            shaped_raw * density.above_surface_softening
        };
        let base_3d = self.evaluate_base_3d(wx, wy, wz, density);
        let pre_slide = density.composition_scale * shaped + base_3d;
        slide(pre_slide, wy, density)
    }

    /// Walk `(wx, wz)` top-down through the density function and
    /// return the first voxel `wy` where `density > 0` (topmost solid).
    ///
    /// Used by tree placement to find anchor points under 3D-density
    /// surface jitter that displaced the surface from the flat `h_target`.
    pub fn topmost_solid(
        &self,
        h_target: f32,
        wx: i32,
        wz: i32,
        search_top: i32,
        comp: DensityComposition,
        density: &DensityConfig,
    ) -> Option<i32> {
        let top = search_top.min(h_target as i32 + SURFACE_BAND);
        let bottom = (h_target as i32 - SURFACE_BAND).max(CAVE_FLOOR_Y);
        for wy in (bottom..=top).rev() {
            if self.evaluate(wx, wy, wz, comp, density) > 0.0 {
                return Some(wy);
            }
        }
        Some(bottom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::config::WorldgenConfig;

    fn test_cfg() -> WorldgenConfig {
        WorldgenConfig::bundled_default().unwrap()
    }

    #[test]
    fn density_at_surface_near_zero() {
        let cfg = test_cfg();
        let d = DensityNoise::new(42, &cfg.density);
        // At y = midpoint, with offset=0 / factor=4 / jagged=0,
        // the depth term is exactly 0 so density = base_3d_noise.
        let mid_y = (cfg.density.y_min + cfg.density.y_max) / 2;
        let mut sum = 0.0_f32;
        for wx in 0..16 {
            for wz in 0..16 {
                sum += d.evaluate(
                    wx,
                    mid_y,
                    wz,
                    DensityComposition {
                        offset: 0.0,
                        factor: 4.0,
                        jagged: 0.0,
                    },
                    &cfg.density,
                );
            }
        }
        let avg = sum / 256.0;
        assert!(
            avg.abs() < 2.0,
            "average density at midpoint with offset=0 should be ~0 (got {avg})",
        );
    }

    #[test]
    fn density_below_surface_is_solid() {
        let cfg = test_cfg();
        let d = DensityNoise::new(42, &cfg.density);
        // 30 blocks below midpoint with offset=0 → depth strongly positive → solid.
        let mid_y = (cfg.density.y_min + cfg.density.y_max) / 2;
        let v = d.evaluate(
            0,
            mid_y - 30,
            0,
            DensityComposition {
                offset: 0.0,
                factor: 4.0,
                jagged: 0.0,
            },
            &cfg.density,
        );
        assert!(v > 1.0, "density below surface should be > 1 (got {v})");
    }

    #[test]
    fn density_above_surface_is_air() {
        let cfg = test_cfg();
        let d = DensityNoise::new(42, &cfg.density);
        let mid_y = (cfg.density.y_min + cfg.density.y_max) / 2;
        let v = d.evaluate(
            0,
            mid_y + 30,
            0,
            DensityComposition {
                offset: 0.0,
                factor: 4.0,
                jagged: 0.0,
            },
            &cfg.density,
        );
        assert!(v < 0.0, "density above surface should be < 0 (got {v})");
    }

    #[test]
    fn density_at_world_top_pulled_to_air() {
        let cfg = test_cfg();
        let d = DensityNoise::new(42, &cfg.density);
        let v = d.evaluate(
            0,
            cfg.density.y_max,
            0,
            DensityComposition {
                offset: 0.0,
                factor: 4.0,
                jagged: 0.0,
            },
            &cfg.density,
        );
        assert!(
            (v - cfg.density.slide_top_target).abs() < 0.5,
            "at y_max density should be near slide_top_target ({}), got {v}",
            cfg.density.slide_top_target,
        );
    }

    #[test]
    fn density_at_world_bottom_pulled_to_solid() {
        let cfg = test_cfg();
        let d = DensityNoise::new(42, &cfg.density);
        let v = d.evaluate(
            0,
            cfg.density.y_min,
            0,
            DensityComposition {
                offset: 0.0,
                factor: 4.0,
                jagged: 0.0,
            },
            &cfg.density,
        );
        assert!(
            (v - cfg.density.slide_bottom_target).abs() < 0.5,
            "at y_min density should be near slide_bottom_target ({}), got {v}",
            cfg.density.slide_bottom_target,
        );
    }
}
