//! Spline-driven heightmap (PR 3).
//!
//! Per the migration plan, the world's surface height is no longer
//! derived from plate-mosaic primitives (shelf_base + ridge_lift +
//! warped FBM). Instead, three 2D inputs feed nested cubic Hermite
//! splines whose output is the *offset* term in the density
//! composition; converting that offset to world Y gives the surface
//! height for hydrology, tree placement, and the `h_target` value
//! used by [`DensityNoise::evaluate_v2`].
//!
//! Inputs:
//!
//! * `continentalness` ∈ `[-1, 1]` — derived from the plate Voronoi
//!   distance field. Continental plates contribute `+plate_t`,
//!   oceanic plates `-plate_t`; clamped to `[-1, 1]`.
//! * `terrain_shape` ∈ `[-1, 1]` — a low-frequency 2D FBM
//!   (~2000-block period) plus a per-plate *roughness bias* that
//!   preserves "rocky vs gentle continent" identity.
//! * `ridges_pv` ∈ `[-1, 1]` — the high-frequency raw ridge FBM
//!   folded through `peaks_and_valleys` to give a peaks/valleys
//!   triangle wave.
//!
//! Outputs (per-column):
//!
//! * `offset` (from `ClimateConfig::offset_spline`) — additive to
//!   the y-gradient.
//! * `factor` (from `factor_spline`) — multiplier on the
//!   `(depth + jagged*noise) * factor` term.
//! * `jaggedness` (from `jaggedness_spline`) — amplitude of the
//!   per-voxel high-frequency rider applied near peaks.
//!
//! All three are sampled once per column and broadcast across the
//! voxel loop (PR 5 wraps this in `FlatCache2D` for cell-grid
//! evaluation; PR 3 just reads per-voxel).
//!
//! See `docs/book/content/part-3-region-build/3.3-heightmap.mdx` and
//! `docs/superpowers/specs/2026-05-19-worldgen-3d-design.md`.

use super::math::{offset_to_world_y, peaks_and_valleys, smooth_plate_contribution};
use crate::worldgen::config::{ClimateConfig, DensityConfig};
use crate::worldgen::plates::PlateLookup;
use crate::worldgen::tuning::*;
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

/// Bundle of noise fields needed for the spline pipeline. Built once
/// per [`crate::worldgen::Generator`] from a snapshot of
/// [`ClimateConfig`] — frequencies are baked at construction, so
/// hot-reloading `terrain_shape_period` / `ridges_period` requires a
/// generator restart. All other climate knobs (knot tables,
/// roughness bias range, amplitudes) hot-reload freely.
pub struct HeightmapNoise {
    /// Low-frequency 2D FBM (MC "erosion" equivalent). Combined with
    /// the per-plate roughness bias to produce `terrain_shape`.
    terrain_shape: Fbm<Simplex>,
    /// Mid-frequency 2D FBM whose raw output is folded through
    /// `peaks_and_valleys` to produce `ridges_pv`.
    ridges_raw: Fbm<Simplex>,
}

impl HeightmapNoise {
    /// Build the noise fields from a [`ClimateConfig`] snapshot. The
    /// frequencies are read at construction; later edits to those
    /// fields require a Generator rebuild.
    pub fn new(seed: u64, cfg: &ClimateConfig) -> Self {
        let terrain_shape = Fbm::<Simplex>::new(seed.wrapping_add(303) as u32)
            .set_octaves(3)
            .set_frequency(1.0 / cfg.terrain_shape_period as f64)
            .set_persistence(0.5);
        let ridges_raw = Fbm::<Simplex>::new(seed.wrapping_add(404) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / cfg.ridges_period as f64)
            .set_persistence(0.5);
        Self {
            terrain_shape,
            ridges_raw,
        }
    }

    /// Sample the climate triple `(continentalness, terrain_shape, ridges_pv)`
    /// at a column. The lookup also returns the plate lookup so callers
    /// (e.g. the cliff detector) can access plate identity without
    /// re-running `plate_at`.
    pub fn climate(
        &self,
        seed: u64,
        wx: f32,
        wz: f32,
        cfg: &ClimateConfig,
    ) -> (f32, f32, f32, PlateLookup) {
        let look = crate::worldgen::plates::plate_at(seed, wx as i32, wz as i32);
        let (cont, bias) = smooth_plate_contribution(seed, wx as i32, wz as i32, cfg);
        let shape_noise =
            (self.terrain_shape.get([wx as f64, wz as f64]) as f32) * cfg.terrain_shape_amplitude;
        let shape = (shape_noise + bias).clamp(-1.0, 1.0);
        let ridges = (self.ridges_raw.get([wx as f64, wz as f64]) as f32) * cfg.ridges_amplitude;
        let ridges_pv = peaks_and_valleys(ridges);
        (cont.clamp(-1.0, 1.0), shape, ridges_pv, look)
    }

    /// Pre-river heightmap (blocks, world-Y).
    ///
    /// The surface height is derived from the offset spline:
    /// ```text
    ///   (c, s, r)    = climate(wx, wz)
    ///   offset       = offset_spline(c, s, r)
    ///   surface_y    = offset_to_world_y(offset)
    /// ```
    /// Hydrology consumes this height to compute drainage. Tree
    /// placement and chunk fill use it as `h_target` for the
    /// per-voxel density composition.
    pub fn h_pre(
        &self,
        seed: u64,
        wx: f32,
        wz: f32,
        cfg: &ClimateConfig,
        density: &DensityConfig,
    ) -> f32 {
        let (c, s, r, _) = self.climate(seed, wx, wz, cfg);
        let offset = cfg.offset_spline.evaluate(c, s, r);
        offset_to_world_y(offset, density).clamp((CAVE_FLOOR_Y + 8) as f32, MAX_TERRAIN_Y as f32)
    }

    /// Magnitude of the horizontal gradient of `h_pre` at `(wx, wz)`,
    /// in blocks per block. Cheap finite-difference over an 8-block
    /// window (±4 each axis) — wide enough that small-scale FBM
    /// jitter doesn't register as a cliff.
    pub fn slope_at(
        &self,
        seed: u64,
        wx: f32,
        wz: f32,
        cfg: &ClimateConfig,
        density: &DensityConfig,
    ) -> f32 {
        let step = 4.0;
        let hxp = self.h_pre(seed, wx + step, wz, cfg, density);
        let hxn = self.h_pre(seed, wx - step, wz, cfg, density);
        let hzp = self.h_pre(seed, wx, wz + step, cfg, density);
        let hzn = self.h_pre(seed, wx, wz - step, cfg, density);
        let gx = (hxp - hxn).abs() / (2.0 * step);
        let gz = (hzp - hzn).abs() / (2.0 * step);
        gx.max(gz)
    }

    /// True if any of a wider stencil's samples around `(wx, wz)`
    /// dips below sea level. Coastal columns get the extended dirt
    /// cap and skip the cliff treatment so the shoreline doesn't
    /// render as a continuous stone wall.
    pub fn is_coastal(
        &self,
        seed: u64,
        wx: f32,
        wz: f32,
        cfg: &ClimateConfig,
        density: &DensityConfig,
    ) -> bool {
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
                if self.h_pre(seed, wx + dx, wz + dz, cfg, density) < sea {
                    return true;
                }
            }
        }
        false
    }

    /// True iff the column is a cliff face — slope above
    /// [`CLIFF_SLOPE_THRESH`] AND no stencil sample dips below sea
    /// level (coastal columns don't cliff so shorelines don't read
    /// as stone walls). The pre-PR-3 `CLIFF_MIN_HEIGHT` gate is
    /// gone: spline-driven height already places mountains far from
    /// the coast by design, so the gate is redundant.
    pub fn is_cliff(
        &self,
        seed: u64,
        wx: f32,
        wz: f32,
        cfg: &ClimateConfig,
        density: &DensityConfig,
    ) -> bool {
        let step = 4.0;
        let hxp = self.h_pre(seed, wx + step, wz, cfg, density);
        let hxn = self.h_pre(seed, wx - step, wz, cfg, density);
        let hzp = self.h_pre(seed, wx, wz + step, cfg, density);
        let hzn = self.h_pre(seed, wx, wz - step, cfg, density);
        let sea = SEA_LEVEL as f32;
        if hxp < sea || hxn < sea || hzp < sea || hzn < sea {
            return false;
        }
        let gx = (hxp - hxn).abs() / (2.0 * step);
        let gz = (hzp - hzn).abs() / (2.0 * step);
        gx.max(gz) > CLIFF_SLOPE_THRESH
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
    fn peaks_and_valleys_zero_at_unit() {
        // |w|=0 → -(|0 - 2/3| - 1/3)*3 = -(2/3 - 1/3)*3 = -1
        assert!((peaks_and_valleys(0.0) - (-1.0)).abs() < 1e-5);
        // |w|=1 → -(|1 - 2/3| - 1/3)*3 = -(1/3 - 1/3)*3 = 0
        // Wait: -(1/3 - 1/3)*3 = 0 not -1. Re-check:
        // |w|=1: |w| - 2/3 = 1/3, |1/3| - 1/3 = 0, *(-3) = 0. So at |w|=1, PV=0... no wait, MC says peaks at |w|=2/3 with value +1.
        // Let me recompute:
        // |w|=2/3: ||2/3-2/3|-1/3| * (-3) = |0-1/3|*(-3) = (1/3)*(-3) = -1. Hmm.
        // Actually MC's formula is `-(abs(abs(w) - 2/3) - 1/3) * 3`.
        // At |w|=2/3: -(0 - 1/3)*3 = 1. So peaks at |w|=2/3 with PV=+1. Correct.
        assert!((peaks_and_valleys(2.0 / 3.0) - 1.0).abs() < 1e-5);
        // At |w|=0: -(2/3 - 1/3)*3 = -1. Valley.
        assert!((peaks_and_valleys(0.0) - (-1.0)).abs() < 1e-5);
        // At |w|=1: -(1/3 - 1/3)*3 = 0. Mid.
        assert!((peaks_and_valleys(1.0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn signed_continentalness_polarity() {
        use crate::worldgen::density::math::signed_continentalness;
        use crate::worldgen::plates;
        use crate::worldgen::plates::PlateKind;
        // Plate at origin (seed 42): could be continental or oceanic.
        let look = plates::plate_at(42, 0, 0);
        let c = signed_continentalness(&look);
        match look.a.kind {
            PlateKind::Continental => assert!(c >= 0.0),
            PlateKind::Oceanic => assert!(c <= 0.0),
        }
    }

    #[test]
    fn offset_to_world_y_at_zero_offset() {
        let density = &test_cfg().density;
        // offset=0 → y at midpoint of [y_min, y_max].
        let y = offset_to_world_y(0.0, density);
        let mid = (density.y_min + density.y_max) as f32 * 0.5;
        assert!((y - mid).abs() < 1e-3);
    }

    #[test]
    fn offset_to_world_y_extremes() {
        let density = &test_cfg().density;
        // offset = +amp → top of world.
        let y_top = offset_to_world_y(density.y_gradient_amplitude, density);
        assert!((y_top - density.y_max as f32).abs() < 1e-3);
        // offset = -amp → bottom.
        let y_bot = offset_to_world_y(-density.y_gradient_amplitude, density);
        assert!((y_bot - density.y_min as f32).abs() < 1e-3);
    }

    #[test]
    fn h_pre_within_world_range() {
        let cfg = test_cfg();
        let n = HeightmapNoise::new(42, &cfg.climate);
        for wx in (-256..256).step_by(32) {
            for wz in (-256..256).step_by(32) {
                let h = n.h_pre(42, wx as f32, wz as f32, &cfg.climate, &cfg.density);
                assert!(
                    h >= (CAVE_FLOOR_Y + 8) as f32 && h <= MAX_TERRAIN_Y as f32,
                    "h_pre at ({wx},{wz}) = {h} out of clamp range",
                );
            }
        }
    }

    #[test]
    fn h_pre_is_deterministic() {
        let cfg = test_cfg();
        let n = HeightmapNoise::new(42, &cfg.climate);
        let a = n.h_pre(42, 100.0, 200.0, &cfg.climate, &cfg.density);
        let b = n.h_pre(42, 100.0, 200.0, &cfg.climate, &cfg.density);
        assert_eq!(a, b);
    }
}
