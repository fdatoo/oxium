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

use crate::worldgen::config::{ClimateConfig, DensityConfig};
use crate::worldgen::plates::{plate_at, Plate, PlateKind, PlateLookup};
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
        let look = plate_at(seed, wx as i32, wz as i32);
        let cont = signed_continentalness(&look);
        let shape_noise =
            (self.terrain_shape.get([wx as f64, wz as f64]) as f32) * cfg.terrain_shape_amplitude;
        // Blend roughness bias between the two nearest plates at same-kind
        // boundaries (C-C / O-O) so adjacent plates with different roughness
        // don't produce a step discontinuity in `shape`, which the offset
        // spline at c≈±1 turns into a vertical cliff. C-O boundaries are
        // left as a hard switch: signed_continentalness ramps c through 0
        // there, which is what makes the coastline read as a real shore.
        let bias = if look.a.kind == look.b.kind {
            let wa = 0.5 + 0.5 * look.t;
            plate_roughness_bias(&look.a, cfg) * wa
                + plate_roughness_bias(&look.b, cfg) * (1.0 - wa)
        } else {
            plate_roughness_bias(&look.a, cfg)
        };
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
        offset_to_world_y(offset, density)
            .clamp((CAVE_FLOOR_Y + 8) as f32, MAX_TERRAIN_Y as f32)
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
    /// for this column. `offset` comes from `offset_spline`, `factor`
    /// from `factor_spline`, `jagged` is the magnitude of the
    /// high-frequency rider near peaks.
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
    /// `J_noise` reuses the base 3D noise sampled at an isotropic
    /// (no y-scale) frequency tweak — for PR 3 we use the same
    /// `relief` field, sampled at the voxel position without
    /// modification. Above the surface (negative `depth`), the
    /// `quarter_negative` softening reduces the magnitude so the 3D
    /// noise can carve overhangs without piercing solid ground.
    /// Single-voxel density evaluation. Used by [`Self::topmost_solid`]
    /// for tree placement; the chunk-fill hot path uses
    /// [`crate::worldgen::density_graph::CellEvaluator`] which
    /// samples a 9×9×9 corner lattice and trilerps.
    pub fn evaluate(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        offset: f32,
        factor: f32,
        jagged: f32,
        density: &DensityConfig,
    ) -> f32 {
        let t = (wy - density.y_min) as f32 / (density.y_max - density.y_min) as f32;
        let y_gradient = density.y_gradient_amplitude * (1.0 - 2.0 * t);
        let depth = y_gradient + offset;
        let j_noise = if jagged.abs() > 1e-6 {
            self.evaluate_base_3d(wx, wy, wz, density)
        } else {
            0.0
        };
        let shaped_raw = (depth + jagged * j_noise) * factor;
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
    /// return the first voxel `wy` where `density > 0` (topmost
    /// solid). Used by tree placement to find anchor points under
    /// 3D-density surface jitter.
    pub fn topmost_solid(
        &self,
        h_target: f32,
        wx: i32,
        wz: i32,
        search_top: i32,
        offset: f32,
        factor: f32,
        jagged: f32,
        density: &DensityConfig,
    ) -> Option<i32> {
        let top = search_top.min(h_target as i32 + SURFACE_BAND);
        let bottom = (h_target as i32 - SURFACE_BAND).max(CAVE_FLOOR_Y);
        for wy in (bottom..=top).rev() {
            if self.evaluate(wx, wy, wz, offset, factor, jagged, density) > 0.0 {
                return Some(wy);
            }
        }
        Some(bottom)
    }
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
    let bot_f =
        ((bot_end - wy) as f32 / cfg.slide_bottom_blocks.max(1) as f32).clamp(0.0, 1.0);
    after_top + (cfg.slide_bottom_target - after_top) * bot_f
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
        use crate::worldgen::plates;
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

    #[test]
    fn density_at_surface_near_zero() {
        let cfg = test_cfg();
        let d = DensityNoise::new(42, &cfg.density);
        // At y = midpoint, with offset=0 / factor=4 / jagged=0,
        // the depth term is exactly 0 so density = base_3d_noise.
        let mid_y = ((cfg.density.y_min + cfg.density.y_max) / 2) as i32;
        let mut sum = 0.0_f32;
        for wx in 0..16 {
            for wz in 0..16 {
                sum += d.evaluate(wx, mid_y, wz, 0.0, 4.0, 0.0, &cfg.density);
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
        let v = d.evaluate(0, mid_y - 30, 0, 0.0, 4.0, 0.0, &cfg.density);
        assert!(v > 1.0, "density below surface should be > 1 (got {v})");
    }

    #[test]
    fn density_above_surface_is_air() {
        let cfg = test_cfg();
        let d = DensityNoise::new(42, &cfg.density);
        let mid_y = (cfg.density.y_min + cfg.density.y_max) / 2;
        let v = d.evaluate(0, mid_y + 30, 0, 0.0, 4.0, 0.0, &cfg.density);
        assert!(v < 0.0, "density above surface should be < 0 (got {v})");
    }

    #[test]
    fn density_at_world_top_pulled_to_air() {
        let cfg = test_cfg();
        let d = DensityNoise::new(42, &cfg.density);
        let v = d.evaluate(0, cfg.density.y_max, 0, 0.0, 4.0, 0.0, &cfg.density);
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
        let v = d.evaluate(0, cfg.density.y_min, 0, 0.0, 4.0, 0.0, &cfg.density);
        assert!(
            (v - cfg.density.slide_bottom_target).abs() < 0.5,
            "at y_min density should be near slide_bottom_target ({}), got {v}",
            cfg.density.slide_bottom_target,
        );
    }
}
