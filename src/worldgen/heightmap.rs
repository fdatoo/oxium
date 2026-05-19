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

    /// True if any of the ±4-block stencil samples around `(wx, wz)`
    /// dips below sea level. Used by the subsurface block selector
    /// to extend the dirt cap of coastal columns down to sea level
    /// so their water-facing sides don't reveal the underlying
    /// stone bedrock.
    pub fn is_coastal(&self, seed: u64, wx: f32, wz: f32) -> bool {
        let step = 4.0;
        let sea = SEA_LEVEL as f32;
        self.h_pre(seed, wx + step, wz) < sea
            || self.h_pre(seed, wx - step, wz) < sea
            || self.h_pre(seed, wx, wz + step) < sea
            || self.h_pre(seed, wx, wz - step) < sea
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

#[cfg(test)]
mod tests {
    use super::*;

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
