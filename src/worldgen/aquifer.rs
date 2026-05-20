//! MC-1.18+ style aquifer system (PR 7).
//!
//! An *aquifer* is a region of the world where caves and rock can be
//! flooded with a fluid (water or lava). Without this, every cave
//! carve produces a dry void; with it, the caves you walk into may be
//! half-submerged, dotted with lava lakes, or even partially solid
//! where the aquifer pressure isn't strong enough to break through
//! the rock.
//!
//! ### Cells
//!
//! World space is partitioned into rectangular cells of size
//! [`AQUIFER_CELL_X`] × [`AQUIFER_CELL_Y`] × [`AQUIFER_CELL_Z`]
//! (16 × 12 × 16). Each cell carries a deterministic
//! [`AquiferCell`]: a jittered center point, a fluid kind
//! (Water/Lava), and a `y_top` — the world Y at which the cell's
//! fluid surface sits.
//!
//! Cell properties are derived from `hash::mix` of `(seed,
//! cell_x, cell_y, cell_z, salt)`, so a cell's value is byte-stable
//! across regenerations. No persistent storage is required.
//!
//! ### Substance lookup
//!
//! For an arbitrary `(wx, wy, wz)`, [`AquiferSystem::compute`] walks
//! the 27-cell neighbourhood of the cell containing the point and
//! finds the three nearest by squared distance to their jittered
//! centers. It then:
//!
//! * Looks at the **closest** cell to decide whether the point is
//!   *inside* a fluid column (`wy ≤ y_top`).
//! * If unanimous across the 3 nearest (same fluid, same
//!   above/below decision), the point gets the cell's fluid kind.
//! * If not unanimous, the point's *barrier pressure* (an
//!   asymmetric function of `wy − y_top` plus a 3D noise sample)
//!   decides whether the aquifer wins (fluid) or the rock wins
//!   (solid/air).
//!
//! The exposed contract is [`AquiferSystem::substance`], which
//! takes the *density* the caller already computed and returns
//! either a [`Substance::Block`] (the caller must place the named
//! block) or [`Substance::Density`] (the caller should keep the
//! pure density decision: positive ⇒ Stone, non-positive ⇒ Air).
//!
//! ### Why this is asymmetric
//!
//! Real water aquifers extend *down* indefinitely (water sinks) but
//! cut off sharply *above* the water table (water doesn't pile up).
//! [`asymmetric_pressure`] uses a steep quadratic falloff for
//! `gap > 0` (above) and a gentle linear taper for `gap ≤ 0`
//! (below). Lava is similar but biased to stay near its source
//! pocket — its `top_falloff` is sharper than water.

use crate::voxel::block::Block;
use crate::worldgen::hash;
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};
use serde::{Deserialize, Serialize};

/// Cell width along X (blocks). Smaller cells → finer-grained
/// aquifer mosaic; 16 matches MC's `Aquifer.SIZE_IN_BLOCKS`.
pub const AQUIFER_CELL_X: i32 = 16;
/// Cell height along Y (blocks). Smaller than X/Z because vertical
/// gradients (water table changes per stratum) are sharper than
/// horizontal.
pub const AQUIFER_CELL_Y: i32 = 12;
/// Cell width along Z (blocks).
pub const AQUIFER_CELL_Z: i32 = 16;

/// Margin (blocks) kept clear of the cell faces when jittering the
/// cell center. Without a margin the center can land right on a face
/// and two adjacent cells' centers may coincide, undermining the
/// nearest-cell distinction.
const CELL_CENTER_MARGIN: i32 = 2;

/// World Y below which lava aquifers can replace water aquifers.
/// Above this, every cell is water.
pub const LAVA_BAND_TOP_Y: i32 = -32;

/// Probability that a cell whose center sits in the lava band rolls
/// as lava (the rest are water). Set just under 0.5 so lava pockets
/// feel like a feature, not the default.
const LAVA_PROB_IN_BAND: f32 = 0.40;

/// Tunable parameters held externally (so the visualizer can rebind
/// without rebuilding the system).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AquiferConfig {
    /// World-space sea level — used as the upper bound on water
    /// aquifer `y_top` values (no aquifer surface above the ocean).
    pub sea_level: i32,
    /// Vertical range over which the per-cell `y_top` jitters from
    /// its band default. Larger → more variation in water-table
    /// height between adjacent cells (more dramatic perched
    /// aquifers; can produce dry/wet ribbon transitions).
    pub y_top_jitter: i32,
    /// Strength of the 3D barrier noise (modulates the pressure
    /// function near the fluid surface). 0 → fluid surfaces sit on
    /// a rigid plane; ~0.5 → wavy boundaries; >1.0 → can flip
    /// blocks on the wrong side of the surface.
    pub barrier_strength: f32,
    /// Period (blocks) of the 3D barrier noise. Smaller → high-
    /// frequency choppy boundaries; larger → smooth long-wave
    /// undulation along the fluid surface.
    pub barrier_period: f32,
    /// Multiplier on the asymmetric-pressure falloff above the
    /// fluid surface. Larger → the aquifer cuts off harder above
    /// `y_top` (less water leaking into elevated caves).
    pub top_falloff: f32,
    /// Multiplier on the asymmetric-pressure falloff below the
    /// fluid surface. Larger → more graceful taper into the depth
    /// (fluid persists further down); smaller → tighter pools.
    pub bottom_falloff: f32,
}

impl Default for AquiferConfig {
    fn default() -> Self {
        Self {
            sea_level: 62,
            y_top_jitter: 6,
            barrier_strength: 0.5,
            barrier_period: 24.0,
            top_falloff: 0.06,
            bottom_falloff: 0.012,
        }
    }
}

/// Per-cell fluid descriptor. Computed deterministically from
/// `(seed, cell_x, cell_y, cell_z)` — no caching required, the
/// reproducible hash means the same cell always returns the same
/// value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AquiferCell {
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
    /// Jittered world-space center used for the nearest-cell
    /// distance metric.
    pub center_x: i32,
    pub center_y: i32,
    pub center_z: i32,
    /// World Y of the fluid surface in this cell. Points strictly
    /// above `y_top` are dry (no fluid contribution); points at or
    /// below contribute fluid pressure.
    pub y_top: i32,
    /// Either [`Block::Water`] or [`Block::Lava`]. The aquifer
    /// system only ever picks these two — any other block is
    /// caller-supplied (the underlying rock).
    pub fluid: Block,
}

/// What the aquifer system says about a particular `(wx, wy, wz)`
/// voxel given the caller's already-computed density value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Substance {
    /// Aquifer is silent here — keep the density-driven decision
    /// (positive ⇒ Stone, non-positive ⇒ Air).
    Density,
    /// Aquifer places this exact block (Water/Lava/Air/Stone).
    Block(Block),
}

/// The aquifer system. Holds the seed (for per-cell hashes), the
/// config, and the FBM noise used by the barrier function.
pub struct AquiferSystem {
    seed: u64,
    pub cfg: AquiferConfig,
    barrier: Fbm<Simplex>,
}

impl AquiferSystem {
    pub fn new(seed: u64, cfg: AquiferConfig) -> Self {
        // 3D barrier noise: shapes the wavy fluid-surface boundary.
        // Use a higher-frequency, fewer-octaves FBM so the boundary
        // feels chiseled rather than blobby.
        let barrier = Fbm::<Simplex>::new(seed.wrapping_add(0xA0F1_BEEF) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / cfg.barrier_period as f64)
            .set_persistence(0.5);
        Self {
            seed,
            cfg,
            barrier,
        }
    }

    /// Resolve the [`AquiferCell`] at integer cell coords. Pure in
    /// `(seed, cx, cy, cz)`.
    pub fn cell_at(&self, cx: i32, cy: i32, cz: i32) -> AquiferCell {
        let jx = (hash::mix(self.seed, &[cx, cy, cz, 0]) as u32)
            % ((AQUIFER_CELL_X - 2 * CELL_CENTER_MARGIN) as u32);
        let jy = (hash::mix(self.seed, &[cx, cy, cz, 1]) as u32)
            % ((AQUIFER_CELL_Y - 2 * CELL_CENTER_MARGIN) as u32);
        let jz = (hash::mix(self.seed, &[cx, cy, cz, 2]) as u32)
            % ((AQUIFER_CELL_Z - 2 * CELL_CENTER_MARGIN) as u32);
        let center_x = cx * AQUIFER_CELL_X + CELL_CENTER_MARGIN + jx as i32;
        let center_y = cy * AQUIFER_CELL_Y + CELL_CENTER_MARGIN + jy as i32;
        let center_z = cz * AQUIFER_CELL_Z + CELL_CENTER_MARGIN + jz as i32;

        // DRY-CELL GATE: only a fraction of cells have an aquifer at
        // all. The rest are sentinel-dry — their `y_top` is so far
        // below any reasonable query that no voxel reads as "in
        // fluid". This makes aquifers a *feature* (occasional flooded
        // caves, lava pools) rather than a default state (every cave
        // is full of water).
        //
        // `dry_roll`: per-cell uniform [0, 1).
        // The kept fraction depends on depth — shallow aquifers are
        // rare because they would conflict with the surface lake /
        // ocean flood; deep aquifers are slightly more common.
        let dry_roll = hash::mix_unit(self.seed, &[cx, cy, cz, 5]);
        // All bands: no aquifers for now. Re-enable selectively
        // (e.g. ~5% lava pools deep underground) once cave shaping
        // is settled — currently the aquifer paired with the
        // bowl-cave problem made it impossible to see what was
        // wrong with cave shape.
        let keep_chance: f32 = 0.0;
        if dry_roll >= keep_chance {
            return AquiferCell {
                cx,
                cy,
                cz,
                center_x,
                center_y,
                center_z,
                y_top: i32::MIN / 2,
                fluid: Block::Water,
            };
        }

        // y_top: nominal band depends on the cell's center Y.
        //   * Near/under sea level: water aquifer well BELOW the cell
        //     (only the bottom few voxels of a cave there see water).
        //   * Deeper: water table near the cell's bottom so each
        //     pocket is a shallow puddle, not a column-filler.
        let band_nominal = if center_y >= LAVA_BAND_TOP_Y {
            // Shallow band: y_top a couple of blocks below the cell
            // center — only the lower portion of a cave catches it.
            center_y - 2
        } else {
            // Deep band: y_top near the cell's bottom.
            cy * AQUIFER_CELL_Y + AQUIFER_CELL_Y / 4
        };
        let jitter_unit = hash::mix_unit(self.seed, &[cx, cy, cz, 3]);
        let jitter = ((jitter_unit - 0.5) * 2.0 * self.cfg.y_top_jitter as f32) as i32;
        let y_top = band_nominal.saturating_add(jitter);

        // Fluid kind: lava only allowed below LAVA_BAND_TOP_Y.
        let fluid_roll = hash::mix_unit(self.seed, &[cx, cy, cz, 4]);
        let fluid = if center_y < LAVA_BAND_TOP_Y && fluid_roll < LAVA_PROB_IN_BAND {
            Block::Lava
        } else {
            Block::Water
        };

        AquiferCell {
            cx,
            cy,
            cz,
            center_x,
            center_y,
            center_z,
            y_top,
            fluid,
        }
    }

    /// Return the primary [`AquiferCell`] for a surface column at
    /// `(wx, wz)`. Uses sea-level as the representative Y so the
    /// returned cell is the one that governs the near-surface aquifer
    /// at this column. Read-only; used by the visualizer probe panel.
    pub fn cell_for_column(&self, wx: i32, wz: i32) -> AquiferCell {
        let wy = self.cfg.sea_level;
        let nearest = self.three_nearest(wx, wy, wz);
        nearest[0]
    }

    /// Walk the 27-cell neighbourhood and return the three cells
    /// whose jittered centers are closest to `(wx, wy, wz)`. Sorted
    /// by ascending squared distance.
    fn three_nearest(&self, wx: i32, wy: i32, wz: i32) -> [AquiferCell; 3] {
        let cx = wx.div_euclid(AQUIFER_CELL_X);
        let cy = wy.div_euclid(AQUIFER_CELL_Y);
        let cz = wz.div_euclid(AQUIFER_CELL_Z);
        let mut best: [(i64, Option<AquiferCell>); 3] = [(i64::MAX, None); 3];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cell = self.cell_at(cx + dx, cy + dy, cz + dz);
                    let dxv = (cell.center_x - wx) as i64;
                    let dyv = (cell.center_y - wy) as i64;
                    let dzv = (cell.center_z - wz) as i64;
                    let d2 = dxv * dxv + dyv * dyv + dzv * dzv;
                    insert_best3(&mut best, d2, cell);
                }
            }
        }
        // Defensive: the 27-cell sweep always populates at least 3 entries.
        [
            best[0].1.expect("three_nearest: slot 0 must be populated"),
            best[1].1.expect("three_nearest: slot 1 must be populated"),
            best[2].1.expect("three_nearest: slot 2 must be populated"),
        ]
    }

    /// 3D barrier noise sample in `[-strength, +strength]`. The
    /// noise gives the fluid surface a chiseled, ribbon-like
    /// boundary instead of a flat plane.
    fn barrier_noise(&self, wx: i32, wy: i32, wz: i32) -> f32 {
        let v = self.barrier.get([wx as f64, wy as f64, wz as f64]) as f32;
        v * self.cfg.barrier_strength
    }

    /// Decide what block goes at `(wx, wy, wz)` given the density
    /// the caller already evaluated. `density > 0.0` is the
    /// caller's "solid" decision; `density ≤ 0.0` is "air".
    ///
    /// Returns:
    /// * `Substance::Density` — aquifer is silent; caller keeps its
    ///   own decision.
    /// * `Substance::Block(b)` — aquifer overrides; place `b`.
    pub fn substance(&self, wx: i32, wy: i32, wz: i32, density: f32) -> Substance {
        let nearest = self.three_nearest(wx, wy, wz);
        let a = nearest[0];
        let in_fluid_a = wy <= a.y_top;
        let want_solid = density > 0.0;

        if want_solid {
            if !in_fluid_a {
                return Substance::Density;
            }
            // Aquifer wants to flood — check if its pressure can
            // overcome the rock. Both the asymmetric falloff and the
            // 3D barrier are summed before subtracting from density.
            let pressure = asymmetric_pressure(
                wy,
                a.y_top,
                self.cfg.top_falloff,
                self.cfg.bottom_falloff,
            ) + self.barrier_noise(wx, wy, wz);
            if density - pressure > 0.0 {
                Substance::Density // rock holds; solid
            } else {
                Substance::Block(a.fluid)
            }
        } else {
            if !in_fluid_a {
                return Substance::Density; // would be air
            }
            // Density says air; we're in a cave at or below the
            // water table. Use 3-nearest unanimity to decide.
            let b = nearest[1];
            let c = nearest[2];
            let agree_b = b.fluid == a.fluid && wy <= b.y_top;
            let agree_c = c.fluid == a.fluid && wy <= c.y_top;
            if agree_b && agree_c {
                Substance::Block(a.fluid)
            } else {
                // Disagreement near the boundary — barrier noise
                // breaks the tie. If pressure stays positive the
                // fluid wins; otherwise leave it as air (this is
                // how dry cave pockets appear inside otherwise
                // wet zones).
                let pressure = asymmetric_pressure(
                    wy,
                    a.y_top,
                    self.cfg.top_falloff,
                    self.cfg.bottom_falloff,
                ) + self.barrier_noise(wx, wy, wz);
                if pressure > 0.0 {
                    Substance::Block(a.fluid)
                } else {
                    Substance::Density // air pocket
                }
            }
        }
    }
}

/// Asymmetric pressure as a function of vertical gap from the fluid
/// surface. Returns a value used to oppose the rock's density:
///
/// * `wy > y_top`: large negative (no fluid above the surface). Curve
///   is quadratic so it cuts off quickly — water doesn't sit above
///   its own table.
/// * `wy ≤ y_top`: positive (fluid wants to be here). Curve is
///   linear and gentle — water persists far below the table.
///
/// The two slopes are tuned via `top_falloff` and `bottom_falloff`
/// in [`AquiferConfig`]; the defaults give a sharp upper cutoff and
/// a slow downward taper.
pub fn asymmetric_pressure(wy: i32, y_top: i32, top_falloff: f32, bottom_falloff: f32) -> f32 {
    let gap = (wy - y_top) as f32;
    if gap > 0.0 {
        // Above table — quadratic negative penalty.
        -top_falloff * gap * gap
    } else {
        // Below table — linear positive pressure, asymptotically
        // saturating at +1.0 so deep water never flips solid by
        // overflow.
        (1.0 - bottom_falloff * (-gap)).max(-1.0)
    }
}

/// Insert `(d2, cell)` into the top-3-min-distance slot array if it
/// belongs there. Slots are sorted ascending after each insertion.
fn insert_best3(best: &mut [(i64, Option<AquiferCell>); 3], d2: i64, cell: AquiferCell) {
    if d2 < best[0].0 {
        best[2] = best[1];
        best[1] = best[0];
        best[0] = (d2, Some(cell));
    } else if d2 < best[1].0 {
        best[2] = best[1];
        best[1] = (d2, Some(cell));
    } else if d2 < best[2].0 {
        best[2] = (d2, Some(cell));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys() -> AquiferSystem {
        AquiferSystem::new(42, AquiferConfig::default())
    }

    #[test]
    fn cell_at_is_deterministic() {
        let s = sys();
        let a = s.cell_at(0, 0, 0);
        let b = s.cell_at(0, 0, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn cell_center_inside_cell_bounds() {
        let s = sys();
        for cx in -2..=2 {
            for cy in -4..=4 {
                for cz in -2..=2 {
                    let c = s.cell_at(cx, cy, cz);
                    assert!(c.center_x >= cx * AQUIFER_CELL_X + CELL_CENTER_MARGIN);
                    assert!(c.center_x < (cx + 1) * AQUIFER_CELL_X - CELL_CENTER_MARGIN + 1);
                    assert!(c.center_y >= cy * AQUIFER_CELL_Y + CELL_CENTER_MARGIN);
                    assert!(c.center_y < (cy + 1) * AQUIFER_CELL_Y - CELL_CENTER_MARGIN + 1);
                    assert!(c.center_z >= cz * AQUIFER_CELL_Z + CELL_CENTER_MARGIN);
                    assert!(c.center_z < (cz + 1) * AQUIFER_CELL_Z - CELL_CENTER_MARGIN + 1);
                }
            }
        }
    }

    #[test]
    fn high_altitude_cells_are_dry() {
        // Cells whose center sits above sea level have y_top
        // pegged at the sentinel so no point above them reads as
        // "in fluid".
        let s = sys();
        for cx in 0..4 {
            for cz in 0..4 {
                let c = s.cell_at(cx, 12, cz);
                // 12 * 12 = 144 > sea_level (62) → sentinel
                assert!(c.y_top < 0, "cell {cx},{cz} y_top={} >= 0", c.y_top);
            }
        }
    }

    #[test]
    fn shallow_cells_pick_water_only() {
        // Cells above the lava band are always water.
        let s = sys();
        let c = s.cell_at(0, 4, 0);
        // 4 * 12 = 48 > LAVA_BAND_TOP_Y (-32) → water
        assert_eq!(c.fluid, Block::Water);
    }

    #[test]
    #[ignore = "all-cells-dry temp override; re-enable when aquifers are tuned back on"]
    fn deep_cells_can_be_lava() {
        // Stochastic, but over a strip of deep cells we should see
        // at least one lava cell.
        let s = sys();
        let mut any_lava = false;
        for cx in 0..16 {
            for cz in 0..16 {
                let c = s.cell_at(cx, -10, cz);
                if c.fluid == Block::Lava {
                    any_lava = true;
                    break;
                }
            }
        }
        assert!(any_lava, "no lava cells found across deep strip");
    }

    #[test]
    fn three_nearest_returns_ascending_distance() {
        let s = sys();
        let near = s.three_nearest(7, 5, 9);
        let d = |c: AquiferCell| {
            let dx = (c.center_x - 7) as i64;
            let dy = (c.center_y - 5) as i64;
            let dz = (c.center_z - 9) as i64;
            dx * dx + dy * dy + dz * dz
        };
        let d0 = d(near[0]);
        let d1 = d(near[1]);
        let d2 = d(near[2]);
        assert!(d0 <= d1, "d0={} > d1={}", d0, d1);
        assert!(d1 <= d2, "d1={} > d2={}", d1, d2);
    }

    #[test]
    fn asymmetric_pressure_above_is_negative() {
        // Above y_top → quadratic cutoff.
        let p = asymmetric_pressure(80, 70, 0.06, 0.012);
        assert!(p < 0.0, "above-table pressure should be negative, got {p}");
    }

    #[test]
    fn asymmetric_pressure_below_is_positive() {
        // Just below y_top → near +1.
        let p = asymmetric_pressure(69, 70, 0.06, 0.012);
        assert!(p > 0.9, "just-below pressure should be ~1, got {p}");
    }

    #[test]
    fn asymmetric_pressure_far_below_saturates() {
        // Far below → still positive but reduced; bounded at -1.0.
        let p = asymmetric_pressure(-100, 50, 0.06, 0.012);
        assert!(p >= -1.0);
    }

    #[test]
    fn substance_solid_far_above_table_keeps_density() {
        let s = sys();
        // High in the air, density says solid.
        let r = s.substance(0, 100, 0, 5.0);
        assert_eq!(r, Substance::Density);
    }

    #[test]
    fn substance_air_far_above_table_keeps_air() {
        let s = sys();
        let r = s.substance(0, 100, 0, -5.0);
        assert_eq!(r, Substance::Density);
    }

    #[test]
    #[ignore = "all-cells-dry temp override; re-enable when aquifers are tuned back on"]
    fn substance_below_table_in_cave_becomes_fluid() {
        // With sparse aquifers (most cells are dry), we have to
        // scan to find a wet cell whose y_top is above a candidate
        // cave voxel, then verify that voxel floods.
        let s = sys();
        let mut found_fluid = false;
        'outer: for cx in 0..32 {
            for cz in 0..32 {
                // Probe each deep cell; pick a wy 2 below its y_top.
                let cell = s.cell_at(cx, -5, cz);
                if cell.y_top < -1_000_000 {
                    continue; // sentinel-dry cell
                }
                let wy = cell.y_top - 1;
                let r = s.substance(cell.center_x, wy, cell.center_z, -5.0);
                if matches!(r, Substance::Block(Block::Water) | Substance::Block(Block::Lava)) {
                    found_fluid = true;
                    break 'outer;
                }
            }
        }
        assert!(found_fluid, "no fluid found in any of 1024 deep aquifer cells");
    }
}
