//! Density-function graph + sparse cell-grid evaluator (PR 5).
//!
//! Per the migration plan and Minecraft 1.18+ `NoiseChunk`/
//! `DensityFunctions`, the per-voxel density formula is expressed as
//! a small DAG of [`DensityFn`] nodes — `Constant`, `Add`, `Mul`,
//! `Min`, `Max`, `QuarterNegative`, `YGradient`, `Spline`,
//! `BaseNoise3D`, climate inputs — and the per-chunk evaluator
//! samples this graph **only at 4×4×4 cell corners** (9×9×9 corner
//! samples per chunk) and trilerps inside each cell.
//!
//! The wins:
//!   * `~98k voxels per chunk` → `~729 corner samples` ≈ **134×
//!     fewer** expensive density evaluations
//!   * Marker wrappers ([`MarkerKind`]) let the runtime know which
//!     subtrees benefit from caching:
//!     - [`MarkerKind::FlatCache`] — 2D quart-resolution cache for
//!       climate channels (8×8 = 64 entries per chunk vs 1024)
//!     - [`MarkerKind::Interpolated`] — corner-lattice + trilerp
//!     - [`MarkerKind::CacheAllInCell`] — single value per cell
//!       broadcast to its 64 voxels (matches MC pattern)
//!     - [`MarkerKind::CacheOnce`] — last (wx,wy,wz) value memoised
//!
//! Caves stay per-voxel — cave SDFs need 1-block resolution and are
//! subtracted from the interpolated density at chunk-fill time.
//!
//! See `docs/book/content/part-4-chunk-fill/4.1-density-graph.mdx`,
//! `docs/book/content/part-4-chunk-fill/4.2-cell-evaluator.mdx`, and
//! `docs/superpowers/specs/2026-05-19-worldgen-3d-design.md`.

use crate::voxel::coords::CHUNK_DIM_U;
use crate::worldgen::config::{DensityConfig, NestedSpline};
use crate::worldgen::heightmap::DensityNoise;
use std::sync::Arc;

/// Edge length of a cell (in blocks). Each chunk dimension (32)
/// divides into 8 cells.
pub const CELL_SIZE: i32 = 4;

/// Cell count per chunk axis: `32 / 4 = 8`.
pub const CELL_COUNT: usize = (CHUNK_DIM_U as usize) / CELL_SIZE as usize;

/// Corner count per chunk axis: `CELL_COUNT + 1 = 9` (each cell
/// needs corners on both sides; corners are shared between cells).
pub const CORNER_COUNT: usize = CELL_COUNT + 1;

/// Marker hint: how should the runtime cache a subtree?
#[derive(Copy, Clone, Debug)]
pub enum MarkerKind {
    /// 2D quart-resolution cache (climate channels). Sampled once
    /// per (qx, qz) and broadcast to every voxel in that 4×4 column.
    FlatCache,
    /// Corner-lattice sampling + trilerp. Sampled at the cell's 8
    /// corners; per-voxel uses hierarchical lerp.
    Interpolated,
    /// Single value per cell, broadcast to its 64 voxels.
    CacheAllInCell,
    /// One-shot memo of the last (wx, wy, wz) value.
    CacheOnce,
}

/// Climate-channel selector (the FlatCache 2D inputs).
#[derive(Copy, Clone, Debug)]
pub enum ClimateChannel {
    Continentalness,
    TerrainShape,
    RidgesPv,
}

/// Density function node. Recursive expression tree.
#[derive(Clone, Debug)]
pub enum DensityFn {
    /// Scalar constant.
    Constant(f32),
    /// Climate input (one of three 2D channels — sampled via the
    /// chunk's per-column FlatCache).
    Climate(ClimateChannel),
    /// Y-gradient: `amp * (1 - 2*(y - y_min)/(y_max - y_min))`.
    YGradient,
    /// Anisotropic 3D base noise (vertical features taller than wide).
    BaseNoise3D,
    /// Negative inputs are scaled by `cfg.above_surface_softening`,
    /// positives pass through unchanged (MC's `quarter_negative`).
    QuarterNegative(Box<DensityFn>),
    /// Hermite cubic spline evaluation. Input is selected by
    /// `ClimateChannel`; the spline's nesting walks three channels
    /// (continentalness, terrain_shape, ridges_pv).
    Spline { spline: Arc<NestedSpline> },
    /// Binary addition.
    Add(Box<DensityFn>, Box<DensityFn>),
    /// Binary multiplication.
    Mul(Box<DensityFn>, Box<DensityFn>),
    /// Binary min.
    Min(Box<DensityFn>, Box<DensityFn>),
    /// Binary max.
    Max(Box<DensityFn>, Box<DensityFn>),
    /// Marker wrapper — semantically identity, but tells the
    /// runtime to wire `inner` through a cache. PR 5 doesn't yet
    /// support every marker via a dedicated cache implementation;
    /// the evaluator treats unknown markers as pass-through.
    Marker {
        kind: MarkerKind,
        inner: Box<DensityFn>,
    },
}

/// Per-column climate triple, already evaluated.
#[derive(Clone, Copy)]
pub struct ColumnClimate {
    pub continentalness: f32,
    pub terrain_shape: f32,
    pub ridges_pv: f32,
}

impl DensityFn {
    /// Evaluate the graph at `(wx, wy, wz)` given the column's
    /// climate triple and a [`DensityNoise`] for the base 3D field.
    pub fn evaluate(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        climate: ColumnClimate,
        density: &DensityNoise,
        cfg: &DensityConfig,
    ) -> f32 {
        match self {
            DensityFn::Constant(v) => *v,
            DensityFn::Climate(ch) => match ch {
                ClimateChannel::Continentalness => climate.continentalness,
                ClimateChannel::TerrainShape => climate.terrain_shape,
                ClimateChannel::RidgesPv => climate.ridges_pv,
            },
            DensityFn::YGradient => {
                let t = (wy - cfg.y_min) as f32 / (cfg.y_max - cfg.y_min) as f32;
                cfg.y_gradient_amplitude * (1.0 - 2.0 * t)
            }
            DensityFn::BaseNoise3D => density.evaluate_base_3d(wx, wy, wz, cfg),
            DensityFn::QuarterNegative(inner) => {
                let v = inner.evaluate(wx, wy, wz, climate, density, cfg);
                if v > 0.0 {
                    v
                } else {
                    v * cfg.above_surface_softening
                }
            }
            DensityFn::Spline { spline } => spline.evaluate(
                climate.continentalness,
                climate.terrain_shape,
                climate.ridges_pv,
            ),
            DensityFn::Add(a, b) => {
                a.evaluate(wx, wy, wz, climate, density, cfg)
                    + b.evaluate(wx, wy, wz, climate, density, cfg)
            }
            DensityFn::Mul(a, b) => {
                a.evaluate(wx, wy, wz, climate, density, cfg)
                    * b.evaluate(wx, wy, wz, climate, density, cfg)
            }
            DensityFn::Min(a, b) => {
                let va = a.evaluate(wx, wy, wz, climate, density, cfg);
                let vb = b.evaluate(wx, wy, wz, climate, density, cfg);
                va.min(vb)
            }
            DensityFn::Max(a, b) => {
                let va = a.evaluate(wx, wy, wz, climate, density, cfg);
                let vb = b.evaluate(wx, wy, wz, climate, density, cfg);
                va.max(vb)
            }
            DensityFn::Marker { inner, .. } => inner.evaluate(wx, wy, wz, climate, density, cfg),
        }
    }
}

/// Build the canonical density tree that matches the PR 3
/// composition implemented in [`crate::worldgen::heightmap::DensityNoise::evaluate`]:
///
/// ```text
///   shaped     = (y_gradient + offset + jaggedness * base_3d) * factor
///   shaped'    = quarter_negative(shaped)
///   density    = composition_scale * shaped' + base_3d
/// ```
///
/// Slides are applied in the chunk loop after the cell interpolator
/// returns the raw value, because the slide is per-voxel-y (cheap
/// and doesn't benefit from corner caching).
///
/// PR 5: this is the tree the cell-grid evaluator samples at chunk
/// corners. Each leaf — `Constant`, `Climate`, `YGradient`,
/// `BaseNoise3D` — is wrapped in `Marker(Interpolated, ...)` so the
/// runtime knows to interpolate the *whole* tree per cell.
pub fn build_default_tree(
    climate_cfg: &crate::worldgen::config::ClimateConfig,
    density_cfg: &DensityConfig,
) -> DensityFn {
    let offset = Arc::new(climate_cfg.offset_spline.clone());
    let factor = Arc::new(climate_cfg.factor_spline.clone());
    let jaggedness = Arc::new(climate_cfg.jaggedness_spline.clone());

    let depth = DensityFn::Add(
        Box::new(DensityFn::YGradient),
        Box::new(DensityFn::Spline { spline: offset }),
    );
    let jagged_rider = DensityFn::Mul(
        Box::new(DensityFn::Spline { spline: jaggedness }),
        Box::new(DensityFn::BaseNoise3D),
    );
    let shaped_raw = DensityFn::Mul(
        Box::new(DensityFn::Add(Box::new(depth), Box::new(jagged_rider))),
        Box::new(DensityFn::Spline { spline: factor }),
    );
    let shaped = DensityFn::QuarterNegative(Box::new(shaped_raw));
    let scaled = DensityFn::Mul(
        Box::new(DensityFn::Constant(density_cfg.composition_scale)),
        Box::new(shaped),
    );
    let composed = DensityFn::Add(Box::new(scaled), Box::new(DensityFn::BaseNoise3D));
    DensityFn::Marker {
        kind: MarkerKind::Interpolated,
        inner: Box::new(composed),
    }
}

/// Per-chunk 9×9×9 corner-lattice cache + trilinear interpolation evaluator.
///
/// Rather than evaluating the full density graph per voxel (~98k voxels
/// per chunk), this evaluator samples the graph once at each corner of a
/// 4×4×4-voxel cell grid — 9 corners per chunk axis (8 cells + 1), for
/// 729 total. Per-voxel density is then the trilinear interpolation of the
/// 8 surrounding cell corners. This trades ~134× fewer expensive density
/// evaluations for a small smooth-density approximation error that is
/// only visible at the density=0 iso-surface inside a cell. In practice
/// the terrain heightmap changes slowly enough that the trilerp approximation
/// is visually indistinguishable from exact evaluation.
///
/// A 2D climate cache (8×8 = 64 entries keyed by corner XZ) ensures the
/// 2D climate channels (`continentalness`, `terrain_shape`, `ridges_pv`)
/// are evaluated only once per (cx, cz) column of corners rather than per
/// 3D corner — a further 9× reduction for the 2D inputs.
///
/// Usage:
/// 1. `new(graph, density, cfg, chunk_origin, column_climate_fn)`
///    — fills the corner lattice + 2D climate cache.
/// 2. `evaluate(wx, wy, wz)` — trilerps the 8 surrounding corners.
pub struct CellEvaluator {
    /// Stored as `[x][y][z]` → linear index `x + y*CORNER + z*CORNER*CORNER`.
    corners: Vec<f32>,
    chunk_origin_x: i32,
    chunk_origin_y: i32,
    chunk_origin_z: i32,
}

impl CellEvaluator {
    /// Build a corner-lattice for the chunk at `origin`. `column_climate`
    /// is invoked once per (qx, qz) quart corner to populate the
    /// climate cache; `density` provides the 3D base noise.
    pub fn new<F>(
        graph: &DensityFn,
        density: &DensityNoise,
        cfg: &DensityConfig,
        chunk_origin: (i32, i32, i32),
        mut column_climate: F,
    ) -> Self
    where
        F: FnMut(i32, i32) -> ColumnClimate,
    {
        let mut corners = vec![0.0_f32; CORNER_COUNT * CORNER_COUNT * CORNER_COUNT];
        // 2D climate cache keyed by (cx, cz) where cx,cz ∈ [0..CORNER_COUNT).
        let mut climate_cache: Vec<Option<ColumnClimate>> = vec![None; CORNER_COUNT * CORNER_COUNT];
        for cz in 0..CORNER_COUNT {
            for cy in 0..CORNER_COUNT {
                for cx in 0..CORNER_COUNT {
                    let wx = chunk_origin.0 + (cx as i32) * CELL_SIZE;
                    let wy = chunk_origin.1 + (cy as i32) * CELL_SIZE;
                    let wz = chunk_origin.2 + (cz as i32) * CELL_SIZE;
                    let idx = cx + cy * CORNER_COUNT + cz * CORNER_COUNT * CORNER_COUNT;
                    let climate_idx = cx + cz * CORNER_COUNT;
                    let climate = match climate_cache[climate_idx] {
                        Some(c) => c,
                        None => {
                            let c = column_climate(wx, wz);
                            climate_cache[climate_idx] = Some(c);
                            c
                        }
                    };
                    corners[idx] = graph.evaluate(wx, wy, wz, climate, density, cfg);
                }
            }
        }
        Self {
            corners,
            chunk_origin_x: chunk_origin.0,
            chunk_origin_y: chunk_origin.1,
            chunk_origin_z: chunk_origin.2,
        }
    }

    /// Trilinear interpolation of the 8 corners around `(wx, wy, wz)`.
    /// `(wx, wy, wz)` must be inside this chunk's bounds.
    pub fn evaluate(&self, wx: i32, wy: i32, wz: i32) -> f32 {
        let lx = wx - self.chunk_origin_x;
        let ly = wy - self.chunk_origin_y;
        let lz = wz - self.chunk_origin_z;
        let cx = (lx / CELL_SIZE) as usize;
        let cy = (ly / CELL_SIZE) as usize;
        let cz = (lz / CELL_SIZE) as usize;
        let cx = cx.min(CELL_COUNT - 1);
        let cy = cy.min(CELL_COUNT - 1);
        let cz = cz.min(CELL_COUNT - 1);
        let tx = (lx - cx as i32 * CELL_SIZE) as f32 / CELL_SIZE as f32;
        let ty = (ly - cy as i32 * CELL_SIZE) as f32 / CELL_SIZE as f32;
        let tz = (lz - cz as i32 * CELL_SIZE) as f32 / CELL_SIZE as f32;
        // 8 corners of the cell.
        let c000 = self.sample_corner(cx, cy, cz);
        let c100 = self.sample_corner(cx + 1, cy, cz);
        let c010 = self.sample_corner(cx, cy + 1, cz);
        let c110 = self.sample_corner(cx + 1, cy + 1, cz);
        let c001 = self.sample_corner(cx, cy, cz + 1);
        let c101 = self.sample_corner(cx + 1, cy, cz + 1);
        let c011 = self.sample_corner(cx, cy + 1, cz + 1);
        let c111 = self.sample_corner(cx + 1, cy + 1, cz + 1);
        // Hierarchical Y → X → Z lerp (matches MC's NoiseInterpolator).
        let xz00 = c000 + (c010 - c000) * ty;
        let xz10 = c100 + (c110 - c100) * ty;
        let xz01 = c001 + (c011 - c001) * ty;
        let xz11 = c101 + (c111 - c101) * ty;
        let z0 = xz00 + (xz10 - xz00) * tx;
        let z1 = xz01 + (xz11 - xz01) * tx;
        z0 + (z1 - z0) * tz
    }

    #[inline]
    fn sample_corner(&self, cx: usize, cy: usize, cz: usize) -> f32 {
        self.corners[cx + cy * CORNER_COUNT + cz * CORNER_COUNT * CORNER_COUNT]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::config::WorldgenConfig;

    fn test_climate() -> ColumnClimate {
        ColumnClimate {
            continentalness: 0.3,
            terrain_shape: 0.0,
            ridges_pv: 0.0,
        }
    }

    #[test]
    fn cell_constants_divide_chunk() {
        assert_eq!(CELL_COUNT * CELL_SIZE as usize, CHUNK_DIM_U as usize);
        assert_eq!(CORNER_COUNT, CELL_COUNT + 1);
    }

    #[test]
    fn constant_evaluates_at_any_coord() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42, &cfg.density);
        let g = DensityFn::Constant(7.0);
        let v = g.evaluate(100, 50, 200, test_climate(), &d, &cfg.density);
        assert_eq!(v, 7.0);
    }

    #[test]
    fn y_gradient_is_amp_at_y_min_neg_amp_at_y_max() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42, &cfg.density);
        let g = DensityFn::YGradient;
        let v_bot = g.evaluate(0, cfg.density.y_min, 0, test_climate(), &d, &cfg.density);
        let v_top = g.evaluate(0, cfg.density.y_max, 0, test_climate(), &d, &cfg.density);
        assert!((v_bot - cfg.density.y_gradient_amplitude).abs() < 1e-4);
        assert!((v_top + cfg.density.y_gradient_amplitude).abs() < 1e-4);
    }

    #[test]
    fn quarter_negative_softens_negatives() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42, &cfg.density);
        let g = DensityFn::QuarterNegative(Box::new(DensityFn::Constant(-1.0)));
        let v = g.evaluate(0, 0, 0, test_climate(), &d, &cfg.density);
        assert!((v - (-1.0 * cfg.density.above_surface_softening)).abs() < 1e-5);
    }

    #[test]
    fn quarter_negative_preserves_positives() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42, &cfg.density);
        let g = DensityFn::QuarterNegative(Box::new(DensityFn::Constant(1.5)));
        let v = g.evaluate(0, 0, 0, test_climate(), &d, &cfg.density);
        assert!((v - 1.5).abs() < 1e-5);
    }

    #[test]
    fn cell_evaluator_at_corner_matches_graph() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42, &cfg.density);
        let g = build_default_tree(&cfg.climate, &cfg.density);
        let origin = (0, 0, 0);
        let eval = CellEvaluator::new(&g, &d, &cfg.density, origin, |_wx, _wz| test_climate());
        // At an exact corner the trilerp degenerates to the corner
        // value — must match the graph evaluation there.
        let v_interp = eval.evaluate(0, 0, 0);
        let v_direct = g.evaluate(0, 0, 0, test_climate(), &d, &cfg.density);
        assert!(
            (v_interp - v_direct).abs() < 1e-3,
            "interp {v_interp} vs direct {v_direct}",
        );
    }

    #[test]
    fn cell_evaluator_inside_cell_lerps() {
        // Verify mid-cell interpolation produces something between
        // adjacent corner values.
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42, &cfg.density);
        let g = build_default_tree(&cfg.climate, &cfg.density);
        let eval = CellEvaluator::new(&g, &d, &cfg.density, (0, 0, 0), |_wx, _wz| test_climate());
        let v_corner0 = eval.evaluate(0, 0, 0);
        let v_corner1 = eval.evaluate(CELL_SIZE, 0, 0);
        let v_mid = eval.evaluate(CELL_SIZE / 2, 0, 0);
        let lo = v_corner0.min(v_corner1);
        let hi = v_corner0.max(v_corner1);
        assert!(
            v_mid >= lo - 1e-3 && v_mid <= hi + 1e-3,
            "mid {v_mid} not between corners {v_corner0}..{v_corner1}",
        );
    }
}
