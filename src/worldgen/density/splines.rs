//! Climate-channel spline pipeline components.
//!
//! Provides the density function node enum ([`DensityFn`]), the
//! climate-channel selector ([`ClimateChannel`]), the marker hint
//! ([`MarkerKind`]), the per-column climate triple ([`ColumnClimate`]),
//! and [`build_default_tree`] which assembles the canonical PR 3/5
//! density expression tree.
//!
//! See `docs/book/content/part-4-chunk-fill/4.1-density-graph.mdx` and
//! `docs/superpowers/specs/2026-05-19-worldgen-3d-design.md`.

use super::heightmap::DensityNoise;
use crate::worldgen::config::{DensityConfig, NestedSpline};
use std::sync::Arc;

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
/// composition implemented in [`super::heightmap::DensityNoise::evaluate`]:
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
