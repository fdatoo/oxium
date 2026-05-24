//! Density evaluation pipeline.
//!
//! Covers two concerns:
//!
//! 1. The 2D heightmap noise that gives each column its target Y
//!    ([`heightmap`]): `HeightmapNoise` and `DensityNoise` drive the
//!    spline-based surface height and per-voxel density respectively.
//!
//! 2. The 3D density volume noise evaluated via a 9³ corner-lattice
//!    trilerp cache ([`cell_evaluator`]): `CellEvaluator` samples the
//!    density expression tree at 729 cell corners per chunk instead of
//!    per-voxel, achieving ~134× fewer expensive evaluations.
//!
//! The density expression tree itself lives in [`splines`], alongside
//! the climate-channel selectors and marker hints. Standalone math
//! helpers (peaks-and-valleys fold, slide, `offset_to_world_y`, etc.)
//! are in [`math`].
//!
//! See `docs/book/content/part-3-region-build/3.3-heightmap.mdx`,
//! `docs/book/content/part-4-chunk-fill/4.1-density-graph.mdx`,
//! `docs/book/content/part-4-chunk-fill/4.2-cell-evaluator.mdx`, and
//! `docs/superpowers/specs/2026-05-19-worldgen-3d-design.md`.

pub mod cell_evaluator;
pub mod density_3d;
pub mod heightmap;
pub mod math;
pub mod splines;

// Re-export everything that was public in the original heightmap.rs
// and density_graph.rs files so callers at crate::worldgen::density::*
// find what they need without digging into submodules.

pub use cell_evaluator::{CELL_COUNT, CELL_SIZE, CORNER_COUNT, CellEvaluator};
pub use density_3d::{DensityComposition, DensityNoise};
pub use heightmap::HeightmapNoise;
pub use math::{
    offset_to_world_y, peaks_and_valleys, plate_roughness_bias, signed_continentalness, slide,
    smooth_plate_contribution,
};
pub use splines::{ClimateChannel, ColumnClimate, DensityFn, MarkerKind, build_default_tree};
