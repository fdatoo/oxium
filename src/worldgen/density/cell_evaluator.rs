//! Per-chunk 9×9×9 corner-lattice cache + trilinear interpolation evaluator.
//!
//! Rather than evaluating the full density graph per voxel (~98k voxels
//! per chunk), this evaluator samples the graph once at each corner of a
//! 4×4×4-voxel cell grid — 9 corners per chunk axis (8 cells + 1), for
//! 729 total. Per-voxel density is then the trilinear interpolation of the
//! 8 surrounding cell corners. This trades ~134× fewer expensive density
//! evaluations for a small smooth-density approximation error that is
//! only visible at the density=0 iso-surface inside a cell. In practice
//! the terrain heightmap changes slowly enough that the trilerp approximation
//! is visually indistinguishable from exact evaluation.
//!
//! A 2D climate cache (8×8 = 64 entries keyed by corner XZ) ensures the
//! 2D climate channels (`continentalness`, `terrain_shape`, `ridges_pv`)
//! are evaluated only once per (cx, cz) column of corners rather than per
//! 3D corner — a further 9× reduction for the 2D inputs.
//!
//! See `docs/book/content/part-4-chunk-fill/4.2-cell-evaluator.mdx` and
//! `docs/superpowers/specs/2026-05-19-worldgen-3d-design.md`.

use super::heightmap::DensityNoise;
use super::splines::{ColumnClimate, DensityFn};
use crate::voxel::coords::CHUNK_DIM_U;
use crate::worldgen::config::DensityConfig;

/// Edge length of a cell (in blocks). Each chunk dimension (32)
/// divides into 8 cells.
pub const CELL_SIZE: i32 = 4;

/// Cell count per chunk axis: `32 / 4 = 8`.
pub const CELL_COUNT: usize = (CHUNK_DIM_U as usize) / CELL_SIZE as usize;

/// Corner count per chunk axis: `CELL_COUNT + 1 = 9` (each cell
/// needs corners on both sides; corners are shared between cells).
pub const CORNER_COUNT: usize = CELL_COUNT + 1;

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
    use crate::worldgen::density::splines::build_default_tree;

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
