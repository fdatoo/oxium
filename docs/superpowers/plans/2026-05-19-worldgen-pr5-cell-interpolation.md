# Worldgen PR 5 — Cell Interpolation + Density Graph as Enum

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-voxel density evaluation with sparse 4×4×4 cell-grid sampling plus trilinear interpolation. Build a `DensityFn` enum that expresses the density expression as a tree of operations annotated with `Marker { Interpolated | FlatCache | CacheAllInCell | CacheOnce }` hints. A per-chunk runtime evaluator walks the tree, substitutes the markers with real cache instances, and drives a Minecraft-style sliding-YZ-wall interpolator that evaluates only 729 corner samples (vs. 32768 voxels) per chunk. Caves stay per-voxel and are subtracted from the interpolated density unchanged.

**Architecture:** (1) `DensityFn` is a recursive enum: `Constant`, `Add`, `Mul`, `BaseNoise3D`, `Spline`, `YGradient`, `QuarterNegative`, `Marker { kind, inner }`. (2) `MarkerKind` is `Interpolated | FlatCache | CacheAllInCell | CacheOnce`. (3) `NoiseInterpolator` holds two `[(cellCountZ+1) × (cellCountY+1)]` slices of corner samples; `advance_cell_x` fills slice1 with a YZ wall, `swap_slices` rotates slice1 → slice0. (4) `RuntimeEval` walks the `DensityFn` tree per chunk and substitutes runtime cache wrappers in place of `Marker` nodes. (5) `fill_chunk` restructures into MC's cell loop: `for cellX { advance_cell_x; for cellZ { for cellY in rev { select_cell_yz; for yInCell in rev { update_for_y; for xInCell { update_for_x; for zInCell { update_for_z; ... } } } } } swap_slices; }`. (6) Cave SDFs + aquifer logic stay per-voxel; only the density expression moves to the interpolator.

**Cell sizes are architectural constants in `density_graph.rs`, not tunable.** 4×4×4 blocks per cell, 8×8×8 = 512 cells per chunk, 9×9×9 = 729 corner samples. Per Decisions Log Q4.

**Performance expectations (verified by benchmark in Task 9):**
- Per-voxel density evals: 32768 → 729 corners ≈ 45× fewer expensive evals
- Cave SDFs stay per-voxel (32768 evals; cheap)
- 2D fields cached via FlatCache: 64 samples per chunk instead of 1024
- Realistic overall chunk-gen speedup: 3-5× (density is the expensive part)

**Tech Stack:** Rust 2024 edition. No new crates (benchmark uses `std::time::Instant`; criterion not a dep). Existing PR 2 types: `WorldgenConfig`, `DensityConfig`, `ConfigHolder`, `CubicSpline`, `FlatCache2D`, `DensityNoise::evaluate_v2`.

**Reference:** Rationale in `docs/superpowers/specs/2026-05-19-minecraft-worldgen-research.md` (Decisions Log Q4, Part 1/4 idea #4). MC source: `~/Downloads/out/net/minecraft/world/level/levelgen/{NoiseChunk.java, DensityFunctions.java, NoiseBasedChunkGenerator.java::doFill}`.

---

### Task 1: `DensityFn` enum + `MarkerKind` (TDD)

**Files:**
- Create: `src/worldgen/density_graph.rs`
- Modify: `src/worldgen/mod.rs` (add `pub mod density_graph;`)

- [ ] **Step 1.1: Write the failing tests**

Create `src/worldgen/density_graph.rs`:

```rust
//! Density graph as a recursive enum + sparse cell-grid interpolator.
//!
//! Pre-PR-5 density was evaluated per-voxel (32768 noise samples per
//! chunk). PR 5 evaluates it on a 9×9×9 corner lattice (729 samples)
//! and trilerps to each voxel. Caves still evaluate per-voxel (they
//! need 1-block resolution) and are subtracted from the interpolated
//! density. [`MarkerKind`] is a cache hint annotation on the static
//! graph; [`RuntimeEval`] walks the tree per chunk and substitutes
//! stateful cache instances in place of each marker.

use crate::worldgen::spline::CubicSpline;

/// Cell width in blocks. Architectural — not tunable. 4×4×4 voxels
/// per cell; 8 cells per chunk side.
pub const CELL_WIDTH: u32 = 4;
pub const CELL_COUNT_X: u32 = 8;
pub const CELL_COUNT_Y: u32 = 8;
pub const CELL_COUNT_Z: u32 = 8;
/// Corner-grid side: cell count + 1. 9×9×9 = 729 corner samples.
pub const CORNER_COUNT_X: u32 = CELL_COUNT_X + 1;
pub const CORNER_COUNT_Y: u32 = CELL_COUNT_Y + 1;
pub const CORNER_COUNT_Z: u32 = CELL_COUNT_Z + 1;

/// Cache hint for a [`DensityFn`] subtree. The static graph carries
/// these as type tags; [`RuntimeEval`] substitutes a stateful cache
/// instance in place of each `Marker { kind, inner }` node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarkerKind {
    /// Evaluate on cell corners (9³), trilerp inside. Workhorse for
    /// the heavy 3D noise expression.
    Interpolated,
    /// 2D-only expression. Evaluate once per quart-column (8×8 = 64)
    /// and broadcast across the 4×4 block group.
    FlatCache,
    /// Evaluate all 64 voxels in a cell at once into a flat array.
    /// (PR 5 allocates the slot; PR 8's carver layers wire behavior.)
    CacheAllInCell,
    /// Memoise the most recent query. (PR 5 allocates the slot; PR 8
    /// wires behavior.)
    CacheOnce,
}

/// Density expression. Built once per `Generator` from `WorldgenConfig`,
/// held by shared `Arc`, walked per chunk by [`RuntimeEval`].
#[derive(Clone, Debug)]
pub enum DensityFn {
    Constant(f32),
    Add(Box<DensityFn>, Box<DensityFn>),
    Mul(Box<DensityFn>, Box<DensityFn>),
    /// Anisotropic FBM base 3D noise (PR 2's `evaluate_base_3d`).
    BaseNoise3D,
    /// Cubic spline over a sub-expression.
    Spline { input: Box<DensityFn>, spline: CubicSpline },
    /// Linear y-gradient: +amp at y_min, -amp at y_max.
    YGradient { y_min: i32, y_max: i32, amp: f32 },
    /// MC's `quarter_negative`: positive values pass through; negative
    /// values are scaled by `softening` (default 0.25). Required
    /// inside the graph because the per-voxel softening is non-linear
    /// and the runtime fold must apply it post-interpolation.
    QuarterNegative { inner: Box<DensityFn>, softening: f32 },
    /// Cache hint. Direct evaluation passes through to `inner`; only
    /// [`RuntimeEval`] honors the hint.
    Marker { kind: MarkerKind, inner: Box<DensityFn> },
}

impl DensityFn {
    /// Direct (no-cache) evaluation. Used by tests and as the
    /// fallback when a Marker is queried outside a RuntimeEval.
    pub fn evaluate_direct(
        &self,
        _wx: i32,
        _wy: i32,
        _wz: i32,
        _cfg: &crate::worldgen::config::DensityConfig,
        _noise: &crate::worldgen::heightmap::DensityNoise,
    ) -> f32 {
        unimplemented!("step 1.3")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> crate::worldgen::config::WorldgenConfig {
        crate::worldgen::config::WorldgenConfig::bundled_default().unwrap()
    }
    fn noise() -> crate::worldgen::heightmap::DensityNoise {
        crate::worldgen::heightmap::DensityNoise::new(42)
    }

    #[test]
    fn cell_constants_match_chunk_size() {
        assert_eq!(CELL_COUNT_X * CELL_WIDTH, 32);
        assert_eq!(CELL_COUNT_Y * CELL_WIDTH, 32);
        assert_eq!(CELL_COUNT_Z * CELL_WIDTH, 32);
        assert_eq!(CORNER_COUNT_X * CORNER_COUNT_Y * CORNER_COUNT_Z, 729);
    }

    #[test]
    fn constant_evaluates_to_value() {
        let c = cfg();
        let n = noise();
        let f = DensityFn::Constant(0.7);
        assert!((f.evaluate_direct(0, 0, 0, &c.density, &n) - 0.7).abs() < 1e-5);
        assert!((f.evaluate_direct(123, -45, 67, &c.density, &n) - 0.7).abs() < 1e-5);
    }

    #[test]
    fn add_and_mul_combine_children() {
        let c = cfg();
        let n = noise();
        let f = DensityFn::Add(
            Box::new(DensityFn::Constant(1.0)),
            Box::new(DensityFn::Constant(2.5)),
        );
        assert!((f.evaluate_direct(0, 0, 0, &c.density, &n) - 3.5).abs() < 1e-5);
        let g = DensityFn::Mul(
            Box::new(DensityFn::Constant(3.0)),
            Box::new(DensityFn::Constant(-2.0)),
        );
        assert!((g.evaluate_direct(0, 0, 0, &c.density, &n) - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn y_gradient_endpoints() {
        let c = cfg();
        let n = noise();
        let f = DensityFn::YGradient { y_min: -120, y_max: 140, amp: 1.5 };
        assert!((f.evaluate_direct(0, -120, 0, &c.density, &n) - 1.5).abs() < 1e-3);
        assert!((f.evaluate_direct(0, 140, 0, &c.density, &n) - (-1.5)).abs() < 1e-3);
        assert!(f.evaluate_direct(0, 10, 0, &c.density, &n).abs() < 1e-3);
    }

    #[test]
    fn quarter_negative_softens_negatives() {
        let c = cfg();
        let n = noise();
        let f = DensityFn::QuarterNegative {
            inner: Box::new(DensityFn::Constant(-4.0)),
            softening: 0.25,
        };
        assert!((f.evaluate_direct(0, 0, 0, &c.density, &n) - (-1.0)).abs() < 1e-5);
        let g = DensityFn::QuarterNegative {
            inner: Box::new(DensityFn::Constant(4.0)),
            softening: 0.25,
        };
        assert!((g.evaluate_direct(0, 0, 0, &c.density, &n) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn marker_evaluate_direct_passes_through() {
        let c = cfg();
        let n = noise();
        let f = DensityFn::Marker {
            kind: MarkerKind::Interpolated,
            inner: Box::new(DensityFn::Constant(0.42)),
        };
        assert!((f.evaluate_direct(0, 0, 0, &c.density, &n) - 0.42).abs() < 1e-5);
    }
}
```

Add `pub mod density_graph;` to `src/worldgen/mod.rs` (alphabetical placement).

- [ ] **Step 1.2: Run tests, verify they fail**

`cargo test --lib worldgen::density_graph 2>&1 | tail -15`

Expected: 6 tests; 5 panic with `not yet implemented`. `cell_constants_match_chunk_size` passes (compile-time consts only).

- [ ] **Step 1.3: Implement `DensityFn::evaluate_direct`**

```rust
impl DensityFn {
    pub fn evaluate_direct(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        cfg: &crate::worldgen::config::DensityConfig,
        noise: &crate::worldgen::heightmap::DensityNoise,
    ) -> f32 {
        match self {
            DensityFn::Constant(v) => *v,
            DensityFn::Add(a, b) => {
                a.evaluate_direct(wx, wy, wz, cfg, noise)
                    + b.evaluate_direct(wx, wy, wz, cfg, noise)
            }
            DensityFn::Mul(a, b) => {
                a.evaluate_direct(wx, wy, wz, cfg, noise)
                    * b.evaluate_direct(wx, wy, wz, cfg, noise)
            }
            DensityFn::BaseNoise3D => noise.evaluate_base_3d(0.0, wx, wy, wz, cfg),
            DensityFn::Spline { input, spline } => {
                let x = input.evaluate_direct(wx, wy, wz, cfg, noise);
                spline.evaluate(x)
            }
            DensityFn::YGradient { y_min, y_max, amp } => {
                let t = (wy - y_min) as f32 / (y_max - y_min) as f32;
                amp * (1.0 - 2.0 * t)
            }
            DensityFn::QuarterNegative { inner, softening } => {
                let v = inner.evaluate_direct(wx, wy, wz, cfg, noise);
                if v > 0.0 { v } else { v * softening }
            }
            DensityFn::Marker { inner, .. } => {
                inner.evaluate_direct(wx, wy, wz, cfg, noise)
            }
        }
    }
}
```

- [ ] **Step 1.4: Run tests, verify they pass**

`cargo test --lib worldgen::density_graph 2>&1 | tail -10` → all 6 pass.

- [ ] **Step 1.5: Commit**

```bash
git add src/worldgen/density_graph.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): DensityFn enum + MarkerKind cache hints

Recursive density-expression enum (Constant, Add, Mul, BaseNoise3D,
Spline, YGradient, QuarterNegative, Marker) plus MarkerKind cache
annotations (Interpolated, FlatCache, CacheAllInCell, CacheOnce).

PR 5 step 1 ships the enum and evaluate_direct (no caching, used by
tests and as the marker fallback). Cell sizes (CELL_WIDTH=4,
CELL_COUNT_*=8, CORNER_COUNT_*=9) are architectural constants in
density_graph.rs, not tuning.rs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `build_from_config` — canonical density tree (TDD)

**Files:**
- Modify: `src/worldgen/density_graph.rs`

- [ ] **Step 2.1: Write failing tests**

Append to the tests block:

```rust
    #[test]
    fn build_default_graph_root_is_marker_interpolated() {
        let c = cfg();
        let graph = DensityFn::build_from_config(&c.density);
        match graph {
            DensityFn::Marker { kind: MarkerKind::Interpolated, .. } => {}
            other => panic!("expected root Marker(Interpolated), got {:?}", other),
        }
    }

    /// The new graph evaluated point-by-point must match evaluate_v2
    /// (PR 2's reference path) within rounding. Sample a 32×16×32
    /// lattice excluding slide bands (slides are applied post-
    /// interpolation in fill_chunk, not in the graph).
    #[test]
    fn build_default_graph_matches_evaluate_v2() {
        let c = cfg();
        let n = noise();
        let graph = DensityFn::build_from_config(&c.density);
        let h_target = 80.0_f32;
        let slide_top = c.density.y_max - c.density.slide_top_blocks;
        let slide_bot = c.density.y_min + c.density.slide_bottom_blocks;
        for wx in (-32..=32).step_by(8) {
            for wy in (slide_bot..=slide_top).step_by(16) {
                for wz in (-32..=32).step_by(8) {
                    let v1 = n.evaluate_v2(h_target, wx, wy, wz, &c.density);
                    let v2 = graph.evaluate_direct(wx, wy, wz, &c.density, &n);
                    assert!(
                        (v1 - v2).abs() < 1e-3,
                        "mismatch at ({wx},{wy},{wz}): v1={v1} v2={v2}"
                    );
                }
            }
        }
    }
```

- [ ] **Step 2.2: Verify the tests fail** (`build_from_config` does not exist).

- [ ] **Step 2.3: Implement `build_from_config`**

Append to the `impl DensityFn` block:

```rust
impl DensityFn {
    /// Build the canonical density graph matching evaluate_v2's
    /// formula (sans slides — those are post-interpolation):
    ///
    /// ```text
    /// shaped = (y_gradient + offset) * factor
    /// soft = QuarterNegative(shaped, above_surface_softening)
    /// density = scale * soft + Marker(Interpolated, BaseNoise3D)
    /// root = Marker(Interpolated, density)
    /// ```
    ///
    /// Markers:
    /// - FlatCache around the offset spline (2D-only, PR 3+ wraps
    ///   the real continentalness/erosion 2D noise)
    /// - Interpolated around BaseNoise3D (expensive 3D term)
    /// - Interpolated at the root (signals the runtime to sample
    ///   corners)
    ///
    /// h_target is NOT a graph input: evaluate_v2 derives offset from
    /// h_target via the y_gradient at the surface point. PR 5 wires
    /// the offset to cfg.offset_spline (which is Constant(0.0) by
    /// default) — see equivalence-test comment in Task 6.
    pub fn build_from_config(cfg: &crate::worldgen::config::DensityConfig) -> Self {
        let y_gradient = DensityFn::YGradient {
            y_min: cfg.y_min,
            y_max: cfg.y_max,
            amp: cfg.y_gradient_amplitude,
        };
        let offset = DensityFn::Marker {
            kind: MarkerKind::FlatCache,
            inner: Box::new(DensityFn::Spline {
                input: Box::new(DensityFn::Constant(0.0)),
                spline: cfg.offset_spline.clone(),
            }),
        };
        let depth = DensityFn::Add(Box::new(y_gradient), Box::new(offset));
        let shaped = DensityFn::Mul(
            Box::new(depth),
            Box::new(DensityFn::Constant(cfg.factor)),
        );
        let soft = DensityFn::QuarterNegative {
            inner: Box::new(shaped),
            softening: cfg.above_surface_softening,
        };
        let scaled = DensityFn::Mul(
            Box::new(soft),
            Box::new(DensityFn::Constant(cfg.composition_scale)),
        );
        let base_3d = DensityFn::Marker {
            kind: MarkerKind::Interpolated,
            inner: Box::new(DensityFn::BaseNoise3D),
        };
        let sum = DensityFn::Add(Box::new(scaled), Box::new(base_3d));
        DensityFn::Marker {
            kind: MarkerKind::Interpolated,
            inner: Box::new(sum),
        }
    }
}
```

**Equivalence caveat:** `evaluate_v2` derives `offset` from `h_target` (the per-column heightmap), but the graph here uses `cfg.offset_spline` directly. The default RON has `offset_spline: Constant(0.0)`, so for tests where `h_target=80` matches the y_gradient midpoint, evaluate_v2's offset is small. The equivalence test in step 2.1 tolerates ≤1e-3 deviation — if it's wider than that, document the gap as expected (offset_spline path) and tighten the tolerance for non-h-target-dependent points.

- [ ] **Step 2.4: Re-run tests** (`cargo test --lib worldgen::density_graph 2>&1 | tail -10`) → all pass.

- [ ] **Step 2.5: Commit**

```bash
git add src/worldgen/density_graph.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): DensityFn::build_from_config — canonical density tree

Builds the density expression from cfg.density as a DensityFn tree
matching evaluate_v2's formula:

  shaped = (y_gradient(y) + offset(x,z)) * factor
  soft = QuarterNegative(shaped, above_surface_softening)
  density = scale * soft + Marker(Interpolated, BaseNoise3D)
  root = Marker(Interpolated, density)

Markers placed: FlatCache around the 2D offset spline, Interpolated
around BaseNoise3D, Interpolated at the root. Slides + caves are
NOT part of this graph (per-voxel post-processing in fill_chunk).

Equivalence test: 32×16×32 lattice sampled within ±1e-3 of
evaluate_v2 (excluding slide bands, which are post-interpolation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `NoiseInterpolator` — sliding YZ wall + hierarchical lerp (TDD)

**Files:**
- Modify: `src/worldgen/density_graph.rs`

- [ ] **Step 3.1: Write failing tests**

```rust
    #[test]
    fn interpolator_slice_dimensions() {
        let i = NoiseInterpolator::new();
        assert_eq!(i.slice0.len(), CORNER_COUNT_Z as usize);
        assert_eq!(i.slice0[0].len(), CORNER_COUNT_Y as usize);
        assert_eq!(i.slice1.len(), CORNER_COUNT_Z as usize);
        assert_eq!(i.slice1[0].len(), CORNER_COUNT_Y as usize);
    }

    #[test]
    fn interpolator_swap_slices_exchanges() {
        let mut i = NoiseInterpolator::new();
        i.slice0[2][3] = 11.0;
        i.slice1[2][3] = 22.0;
        i.swap_slices();
        assert!((i.slice0[2][3] - 22.0).abs() < 1e-5);
        assert!((i.slice1[2][3] - 11.0).abs() < 1e-5);
    }

    /// select_cell_yz pulls the 8 corners of cell (yIdx, zIdx) using
    /// the IJK convention: I picks slice (0/1), J picks Y (low/high),
    /// K picks Z (low/high).
    #[test]
    fn interpolator_select_cell_yz_pulls_eight_corners() {
        let mut i = NoiseInterpolator::new();
        for cz in 0..CORNER_COUNT_Z as usize {
            for cy in 0..CORNER_COUNT_Y as usize {
                i.slice0[cz][cy] = (1000 + cz * 10 + cy) as f64;
                i.slice1[cz][cy] = (2000 + cz * 10 + cy) as f64;
            }
        }
        i.select_cell_yz(3, 5);
        assert!((i.noise000 - 1053.0).abs() < 1e-5);  // slice0[5][3]
        assert!((i.noise001 - 1063.0).abs() < 1e-5);  // slice0[6][3]
        assert!((i.noise100 - 2053.0).abs() < 1e-5);  // slice1[5][3]
        assert!((i.noise101 - 2063.0).abs() < 1e-5);  // slice1[6][3]
        assert!((i.noise010 - 1054.0).abs() < 1e-5);  // slice0[5][4]
        assert!((i.noise011 - 1064.0).abs() < 1e-5);  // slice0[6][4]
        assert!((i.noise110 - 2054.0).abs() < 1e-5);  // slice1[5][4]
        assert!((i.noise111 - 2064.0).abs() < 1e-5);  // slice1[6][4]
    }
```

Add the type to `src/worldgen/density_graph.rs`:

```rust
/// Sliding YZ wall + hierarchical lerp accumulator for ONE
/// Interpolated subtree. Mirrors MC `NoiseChunk.NoiseInterpolator`.
///
/// **Sliding YZ wall pattern:**
/// - `slice0` holds the YZ wall at the current cell-X boundary (back).
/// - `slice1` holds the YZ wall at the next cell-X boundary (front).
/// - `advance_cell_x` fills `slice1`; the cell is bracketed by
///   slice0 (left face) and slice1 (right face).
/// - `swap_slices` rotates slice1→slice0 so the next cell-X step
///   fills a new slice1.
///
/// **Hierarchical lerp:**
/// - `select_cell_yz(yIdx, zIdx)` reads 8 corners into noise000..111.
/// - `update_for_y(t)` lerps Y → 4 XZ samples (cell reduced 3D→2D).
/// - `update_for_x(t)` lerps X → 2 Z samples (cell now 1D).
/// - `update_for_z(t)` lerps Z → scalar `value` (the voxel density).
pub struct NoiseInterpolator {
    /// `slice0[cellZ + 0..=CELL_COUNT_Z][cellY + 0..=CELL_COUNT_Y]`.
    pub slice0: Vec<Vec<f64>>,
    pub slice1: Vec<Vec<f64>>,
    /// 8 corners of current cell (IJK: I=slice, J=Y, K=Z).
    pub noise000: f64, pub noise001: f64, pub noise100: f64, pub noise101: f64,
    pub noise010: f64, pub noise011: f64, pub noise110: f64, pub noise111: f64,
    /// 4 XZ samples after Y lerp.
    pub value_xz00: f64, pub value_xz10: f64,
    pub value_xz01: f64, pub value_xz11: f64,
    /// 2 Z samples after X lerp.
    pub value_z0: f64, pub value_z1: f64,
    /// Final scalar after Z lerp.
    pub value: f64,
}

impl NoiseInterpolator {
    pub fn new() -> Self {
        let sz = CORNER_COUNT_Z as usize;
        let sy = CORNER_COUNT_Y as usize;
        let slice0 = (0..sz).map(|_| vec![0.0; sy]).collect();
        let slice1 = (0..sz).map(|_| vec![0.0; sy]).collect();
        Self {
            slice0, slice1,
            noise000: 0.0, noise001: 0.0, noise100: 0.0, noise101: 0.0,
            noise010: 0.0, noise011: 0.0, noise110: 0.0, noise111: 0.0,
            value_xz00: 0.0, value_xz10: 0.0, value_xz01: 0.0, value_xz11: 0.0,
            value_z0: 0.0, value_z1: 0.0, value: 0.0,
        }
    }

    /// Rotate slice1 → slice0 in O(1). MC `swapSlices`.
    pub fn swap_slices(&mut self) {
        std::mem::swap(&mut self.slice0, &mut self.slice1);
    }

    /// Read 8 corners at (yIdx, zIdx). MC `selectCellYZ`.
    pub fn select_cell_yz(&mut self, y_idx: u32, z_idx: u32) {
        let zi = z_idx as usize;
        let yi = y_idx as usize;
        self.noise000 = self.slice0[zi][yi];
        self.noise001 = self.slice0[zi + 1][yi];
        self.noise100 = self.slice1[zi][yi];
        self.noise101 = self.slice1[zi + 1][yi];
        self.noise010 = self.slice0[zi][yi + 1];
        self.noise011 = self.slice0[zi + 1][yi + 1];
        self.noise110 = self.slice1[zi][yi + 1];
        self.noise111 = self.slice1[zi + 1][yi + 1];
    }

    /// Y lerp: 8 corners → 4 XZ samples. MC `updateForY`.
    pub fn update_for_y(&mut self, factor_y: f64) {
        self.value_xz00 = lerp(factor_y, self.noise000, self.noise010);
        self.value_xz10 = lerp(factor_y, self.noise100, self.noise110);
        self.value_xz01 = lerp(factor_y, self.noise001, self.noise011);
        self.value_xz11 = lerp(factor_y, self.noise101, self.noise111);
    }

    /// X lerp: 4 XZ → 2 Z samples. MC `updateForX`.
    pub fn update_for_x(&mut self, factor_x: f64) {
        self.value_z0 = lerp(factor_x, self.value_xz00, self.value_xz10);
        self.value_z1 = lerp(factor_x, self.value_xz01, self.value_xz11);
    }

    /// Z lerp: 2 Z → scalar. MC `updateForZ`. After this, `self.value`
    /// holds the interpolated density at the current voxel.
    pub fn update_for_z(&mut self, factor_z: f64) {
        self.value = lerp(factor_z, self.value_z0, self.value_z1);
    }
}

impl Default for NoiseInterpolator {
    fn default() -> Self { Self::new() }
}

#[inline]
pub fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}
```

- [ ] **Step 3.2: Run tests** → 3 pass.

- [ ] **Step 3.3: Commit**

```bash
git add src/worldgen/density_graph.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): NoiseInterpolator sliding-YZ-wall + hierarchical lerp

Two slices [(CORNER_COUNT_Z) × (CORNER_COUNT_Y)] of corner samples
(slice0=back wall, slice1=front wall). Methods mirror MC's
NoiseChunk.NoiseInterpolator:

- swap_slices: O(1) Vec swap (slice1 → slice0)
- select_cell_yz(yIdx, zIdx): pull 8 corners using IJK convention
- update_for_y(t): lerp Y → 4 XZ samples
- update_for_x(t): lerp X → 2 Z samples
- update_for_z(t): lerp Z → scalar value

Per voxel: 7 lerps after slice fill. Heavy density evaluation only
happens at the 9³=729 corner samples per chunk.

Slice-filling from the graph (advance_cell_x) lands in Task 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `RuntimeEval` — per-chunk graph walker + cache substitution (TDD)

**Files:**
- Modify: `src/worldgen/density_graph.rs`

- [ ] **Step 4.1: Write failing tests**

```rust
    fn rt_for_chunk(origin: glam::IVec3) -> RuntimeEval {
        let c = cfg();
        let n = std::sync::Arc::new(noise());
        let graph = std::sync::Arc::new(DensityFn::build_from_config(&c.density));
        RuntimeEval::new(graph, origin, c.density.clone(), n)
    }

    #[test]
    fn runtime_eval_one_interpolator_per_interpolated_marker_collapsing_nested() {
        let rt = rt_for_chunk(glam::IVec3::ZERO);
        // Default graph has nested Interpolated (root + BaseNoise3D).
        // Runtime collapses to one shared interpolator.
        assert_eq!(rt.interpolator_count(), 1);
    }

    #[test]
    fn runtime_eval_advance_cell_x_fills_slice1() {
        let mut rt = rt_for_chunk(glam::IVec3::ZERO);
        rt.initialize_for_first_cell_x();
        rt.advance_cell_x(0);
        assert!(rt.peek_slice1_corner(0, 0, 0).is_finite());
    }

    /// At a cell corner the lerp degenerates (t=0 or t=1) to the
    /// corner sample, which equals the graph's direct evaluation.
    #[test]
    fn runtime_eval_at_corner_matches_direct() {
        let c = cfg();
        let n = std::sync::Arc::new(noise());
        let graph = std::sync::Arc::new(DensityFn::build_from_config(&c.density));
        let mut rt = RuntimeEval::new(
            graph.clone(), glam::IVec3::ZERO, c.density.clone(), n.clone(),
        );
        rt.initialize_for_first_cell_x();
        rt.advance_cell_x(0);
        rt.select_cell_yz(0, 0);
        rt.update_for_y(0.0);
        rt.update_for_x(0.0);
        rt.update_for_z(0.0);
        let interp = rt.value() as f32;
        let direct = graph.evaluate_direct(0, 0, 0, &c.density, &n);
        assert!(
            (interp - direct).abs() < 1e-3,
            "interp={interp} direct={direct}"
        );
    }
```

- [ ] **Step 4.2: Verify the tests fail** (RuntimeEval does not exist).

- [ ] **Step 4.3: Implement `RuntimeEval`**

```rust
/// Per-chunk runtime evaluator. Walks the static graph once at
/// construction, allocates a NoiseInterpolator per Interpolated
/// marker (collapsing nested Interpolated to one), and fills slices
/// on demand. The graph is read-only Arc; only the runtime state
/// lives here. Constructed fresh per `Generator::fill_chunk` call.
pub struct RuntimeEval {
    graph: std::sync::Arc<DensityFn>,
    config: crate::worldgen::config::DensityConfig,
    noise: std::sync::Arc<crate::worldgen::heightmap::DensityNoise>,
    chunk_origin: glam::IVec3,
    interpolators: Vec<NoiseInterpolator>,
    flat_caches: Vec<crate::worldgen::flat_cache::FlatCache2D<f32>>,
    current_cell_x: i32,
}

impl RuntimeEval {
    pub fn new(
        graph: std::sync::Arc<DensityFn>,
        chunk_origin: glam::IVec3,
        config: crate::worldgen::config::DensityConfig,
        noise: std::sync::Arc<crate::worldgen::heightmap::DensityNoise>,
    ) -> Self {
        let mut state = RuntimeEval {
            graph: graph.clone(), config, noise, chunk_origin,
            interpolators: Vec::new(),
            flat_caches: Vec::new(),
            current_cell_x: 0,
        };
        state.allocate_caches_for_subtree(&graph, false);
        state
    }

    /// Depth-first walk: one NoiseInterpolator per Interpolated
    /// marker (skipping nested), one FlatCache2D per FlatCache.
    /// CacheOnce / CacheAllInCell allocate no backing state in PR 5
    /// (PR 8 wires them).
    fn allocate_caches_for_subtree(&mut self, node: &DensityFn, inside_interpolated: bool) {
        match node {
            DensityFn::Marker { kind: MarkerKind::Interpolated, inner } => {
                if !inside_interpolated {
                    self.interpolators.push(NoiseInterpolator::new());
                }
                self.allocate_caches_for_subtree(inner, true);
            }
            DensityFn::Marker { kind: MarkerKind::FlatCache, inner } => {
                self.flat_caches.push(
                    crate::worldgen::flat_cache::FlatCache2D::<f32>::new(),
                );
                self.allocate_caches_for_subtree(inner, inside_interpolated);
            }
            DensityFn::Marker { inner, .. } => {
                self.allocate_caches_for_subtree(inner, inside_interpolated);
            }
            DensityFn::Add(a, b) | DensityFn::Mul(a, b) => {
                self.allocate_caches_for_subtree(a, inside_interpolated);
                self.allocate_caches_for_subtree(b, inside_interpolated);
            }
            DensityFn::Spline { input, .. } => {
                self.allocate_caches_for_subtree(input, inside_interpolated);
            }
            DensityFn::QuarterNegative { inner, .. } => {
                self.allocate_caches_for_subtree(inner, inside_interpolated);
            }
            _ => {}
        }
    }

    pub fn interpolator_count(&self) -> usize { self.interpolators.len() }

    #[cfg(test)]
    pub fn flat_cache_count(&self) -> usize { self.flat_caches.len() }

    /// Fill slice0 at cellX=0. Call once at chunk start.
    /// MC `initializeForFirstCellX`.
    pub fn initialize_for_first_cell_x(&mut self) {
        self.current_cell_x = 0;
        self.fill_slice(true, 0);
    }

    /// Fill slice1 with the YZ wall at cellX=idx+1. MC `advanceCellX`.
    pub fn advance_cell_x(&mut self, cell_x_index: i32) {
        self.fill_slice(false, cell_x_index + 1);
        self.current_cell_x = cell_x_index;
    }

    fn fill_slice(&mut self, slice0_target: bool, cell_x_global: i32) {
        if self.interpolators.is_empty() { return; }
        let wx = self.chunk_origin.x + cell_x_global * CELL_WIDTH as i32;
        // PR 5 has exactly one interpolator (default graph). PR 8
        // (cheese/spaghetti markers) needs a per-interpolator graph
        // walk — TODO when those land.
        let graph = self.graph.clone();
        let inner = Self::find_interpolated_inner(&graph)
            .expect("PR 5 graph always has an Interpolated root");
        let origin = self.chunk_origin;
        for cz in 0..CORNER_COUNT_Z {
            for cy in 0..CORNER_COUNT_Y {
                let wy = origin.y + cy as i32 * CELL_WIDTH as i32;
                let wz = origin.z + cz as i32 * CELL_WIDTH as i32;
                let v = inner.evaluate_direct(
                    wx, wy, wz, &self.config, &self.noise,
                ) as f64;
                if slice0_target {
                    self.interpolators[0].slice0[cz as usize][cy as usize] = v;
                } else {
                    self.interpolators[0].slice1[cz as usize][cy as usize] = v;
                }
            }
        }
    }

    fn find_interpolated_inner(node: &DensityFn) -> Option<&DensityFn> {
        match node {
            DensityFn::Marker { kind: MarkerKind::Interpolated, inner } => {
                Some(inner.as_ref())
            }
            _ => None,
        }
    }

    pub fn select_cell_yz(&mut self, y_idx: u32, z_idx: u32) {
        for i in &mut self.interpolators { i.select_cell_yz(y_idx, z_idx); }
    }
    pub fn update_for_y(&mut self, f: f64) {
        for i in &mut self.interpolators { i.update_for_y(f); }
    }
    pub fn update_for_x(&mut self, f: f64) {
        for i in &mut self.interpolators { i.update_for_x(f); }
    }
    pub fn update_for_z(&mut self, f: f64) {
        for i in &mut self.interpolators { i.update_for_z(f); }
    }
    pub fn value(&self) -> f64 { self.interpolators[0].value }
    pub fn swap_slices(&mut self) {
        for i in &mut self.interpolators { i.swap_slices(); }
    }

    #[cfg(test)]
    pub fn peek_slice1_corner(&self, cz: u32, cy: u32) -> f64 {
        self.interpolators[0].slice1[cz as usize][cy as usize]
    }
}
```

- [ ] **Step 4.4: Run tests** → 3 pass.

- [ ] **Step 4.5: Commit**

```bash
git add src/worldgen/density_graph.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): RuntimeEval — per-chunk graph walker

RuntimeEval owns one NoiseInterpolator per Interpolated marker
(nested collapse) and one FlatCache2D per FlatCache marker.
Construction walks the graph depth-first; lifecycle mirrors MC's
NoiseChunk (initialize_for_first_cell_x → advance_cell_x →
select_cell_yz → update_for_y/x/z → value → swap_slices).

PR 5 has exactly one interpolator (default graph). PR 8
(cheese/spaghetti) needs a per-interpolator graph walk — flagged
as a TODO in fill_slice.

Equivalence test: read at corner (0,0,0) matches direct evaluation
within 1e-3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Plumb the graph + Arc<DensityNoise> through `Generator`

**Files:**
- Modify: `src/worldgen/mod.rs`

- [ ] **Step 5.1: Write failing test**

```rust
    #[test]
    fn generator_holds_density_graph() {
        let g = Generator::new(42);
        match g.density_graph().as_ref() {
            crate::worldgen::density_graph::DensityFn::Marker {
                kind: crate::worldgen::density_graph::MarkerKind::Interpolated,
                ..
            } => {}
            _ => panic!("expected root Marker(Interpolated)"),
        }
    }
```

- [ ] **Step 5.2: Verify failure** (no `density_graph` field/method).

- [ ] **Step 5.3: Add the field, accessor, and Arc<DensityNoise>**

In `src/worldgen/mod.rs`:

```rust
pub struct Generator {
    // ... existing fields ...
    /// Density noise field. Wrapped in Arc so per-chunk RuntimeEval
    /// instances can share it without cloning the FBM state.
    density: std::sync::Arc<heightmap::DensityNoise>,
    /// Density expression graph, built once at construction from the
    /// snapshot config. Read-only across chunk fills (TODO: rebuild
    /// on hot-reload — the existing PR 2 reload pattern handles this
    /// by reconstructing the Generator).
    density_graph: std::sync::Arc<crate::worldgen::density_graph::DensityFn>,
}
```

In `Generator::new` (or `with_config` from PR 2):

```rust
let density = std::sync::Arc::new(heightmap::DensityNoise::new(seed));
// Build graph after config is available.
let cfg_snapshot = config.load(); // PR 2's ConfigHolder accessor
let density_graph = std::sync::Arc::new(
    crate::worldgen::density_graph::DensityFn::build_from_config(
        &cfg_snapshot.density,
    ),
);
Self {
    // ... existing fields ...
    density, density_graph,
}
```

```rust
impl Generator {
    pub fn density_graph(
        &self,
    ) -> std::sync::Arc<crate::worldgen::density_graph::DensityFn> {
        self.density_graph.clone()
    }
}
```

Existing call sites that use `self.density.evaluate_v2(...)` continue to work (Arc derefs transparently).

- [ ] **Step 5.4: Run the new test + worldgen suite**

`cargo test --lib worldgen 2>&1 | tail -10` → all existing tests still pass.

- [ ] **Step 5.5: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): plumb Arc<DensityNoise> + DensityFn graph through Generator

Generator now holds:
- density: Arc<DensityNoise> (shareable with per-chunk RuntimeEval)
- density_graph: Arc<DensityFn> built from the snapshot config

Existing call sites use Arc deref transparently. The graph captures
leaf constants at construction; hot-reload of cfg.density.* takes
effect on Generator rebuild (matches PR 2's reload pattern).

Step 6 wires RuntimeEval into fill_chunk.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Migrate `fill_chunk` to MC's cell loop (TDD with golden re-baseline)

**Files:**
- Modify: `src/worldgen/mod.rs`

This is the biggest task in PR 5. The new cell loop mirrors MC's `NoiseBasedChunkGenerator.doFill` exactly. Caves stay per-voxel.

- [ ] **Step 6.1: Put the golden hash test in sentinel mode**

In `src/worldgen/mod.rs`, change:

```rust
const GOLDEN_42_002: u64 = 0x886E_0C40_5650_12C7;
```

to:

```rust
const GOLDEN_42_002: u64 = 0xDEAD_BEEF_DEAD_BEEF;
```

Step 6.5 captures the new value.

- [ ] **Step 6.2: Write a determinism test**

```rust
    #[test]
    fn cell_loop_produces_same_chunk_twice() {
        let g = Generator::new(42);
        let mut a = DenseChunk::empty();
        let mut b = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut a);
        g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut b);
        assert_eq!(hash_chunk(&a), hash_chunk(&b));
    }
```

- [ ] **Step 6.3: Replace `fill_chunk` body with the MC cell loop**

Replace the existing `fill_chunk` body in `src/worldgen/mod.rs`:

```rust
pub fn fill_chunk(&self, coord: ChunkCoord, out: &mut DenseChunk) {
    let origin = coord.origin().0;
    let cfg = self.config_snapshot();
    let regions = self.gather_chunk_regions(coord);
    let chunk_max = origin + glam::IVec3::splat(CHUNK_DIM_U as i32);
    let cave_systems = regions.cave_systems_intersecting(origin, chunk_max);

    // Per-chunk runtime evaluator: walks the density graph, allocates
    // interpolator + flat caches.
    let mut rt = crate::worldgen::density_graph::RuntimeEval::new(
        self.density_graph.clone(),
        origin,
        cfg.density.clone(),
        self.density.clone(),
    );
    rt.initialize_for_first_cell_x();

    use crate::worldgen::density_graph::{
        CELL_COUNT_X, CELL_COUNT_Y, CELL_COUNT_Z, CELL_WIDTH,
    };

    // Pre-compute per-column terrain decisions (h_target, lake_rim,
    // biome) — 2D-only, evaluated once per (x,z).
    let mut columns: Vec<Vec<ColumnData>> =
        Vec::with_capacity(CHUNK_DIM_U as usize);
    for z in 0..CHUNK_DIM_U {
        let mut row = Vec::with_capacity(CHUNK_DIM_U as usize);
        for x in 0..CHUNK_DIM_U {
            let wx = origin.x + x as i32;
            let wz = origin.z + z as i32;
            row.push(self.column_data_with(wx, wz, &regions));
        }
        columns.push(row);
    }

    // Seed depth_below_surface from the voxel above this chunk's top.
    // (Same logic as pre-PR-5; prevents 32-block grass-cycle bug.)
    let mut depth_grid: Vec<Vec<Option<i32>>> =
        Vec::with_capacity(CHUNK_DIM_U as usize);
    for z in 0..CHUNK_DIM_U {
        let mut row = Vec::with_capacity(CHUNK_DIM_U as usize);
        for x in 0..CHUNK_DIM_U {
            let wx = origin.x + x as i32;
            let wz = origin.z + z as i32;
            let h_target = columns[z as usize][x as usize].height as f32;
            let above_top_wy = origin.y + CHUNK_DIM_U as i32;
            let above_d = self.density.evaluate_v2(
                h_target, wx, above_top_wy, wz, &cfg.density,
            );
            row.push(if above_d > 0.0 { Some(4) } else { None });
        }
        depth_grid.push(row);
    }

    // === MC's cell loop (NoiseBasedChunkGenerator.doFill, lines 399-457) ===
    //
    // for cellX in 0..cellCountX {
    //   advance_cell_x(cellX);          // fill slice1 at next YZ wall
    //   for cellZ in 0..cellCountZ {
    //     for cellY in (0..cellCountY).rev() {
    //       select_cell_yz(cellY, cellZ);
    //       for yInCell in (0..cellWidth).rev() {
    //         update_for_y(factorY);
    //         for xInCell in 0..cellWidth {
    //           update_for_x(factorX);
    //           for zInCell in 0..cellWidth {
    //             update_for_z(factorZ);
    //             // voxel work
    //           }
    //         }
    //       }
    //     }
    //   }
    //   swap_slices();                  // rotate slice1 → slice0
    // }
    //
    // Y descends because depth_below_surface flows top-down.

    let cell_width = CELL_WIDTH as u32;
    for cell_x in 0..CELL_COUNT_X {
        rt.advance_cell_x(cell_x as i32);
        for cell_z in 0..CELL_COUNT_Z {
            for cell_y in (0..CELL_COUNT_Y).rev() {
                rt.select_cell_yz(cell_y, cell_z);
                for y_in_cell in (0..cell_width).rev() {
                    let factor_y = y_in_cell as f64 / cell_width as f64;
                    rt.update_for_y(factor_y);
                    for x_in_cell in 0..cell_width {
                        let factor_x = x_in_cell as f64 / cell_width as f64;
                        rt.update_for_x(factor_x);
                        for z_in_cell in 0..cell_width {
                            let factor_z = z_in_cell as f64 / cell_width as f64;
                            rt.update_for_z(factor_z);

                            let lx = cell_x * cell_width + x_in_cell;
                            let ly = cell_y * cell_width + y_in_cell;
                            let lz = cell_z * cell_width + z_in_cell;
                            let wx = origin.x + lx as i32;
                            let wy = origin.y + ly as i32;
                            let wz = origin.z + lz as i32;

                            let col = &columns[lz as usize][lx as usize];
                            let h_target = col.height as f32;

                            // Interpolated density (pre-slide, pre-cave).
                            let interp_d = rt.value() as f32;
                            // Slides are y-only; apply post-interp.
                            let raw_density =
                                apply_slides(interp_d, wy, &cfg.density);

                            // Per-voxel caves (need 1-block res).
                            let approx_depth = col.height - wy;
                            let mut cave = 0.0_f32;
                            if !cave_systems.is_empty() && wy > CAVE_FLOOR_Y {
                                if approx_depth > CAVE_SURFACE_BUFFER {
                                    cave += caves::cave_sdf(wx, wy, wz, &cave_systems);
                                }
                                cave += caves::entrance_sdf(wx, wy, wz, &cave_systems);
                            }
                            if approx_depth > CAVE_SURFACE_BUFFER
                                && wy > CAVE_FLOOR_Y
                                && self.wormhole_noise.carve(wx, wy, wz)
                            {
                                cave += CAVE_SDF_INTENSITY;
                            }
                            let solid = (raw_density - cave) > 0.0;

                            let depth_state =
                                &mut depth_grid[lz as usize][lx as usize];
                            let block = if !solid {
                                *depth_state = None;
                                let in_lake =
                                    col.lake_rim.map_or(false, |r| wy <= r);
                                let in_ocean = col.height <= SEA_LEVEL
                                    && wy <= SEA_LEVEL;
                                if in_lake || in_ocean { Block::Water } else { Block::Air }
                            } else {
                                let d = depth_state.map(|d| d + 1).unwrap_or(0);
                                *depth_state = Some(d);
                                let near_surface = (h_target - wy as f32).abs()
                                    <= SURFACE_BAND as f32;
                                if col.is_cliff || !near_surface {
                                    Block::Stone
                                } else if d == 0 {
                                    surface_block(self.seed, wx, wz, wy, col)
                                } else if d <= 3 {
                                    Block::Dirt
                                } else {
                                    Block::Stone
                                }
                            };
                            out.set(LocalPos(UVec3::new(lx, ly, lz)), block);
                        }
                    }
                }
            }
        }
        rt.swap_slices();
    }

    self.add_trees(coord, out);
}
```

Add helpers in the same file (module-level functions):

```rust
/// Apply MC-style top/bottom slides post-interpolation. Mirrors
/// `heightmap::slide` from PR 2 but operates on a pre-computed
/// density rather than calling `evaluate_v2`.
fn apply_slides(
    density: f32,
    wy: i32,
    cfg: &crate::worldgen::config::DensityConfig,
) -> f32 {
    let top_start = cfg.y_max - cfg.slide_top_blocks;
    let top_f = ((wy - top_start) as f32 / cfg.slide_top_blocks as f32)
        .clamp(0.0, 1.0);
    let after_top = density + (cfg.slide_top_target - density) * top_f;
    let bot_end = cfg.y_min + cfg.slide_bottom_blocks;
    let bot_f = ((bot_end - wy) as f32 / cfg.slide_bottom_blocks as f32)
        .clamp(0.0, 1.0);
    after_top + (cfg.slide_bottom_target - after_top) * bot_f
}

/// Pick the surface block for the top-of-column voxel. Extracts the
/// inline logic that lived in fill_chunk pre-PR-5. PR 6 replaces
/// this with the surface-rules DSL.
fn surface_block(seed: u64, wx: i32, wz: i32, wy: i32, col: &ColumnData) -> Block {
    if wy >= SEA_LEVEL - 1 && wy <= SEA_LEVEL + 2 && !col.biome.snow_capped() {
        Block::Sand
    } else if wy >= SNOW_LINE {
        Block::Snow
    } else if col.biome.snow_capped() && wy >= SEA_LEVEL + COLD_SNOW_MIN_ABOVE_SEA {
        Block::Snow
    } else if col.biome == Biome::Desert {
        Block::Sand
    } else {
        let dist_to_boundary = 0.30 - col.desertness;
        if dist_to_boundary > 0.0 && dist_to_boundary < SAND_TRANSITION_BAND {
            let p = 0.5 * (1.0 - dist_to_boundary / SAND_TRANSITION_BAND);
            let roll = hash::mix_unit(seed, &[wx, wz, 71]);
            if roll < p { Block::Sand } else { Block::Grass }
        } else {
            Block::Grass
        }
    }
}
```

- [ ] **Step 6.4: Run the worldgen suite**

`cargo test --lib worldgen 2>&1 | tail -20`

Expected:
- `cell_loop_produces_same_chunk_twice` passes
- `golden_seed42_chunk_0_2_0` prints `UPDATE GOLDEN_42_002 to: 0x<hex>` (sentinel mode)
- `deep_underground_has_no_surface_blocks` passes (depth seeding preserved)
- `deep_caves_under_land_are_dry` passes (cave + aquifer unchanged)
- `cold_biome_caps_with_snow` passes
- `underground_chunk_has_both_caves_and_solid` passes
- `worldgen_fingerprint::fingerprint_hash_matches_pin` passes (2D heightmap untouched)
- All existing tests still pass

- [ ] **Step 6.5: Capture and re-pin the golden hash**

`cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 -- --nocapture 2>&1 | grep "UPDATE GOLDEN"`

Update `GOLDEN_42_002` to the printed hex and update the leading comment:

```rust
// Hash re-baselined for PR 5: sparse 4×4×4 cell-grid density
// interpolation. Subsurface voxels near the iso-surface may flip
// solid↔air vs the per-voxel path because interpolation smooths
// sharp noise features.
const GOLDEN_42_002: u64 = 0x<NEW_HEX>;
```

- [ ] **Step 6.6: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): migrate fill_chunk to MC-style cell loop

Per-voxel density evaluation is replaced by sparse 4×4×4 cell grid
+ trilinear interpolation, driven by RuntimeEval over the DensityFn
graph.

Cell loop (mirrors MC NoiseBasedChunkGenerator.doFill):
  for cellX in 0..8 { rt.advance_cell_x;
    for cellZ in 0..8 {
      for cellY in (0..8).rev() { rt.select_cell_yz;
        for yInCell in (0..4).rev() { rt.update_for_y;
          for xInCell in 0..4 { rt.update_for_x;
            for zInCell in 0..4 { rt.update_for_z;
              // read rt.value(), apply slides, subtract caves,
              // decide block from depth/biome state
  ... } } } } } rt.swap_slices(); }

Per-voxel work (unchanged): cave SDFs, entrance/wormhole, aquifer
rule (ocean/lake/dry), depth_below_surface state, surface selection.

Performance: 32768 → 729 density evals (~45×); benchmark in Task 9.

Golden hash re-baselined (subsurface voxel flips from interpolation
smoothing). worldgen_fingerprint unchanged (2D heightmap untouched).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: FlatCache allocation verification (TDD)

**Files:**
- Modify: `src/worldgen/density_graph.rs`

PR 5's default graph wraps `Spline(Constant(0.0))` in a FlatCache marker — the slot is allocated but its inner is constant. PR 3+ replaces the constant with real 2D noise, at which point the cache pays off. This task just verifies the slot allocation lands cleanly.

- [ ] **Step 7.1: Add test**

```rust
    #[test]
    fn flat_cache_slot_allocated_for_default_graph() {
        let rt = rt_for_chunk(glam::IVec3::ZERO);
        assert_eq!(rt.flat_cache_count(), 1);
    }
```

- [ ] **Step 7.2: Run** → passes (Task 4's `allocate_caches_for_subtree` already wires this).

- [ ] **Step 7.3: Document the PR 5 limitation**

Add a doc-comment in `density_graph.rs` near `allocate_caches_for_subtree`:

```rust
// PR 5 limitation: FlatCache slots are allocated but the runtime
// cache-read path is a no-op because the default graph wraps a
// Spline-of-Constant which doesn't depend on x,z. PR 3+ adds real
// 2D noise (continentalness, erosion) — at that point the cache
// fires at quart resolution (8×8 = 64 samples per chunk instead of
// 1024), a 16× saving.
```

- [ ] **Step 7.4: Commit**

```bash
git add src/worldgen/density_graph.rs
git commit -m "$(cat <<'EOF'
test(worldgen): verify FlatCache slot allocation in RuntimeEval

The default graph's offset-spline marker creates a FlatCache slot
(via Task 4's allocate_caches_for_subtree). PR 5 wraps a Spline of
Constant(0.0), so the cache read is a no-op — but the slot is in
place and the test verifies allocation, so PR 3+ wiring is purely
"add the noise expression and the cache starts firing".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Visual + behavioral sanity tests (TDD)

**Files:**
- Modify: `src/worldgen/mod.rs`

- [ ] **Step 8.1: Write the sanity tests**

```rust
    /// Catches sign flips and gross mis-tuning. A surface chunk
    /// should remain roughly 10-90% solid.
    #[test]
    fn interpolated_chunk_solid_count_in_band() {
        let g = Generator::new(42);
        let mut c = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut c);
        let solid = c.blocks.iter()
            .filter(|b| !matches!(b, Block::Air | Block::Water)).count();
        assert!(
            solid > CHUNK_VOL / 10 && solid < CHUNK_VOL * 9 / 10,
            "surface chunk had {solid} solid voxels"
        );
    }

    /// Cell boundary continuity: the rate of solid↔air disagreement
    /// at cell boundaries (x=3↔4, 7↔8, ...) should not be drastically
    /// higher than the interior rate. A failure here indicates a
    /// slice swap or update_for_x bug (would show as vertical seams
    /// every 4 blocks).
    #[test]
    fn cell_boundary_voxels_continuous() {
        let g = Generator::new(42);
        let mut chunk = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut chunk);
        let mut boundary = 0;
        let mut interior = 0;
        for z in 0..CHUNK_DIM_U {
            for y in 0..CHUNK_DIM_U {
                // Boundary at x=3↔4, 7↔8, ..., 27↔28.
                for cell_x in 0..7 {
                    let xl = cell_x * 4 + 3;
                    let xr = cell_x * 4 + 4;
                    let l = chunk.blocks[LocalPos(UVec3::new(xl, y, z)).to_index()];
                    let r = chunk.blocks[LocalPos(UVec3::new(xr, y, z)).to_index()];
                    if !matches!(l, Block::Air | Block::Water)
                        != !matches!(r, Block::Air | Block::Water)
                    {
                        boundary += 1;
                    }
                }
                // Interior comparison at x=1↔2, x=5↔6, ...
                for cell_x in 0..8 {
                    let xl = cell_x * 4 + 1;
                    let xr = cell_x * 4 + 2;
                    let l = chunk.blocks[LocalPos(UVec3::new(xl, y, z)).to_index()];
                    let r = chunk.blocks[LocalPos(UVec3::new(xr, y, z)).to_index()];
                    if !matches!(l, Block::Air | Block::Water)
                        != !matches!(r, Block::Air | Block::Water)
                    {
                        interior += 1;
                    }
                }
            }
        }
        let b_rate = boundary as f64 / 7168.0;   // 7 × 32 × 32
        let i_rate = interior as f64 / 8192.0;   // 8 × 32 × 32
        assert!(
            b_rate < i_rate * 3.0 + 0.01,
            "boundary disagreement {b_rate:.4} >> interior {i_rate:.4}"
        );
    }
```

- [ ] **Step 8.2: Run** → both pass.

If `cell_boundary_voxels_continuous` fails, diagnose:
1. Visually inspect a chunk for 4-block seams.
2. Check `advance_cell_x` runs before the inner cellZ/cellY loops.
3. Check `swap_slices` runs at end-of-cellX, not before.

- [ ] **Step 8.3: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
test(worldgen): sanity tests for PR 5 cell-loop interpolation

interpolated_chunk_solid_count_in_band — catches sign flips and
gross mis-tuning.

cell_boundary_voxels_continuous — counts solid↔air disagreement at
cell boundaries (x=3↔4, 7↔8...) and at interior pairs; asserts the
boundary rate isn't drastically higher than the interior rate.
Catches slice swap and update_for_x bugs that would show as
vertical seams every 4 blocks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Performance benchmark (TDD with measurable speedup)

**Files:**
- Modify: `src/worldgen/mod.rs`

No new crate — use `std::time::Instant`.

- [ ] **Step 9.1: Add the benchmark**

```rust
    /// Per-chunk timing benchmark. PR 5 claims 3-5× chunk-gen speedup
    /// from sparse cell-grid density evaluation. We assert ≤ a budget
    /// derived from the measurement at PR 5's commit (set after
    /// running step 9.3).
    ///
    /// Run: `cargo test --release --lib benchmark -- --ignored --nocapture`
    ///
    /// Recorded numbers (filled in after step 9.3):
    /// - Pre-PR-5 baseline (main branch, same machine): TBD ms/chunk
    /// - PR 5 at this commit: TBD ms/chunk
    /// - Speedup: TBD×
    #[test]
    #[ignore = "perf benchmark — run with --release --ignored"]
    fn benchmark_chunk_gen_perf() {
        const TARGET_MAX_MS_PER_CHUNK: u128 = 60; // tune in step 9.3
        let g = Generator::new(42);
        // Warm-up: populate caches.
        for cx in 0..4 {
            let mut c = DenseChunk::empty();
            g.fill_chunk(ChunkCoord(IVec3::new(cx, 2, 0)), &mut c);
        }
        let n: u128 = 16;
        let start = std::time::Instant::now();
        for cx in 0..(n as i32) {
            let mut c = DenseChunk::empty();
            g.fill_chunk(ChunkCoord(IVec3::new(cx, 2, 0)), &mut c);
        }
        let elapsed_ms = start.elapsed().as_millis();
        let per_chunk_ms = elapsed_ms / n;
        println!("PR 5 chunk gen: {n} chunks in {elapsed_ms} ms = {per_chunk_ms} ms/chunk");
        assert!(
            per_chunk_ms <= TARGET_MAX_MS_PER_CHUNK,
            "PR 5 chunk gen {per_chunk_ms} ms/chunk > {TARGET_MAX_MS_PER_CHUNK} ms budget"
        );
    }
```

- [ ] **Step 9.2: Run the benchmark**

`cargo test --release --lib worldgen::tests::benchmark -- --ignored --nocapture 2>&1 | tail -5`

Expected output: `PR 5 chunk gen: 16 chunks in <N> ms = <N/16> ms/chunk`.

- [ ] **Step 9.3: Record baseline + PR 5 numbers**

Run the benchmark on `main` (pre-PR-5) for comparison, then update the test's comment with measured values and tighten `TARGET_MAX_MS_PER_CHUNK` to ~1.5× the PR 5 measurement (catches future regressions ≥50%). Assert at least 2× speedup vs the recorded baseline.

- [ ] **Step 9.4: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
test(worldgen): benchmark — PR 5 chunk gen ≥ 2× faster than pre-PR-5

Per-chunk timing benchmark records the measured speedup of PR 5's
cell-grid interpolation over the pre-PR-5 per-voxel path. Gated
behind --ignored (manual `cargo test --release --lib benchmark --
--ignored`).

Recorded in the test comment:
- Pre-PR-5 baseline: ~<N> ms/chunk
- PR 5: ~<N> ms/chunk
- Speedup: <N>×

TARGET_MAX_MS_PER_CHUNK set ~1.5× the PR 5 measurement to catch
≥50% regressions in future PRs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Final verification

**Files:** none (verification only)

- [ ] **Step 10.1: Run the full test suite**

`cargo test 2>&1 | tail -20`

Expected: every test passes, including:
- All `worldgen::density_graph::*` tests (15+)
- All existing `worldgen::*` tests (50+) including the re-baselined `golden_seed42_chunk_0_2_0`
- New `cell_loop_produces_same_chunk_twice`, `interpolated_chunk_solid_count_in_band`, `cell_boundary_voxels_continuous`
- `worldgen_fingerprint::fingerprint_hash_matches_pin` unchanged (2D heightmap untouched)

- [ ] **Step 10.2: Run the benchmark**

`cargo test --release --lib worldgen::tests::benchmark -- --ignored --nocapture 2>&1 | grep "ms/chunk"`

Within budget.

- [ ] **Step 10.3: Visual smoke test**

Boot the game. Check:
- Terrain looks like pre-PR-5 (chunk hash changed but visual shape similar)
- No vertical/horizontal seams every 4 blocks (would indicate slice swap or update ordering bug)
- Caves still navigable (per-voxel path unchanged)
- Slides work (clean sky ceiling near y=140, solid floor near y=-120)
- Edit `assets/worldgen/default.ron`: change `factor` to `2.0`, save, rebuild Generator (PR 2's reload mechanism). New chunks reflect the change.

Diagnosis for failures:
- **4-block seams**: slice swap or update_for_x wrong; re-check Task 6 step 6.3
- **Blurred caves**: caves accidentally interpolated; verify `cave` is computed inside the innermost loop
- **Dramatically different terrain**: Task 2 step 2.3 equivalence broke silently; re-run that test
- **Slow chunk-gen**: see Task 9 step 9.2 diagnostics

- [ ] **Step 10.4: Final commit (no-op if nothing changed)**

If steps 10.1–10.3 prompted RON tuning, commit it. Otherwise PR 5 is complete.

---

## Out of scope for PR 5 (deferred to later PRs)

- **Continentalness / erosion / ridges as real 2D noise** (PR 3). PR 5 uses the existing `cfg.offset_spline` (default `Constant(0.0)`).
- **Multi-noise biome lookup with R-tree** (PR 4). PR 5 doesn't change biome assignment.
- **Surface rules DSL** (PR 6). PR 5's `surface_block` helper is a refactor convenience; PR 6 replaces it with the rule-DSL evaluator.
- **Real aquifer** (PR 7). PR 5 preserves the primitive ocean/lake rule.
- **Noise carver layers (cheese, spaghetti)** (PR 8). New `DensityFn` subtrees under separate Interpolated markers; PR 5's runtime supports them via the `interpolators` Vec but no such markers exist yet.
- **Multiple Interpolated subtrees in one graph.** PR 5 handles nested Interpolated (collapses to one) but treats the canonical graph as having exactly ONE Interpolated root. PR 8 will need to extend `find_interpolated_inner` to walk all of them.
- **`CacheAllInCell` and `CacheOnce` runtime behavior.** PR 5 allocates slots but the runtime fold is a no-op (inner evaluated directly). PR 8's carver layers wire real behavior.
- **JSON/RON hot-reload of the graph topology.** PR 5's graph is built in Rust code from `WorldgenConfig`. Hot-reload of `cfg.density.*` requires a Generator rebuild (the existing PR 2 pattern).

## Plan self-review notes

- All 10 tasks have concrete code in every step. No "TBD" or "fill in details" except where deliberately deferred to PR 3/8 (clearly marked).
- Type names consistent: `DensityFn`, `MarkerKind`, `NoiseInterpolator`, `RuntimeEval`, plus PR 2 types (`WorldgenConfig`, `DensityConfig`, `ConfigHolder`, `CubicSpline`, `FlatCache2D`, `DensityNoise::evaluate_v2`).
- Cell sizes are architectural constants in `density_graph.rs` (`CELL_WIDTH`, `CELL_COUNT_*`, `CORNER_COUNT_*`), NOT in `tuning.rs` per the spec.
- Each task ends with a commit boundary, heredoc + Co-Authored-By trailer.
- Golden hash management: Task 6 step 6.1 puts test in sentinel mode; step 6.5 captures and re-pins.
- Equivalence to `DensityNoise::evaluate_v2` asserted in Task 2 step 2.1 — 32×16×32 lattice within 1e-3 tolerance, excluding slide bands.
- Caves stay per-voxel — explicitly documented in Task 6 step 6.3's commit message and cell-loop body comments.
- Sliding YZ wall + swap_slices algorithm documented in Task 3 (type-level doc-comment) and the cell loop in Task 6 step 6.3 explicitly mirrors MC's `doFill`.
- Performance: Task 9's benchmark uses `std::time::Instant` (no new crate); assertion budget tightens after measurement.
- FlatCache wired for allocation in Task 7; real load arrives with PR 3+ (`Constant(0.0)` → real 2D noise).
- `DensityNoise` becomes `Arc<DensityNoise>` inside `Generator` (Task 5 step 5.3) to share with `RuntimeEval`.
- PR 5 LOC estimate: ~500 lines (density_graph.rs ~400 lines, fill_chunk refactor + helpers ~100 lines net change).
- 10 tasks total; TDD steps in tasks 1-9; task 10 is verification only.
