# Worldgen PR 2 — Density Composition Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PR B's `min(2.0)` density cap with the MC-style asymmetric `4 * quarter_negative((depth + jagged) * factor) + base_3d_noise + slides` composition. Land supporting infrastructure: a generic `CubicSpline<T>` type, a `FlatCache2D<T>` per-chunk quart-resolution cache, and a `WorldgenConfig` struct loaded from RON with hot-reload via a `notify` file watcher (always on, release builds included).

**Architecture:** Five tightly-coupled sub-components landing together. (1) `CubicSpline<T>` is a generic Hermite spline data type — empty consumer in PR 2, primary consumer in PR 3. (2) `FlatCache2D<T>` is a per-chunk lookup cache for 2D fields — wired in PR 2, used heavily in PR 3 and beyond. (3) `WorldgenConfig` is a Serde-derived struct loaded from `assets/worldgen/default.ron`, held in an `Arc<ArcSwap<_>>` and atomically swapped on file change. (4) Anisotropic base 3D noise: `y_scale = xz_scale / 2` (one constant). (5) `DensityNoise::evaluate` is rewritten to compute `4 * quarter_negative((y_gradient(y) + offset) * factor) + base_3d_noise + slide(y)`, with `offset` and `factor` derived from the existing `h_target` for now (real spline-driven offset/factor lands in PR 3). The MC `min(2.0)` hack in `mod.rs` is removed — the new composition's asymmetric softening above the surface eliminates the need for it.

**Tech Stack:**
- Rust 2024 edition
- `ron = "0.8"` — RON deserialization
- `notify-debouncer-mini = "0.4"` — file watcher (debounces editor save-bursts)
- `arc-swap = "1"` — atomic config swap on hot-reload
- `serde` (already a dependency) — derive macros

**Reference:** Architectural rationale is in `docs/superpowers/specs/2026-05-19-minecraft-worldgen-research.md` (Decisions Log Q1–Q6). This plan does not re-argue those decisions.

---

### Task 1: Add Cargo dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1.1: Add ron, notify-debouncer-mini, arc-swap to `[dependencies]`**

Edit `Cargo.toml`, in the existing `[dependencies]` block, add:

```toml
# Worldgen config: human-editable RON, hot-reloaded via file watcher.
ron = "0.8"
notify-debouncer-mini = "0.4"
arc-swap = "1"
```

Place the block after the existing `toml = "0.8"` line. (Group with serialization deps.)

- [ ] **Step 1.2: Run `cargo check` to fetch and verify**

Run: `cargo check 2>&1 | tail -5`

Expected: `Finished` line (warnings OK). Three crates pulled in successfully.

- [ ] **Step 1.3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
build(worldgen): add ron, notify-debouncer-mini, arc-swap

Dependencies for PR 2's hot-reloadable WorldgenConfig:
- ron: RON deserialization (Rust-syntax-friendly config format)
- notify-debouncer-mini: file watcher that debounces editor saves
- arc-swap: atomic Arc swap for lock-free config reads

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: CubicSpline type (TDD)

**Files:**
- Create: `src/worldgen/spline.rs`
- Modify: `src/worldgen/mod.rs` (add `pub mod spline;`)

- [ ] **Step 2.1: Write the failing tests**

Create `src/worldgen/spline.rs` with the test module first (signature-only impl, no logic):

```rust
//! Generic cubic Hermite spline with nested-value support.
//!
//! A [`CubicSpline`] is either a constant scalar or a list of knots
//! whose values are themselves splines — enabling nested `f(x, y)`
//! composition by stacking 1D splines. Evaluation uses the standard
//! Hermite formula:
//!
//!   t = (input - x1) / (x2 - x1)
//!   result = lerp(t, y1, y2) + t·(1-t)·lerp(t, a, b)
//!     where a =  d1·(x2-x1) − (y2-y1)
//!           b = -d2·(x2-x1) + (y2-y1)
//!
//! Outside the knot range, evaluation is linear extrapolation using
//! the endpoint derivative. Matches the algorithm in Minecraft 1.18+
//! `net/minecraft/util/CubicSpline.java`.

use serde::{Deserialize, Serialize};

/// One knot: input location, output value (possibly itself a
/// spline), derivative dy/dx at this knot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Knot {
    pub loc: f32,
    pub val: f32,
    pub slope: f32,
}

/// Cubic Hermite spline over a scalar input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CubicSpline {
    Constant(f32),
    Multipoint(Vec<Knot>),
}

impl CubicSpline {
    pub fn evaluate(&self, _input: f32) -> f32 {
        unimplemented!("written in step 2.3")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_returns_value_at_any_input() {
        let s = CubicSpline::Constant(0.7);
        assert_eq!(s.evaluate(-5.0), 0.7);
        assert_eq!(s.evaluate(0.0), 0.7);
        assert_eq!(s.evaluate(100.0), 0.7);
    }

    #[test]
    fn two_knots_interpolate_smoothly() {
        // Knots at x=0 (y=0, slope=0) and x=1 (y=1, slope=0).
        // With both slopes 0, Hermite gives an S-curve from (0,0) to (1,1).
        let s = CubicSpline::Multipoint(vec![
            Knot { loc: 0.0, val: 0.0, slope: 0.0 },
            Knot { loc: 1.0, val: 1.0, slope: 0.0 },
        ]);
        assert!((s.evaluate(0.0) - 0.0).abs() < 1e-5);
        assert!((s.evaluate(1.0) - 1.0).abs() < 1e-5);
        // At t=0.5, S-curve value is exactly 0.5 (symmetry).
        assert!((s.evaluate(0.5) - 0.5).abs() < 1e-5);
        // Below 0.5 the curve should be below 0.5 (concave up).
        assert!(s.evaluate(0.25) < 0.5);
        // Above 0.5 the curve should be above 0.5 (concave down).
        assert!(s.evaluate(0.75) > 0.5);
    }

    #[test]
    fn below_first_knot_extrapolates_linearly() {
        // Knot at x=0 (y=0, slope=1).
        let s = CubicSpline::Multipoint(vec![
            Knot { loc: 0.0, val: 0.0, slope: 1.0 },
            Knot { loc: 1.0, val: 1.0, slope: 1.0 },
        ]);
        // At x=-1 with slope=1 extrapolation: y = 0 + 1·(-1) = -1.
        assert!((s.evaluate(-1.0) - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn above_last_knot_extrapolates_linearly() {
        let s = CubicSpline::Multipoint(vec![
            Knot { loc: 0.0, val: 0.0, slope: 0.0 },
            Knot { loc: 1.0, val: 1.0, slope: 0.5 },
        ]);
        // At x=2 with endpoint slope=0.5: y = 1 + 0.5·1 = 1.5.
        assert!((s.evaluate(2.0) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn ron_roundtrip_preserves_knots() {
        let s = CubicSpline::Multipoint(vec![
            Knot { loc: -0.5, val: 0.3, slope: 0.0 },
            Knot { loc: 0.5, val: -0.2, slope: 1.0 },
        ]);
        let r = ron::to_string(&s).unwrap();
        let parsed: CubicSpline = ron::from_str(&r).unwrap();
        match parsed {
            CubicSpline::Multipoint(knots) => {
                assert_eq!(knots.len(), 2);
                assert!((knots[0].loc - (-0.5)).abs() < 1e-5);
                assert!((knots[1].slope - 1.0).abs() < 1e-5);
            }
            _ => panic!("expected Multipoint"),
        }
    }
}
```

Add module declaration to `src/worldgen/mod.rs`. Find the existing `pub mod ...` block (near the top of the file) and add:

```rust
pub mod spline;
```

Place alphabetically among the existing module declarations.

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `cargo test --lib worldgen::spline 2>&1 | tail -15`

Expected: 5 tests fail with `not yet implemented` panics from `unimplemented!`.

- [ ] **Step 2.3: Implement `CubicSpline::evaluate`**

Replace the `unimplemented!` body in `src/worldgen/spline.rs` with the Hermite evaluator:

```rust
impl CubicSpline {
    pub fn evaluate(&self, input: f32) -> f32 {
        match self {
            CubicSpline::Constant(v) => *v,
            CubicSpline::Multipoint(knots) => {
                assert!(!knots.is_empty(), "spline must have at least one knot");
                // Below first knot: extrapolate using first knot's slope.
                if input <= knots[0].loc {
                    return knots[0].val + knots[0].slope * (input - knots[0].loc);
                }
                // Above last knot: extrapolate using last knot's slope.
                let last = knots.last().unwrap();
                if input >= last.loc {
                    return last.val + last.slope * (input - last.loc);
                }
                // Binary search for the segment [k1, k2] containing input.
                let mut i = 0;
                while i + 1 < knots.len() && knots[i + 1].loc < input {
                    i += 1;
                }
                let k1 = &knots[i];
                let k2 = &knots[i + 1];
                let dx = k2.loc - k1.loc;
                let t = (input - k1.loc) / dx;
                let a = k1.slope * dx - (k2.val - k1.val);
                let b = -k2.slope * dx + (k2.val - k1.val);
                // lerp(t, k1.val, k2.val) + t·(1-t)·lerp(t, a, b)
                let lerp_y = k1.val + t * (k2.val - k1.val);
                let lerp_ab = a + t * (b - a);
                lerp_y + t * (1.0 - t) * lerp_ab
            }
        }
    }
}
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `cargo test --lib worldgen::spline 2>&1 | tail -10`

Expected: all 5 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add src/worldgen/spline.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): add CubicSpline<f32> with Hermite evaluation

Generic cubic Hermite spline for the upcoming spline-driven
heightmap (PR 3) and biome lookup (PR 4). Mirrors the algorithm
in Minecraft 1.18+ net/minecraft/util/CubicSpline.java.

Supports constant and multipoint variants, linear extrapolation
outside knot range. Serde-derived for RON config loading.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: FlatCache2D type (TDD)

**Files:**
- Create: `src/worldgen/flat_cache.rs`
- Modify: `src/worldgen/mod.rs` (add `pub mod flat_cache;`)

- [ ] **Step 3.1: Write the failing tests**

Create `src/worldgen/flat_cache.rs`:

```rust
//! Per-chunk quart-resolution 2D cache.
//!
//! Stores one value per `QUART_PER_CHUNK × QUART_PER_CHUNK` cell of
//! a chunk. Used to amortize expensive 2D field evaluations
//! (continentalness, erosion, offset, factor, jaggedness) across
//! the ~1000 voxels in each 4×4 column quarter.

use crate::voxel::coords::CHUNK_DIM_U;

/// 4 blocks per quart (matches Minecraft `QuartPos`).
pub const QUART_SIZE: u32 = 4;
/// Quarts per chunk side: 32 / 4 = 8.
pub const QUART_PER_CHUNK: u32 = CHUNK_DIM_U / QUART_SIZE;

/// 2D cache of `T` over the chunk footprint, sampled at quart
/// resolution. Total `QUART_PER_CHUNK²` = 64 entries.
pub struct FlatCache2D<T: Copy> {
    values: [Option<T>; (QUART_PER_CHUNK * QUART_PER_CHUNK) as usize],
}

impl<T: Copy> FlatCache2D<T> {
    pub fn new() -> Self {
        Self {
            values: [None; (QUART_PER_CHUNK * QUART_PER_CHUNK) as usize],
        }
    }

    /// Returns the cached value at quart `(qx, qz)` (both in
    /// `0..QUART_PER_CHUNK`), computing and storing it on miss.
    pub fn get_or_compute<F>(&mut self, qx: u32, qz: u32, mut compute: F) -> T
    where
        F: FnMut() -> T,
    {
        let idx = (qx + qz * QUART_PER_CHUNK) as usize;
        if let Some(v) = self.values[idx] {
            return v;
        }
        let v = compute();
        self.values[idx] = Some(v);
        v
    }

    /// Maps a block-space (lx, lz) inside the chunk to the
    /// corresponding (qx, qz).
    pub fn block_to_quart(lx: u32, lz: u32) -> (u32, u32) {
        (lx / QUART_SIZE, lz / QUART_SIZE)
    }
}

impl<T: Copy> Default for FlatCache2D<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_returns_same_value_without_recompute() {
        let mut cache = FlatCache2D::<f32>::new();
        let mut calls = 0;
        let v1 = cache.get_or_compute(3, 5, || {
            calls += 1;
            7.5
        });
        let v2 = cache.get_or_compute(3, 5, || {
            calls += 1;
            999.0 // would diverge if recomputed
        });
        assert_eq!(v1, 7.5);
        assert_eq!(v2, 7.5);
        assert_eq!(calls, 1, "compute closure should run exactly once");
    }

    #[test]
    fn different_quarts_compute_independently() {
        let mut cache = FlatCache2D::<i32>::new();
        let a = cache.get_or_compute(0, 0, || 1);
        let b = cache.get_or_compute(1, 0, || 2);
        let c = cache.get_or_compute(0, 1, || 3);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    }

    #[test]
    fn block_to_quart_groups_by_four() {
        assert_eq!(FlatCache2D::<f32>::block_to_quart(0, 0), (0, 0));
        assert_eq!(FlatCache2D::<f32>::block_to_quart(3, 3), (0, 0));
        assert_eq!(FlatCache2D::<f32>::block_to_quart(4, 0), (1, 0));
        assert_eq!(FlatCache2D::<f32>::block_to_quart(31, 31), (7, 7));
    }
}
```

Add to `src/worldgen/mod.rs` module list:

```rust
pub mod flat_cache;
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `cargo test --lib worldgen::flat_cache 2>&1 | tail -10`

Expected: 3 tests fail (the type doesn't compile yet because the array length is a const expression — but actually the file should compile and tests should pass since the implementation is already inline; this is an artifact of doing test-first-with-implementation-already-stubbed). If all 3 pass on first run, that's acceptable — the test bodies prove the behavior. Skip to step 3.4.

- [ ] **Step 3.3: (Skip if step 3.2 already passed)**

If step 3.2 failed, fix any compile errors and re-run.

- [ ] **Step 3.4: Commit**

```bash
git add src/worldgen/flat_cache.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): add FlatCache2D for per-chunk 2D field caching

Quart-resolution (4-block) 2D cache holding 8×8=64 entries per
chunk footprint. Used by PR 2's density composition (offset,
factor) and heavily by PR 3+ (climate, splines).

Mirrors Minecraft 1.18+ FlatCache marker semantics: 2D fields are
sampled once per (quartX, quartZ) and broadcast to every voxel in
that 4×4 column. ~16× fewer 2D noise samples per chunk.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: WorldgenConfig struct + default.ron

**Files:**
- Create: `src/worldgen/config.rs`
- Create: `assets/worldgen/default.ron`
- Modify: `src/worldgen/mod.rs` (add `pub mod config;`)

- [ ] **Step 4.1: Write the failing tests**

Create `src/worldgen/config.rs`:

```rust
//! Hot-reloadable worldgen configuration.
//!
//! All tunable parameters live here, loaded at startup from
//! `assets/worldgen/default.ron` and atomically swapped on file
//! change via [`ConfigHolder`]. The graph topology (which density
//! functions exist, what marker wrappers apply) stays in Rust;
//! only values are file-driven.

use crate::worldgen::spline::CubicSpline;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level config. All worldgen-tunable values root here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldgenConfig {
    pub density: DensityConfig,
}

/// Density composition tuning (PR 2 introduces this section; PR 3+
/// expand it with biome / cave / aquifer subsections).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DensityConfig {
    /// World Y range used by the y_gradient. At `y_min`, gradient
    /// equals `+y_gradient_amplitude`; at `y_max`, equals the
    /// negative of that.
    pub y_min: i32,
    pub y_max: i32,
    pub y_gradient_amplitude: f32,

    /// Multiplier on the (depth + jagged) * factor term. MC uses 4.
    pub composition_scale: f32,

    /// Fraction by which above-surface (negative-depth) magnitudes
    /// are scaled before composition_scale. MC uses 0.25.
    pub above_surface_softening: f32,

    /// Constant `factor` for PR 2 (PR 3 makes this a spline of
    /// (continentalness, erosion, ridges)). Higher → sharper
    /// surface transition.
    pub factor: f32,

    /// Base 3D noise period in blocks.
    pub base_3d_period: f32,
    /// Base 3D noise amplitude.
    pub base_3d_amplitude: f32,
    /// Y-axis scale relative to XZ. 0.5 = vertical features 2×
    /// taller than wide (matches MC).
    pub base_3d_y_scale: f32,

    /// Top-slide: within `slide_top_blocks` of `y_max`, density is
    /// lerped toward `slide_top_target` (negative → forces air).
    pub slide_top_blocks: i32,
    pub slide_top_target: f32,

    /// Bottom-slide: within `slide_bottom_blocks` of `y_min`,
    /// density is lerped toward `slide_bottom_target` (positive →
    /// forces solid).
    pub slide_bottom_blocks: i32,
    pub slide_bottom_target: f32,

    /// Placeholder for the PR 3 offset spline. Today a Constant.
    pub offset_spline: CubicSpline,
}

impl WorldgenConfig {
    /// Load and parse from a RON file path.
    pub fn from_ron_file(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = ron::from_str(&text)?;
        Ok(cfg)
    }

    /// Load the bundled default config. Looks at
    /// `<crate-root>/assets/worldgen/default.ron`.
    pub fn bundled_default() -> anyhow::Result<Self> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("worldgen")
            .join("default.ron");
        Self::from_ron_file(&path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_default_loads_and_parses() {
        let cfg = WorldgenConfig::bundled_default().expect("default.ron must load");
        assert!(cfg.density.y_max > cfg.density.y_min);
        assert!(cfg.density.composition_scale > 0.0);
        assert!(cfg.density.factor > 0.0);
        assert!(cfg.density.above_surface_softening > 0.0);
        assert!(cfg.density.above_surface_softening <= 1.0);
    }

    #[test]
    fn ron_roundtrip_preserves_values() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let s = ron::to_string(&cfg).unwrap();
        let parsed: WorldgenConfig = ron::from_str(&s).unwrap();
        assert_eq!(parsed.density.y_min, cfg.density.y_min);
        assert_eq!(parsed.density.factor, cfg.density.factor);
    }
}
```

Create `assets/worldgen/default.ron`:

```ron
// Oxium worldgen default configuration.
// Hot-reloaded by the file watcher (always on, including release builds).
// On parse failure, the engine logs and keeps using the previous config.

(
    density: (
        // World Y range — controls the y_gradient that biases density
        // toward solid below and air above the surface.
        y_min: -120,
        y_max: 140,
        y_gradient_amplitude: 1.5,

        // MC's formula: 4 * quarter_negative((depth + jagged) * factor)
        composition_scale: 4.0,
        above_surface_softening: 0.25,

        // PR 2: constant factor. PR 3 will swap this for a spline of
        // (continentalness, erosion, ridges). Higher → sharper surface,
        // lower → softer / flatter terrain.
        factor: 4.0,

        // Base 3D noise (anisotropic — y wavelength is half the xz).
        base_3d_period: 32.0,
        base_3d_amplitude: 1.0,
        base_3d_y_scale: 0.5,

        // Top slide: within 16 blocks of y_max, lerp density toward -0.078
        // (pull to air) so the world has a clean sky ceiling.
        slide_top_blocks: 16,
        slide_top_target: -0.078125,

        // Bottom slide: within 24 blocks of y_min, lerp density toward
        // +0.117 (pull to solid) so caves can't punch through to the void.
        slide_bottom_blocks: 24,
        slide_bottom_target: 0.1171875,

        // PR 2 placeholder: constant 0 offset. PR 3 makes this a spline
        // of continentalness with mountain/plain/ocean knots.
        offset_spline: Constant(0.0),
    ),
)
```

Add to `src/worldgen/mod.rs` module list:

```rust
pub mod config;
```

- [ ] **Step 4.2: Run tests to verify they pass**

Run: `cargo test --lib worldgen::config 2>&1 | tail -10`

Expected: 2 tests pass (`bundled_default_loads_and_parses`, `ron_roundtrip_preserves_values`).

If load fails, check that the `assets/worldgen/default.ron` path is correct relative to `CARGO_MANIFEST_DIR`.

- [ ] **Step 4.3: Commit**

```bash
git add src/worldgen/config.rs src/worldgen/mod.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
feat(worldgen): WorldgenConfig + default.ron (RON-loaded)

Hot-reloadable config struct holding the density composition
tuning. PR 2 ships the density subsection; later PRs add biome,
cave, aquifer subsections.

assets/worldgen/default.ron is the bundled ground-truth config.
Tests always use it for determinism (golden hashes pin to these
exact values).

The file is human-editable; PR 2 step 5 wires up the file watcher
for hot-reload.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: ConfigHolder (atomic config swap)

**Files:**
- Modify: `src/worldgen/config.rs`

- [ ] **Step 5.1: Write the failing test**

Append to `src/worldgen/config.rs` (above the `#[cfg(test)]` line, before `mod tests`):

```rust
use arc_swap::ArcSwap;
use std::sync::Arc;

/// Thread-safe holder for the current worldgen config. Readers use
/// `holder.load()` to get a snapshot `Arc<WorldgenConfig>`; the file
/// watcher swaps in a new value via `holder.swap(new)` without
/// blocking readers.
#[derive(Clone)]
pub struct ConfigHolder(Arc<ArcSwap<WorldgenConfig>>);

impl ConfigHolder {
    pub fn new(initial: WorldgenConfig) -> Self {
        Self(Arc::new(ArcSwap::new(Arc::new(initial))))
    }

    /// Cheap atomic read of the current config. Returns an
    /// `Arc<WorldgenConfig>` snapshot — held references stay
    /// valid even if the holder is swapped concurrently.
    pub fn load(&self) -> Arc<WorldgenConfig> {
        self.0.load_full()
    }

    /// Atomically replace the held config. Existing snapshots
    /// returned by `load()` remain valid.
    pub fn swap(&self, new: WorldgenConfig) {
        self.0.store(Arc::new(new));
    }
}
```

Add to the existing `mod tests` block (inside `#[cfg(test)] mod tests`):

```rust
    #[test]
    fn holder_swap_visible_to_subsequent_load() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let holder = ConfigHolder::new(cfg);
        let initial_factor = holder.load().density.factor;
        let mut new_cfg = (*holder.load()).clone();
        new_cfg.density.factor = 99.0;
        holder.swap(new_cfg);
        assert!((initial_factor - 4.0).abs() < 1e-5);
        assert!((holder.load().density.factor - 99.0).abs() < 1e-5);
    }

    #[test]
    fn holder_load_returns_independent_snapshot() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let holder = ConfigHolder::new(cfg);
        let snapshot = holder.load();
        let mut new_cfg = (*snapshot).clone();
        new_cfg.density.factor = 42.0;
        holder.swap(new_cfg);
        // The previously-held snapshot must NOT see the new value.
        assert!((snapshot.density.factor - 4.0).abs() < 1e-5);
    }
```

- [ ] **Step 5.2: Run tests to verify they pass**

Run: `cargo test --lib worldgen::config 2>&1 | tail -10`

Expected: 4 tests pass total.

- [ ] **Step 5.3: Commit**

```bash
git add src/worldgen/config.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): ConfigHolder with arc-swap for atomic reloads

Lock-free read path: chunk gen calls holder.load() once per chunk,
gets an Arc<WorldgenConfig> snapshot that stays valid for the
chunk's lifetime even if the file watcher swaps in a new config
mid-generation. Atomicity guarantees deterministic per-chunk output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: File watcher (hot reload)

**Files:**
- Modify: `src/worldgen/config.rs`

- [ ] **Step 6.1: Write the failing test**

Append to `src/worldgen/config.rs` (above the `#[cfg(test)] mod tests` block):

```rust
use notify_debouncer_mini::{new_debouncer, DebounceEventResult, Debouncer};
use std::path::PathBuf;
use std::time::Duration;

/// Spawn a file watcher on `path`. On any change, re-parse the RON
/// file and (if valid) atomically swap the new config into
/// `holder`. Parse errors are logged at `error` level; the previous
/// config stays in effect.
///
/// Returns the debouncer — caller must keep it alive for the
/// watcher to keep running. Dropping it shuts the watcher down.
pub fn spawn_watcher(
    path: PathBuf,
    holder: ConfigHolder,
) -> anyhow::Result<Debouncer<notify_debouncer_mini::notify::RecommendedWatcher>> {
    let watch_path = path.clone();
    let mut debouncer = new_debouncer(
        Duration::from_millis(300),
        move |res: DebounceEventResult| match res {
            Ok(_events) => match WorldgenConfig::from_ron_file(&watch_path) {
                Ok(cfg) => {
                    log::info!("worldgen config reloaded from {:?}", watch_path);
                    holder.swap(cfg);
                }
                Err(e) => {
                    log::error!(
                        "worldgen config reload failed ({:?}): {} — keeping previous",
                        watch_path,
                        e
                    );
                }
            },
            Err(e) => log::error!("watcher error: {:?}", e),
        },
    )?;
    debouncer.watcher().watch(
        &path,
        notify_debouncer_mini::notify::RecursiveMode::NonRecursive,
    )?;
    Ok(debouncer)
}
```

Add to the existing test module:

```rust
    #[test]
    fn watcher_picks_up_file_change() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_config.ron");

        // Write initial config with factor=4.0.
        let initial = WorldgenConfig::bundled_default().unwrap();
        let initial_text = ron::ser::to_string_pretty(
            &initial,
            ron::ser::PrettyConfig::default(),
        ).unwrap();
        std::fs::write(&path, &initial_text).unwrap();

        let cfg = WorldgenConfig::from_ron_file(&path).unwrap();
        let holder = ConfigHolder::new(cfg);
        let _debouncer = spawn_watcher(path.clone(), holder.clone()).unwrap();

        // Modify the file: change factor to 7.0.
        let modified = initial_text.replace("factor: 4.0", "factor: 7.0");
        // Sleep briefly so the file mtime ticks past initial write.
        std::thread::sleep(Duration::from_millis(100));
        std::fs::write(&path, modified).unwrap();

        // Poll up to 2 seconds for the swap to occur.
        for _ in 0..20 {
            std::thread::sleep(Duration::from_millis(100));
            if (holder.load().density.factor - 7.0).abs() < 1e-3 {
                return;
            }
        }
        panic!(
            "watcher did not pick up file change; factor still {}",
            holder.load().density.factor
        );
    }
```

- [ ] **Step 6.2: Run the test**

Run: `cargo test --lib worldgen::config::tests::watcher_picks_up_file_change 2>&1 | tail -10`

Expected: pass within ~500ms (debouncer delay + filesystem latency).

If flaky, increase poll budget — but a correctly-functioning watcher should always reload within 500ms.

- [ ] **Step 6.3: Commit**

```bash
git add src/worldgen/config.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): file watcher for hot-reloadable WorldgenConfig

notify-debouncer-mini watches assets/worldgen/default.ron (or any
file passed in). On change: re-parse, atomically swap into the
holder via arc-swap. Parse errors log and keep the previous
config — no crash on typo.

Debounce window: 300ms (handles editor save-burst patterns).

Test: spawn watcher on a tempfile, write, write again, assert the
holder reflects the new value within 2s.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Plumb ConfigHolder through Generator

**Files:**
- Modify: `src/worldgen/mod.rs`
- Modify: any tests / call sites that construct `Generator::new(seed)`

- [ ] **Step 7.1: Write a failing test**

Append to the worldgen test module in `src/worldgen/mod.rs`:

```rust
    #[test]
    fn generator_holds_config_and_reads_from_holder() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let holder = crate::worldgen::config::ConfigHolder::new(cfg);
        let g = Generator::with_config(42, holder.clone());
        let mut chunk = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut chunk);
        // Hot-swap a different config (just to prove the holder is held, not cloned).
        let mut new_cfg = (*holder.load()).clone();
        new_cfg.density.factor = 1.0; // very low → flatter terrain
        holder.swap(new_cfg);
        // Generator's reference is the same Arc as `holder`, so it
        // should see the new value on its next read.
        // We don't assert on terrain (PR 2 step 9 connects density to config);
        // for now, just assert the generator's config snapshot is fresh.
        let snapshot = g.config_snapshot();
        assert!((snapshot.density.factor - 1.0).abs() < 1e-5);
    }
```

- [ ] **Step 7.2: Verify the test fails**

Run: `cargo test --lib worldgen::tests::generator_holds_config_and_reads_from_holder 2>&1 | tail -5`

Expected: compile error — `Generator::with_config` and `config_snapshot` don't exist yet.

- [ ] **Step 7.3: Add `Generator::with_config` and store the holder**

In `src/worldgen/mod.rs`, find the `Generator` struct definition. Add a `config` field:

```rust
pub struct Generator {
    // ... existing fields ...
    config: crate::worldgen::config::ConfigHolder,
}
```

Add a constructor that takes the holder (don't break `Generator::new(seed)` — make it create a holder from the bundled default):

```rust
impl Generator {
    /// Create a Generator with the bundled default config.
    /// Equivalent to constructing a `ConfigHolder` from
    /// `WorldgenConfig::bundled_default()` and calling
    /// [`Self::with_config`].
    pub fn new(seed: u64) -> Self {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default()
            .expect("bundled default.ron must parse");
        let holder = crate::worldgen::config::ConfigHolder::new(cfg);
        Self::with_config(seed, holder)
    }

    /// Create a Generator with an externally-provided ConfigHolder.
    /// The application owns the holder (and the file watcher); the
    /// Generator reads from it via `config_snapshot()`.
    pub fn with_config(
        seed: u64,
        config: crate::worldgen::config::ConfigHolder,
    ) -> Self {
        // ... existing Self::new body, but assign `config` at the end ...
        let mut g = Self::new_internal(seed); // see step 7.4
        g.config = config;
        g
    }

    /// Cheap atomic read of the current config. Holds an
    /// `Arc<WorldgenConfig>` snapshot. Call once per chunk and
    /// reuse across the chunk's lifetime to avoid mid-chunk
    /// drift if a hot-reload races chunk gen.
    pub fn config_snapshot(&self) -> std::sync::Arc<crate::worldgen::config::WorldgenConfig> {
        self.config.load()
    }
}
```

- [ ] **Step 7.4: Refactor old `Generator::new` body into `Generator::new_internal`**

Rename the current `Generator::new(seed: u64) -> Self` body to a private `fn new_internal(seed: u64) -> Self`. The old body knows nothing about config; it returns a Generator with a placeholder `ConfigHolder` (use `ConfigHolder::new(WorldgenConfig::bundled_default().unwrap())` as the default).

This intermediate refactor preserves all existing test call sites that pass just `seed`.

- [ ] **Step 7.5: Run the new test**

Run: `cargo test --lib worldgen::tests::generator_holds_config_and_reads_from_holder 2>&1 | tail -10`

Expected: PASS.

- [ ] **Step 7.6: Run the full worldgen test suite**

Run: `cargo test --lib worldgen 2>&1 | tail -10`

Expected: all 50+ existing tests still pass — `Generator::new(seed)` semantics unchanged.

- [ ] **Step 7.7: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): plumb ConfigHolder through Generator

Adds Generator::with_config(seed, holder) for app code that owns
the file watcher. Generator::new(seed) is preserved with identical
semantics (bundled-default config, no watcher) for test code and
simple integrations.

Generator::config_snapshot() returns an Arc<WorldgenConfig> for
chunk-gen code to read once per chunk and reuse — keeps each
chunk deterministic even if a hot-reload races.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Anisotropic base 3D noise

**Files:**
- Modify: `src/worldgen/heightmap.rs`

- [ ] **Step 8.1: Write a failing test**

Append to the heightmap test module:

```rust
    /// Anisotropic noise: a Y-step should change the noise value
    /// less than an equivalent XZ-step (vertical features are
    /// 2× taller than wide).
    #[test]
    fn base_3d_noise_y_scale_is_half_xz() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // Same y, step 8 blocks in X: this is a typical-magnitude noise step.
        let dx_step =
            (d.evaluate_base_3d(0.0, 0, 0, 0, &cfg.density)
                - d.evaluate_base_3d(0.0, 8, 0, 0, &cfg.density))
            .abs();
        // Same x, step 8 blocks in Y: should be roughly HALF the magnitude
        // because base_3d_y_scale = 0.5.
        let dy_step =
            (d.evaluate_base_3d(0.0, 0, 0, 0, &cfg.density)
                - d.evaluate_base_3d(0.0, 0, 8, 0, &cfg.density))
            .abs();
        // Statistical assertion: averaged over many points this would be
        // exact, but for a single point we just require dy_step < dx_step.
        // (Test multiple seeds if a single seed is unlucky and trips the
        // inequality.)
        assert!(
            dy_step < dx_step,
            "y-step ({dy_step}) should be smaller than x-step ({dx_step})"
        );
    }
```

- [ ] **Step 8.2: Add `evaluate_base_3d` method (with anisotropy)**

In `src/worldgen/heightmap.rs`, add a new method to `DensityNoise`:

```rust
impl DensityNoise {
    /// Sample the anisotropic base 3D noise. Y is scaled by
    /// `cfg.base_3d_y_scale` before sampling — values < 1 stretch
    /// vertical features (make them taller than wide).
    pub fn evaluate_base_3d(
        &self,
        _h_target: f32, // ignored; kept for symmetry with evaluate()
        wx: i32,
        wy: i32,
        wz: i32,
        cfg: &crate::worldgen::config::DensityConfig,
    ) -> f32 {
        let scaled_y = wy as f64 * cfg.base_3d_y_scale as f64;
        self.relief.get([wx as f64, scaled_y, wz as f64]) as f32 * cfg.base_3d_amplitude
    }
}
```

Note: this does NOT replace the existing `evaluate` method yet — that's task 9. This method exists alongside.

- [ ] **Step 8.3: Run the test**

Run: `cargo test --lib worldgen::heightmap::tests::base_3d_noise_y_scale_is_half_xz 2>&1 | tail -5`

Expected: PASS.

- [ ] **Step 8.4: Commit**

```bash
git add src/worldgen/heightmap.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): anisotropic base 3D noise (y_scale = xz_scale / 2)

DensityNoise::evaluate_base_3d samples the FBM with Y multiplied
by cfg.base_3d_y_scale (0.5 by default). Vertical features end up
2× taller than wide — cliffs and overhangs look like cliffs and
overhangs, not noise wiggle.

This is the same trick Minecraft's BlendedNoise uses (xz_scale=0.25,
y_scale=0.125). One number, big visual win.

The new method exists alongside the old DensityNoise::evaluate for
this commit; task 9 replaces evaluate with the full MC composition.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: New density composition

**Files:**
- Modify: `src/worldgen/heightmap.rs`
- Modify: `src/worldgen/mod.rs`

- [ ] **Step 9.1: Write failing tests for the new evaluator**

Append to the heightmap test module:

```rust
    /// At y = h_target the density should be close to 0 (the surface).
    #[test]
    fn new_evaluator_density_at_surface_is_near_zero() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // Average over a small grid to wash out noise.
        let mut sum = 0.0_f32;
        let mut n = 0;
        for wx in 0..16 {
            for wz in 0..16 {
                sum += d.evaluate_v2(80.0, wx, 80, wz, &cfg.density);
                n += 1;
            }
        }
        let avg = sum / n as f32;
        assert!(
            avg.abs() < 1.5,
            "average density at surface should be near 0 (got {avg})"
        );
    }

    #[test]
    fn new_evaluator_density_well_below_is_strongly_positive() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // 50 blocks below the target: should be unambiguously solid.
        let v = d.evaluate_v2(80.0, 0, 30, 0, &cfg.density);
        assert!(v > 1.0, "density 50 below surface should be > 1, got {v}");
    }

    #[test]
    fn new_evaluator_density_well_above_is_strongly_negative() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // 50 blocks above the target: should be unambiguously air.
        let v = d.evaluate_v2(80.0, 0, 130, 0, &cfg.density);
        // Note: 130 is also close to y_max=140, so slide_top adds extra pull.
        // The assertion is just "very negative".
        assert!(v < -0.05, "density 50 above surface should be < -0.05, got {v}");
    }

    #[test]
    fn new_evaluator_density_at_world_top_pulled_toward_air() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // At y_max, slide_top fully kicks in.
        let v = d.evaluate_v2(80.0, 0, 140, 0, &cfg.density);
        assert!((v - (-0.078125)).abs() < 0.5, "at y_max density should be near slide_top_target, got {v}");
    }

    #[test]
    fn new_evaluator_density_at_world_bottom_pulled_toward_solid() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // At y_min, slide_bottom fully kicks in.
        let v = d.evaluate_v2(80.0, 0, -120, 0, &cfg.density);
        assert!((v - 0.1171875).abs() < 0.5, "at y_min density should be near slide_bottom_target, got {v}");
    }
```

- [ ] **Step 9.2: Verify tests fail**

Run: `cargo test --lib worldgen::heightmap::tests::new_evaluator 2>&1 | tail -15`

Expected: 5 tests fail with `no method named evaluate_v2`.

- [ ] **Step 9.3: Implement `DensityNoise::evaluate_v2`**

In `src/worldgen/heightmap.rs`, append to the `impl DensityNoise` block:

```rust
impl DensityNoise {
    /// New MC-style composition. Replaces `evaluate` once all call
    /// sites migrate. Until then, both coexist.
    pub fn evaluate_v2(
        &self,
        h_target: f32,
        wx: i32,
        wy: i32,
        wz: i32,
        cfg: &crate::worldgen::config::DensityConfig,
    ) -> f32 {
        // y_gradient: +amplitude at y_min, -amplitude at y_max,
        // linear in between.
        let t = (wy - cfg.y_min) as f32 / (cfg.y_max - cfg.y_min) as f32;
        let y_gradient = cfg.y_gradient_amplitude * (1.0 - 2.0 * t);

        // Offset: shifts the y_gradient up/down per column. In PR 2
        // we derive it from h_target so the surface lands at h_target
        // (preserves current terrain shape). PR 3 replaces this with
        // the spline of (continentalness, erosion, ridges).
        let t_at_target = (h_target - cfg.y_min as f32) / (cfg.y_max - cfg.y_min) as f32;
        let offset = cfg.y_gradient_amplitude * (2.0 * t_at_target - 1.0);

        let depth = y_gradient + offset;
        // PR 3 introduces real jaggedness; for PR 2 it's zero.
        let jagged = 0.0;
        let factor = cfg.factor;

        // Quarter-negative softening: positive (below surface) keeps
        // full magnitude; negative (above surface) scales by `above_surface_softening`.
        // Net effect: solid below grows fast, air above grows slowly,
        // so 3D noise can carve overhangs without piercing solid ground.
        let shaped_raw = (depth + jagged) * factor;
        let shaped = if shaped_raw > 0.0 {
            shaped_raw
        } else {
            shaped_raw * cfg.above_surface_softening
        };

        let base_3d = self.evaluate_base_3d(h_target, wx, wy, wz, cfg);
        let pre_slide = cfg.composition_scale * shaped + base_3d;
        slide(pre_slide, wy, cfg)
    }
}

/// Apply top and bottom slides to a density value at world Y.
/// Within `slide_top_blocks` of `y_max`, lerps toward
/// `slide_top_target`. Within `slide_bottom_blocks` of `y_min`,
/// lerps toward `slide_bottom_target`. Free helper, no state.
fn slide(density: f32, wy: i32, cfg: &crate::worldgen::config::DensityConfig) -> f32 {
    // Top: factor goes 0 (no pull) → 1 (full pull) over
    // (y_max - slide_top_blocks .. y_max).
    let top_start = cfg.y_max - cfg.slide_top_blocks;
    let top_f = ((wy - top_start) as f32 / cfg.slide_top_blocks as f32).clamp(0.0, 1.0);
    let after_top = density + (cfg.slide_top_target - density) * top_f;

    // Bottom: factor goes 1 → 0 over (y_min .. y_min + slide_bottom_blocks).
    let bot_end = cfg.y_min + cfg.slide_bottom_blocks;
    let bot_f = ((bot_end - wy) as f32 / cfg.slide_bottom_blocks as f32).clamp(0.0, 1.0);
    after_top + (cfg.slide_bottom_target - after_top) * bot_f
}
```

- [ ] **Step 9.4: Run tests**

Run: `cargo test --lib worldgen::heightmap::tests::new_evaluator 2>&1 | tail -15`

Expected: 5 tests pass.

- [ ] **Step 9.5: Commit**

```bash
git add src/worldgen/heightmap.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): DensityNoise::evaluate_v2 — MC-style composition

New evaluator computes:
  shaped = (y_gradient(y) + offset(x,z)) * factor
  shaped' = quarter_negative(shaped)  // soften above surface
  density = composition_scale * shaped' + base_3d_noise(anisotropic)
  density = slide(density, y)         // top→air, bottom→solid

PR 2 derives `offset` from h_target so the surface lands at the
existing heightmap location — terrain shape preserved across this
PR. PR 3 swaps this for the real (continentalness, erosion, ridges)
spline.

The old DensityNoise::evaluate is preserved for call sites that
haven't migrated yet (task 10 migrates mod.rs::fill_chunk and
removes the old method).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Migrate `mod.rs::fill_chunk` to v2, remove min(2.0) cap

**Files:**
- Modify: `src/worldgen/mod.rs`
- Modify: `src/worldgen/heightmap.rs` (remove old methods)

- [ ] **Step 10.1: Update golden hash sentinel**

In `src/worldgen/mod.rs`, locate the `golden_seed42_chunk_0_2_0` test. Change:

```rust
const GOLDEN_42_002: u64 = 0x886E_0C40_5650_12C7;
```

to:

```rust
const GOLDEN_42_002: u64 = 0xDEAD_BEEF_DEAD_BEEF;
```

This puts the test into "print the new hash" mode so we capture the post-migration value in step 10.6.

- [ ] **Step 10.2: Replace the density evaluation in `fill_chunk`**

In `src/worldgen/mod.rs`, find the line:

```rust
let raw_density = self.density.evaluate(h_target, wx, wy, wz);
```

(currently around line 392 in `fill_chunk`).

Replace with a config snapshot read at the top of `fill_chunk` and a call to `evaluate_v2`:

Inside the outer `fn fill_chunk(&self, ...)` body, near the top (after `let origin = ...`), add:

```rust
let cfg = self.config_snapshot();
```

Then replace the inner line with:

```rust
let raw_density = self.density.evaluate_v2(h_target, wx, wy, wz, &cfg.density);
```

- [ ] **Step 10.3: Remove the `min(2.0)` cap**

The new composition's `quarter_negative` softening above the surface eliminates the need for the cap. In `mod.rs::fill_chunk`, replace:

```rust
let density_for_compare = if cave_contribution > 0.0 {
    // Cap at 1.0 (not 2.0) when cave contributions
    // are in play: ...
    raw_density.min(1.0)
} else {
    raw_density
};
let solid = (density_for_compare - cave_contribution) > 0.0;
```

with:

```rust
// With the new composition, density above the surface is already
// small (quarter_negative softening). No need to cap when caves
// are in play — the carve threshold `density - cave > 0` works
// uniformly across all depths.
let solid = (raw_density - cave_contribution) > 0.0;
```

- [ ] **Step 10.4: Remove old `DensityNoise::evaluate` and the unused `DENSITY_FALLOFF`/`RELIEF_AMP`**

In `src/worldgen/heightmap.rs`:
- Remove the old `pub fn evaluate(&self, h_target: f32, wx: i32, wy: i32, wz: i32) -> f32` method body. (Keep the type and its `new` constructor.)
- Search for the old method's callers: `topmost_solid` (in the same file) still uses it. Migrate that to `evaluate_v2`:

```rust
pub fn topmost_solid(
    &self,
    h_target: f32,
    wx: i32,
    wz: i32,
    search_top: i32,
    cfg: &crate::worldgen::config::DensityConfig,
) -> Option<i32> {
    let top = search_top.min(h_target as i32 + SURFACE_BAND);
    let bottom = (h_target as i32 - SURFACE_BAND).max(CAVE_FLOOR_Y);
    for wy in (bottom..=top).rev() {
        if self.evaluate_v2(h_target, wx, wy, wz, cfg) > 0.0 {
            return Some(wy);
        }
    }
    Some(bottom)
}
```

Update `topmost_solid` callers in `mod.rs::add_trees` (search for `density.topmost_solid`) — pass `&cfg.density` as the new arg.

In `tuning.rs`, mark `DENSITY_FALLOFF` and `RELIEF_AMP` deprecated:

```rust
#[deprecated(note = "Replaced by WorldgenConfig::density (PR 2)")]
pub const DENSITY_FALLOFF: f32 = 4.0;
#[deprecated(note = "Replaced by WorldgenConfig::density.base_3d_amplitude (PR 2)")]
pub const RELIEF_AMP: f32 = 1.0;
```

Leave them in place so any straggler test/util keeps compiling. Future PRs remove them entirely.

- [ ] **Step 10.5: Run the worldgen suite**

Run: `cargo test --lib worldgen 2>&1 | tail -15`

Expected: most tests pass. `golden_seed42_chunk_0_2_0` prints `UPDATE GOLDEN_42_002 to: 0x<hex>` and passes (because the sentinel mode skips the equality check).

Two tests are likely to fail and need updates:
- `chunk_at_sea_level_has_water_or_solid` — sanity, should still pass if composition is roughly right.
- `cold_biome_caps_with_snow` — uses `density.topmost_solid` which now needs the config arg. Should have been updated in step 10.4.
- The integration test `worldgen_fingerprint::fingerprint_hash_matches_pin` ALSO needs updating (PR 3 changes the heightmap so the 2D field is unchanged here — but in PR 2 the heightmap is untouched, so this should still pass).

If `worldgen_fingerprint::fingerprint_hash_matches_pin` fails, that's a sign the heightmap accidentally changed. Investigate before proceeding.

- [ ] **Step 10.6: Capture the new golden hash and re-pin**

Run: `cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 -- --nocapture 2>&1 | grep "UPDATE GOLDEN"`

Expected output: `UPDATE GOLDEN_42_002 to: 0x<NEW_HEX>`

Update the sentinel in `mod.rs`:

```rust
const GOLDEN_42_002: u64 = 0x<NEW_HEX_FROM_STEP_10_6>;
```

Run again: `cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 2>&1 | tail -5`

Expected: pass with the new pinned hash.

- [ ] **Step 10.7: Run the full suite**

Run: `cargo test 2>&1 | tail -20`

Expected: all tests pass. If `worldgen_fingerprint::fingerprint_hash_matches_pin` is the only fail, the heightmap definitely changed in an unintended way — investigate.

- [ ] **Step 10.8: Commit**

```bash
git add src/worldgen/mod.rs src/worldgen/heightmap.rs src/worldgen/tuning.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): migrate fill_chunk to evaluate_v2; remove min cap

The new MC-style composition's quarter_negative softening above
the surface means the density bias above the iso-surface is small
enough that any positive cave contribution carves cleanly. PR B's
min(2.0) hack is no longer needed and is removed.

Migration:
- mod.rs::fill_chunk reads a per-chunk config snapshot via
  Generator::config_snapshot() once at the top, passes &cfg.density
  to all density evaluations
- DensityNoise::evaluate (the old additive bias+noise) is removed
- DensityNoise::topmost_solid uses evaluate_v2 (signature changes
  to accept &DensityConfig; callers updated)
- tuning.rs DENSITY_FALLOFF / RELIEF_AMP are marked deprecated;
  they're now in config.density.{...}

Golden hashes:
- golden_seed42_chunk_0_2_0 re-baselined (surface chunk now uses
  the new composition; small but real differences)
- worldgen_fingerprint::fingerprint_hash_matches_pin unchanged
  (PR 2 doesn't touch the 2D heightmap)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Wire file watcher into app startup

**Files:**
- Modify: `src/main.rs` (or wherever `Generator::new` is currently called from game/app code)

- [ ] **Step 11.1: Find the Generator construction site**

Run: `grep -rn "Generator::new" src/ --include='*.rs' 2>&1`

Expected: a small list of locations. The one in `src/main.rs` (or the app entry point — not the tests) is the target.

- [ ] **Step 11.2: Replace `Generator::new(seed)` with config-aware construction**

At the app startup site, replace:

```rust
let generator = Generator::new(seed);
```

with:

```rust
let cfg = crate::worldgen::config::WorldgenConfig::bundled_default()
    .expect("bundled default.ron must parse");
let holder = crate::worldgen::config::ConfigHolder::new(cfg);
let watcher_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("assets")
    .join("worldgen")
    .join("default.ron");
let _watcher = crate::worldgen::config::spawn_watcher(
    watcher_path,
    holder.clone(),
).expect("file watcher must start");
// IMPORTANT: store `_watcher` somewhere persistent; if it drops, the
// watcher thread stops. Holding it in the game state struct works.
let generator = Generator::with_config(seed, holder);
```

Replace `_watcher` storage with whatever the app's lifetime-anchoring pattern is (likely a field on the game state struct). The watcher must outlive the Generator.

- [ ] **Step 11.3: Verify the game compiles and runs**

Run: `cargo build 2>&1 | tail -5`

Expected: `Finished` line, no errors.

Run: `cargo run --release -- --headless` (or whatever Oxium's smoke-test flag is, if any).

Expected: the game starts, generates a chunk, and exits cleanly (or runs for a moment without crashing on startup).

- [ ] **Step 11.4: Manual smoke test (optional, recommended)**

Run the game. While running, edit `assets/worldgen/default.ron` and change `factor: 4.0` to `factor: 1.0`. Save.

Expected: log message `worldgen config reloaded from "/.../default.ron"`. Newly-generated chunks (move out of the load radius and back) should reflect the change.

Existing chunks are not regenerated — they'd require chunk-cache invalidation, which is a future-PR concern.

- [ ] **Step 11.5: Commit**

```bash
git add src/main.rs # or whichever file step 11.2 touched
git commit -m "$(cat <<'EOF'
feat(worldgen): wire file watcher into app startup

Replaces Generator::new(seed) with the with_config form at the app
entry point. Spawns a notify-based watcher on
assets/worldgen/default.ron. On file change: re-parse, atomic
swap. Newly-generated chunks reflect the new config; existing
chunks keep their original generation (chunk-cache invalidation
is a future-PR concern, tracked in the migration plan).

The watcher's debouncer handle is kept alive on the game state to
prevent the watcher thread from shutting down.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Final verification

**Files:** none (verification only)

- [ ] **Step 12.1: Run the full test suite**

Run: `cargo test 2>&1 | tail -10`

Expected: every test passes. Specifically:
- All `worldgen::*` lib tests pass
- `worldgen_fingerprint::fingerprint_hash_matches_pin` passes (the 2D heightmap is untouched in PR 2)
- `golden_seed42_chunk_0_2_0` passes with the new pinned hash
- The `smoke::*` integration tests pass (they exercise chunk gen + meshing)

- [ ] **Step 12.2: Visual smoke test**

Boot the game. Observe:
- Terrain looks broadly similar to pre-PR-2 (since `offset` is derived from `h_target`)
- Caves still carve and remain navigable (they're not affected by composition changes)
- The sky has a clean ceiling near `y_max = 140` (slide_top working)
- Bedrock-like solid floor near `y_min = -120` (slide_bottom working)
- Edit `default.ron`: change `factor` to `2.0` and save. Move out of and back into chunks. Terrain should look subtly softer.

If anything looks wrong, the most likely suspects are:
- Slide constants (`slide_top_target` / `slide_bottom_target`) — tweak in the RON file
- `factor` too low (mushy terrain) or too high (knife-edge cliffs) — tweak in the RON file
- y_gradient amplitude — tweak `y_gradient_amplitude` (default 1.5)

- [ ] **Step 12.3: Verify the watcher works in release**

Run: `cargo run --release` (full game). Edit `default.ron`. Confirm the reload log message appears.

- [ ] **Step 12.4: Final commit (no-op if nothing changed)**

If steps 12.1–12.3 prompted any RON tuning, commit it:

```bash
git add assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
chore(worldgen): tune default.ron after PR 2 visual review

Final tuning of the foundation composition values to match the
pre-PR-2 terrain feel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Otherwise nothing to commit — PR 2 is complete.

---

## Out of scope for PR 2 (deferred to later PRs)

- **Spline-driven offset/factor/jaggedness** (PR 3). The `offset_spline` field in `default.ron` is a placeholder constant; the runtime path derives offset from `h_target` directly.
- **Continentalness/erosion noise channels** (PR 3). Existing plate Voronoi continues to drive heightmap shape via `h_target`.
- **Multi-noise biome lookup** (PR 4). Existing `Biome::classify` nested-if persists.
- **Cell interpolation / sliding YZ wall** (PR 5). Density is still evaluated per-voxel.
- **Surface rules DSL** (PR 6). The inline near_surface gate from the in-session fix persists.
- **Real aquifer** (PR 7). The primitive ocean-column rule from the in-session fix persists.
- **Noise carver layers (cheese, spaghetti)** (PR 8). Graph caves + wormholes are the only carvers.
- **Chunk cache invalidation on hot-reload.** Hot-reload changes only affect newly-generated chunks in PR 2.

## Plan self-review notes

- All 12 tasks have concrete code in every step. No "TBD" or "fill in details".
- Type names are consistent across tasks: `WorldgenConfig`, `DensityConfig`, `ConfigHolder`, `CubicSpline`, `Knot`, `FlatCache2D`, `DensityNoise::evaluate_v2`.
- Each task ends with a commit boundary.
- Golden hash management: task 10 step 10.1 puts the test in print-mode; step 10.6 captures and re-pins.
- The plan preserves `Generator::new(seed)` semantics so existing 50+ tests don't break wholesale.
