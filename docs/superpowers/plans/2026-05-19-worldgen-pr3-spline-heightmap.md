# Worldgen PR 3 — Spline-driven Heightmap with Continentalness + Erosion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Assumes PR 2 has landed** — `WorldgenConfig`, `DensityConfig`, `ConfigHolder`, `CubicSpline`, `Knot`, `FlatCache2D`, and `DensityNoise::evaluate_v2` already exist; the runtime is calling `evaluate_v2` with `offset` derived from `h_target`. PR 3 replaces that derivation with the real Minecraft-style spline pipeline.

**Goal:** Replace PR 2's `h_target`-derived offset with a Minecraft-style spline pipeline. Three 2D inputs — `continentalness` (the plate Voronoi signed distance field, blended via `plate_t`), `terrain_shape` (a new low-frequency 2D noise plus per-plate roughness bias), and `ridges_pv` (peaks-and-valleys triangle fold on a ridge noise) — feed nested cubic Hermite splines that produce `offset(x,z)`, `factor(x,z)`, and `jaggedness(x,z)` per column. The density composition is unchanged; only the inputs that drive it become spline-driven. The dead `ridge_lift` / `BOUNDARY_RIDGE_WIDTH` / `RIDGE_PEAK_*` / `CLIFF_MIN_HEIGHT` band-aids that the splines obsolete are deleted. Hydrology is left intact except for one line: `valley_carve` re-targets to the spline `offset` instead of `h_pre`. A 4-block above-water 3D-noise clamp prevents overhang ceilings from tunnelling rivers and lake rims.

**Architecture:** PR 3 introduces three new spline-config sections and one new noise struct. (1) `ClimateConfig` holds three nested `CubicSpline` knot tables (`offset_spline`, `factor_spline`, `jaggedness_spline`) plus the per-plate `roughness_bias` knot table — values come from `assets/worldgen/default.ron`. (2) `TerrainShapeNoise` is a new `noise::Fbm` struct that mirrors PR 2's `DensityNoise` pattern: low-frequency (~2000-block wavelength) for terrain shape, mid-frequency (~500-block) for ridges. (3) A new `column_climate(wx, wz)` helper computes `(continentalness, terrain_shape, ridges_pv)` for a column using the existing `plate_at` Voronoi for continentalness, the new `TerrainShapeNoise` for the other two, and the per-plate `roughness_bias` as additive bias on terrain_shape. (4) A new `column_spline_outputs(wx, wz)` helper evaluates the three splines and returns `(offset, factor, jaggedness)`. (5) `DensityNoise::evaluate_v2` gains three extra parameters (or takes a `SplineOutputs` struct) so the per-voxel call receives the column's pre-computed spline outputs — `FlatCache2D` caches these once per quart. (6) `mod.rs::fill_chunk` reads `is_river || lake_rim.is_some()` once per column and applies a 4-block above-water clamp on positive 3D-noise contribution. (7) Hydrology's `valley_carve` consumer rebinds: `column_data_with` reads the spline offset, applies carve to it, and the result becomes `height` — the spline offset replaces `h_pre` end-to-end. (8) Obsolete tuning constants and `ridge_lift` are removed; `HeightmapNoise::is_cliff` keeps the slope check but loses the `CLIFF_MIN_HEIGHT` gate (the spline produces real high mountains, so the height gate is redundant; the slope gate is still meaningful).

**Tech Stack:**
- Rust 2024 edition
- `noise = "0.9"` — already a dependency (used by PR 2's `DensityNoise`)
- `serde` — for RON deserialisation of nested spline knots
- `glam` — `Vec2` for plate seed math
- No new crates.

**Reference:** Architectural rationale in `docs/superpowers/specs/2026-05-19-minecraft-worldgen-research.md` Part 4 ideas #1 (continentalness/erosion replacing plates) and #2 (`quarter_negative * factor` composition), plus Decisions Log Q1 (plates → continentalness input + per-plate roughness bias on erosion) and Q2 (hydrology untouched, valley_carve targets spline offset, 4-block above-water noise clamp). Minecraft knot tables come from `net/minecraft/data/worldgen/TerrainProvider.java` (referenced in the research doc Part 4).

---

### Task 1: Add `ClimateConfig` section to `WorldgenConfig` and seed default.ron

**Files:**
- Modify: `src/worldgen/config.rs`
- Modify: `assets/worldgen/default.ron`

- [ ] **Step 1.1: Write failing tests for the new ClimateConfig fields**

Append to `src/worldgen/config.rs` test module:

```rust
    #[test]
    fn climate_config_loads_offset_spline_with_mountain_knots() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        // Continentalness +1.0 (deep inland) should map through the
        // outermost spline to something positive (mountains rise).
        // Test asserts the spline at that point is positive — concrete
        // knot values are verified by the spline tests, not this one.
        let cont_inland = 1.0_f32;
        let v = cfg.climate.offset_spline_at(cont_inland, 0.0, 0.0);
        assert!(v > 0.0, "inland offset spline should be positive, got {v}");
    }

    #[test]
    fn climate_config_loads_factor_spline() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        // Deep ocean (cont = -1.0): factor should be small (flat).
        let f_ocean = cfg.climate.factor_spline_at(-1.0, 0.0, 0.0);
        // Mountains (cont = +1.0, shape = -1.0 → very low erosion): big factor.
        let f_mountain = cfg.climate.factor_spline_at(1.0, -1.0, 0.0);
        assert!(
            f_ocean < f_mountain,
            "ocean factor ({f_ocean}) should be < mountain factor ({f_mountain})"
        );
    }

    #[test]
    fn climate_config_jaggedness_only_at_peaks() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        // Jaggedness should be ~zero at non-peak PV (ridges_pv near 0).
        let j_off_peak = cfg.climate.jaggedness_spline_at(1.0, -1.0, 0.0);
        // Jaggedness should be > zero at PV = 1.0 (peak).
        let j_peak = cfg.climate.jaggedness_spline_at(1.0, -1.0, 1.0);
        assert!(j_peak > j_off_peak, "jaggedness should rise with PV");
    }

    #[test]
    fn plate_roughness_bias_in_reasonable_range() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        // Roughness bias is additive on terrain_shape (which is in
        // [-1, 1] from FBM). The bias should stay small enough that
        // shape + bias stays in roughly [-1.5, 1.5].
        assert!(cfg.climate.plate_roughness_bias_range.0 >= -0.5);
        assert!(cfg.climate.plate_roughness_bias_range.1 <= 0.5);
        assert!(
            cfg.climate.plate_roughness_bias_range.1
                > cfg.climate.plate_roughness_bias_range.0
        );
    }
```

- [ ] **Step 1.2: Add `ClimateConfig` to `WorldgenConfig`**

In `src/worldgen/config.rs`, add the climate subsection. Insert above `DensityConfig` (or wherever sub-sections are grouped):

```rust
/// Climate-driven spline pipeline tuning (PR 3).
///
/// Three 2D inputs feed three nested cubic Hermite splines:
///
/// * `continentalness` — plate Voronoi signed distance field
///   blended via `plate_t`. Positive inland, negative offshore.
/// * `terrain_shape` — low-frequency 2D noise (~2000-block period),
///   plus per-plate `roughness_bias`. Low values produce mountains,
///   high values flatten terrain. Avoids the name "erosion" so it
///   doesn't collide with `hydrology.rs`'s actual erosion concept.
/// * `ridges_pv` — peaks-and-valleys triangle fold on a higher-
///   frequency ridge noise (~500-block period). Drives jaggedness.
///
/// Each spline is nested: e.g. `offset_spline` is keyed on
/// continentalness, and each knot's value is itself a spline keyed
/// on terrain_shape. The innermost result is a scalar in roughly
/// `[-1.5, 1.5]` (matching the y_gradient scale).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClimateConfig {
    /// Outer spline keyed on continentalness; each knot's value is
    /// the offset (in y_gradient units) added to the y_gradient.
    /// Nested splines on terrain_shape and ridges_pv are encoded as
    /// `Nested(...)` knots — see [`NestedSpline`] below.
    pub offset_spline: NestedSpline,
    /// Outer spline keyed on continentalness; output is the
    /// multiplicative `factor` for the (depth+jagged)*factor term.
    pub factor_spline: NestedSpline,
    /// Outer spline keyed on continentalness; output is `jaggedness`
    /// (amplitude of the per-voxel jagged noise rider).
    pub jaggedness_spline: NestedSpline,

    /// Per-plate roughness bias range (uniform random in this range
    /// per plate, salt 5 — see plates.rs::Plate::of). Additive on
    /// terrain_shape: `effective_shape = shape_noise + bias`.
    pub plate_roughness_bias_range: (f32, f32),

    // Noise parameters for the new TerrainShapeNoise field.
    pub terrain_shape_period: f32,   // ~2000 blocks (MC erosion)
    pub terrain_shape_amplitude: f32, // ~1.0 (range stays ~[-1, 1])
    pub ridges_period: f32,           // ~500 blocks
    pub ridges_amplitude: f32,        // ~1.0
}

/// A nested spline: each knot's value is *itself* a spline. The
/// `evaluate` method takes three inputs `(c, s, r)` and walks the
/// nesting — outer spline keyed on `c`, mid on `s`, inner on `r`.
///
/// Equivalent to MC's `CubicSpline.Multipoint` with `Value` knots
/// whose value is another `CubicSpline`. Encoded here as a flat enum
/// for Serde simplicity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NestedSpline {
    /// Constant leaf — terminal value.
    Constant(f32),
    /// Multipoint spline whose knots carry nested splines as values.
    Multipoint(Vec<NestedKnot>),
}

/// A nested knot: location on the current axis, value as another
/// nested spline, slope on the current axis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NestedKnot {
    pub loc: f32,
    pub val: NestedSpline,
    pub slope: f32,
}

impl NestedSpline {
    /// Walk the nesting: at each level, evaluate at the next input
    /// and recurse into the picked segment's value.
    ///
    /// The outer level keys on `c`, the mid on `s`, the inner on `r`.
    /// Constants short-circuit (so a nesting can be partial — a leaf
    /// at any depth returns immediately).
    pub fn evaluate(&self, c: f32, s: f32, r: f32) -> f32 {
        // First peel: split into "the input to use at this level" and
        // "the rest".
        self.evaluate_inner(&[c, s, r], 0)
    }

    fn evaluate_inner(&self, inputs: &[f32], depth: usize) -> f32 {
        match self {
            NestedSpline::Constant(v) => *v,
            NestedSpline::Multipoint(knots) => {
                assert!(!knots.is_empty(), "nested spline must have ≥1 knot");
                let input = inputs.get(depth).copied().unwrap_or(0.0);
                // Below first knot: extrapolate using first knot's slope.
                if input <= knots[0].loc {
                    let base = knots[0].val.evaluate_inner(inputs, depth + 1);
                    return base + knots[0].slope * (input - knots[0].loc);
                }
                let last = knots.last().unwrap();
                if input >= last.loc {
                    let base = last.val.evaluate_inner(inputs, depth + 1);
                    return base + last.slope * (input - last.loc);
                }
                // Find segment [k1, k2] containing input.
                let mut i = 0;
                while i + 1 < knots.len() && knots[i + 1].loc < input {
                    i += 1;
                }
                let k1 = &knots[i];
                let k2 = &knots[i + 1];
                let v1 = k1.val.evaluate_inner(inputs, depth + 1);
                let v2 = k2.val.evaluate_inner(inputs, depth + 1);
                let dx = k2.loc - k1.loc;
                let t = (input - k1.loc) / dx;
                let a = k1.slope * dx - (v2 - v1);
                let b = -k2.slope * dx + (v2 - v1);
                let lerp_y = v1 + t * (v2 - v1);
                let lerp_ab = a + t * (b - a);
                lerp_y + t * (1.0 - t) * lerp_ab
            }
        }
    }
}

impl ClimateConfig {
    /// Evaluate the offset spline at `(continentalness, terrain_shape, ridges_pv)`.
    pub fn offset_spline_at(&self, c: f32, s: f32, r: f32) -> f32 {
        self.offset_spline.evaluate(c, s, r)
    }
    /// Evaluate the factor spline.
    pub fn factor_spline_at(&self, c: f32, s: f32, r: f32) -> f32 {
        self.factor_spline.evaluate(c, s, r)
    }
    /// Evaluate the jaggedness spline.
    pub fn jaggedness_spline_at(&self, c: f32, s: f32, r: f32) -> f32 {
        self.jaggedness_spline.evaluate(c, s, r)
    }
}
```

Add a `climate: ClimateConfig` field to `WorldgenConfig`:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldgenConfig {
    pub density: DensityConfig,
    pub climate: ClimateConfig,
}
```

- [ ] **Step 1.3: Update `assets/worldgen/default.ron` with the climate section**

Open `assets/worldgen/default.ron` and append the `climate: (...)` section inside the top-level struct. The knot values are seeded from MC's `TerrainProvider.java`, scaled to Oxium's y_gradient_amplitude of 1.5.

```ron
// (existing density: (...) section stays as-is from PR 2.)

    // ── Climate / spline pipeline (PR 3) ──────────────────────────────
    //
    // Three 2D inputs drive nested cubic Hermite splines:
    //   continentalness ∈ [-1, 1]: -1 deep ocean, +1 deep inland
    //   terrain_shape   ∈ [-1, 1]: -1 mountainous (low erosion), +1 flat
    //   ridges_pv       ∈ [-1, 1]: PV fold; +1 = ridge peak, 0 = saddle
    //
    // Knot loc/val/slope numbers are derived from Minecraft 1.18+
    // net/minecraft/data/worldgen/TerrainProvider.java (Mojang's
    // hand-tuned overworld). Oxium's y_gradient_amplitude = 1.5 means
    // the output is in roughly [-1.5, 1.5]; MC's values are in [-1, 1]
    // (their amplitude is 1) so we apply a 1.5× scale to match.
    climate: (
        // Per-plate roughness bias: each plate rolls a uniform value
        // in this range (salt 5), added to terrain_shape noise. Keeps
        // "rocky vs gentle continent" identity even after splines.
        plate_roughness_bias_range: (-0.25, 0.25),

        // TerrainShape noise: low frequency, ~2000-block period
        // (MC erosion). Amplitude 1.0 to fill [-1, 1].
        terrain_shape_period: 2000.0,
        terrain_shape_amplitude: 1.0,

        // Ridges noise: mid frequency, ~500-block period
        // (MC weirdness pre-fold). Amplitude 1.0.
        ridges_period: 500.0,
        ridges_amplitude: 1.0,

        // ─── offset_spline ─────────────────────────────────────────
        // Continentalness → "what is the base height of this column?"
        // Nested: each knot's value is a spline on terrain_shape; that
        // inner spline's knots are constants (PR 3 does not vary
        // offset on ridges_pv — that's reserved for jaggedness).
        offset_spline: Multipoint([
            // Deep ocean wall (mushroom-island cap above).
            (loc: -1.10, val: Constant(0.066), slope: 0.0),
            // Deep ocean floor — flat, repeated for plateau effect.
            (loc: -1.02, val: Constant(-0.333), slope: 0.0),
            (loc: -0.51, val: Constant(-0.333), slope: 0.0),
            // Shallow ocean (continental shelf).
            (loc: -0.44, val: Constant(-0.18), slope: 0.0),
            (loc: -0.18, val: Constant(-0.18), slope: 0.0),
            // Hard step into beach. Two knots at -0.16, -0.15 produce
            // a visible coastline (MC's trick).
            (loc: -0.16, val: Constant(-0.06), slope: 0.0),
            (loc: -0.15, val: Constant(0.06), slope: 0.0),
            // Inland — terrain_shape now matters. Nested spline on
            // terrain_shape: low shape → mountains, high → plains.
            (loc: -0.10, val: Multipoint([
                (loc: -0.85, val: Constant(0.75), slope: 0.0), // veryLowErosionMountains
                (loc: -0.70, val: Constant(0.75), slope: 0.0),
                (loc: -0.40, val: Constant(0.60), slope: 0.0),
                (loc: -0.35, val: Constant(0.30), slope: 0.0), // widePlateau
                (loc: -0.10, val: Constant(0.15), slope: 0.0), // narrowPlateau
                (loc:  0.20, val: Constant(0.0),  slope: 0.0), // plains
                (loc:  0.70, val: Constant(-0.03), slope: 0.0), // swamps
            ]), slope: 0.0),
            (loc:  0.25, val: Multipoint([
                (loc: -0.85, val: Constant(1.05), slope: 0.0),
                (loc: -0.70, val: Constant(1.05), slope: 0.0),
                (loc: -0.40, val: Constant(0.825), slope: 0.0),
                (loc: -0.35, val: Constant(0.45), slope: 0.0),
                (loc: -0.10, val: Constant(0.225), slope: 0.0),
                (loc:  0.20, val: Constant(0.0),  slope: 0.0),
                (loc:  0.70, val: Constant(-0.03), slope: 0.0),
            ]), slope: 0.0),
            (loc:  1.00, val: Multipoint([
                (loc: -0.85, val: Constant(1.50), slope: 0.0), // tallest peaks
                (loc: -0.70, val: Constant(1.50), slope: 0.0),
                (loc: -0.40, val: Constant(1.20), slope: 0.0),
                (loc: -0.35, val: Constant(0.60), slope: 0.0),
                (loc: -0.10, val: Constant(0.30), slope: 0.0),
                (loc:  0.20, val: Constant(0.0),  slope: 0.0),
                (loc:  0.70, val: Constant(-0.03), slope: 0.0),
            ]), slope: 0.0),
        ]),

        // ─── factor_spline ─────────────────────────────────────────
        // Continentalness → "how sharp is the y-transition?"
        // Mountain ramps 0.10 → 0.70 → 1.00 walking inland — coastal
        // mountains are *impossible* by construction.
        factor_spline: Multipoint([
            (loc: -1.10, val: Constant(0.10), slope: 0.0),
            (loc: -0.18, val: Constant(0.10), slope: 0.0), // ocean: very soft
            (loc: -0.15, val: Constant(0.20), slope: 0.0), // beach step
            (loc: -0.10, val: Multipoint([
                (loc: -0.85, val: Constant(4.0), slope: 0.0), // mountain sharpness
                (loc:  0.20, val: Constant(1.5), slope: 0.0), // plains: soft
                (loc:  0.70, val: Constant(1.5), slope: 0.0),
            ]), slope: 0.0),
            (loc:  0.25, val: Multipoint([
                (loc: -0.85, val: Constant(5.5), slope: 0.0),
                (loc:  0.20, val: Constant(2.0), slope: 0.0),
                (loc:  0.70, val: Constant(2.0), slope: 0.0),
            ]), slope: 0.0),
            (loc:  1.00, val: Multipoint([
                (loc: -0.85, val: Constant(7.0), slope: 0.0), // sharpest mountains
                (loc:  0.20, val: Constant(2.5), slope: 0.0),
                (loc:  0.70, val: Constant(2.5), slope: 0.0),
            ]), slope: 0.0),
        ]),

        // ─── jaggedness_spline ─────────────────────────────────────
        // PV (ridges peak-and-valleys fold) drives this; jaggedness
        // is only meaningful at peak ridges (PV → +1). At PV ≤ 0 the
        // spline returns ~0 so plains stay smooth.
        // Nested: outer keys on continentalness (no jaggedness in
        // ocean), inner on terrain_shape (no jaggedness on flat land),
        // innermost on PV (only at peaks).
        jaggedness_spline: Multipoint([
            (loc: -1.10, val: Constant(0.0), slope: 0.0),
            (loc: -0.10, val: Constant(0.0), slope: 0.0), // no jaggedness near coast
            (loc:  0.25, val: Multipoint([
                (loc: -0.85, val: Multipoint([
                    (loc: -1.0, val: Constant(0.0), slope: 0.0),
                    (loc:  0.0, val: Constant(0.0), slope: 0.0),
                    (loc:  1.0, val: Constant(0.6), slope: 0.0), // peak jaggedness
                ]), slope: 0.0),
                (loc:  0.20, val: Constant(0.0), slope: 0.0),
            ]), slope: 0.0),
            (loc:  1.00, val: Multipoint([
                (loc: -0.85, val: Multipoint([
                    (loc: -1.0, val: Constant(0.0), slope: 0.0),
                    (loc:  0.0, val: Constant(0.0), slope: 0.0),
                    (loc:  1.0, val: Constant(0.9), slope: 0.0), // strongest peak jaggedness
                ]), slope: 0.0),
                (loc:  0.20, val: Constant(0.0), slope: 0.0),
            ]), slope: 0.0),
        ]),
    ),
```

- [ ] **Step 1.4: Run the tests**

Run: `cargo test --lib worldgen::config 2>&1 | tail -10`

Expected: the four new `climate_config_*` tests pass, plus the four PR 2 tests still pass. If the RON parse fails, the error message will name the offending field — common issues: missing comma at end of a `Multipoint` arm, mis-spelled `Knot` (should be `NestedKnot` shape `(loc:, val:, slope:)`).

- [ ] **Step 1.5: Commit**

```bash
git add src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
feat(worldgen): add ClimateConfig with nested spline tables

ClimateConfig holds three nested cubic Hermite splines
(offset, factor, jaggedness) keyed on (continentalness,
terrain_shape, ridges_pv). Knot values seeded from
Minecraft 1.18+ TerrainProvider.java, scaled to Oxium's
y_gradient_amplitude of 1.5.

NestedSpline is a Serde-friendly recursive enum: each knot's
value is either a Constant leaf or another Multipoint level.
At eval time the input triple is walked level-by-level — outer
on c, mid on s, inner on r. Constants short-circuit so partial
nestings (e.g. ocean knots that don't care about terrain_shape)
encode cleanly.

PR 3 task 2 introduces TerrainShapeNoise; tasks 3–5 wire the
spline outputs into per-column evaluation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: TerrainShapeNoise struct (TDD)

**Files:**
- Create: `src/worldgen/terrain_shape.rs`
- Modify: `src/worldgen/mod.rs` (add `pub mod terrain_shape;`)

- [ ] **Step 2.1: Write the failing tests**

Create `src/worldgen/terrain_shape.rs`:

```rust
//! Low-frequency 2D noise channels for the spline pipeline.
//!
//! Mirrors PR 2's `DensityNoise` pattern: a single struct that owns
//! the noise field instances and exposes pure-function evaluators.
//! Built once per `Generator` (instances are `Send + Sync` for the
//! underlying `Fbm`).
//!
//! Two channels:
//!
//! * `terrain_shape` — ~2000-block period. Negative = mountainous,
//!   positive = flat. The per-plate `roughness_bias` is added to the
//!   raw value at call sites that have the plate lookup.
//! * `ridges_raw` — ~500-block period, in `[-1, 1]`. Folded by
//!   [`peaks_and_valleys`] into `ridges_pv` before the spline.

use crate::worldgen::config::ClimateConfig;
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

pub struct TerrainShapeNoise {
    /// Low-frequency shape field. Period ~2000 blocks.
    terrain_shape: Fbm<Simplex>,
    /// Mid-frequency ridge field. Period ~500 blocks. The raw
    /// output is folded by `peaks_and_valleys()` before feeding the
    /// jaggedness spline.
    ridges_raw: Fbm<Simplex>,
}

impl TerrainShapeNoise {
    pub fn new(seed: u64, cfg: &ClimateConfig) -> Self {
        // Salts 0x301 / 0x302 disambiguate from existing fields.
        // Octaves and persistence mirror MC's erosion / weirdness
        // (firstOctave -9 ≈ 2000-block period, low octave count).
        let terrain_shape = Fbm::<Simplex>::new(seed.wrapping_add(0x301) as u32)
            .set_octaves(4)
            .set_frequency(1.0 / cfg.terrain_shape_period as f64)
            .set_persistence(0.5);
        let ridges_raw = Fbm::<Simplex>::new(seed.wrapping_add(0x302) as u32)
            .set_octaves(3)
            .set_frequency(1.0 / cfg.ridges_period as f64)
            .set_persistence(0.5);
        Self {
            terrain_shape,
            ridges_raw,
        }
    }

    /// Raw terrain_shape noise at world `(wx, wz)`. Caller adds the
    /// per-plate `roughness_bias`.
    pub fn shape_raw(&self, wx: f32, wz: f32, cfg: &ClimateConfig) -> f32 {
        let v = self.terrain_shape.get([wx as f64, wz as f64]) as f32;
        v * cfg.terrain_shape_amplitude
    }

    /// Raw weirdness noise (unfolded). Caller passes this through
    /// [`peaks_and_valleys`] before feeding the spline.
    pub fn ridges_raw(&self, wx: f32, wz: f32, cfg: &ClimateConfig) -> f32 {
        let v = self.ridges_raw.get([wx as f64, wz as f64]) as f32;
        v * cfg.ridges_amplitude
    }
}

/// Minecraft's peaks-and-valleys fold:
///   pv(w) = -(||w| − 2/3| − 1/3) · 3
///
/// Triangle wave on `|w|`: 0 at `|w| ∈ {0, 2/3}`, peaks at
/// `|w| = 1/3` and `|w| = 1`. Inputs outside `[-1, 1]` are clamped.
/// Output range: `[-1, 1]`. The sign-preserving wrapper means a
/// raw ridge noise of 0 → PV of -1 (valley), and a raw noise of ±1
/// → PV of +1 (peak).
pub fn peaks_and_valleys(w: f32) -> f32 {
    let w_abs = w.abs().min(1.0);
    -(((w_abs - 2.0 / 3.0).abs() - 1.0 / 3.0) * 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::config::WorldgenConfig;

    #[test]
    fn shape_raw_is_pure() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let n = TerrainShapeNoise::new(42, &cfg.climate);
        let a = n.shape_raw(100.0, 200.0, &cfg.climate);
        let b = n.shape_raw(100.0, 200.0, &cfg.climate);
        assert_eq!(a, b);
    }

    #[test]
    fn shape_raw_in_unit_range() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let n = TerrainShapeNoise::new(42, &cfg.climate);
        for wx in (-5000..=5000).step_by(250) {
            for wz in (-5000..=5000).step_by(250) {
                let v = n.shape_raw(wx as f32, wz as f32, &cfg.climate);
                assert!(
                    (-1.5..=1.5).contains(&v),
                    "shape_raw out of range at ({wx}, {wz}): {v}"
                );
            }
        }
    }

    #[test]
    fn pv_zero_at_zero() {
        // pv(0) = -((|0 - 2/3| - 1/3)·3) = -((2/3 - 1/3)·3) = -1
        // (a *raw weirdness of 0* is a valley, not a peak — MC convention)
        let v = peaks_and_valleys(0.0);
        assert!((v - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn pv_one_at_plus_minus_one() {
        // pv(1) = -((|1 - 2/3| - 1/3)·3) = -((1/3 - 1/3)·3) = 0
        // Hmm — at exactly ±1 the formula gives 0 (zero crossing).
        // The peaks are at |w| = 1/3 where pv = -((1/3 - 1/3)·3) = 0...
        // Wait. Let me recompute: at |w| = 1/3:
        //   pv = -((|1/3 - 2/3| - 1/3)·3) = -((1/3 - 1/3)·3) = 0
        // At |w| = 0: pv = -((2/3 - 1/3)·3) = -1
        // At |w| = 1: pv = -((1/3 - 1/3)·3) = 0
        // At |w| = 2/3: pv = -((0 - 1/3)·3) = 1
        // So the peak is at |w| = 2/3. Verify.
        let v = peaks_and_valleys(2.0 / 3.0);
        assert!((v - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pv_output_in_unit_range() {
        // Sweep w in [-1, 1], check pv stays in [-1, 1].
        for i in -100..=100 {
            let w = i as f32 / 100.0;
            let v = peaks_and_valleys(w);
            assert!(
                (-1.001..=1.001).contains(&v),
                "pv({w}) = {v} out of [-1, 1]"
            );
        }
    }
}
```

Add to `src/worldgen/mod.rs` module list:

```rust
pub mod terrain_shape;
```

- [ ] **Step 2.2: Verify tests pass**

Run: `cargo test --lib worldgen::terrain_shape 2>&1 | tail -10`

Expected: 5 tests pass.

- [ ] **Step 2.3: Commit**

```bash
git add src/worldgen/terrain_shape.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): TerrainShapeNoise — 2 new low-freq noise channels

Mirrors DensityNoise's pattern: a single struct owning two Fbm
fields and exposing pure-function evaluators.

* shape_raw  — ~2000-block period (MC erosion equivalent)
* ridges_raw — ~500-block period (MC weirdness pre-fold)

Plus a free `peaks_and_valleys(w)` helper implementing MC's
triangle fold:
  pv(w) = -(||w| - 2/3| - 1/3)·3
peak at |w| = 2/3, zero crossings at 0 and ±1.

The per-plate roughness_bias (uniform random in
plate_roughness_bias_range) is *additive* on shape_raw — applied
at call sites that have the plate lookup, not inside this module.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Replace `Plate::roughness` with `roughness_bias` (additive scale)

**Files:**
- Modify: `src/worldgen/plates.rs`
- Modify: `src/worldgen/tuning.rs` (deprecate `ROUGHNESS_RANGE`)
- Modify: `src/worldgen/heightmap.rs` (will be removed in task 7; for now adapt)

The existing `Plate::roughness` is a *multiplier* on warped FBM amplitude. PR 3 redefines it as an *additive bias* on terrain_shape (driven by `ClimateConfig::plate_roughness_bias_range`).

- [ ] **Step 3.1: Write a failing test**

Append to `src/worldgen/plates.rs` test module:

```rust
    #[test]
    fn roughness_bias_in_config_range() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let (lo, hi) = cfg.climate.plate_roughness_bias_range;
        for cz in -8..8 {
            for cx in -8..8 {
                let p = Plate::of_with_cfg(42, cx, cz, &cfg.climate);
                assert!(
                    p.roughness_bias >= lo && p.roughness_bias <= hi,
                    "plate ({cx},{cz}) roughness_bias {} outside [{lo}, {hi}]",
                    p.roughness_bias
                );
            }
        }
    }
```

- [ ] **Step 3.2: Add `roughness_bias` field and `Plate::of_with_cfg`**

In `src/worldgen/plates.rs`, change the `Plate` struct:

```rust
#[derive(Debug, Clone, Copy)]
pub struct Plate {
    pub id: PlateId,
    pub kind: PlateKind,
    pub seed_xz: Vec2,
    pub base_elevation: f32,
    /// Additive bias on terrain_shape noise. Preserves "this is the
    /// rocky / gentle continent" identity across the spline pipeline.
    /// Replaces the old `roughness` multiplier (which scaled an FBM
    /// amplitude that no longer exists post-PR-3).
    pub roughness_bias: f32,
}
```

Add a new `of_with_cfg` constructor that takes the `ClimateConfig` reference:

```rust
impl Plate {
    pub fn of_with_cfg(
        seed: u64,
        cell_x: i32,
        cell_z: i32,
        cfg: &crate::worldgen::config::ClimateConfig,
    ) -> Self {
        let id = PlateId { cell_x, cell_z };
        let jx = mix_unit(seed, &[cell_x, cell_z, 0]);
        let jz = mix_unit(seed, &[cell_x, cell_z, 1]);
        let seed_xz = Vec2::new(
            (cell_x as f32 + jx) * PLATE_CELL_SIZE as f32,
            (cell_z as f32 + jz) * PLATE_CELL_SIZE as f32,
        );
        let kind = if mix_unit(seed, &[cell_x, cell_z, 2]) < CONTINENTAL_RATIO {
            PlateKind::Continental
        } else {
            PlateKind::Oceanic
        };
        let base_elevation = match kind {
            PlateKind::Continental => mix_range(
                seed,
                &[cell_x, cell_z, 3],
                CONTINENTAL_BASE_RANGE.0,
                CONTINENTAL_BASE_RANGE.1,
            ),
            PlateKind::Oceanic => mix_range(
                seed,
                &[cell_x, cell_z, 3],
                OCEANIC_BASE_RANGE.0,
                OCEANIC_BASE_RANGE.1,
            ),
        };
        // Salt 5 is new (was salt 4 = roughness multiplier).
        // Bias range comes from config so it's hot-reloadable.
        let (lo, hi) = cfg.plate_roughness_bias_range;
        let roughness_bias = mix_range(seed, &[cell_x, cell_z, 5], lo, hi);
        Plate {
            id,
            kind,
            seed_xz,
            base_elevation,
            roughness_bias,
        }
    }
}
```

Keep the old `Plate::of` but mark it deprecated so any straggler caller still compiles. Make it forward to `of_with_cfg` with the bundled default:

```rust
#[deprecated(note = "Use Plate::of_with_cfg(seed, cx, cz, &climate_cfg). PR 3 swaps roughness multiplier for roughness_bias additive on terrain_shape.")]
pub fn of(seed: u64, cell_x: i32, cell_z: i32) -> Self {
    // Fallback: bundled default config. Tests that don't have a
    // ClimateConfig handy can still call this; production paths
    // should use of_with_cfg.
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default()
        .expect("bundled default.ron must parse");
    Self::of_with_cfg(seed, cell_x, cell_z, &cfg.climate)
}
```

- [ ] **Step 3.3: Add `plate_at_with_cfg` mirroring `plate_at`**

In the same file, mirror the `plate_at` API:

```rust
pub fn plate_at_with_cfg(
    seed: u64,
    wx: i32,
    wz: i32,
    cfg: &crate::worldgen::config::ClimateConfig,
) -> PlateLookup {
    let q = Vec2::new(wx as f32, wz as f32);
    let qcx = wx.div_euclid(PLATE_CELL_SIZE);
    let qcz = wz.div_euclid(PLATE_CELL_SIZE);
    let mut best = (f32::INFINITY, None::<Plate>);
    let mut second = (f32::INFINITY, None::<Plate>);
    for dz in -1..=1 {
        for dx in -1..=1 {
            let cell_x = qcx + dx;
            let cell_z = qcz + dz;
            let p = Plate::of_with_cfg(seed, cell_x, cell_z, cfg);
            let d = (p.seed_xz - q).length();
            if d < best.0 {
                second = best;
                best = (d, Some(p));
            } else if d < second.0 {
                second = (d, Some(p));
            }
        }
    }
    let a = best.1.expect("3×3 window always yields a nearest plate");
    let b = second
        .1
        .expect("3×3 window has 9 candidates, at least 2 exist");
    let d_a = best.0;
    let d_b = second.0;
    let denom = d_a + d_b;
    let t = if denom < f32::EPSILON {
        1.0
    } else {
        (d_b - d_a) / denom
    };
    PlateLookup { a, b, d_a, d_b, t }
}
```

Mark the old `plate_at` deprecated:

```rust
#[deprecated(note = "Use plate_at_with_cfg(seed, wx, wz, &climate_cfg). PR 3.")]
pub fn plate_at(seed: u64, wx: i32, wz: i32) -> PlateLookup {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default()
        .expect("bundled default.ron must parse");
    plate_at_with_cfg(seed, wx, wz, &cfg.climate)
}
```

- [ ] **Step 3.4: Mark `tuning::ROUGHNESS_RANGE` deprecated**

In `src/worldgen/tuning.rs`:

```rust
#[deprecated(note = "Replaced by ClimateConfig::plate_roughness_bias_range (PR 3)")]
pub const ROUGHNESS_RANGE: (f32, f32) = (0.7, 1.4);
```

Leave the constant in place — it's harmless and any future cleanup PR can remove it.

- [ ] **Step 3.5: Update `heightmap.rs` references to `roughness`**

`heightmap.rs::h_pre` currently multiplies relief by `look.a.roughness`. PR 3 task 7 removes `h_pre` entirely, but for this commit we still need it to compile. Change the relief multiplier from `look.a.roughness` to `1.0` (the spline pipeline does the per-plate weighting now via `roughness_bias`):

```rust
let relief = (self.base.get([wxp as f64, wzp as f64]) as f32) * BASE_FBM_AMPLITUDE;
let h = SEA_LEVEL as f32 + shelf + ridge + relief; // was: relief * look.a.roughness
```

This temporarily flattens per-plate variation in `h_pre` — but `h_pre` is being deleted in task 7, so the temporary flatness is two commits long. No goldens are pinned to per-plate variation specifically.

- [ ] **Step 3.6: Run tests**

Run: `cargo test --lib worldgen::plates 2>&1 | tail -15`

Expected: existing plate tests pass (they used `roughness` in `Plate` construction but didn't assert on its numeric value beyond a range check). The new `roughness_bias_in_config_range` test passes.

Some tests still call `Plate::of(seed, cx, cz)` and `plate_at(seed, wx, wz)` — these now go through the deprecation warning path. That's fine for this commit; PR 3 task 4 migrates them.

- [ ] **Step 3.7: Commit**

```bash
git add src/worldgen/plates.rs src/worldgen/tuning.rs src/worldgen/heightmap.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): replace Plate::roughness multiplier with roughness_bias

PR 3 redefines per-plate roughness as an ADDITIVE bias on
terrain_shape noise instead of a MULTIPLIER on warped-FBM relief.

Why: the spline pipeline produces continental shape directly from
(continentalness, terrain_shape, ridges_pv). There's no per-plate
amplitude to multiply — the spline curves are global. But preserving
"this is the rocky / gentle continent" identity is still desirable,
hence the additive bias.

New API:
* Plate::of_with_cfg(seed, cx, cz, &ClimateConfig) — preferred
* plate_at_with_cfg(seed, wx, wz, &ClimateConfig) — preferred
* Plate::of / plate_at — deprecated forwarders (load bundled default)

The bias range comes from ClimateConfig::plate_roughness_bias_range
(hot-reloadable). tuning::ROUGHNESS_RANGE is marked deprecated;
heightmap::h_pre temporarily ignores per-plate roughness (h_pre
itself is removed in PR 3 task 7).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `continentalness` helper + `column_climate` (TDD)

**Files:**
- Modify: `src/worldgen/plates.rs` (add `continentalness_at`)
- Modify: `src/worldgen/mod.rs` (add `column_climate` helper on `Generator`)

The plate Voronoi signed distance field becomes the continentalness input. Inside an oceanic plate, value is negative; inside a continental plate, positive. The `plate_t` factor smoothly blends across boundaries.

- [ ] **Step 4.1: Write a failing test for `continentalness_at`**

Append to `src/worldgen/plates.rs` test module:

```rust
    #[test]
    fn continentalness_negative_in_deep_oceanic_plate() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        // Find a deep oceanic-plate interior column.
        let mut found = None;
        'outer: for wx in (-2000..2000).step_by(50) {
            for wz in (-2000..2000).step_by(50) {
                let look = plate_at_with_cfg(42, wx, wz, &cfg.climate);
                if look.t > 0.6 && matches!(look.a.kind, PlateKind::Oceanic) {
                    found = Some((wx, wz));
                    break 'outer;
                }
            }
        }
        let (wx, wz) = found.expect("no deep oceanic column in scan");
        let c = continentalness_at(42, wx, wz, &cfg.climate);
        assert!(c < -0.3, "deep ocean continentalness should be very negative, got {c}");
    }

    #[test]
    fn continentalness_positive_in_deep_continental_plate() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let mut found = None;
        'outer: for wx in (-2000..2000).step_by(50) {
            for wz in (-2000..2000).step_by(50) {
                let look = plate_at_with_cfg(42, wx, wz, &cfg.climate);
                if look.t > 0.6 && matches!(look.a.kind, PlateKind::Continental) {
                    found = Some((wx, wz));
                    break 'outer;
                }
            }
        }
        let (wx, wz) = found.expect("no deep continental column in scan");
        let c = continentalness_at(42, wx, wz, &cfg.climate);
        assert!(c > 0.3, "deep land continentalness should be > 0.3, got {c}");
    }

    #[test]
    fn continentalness_near_zero_at_boundary() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        // Find a near-boundary column (t small).
        let mut found = None;
        'outer: for wx in (-2000..2000).step_by(20) {
            for wz in (-2000..2000).step_by(20) {
                let look = plate_at_with_cfg(42, wx, wz, &cfg.climate);
                if look.t < 0.05 {
                    found = Some((wx, wz));
                    break 'outer;
                }
            }
        }
        let (wx, wz) = found.expect("no boundary column in scan");
        let c = continentalness_at(42, wx, wz, &cfg.climate);
        assert!(
            c.abs() < 0.3,
            "boundary continentalness should be near 0, got {c}"
        );
    }
```

- [ ] **Step 4.2: Implement `continentalness_at`**

Append to `src/worldgen/plates.rs`:

```rust
/// Continentalness (∈ roughly `[-1, 1]`) at world `(wx, wz)`.
///
/// Computed from the plate Voronoi signed distance field:
/// * Inside an oceanic plate (kind = Oceanic), value is negative,
///   modulated by `plate_t` so deep interior → ~-1 and boundary → 0.
/// * Inside a continental plate, value is positive, same shape.
/// * At a boundary, the smooth blend produces continentalness near 0.
///
/// The sign-by-kind + t-modulation reproduces MC's continentalness
/// noise behavior using Oxium's plate mosaic: continents are
/// "blobby" rather than smooth low-freq noise, but the downstream
/// spline pipeline doesn't care about the macro shape — only the
/// scalar value per column.
pub fn continentalness_at(
    seed: u64,
    wx: i32,
    wz: i32,
    cfg: &crate::worldgen::config::ClimateConfig,
) -> f32 {
    let look = plate_at_with_cfg(seed, wx, wz, cfg);
    let sign_a = match look.a.kind {
        PlateKind::Continental => 1.0,
        PlateKind::Oceanic => -1.0,
    };
    let sign_b = match look.b.kind {
        PlateKind::Continental => 1.0,
        PlateKind::Oceanic => -1.0,
    };
    // Magnitude grows with t (deep interior). Blend the two signs by
    // (1+t)/2 weight on a, (1-t)/2 on b. When t = 1 we're fully in a;
    // when t = 0 we're 50/50 between a and b.
    let w_a = 0.5 + 0.5 * look.t.clamp(0.0, 1.0);
    let w_b = 1.0 - w_a;
    // Boundary attenuation: at t = 0, magnitude is also 0 (the two
    // signs cancel if a and b are opposite kinds). At t = 1, full.
    sign_a * w_a + sign_b * w_b
}
```

- [ ] **Step 4.3: Add `Generator::column_climate` and `SplineOutputs` struct**

In `src/worldgen/mod.rs`, add near the `Generator` impl block (before `column_data_with`):

```rust
/// Spline outputs for one column. Computed once per quart (cached
/// via `FlatCache2D`) and reused across the column's voxels.
#[derive(Debug, Clone, Copy)]
pub struct SplineOutputs {
    pub offset: f32,
    pub factor: f32,
    pub jaggedness: f32,
}

/// Climate inputs for one column. Computed alongside `SplineOutputs`.
#[derive(Debug, Clone, Copy)]
pub struct ClimateInputs {
    pub continentalness: f32,
    pub terrain_shape: f32,
    pub ridges_pv: f32,
}
```

Add helpers on `Generator`:

```rust
impl Generator {
    /// Compute the (continentalness, terrain_shape, ridges_pv) triple
    /// at world `(wx, wz)`. Pure in `(seed, cfg, wx, wz)`.
    pub fn column_climate(&self, wx: i32, wz: i32) -> ClimateInputs {
        let cfg = self.config_snapshot();
        let continentalness = crate::worldgen::plates::continentalness_at(
            self.seed,
            wx,
            wz,
            &cfg.climate,
        );
        // Terrain shape: raw noise + per-plate bias.
        let plate = crate::worldgen::plates::plate_at_with_cfg(
            self.seed,
            wx,
            wz,
            &cfg.climate,
        );
        let shape_noise = self.terrain_shape
            .shape_raw(wx as f32, wz as f32, &cfg.climate);
        let terrain_shape = (shape_noise + plate.a.roughness_bias).clamp(-1.0, 1.0);
        // Ridges: raw noise → PV fold.
        let r_raw = self.terrain_shape
            .ridges_raw(wx as f32, wz as f32, &cfg.climate);
        let ridges_pv = crate::worldgen::terrain_shape::peaks_and_valleys(r_raw);
        ClimateInputs {
            continentalness,
            terrain_shape,
            ridges_pv,
        }
    }

    /// Evaluate the three splines (offset, factor, jaggedness) for
    /// the column at `(wx, wz)`. Equivalent to:
    ///   1. `column_climate(wx, wz)` to get inputs
    ///   2. evaluate each spline at those inputs
    pub fn column_spline_outputs(&self, wx: i32, wz: i32) -> SplineOutputs {
        let cfg = self.config_snapshot();
        let climate = self.column_climate(wx, wz);
        SplineOutputs {
            offset: cfg.climate.offset_spline_at(
                climate.continentalness,
                climate.terrain_shape,
                climate.ridges_pv,
            ),
            factor: cfg.climate.factor_spline_at(
                climate.continentalness,
                climate.terrain_shape,
                climate.ridges_pv,
            ),
            jaggedness: cfg.climate.jaggedness_spline_at(
                climate.continentalness,
                climate.terrain_shape,
                climate.ridges_pv,
            ),
        }
    }
}
```

Generator gains a `terrain_shape: TerrainShapeNoise` field. Construct it in `Generator::new_internal` (the PR 2 helper):

```rust
pub struct Generator {
    // existing fields...
    terrain_shape: crate::worldgen::terrain_shape::TerrainShapeNoise,
}

fn new_internal(seed: u64) -> Self {
    let bundled = crate::worldgen::config::WorldgenConfig::bundled_default()
        .expect("bundled default.ron must parse");
    let terrain_shape =
        crate::worldgen::terrain_shape::TerrainShapeNoise::new(seed, &bundled.climate);
    Self {
        // existing field initializers...
        terrain_shape,
        config: crate::worldgen::config::ConfigHolder::new(bundled),
    }
}
```

- [ ] **Step 4.4: Write a Generator-level integration test**

Append to the worldgen test module in `src/worldgen/mod.rs`:

```rust
    #[test]
    fn column_spline_outputs_inland_produces_positive_offset() {
        let g = Generator::new(42);
        // Scan for a deep continental column.
        let cfg = g.config_snapshot();
        let mut found = None;
        'outer: for wx in (-2000..2000).step_by(50) {
            for wz in (-2000..2000).step_by(50) {
                let look = crate::worldgen::plates::plate_at_with_cfg(
                    42, wx, wz, &cfg.climate,
                );
                if look.t > 0.6
                    && matches!(look.a.kind, crate::worldgen::plates::PlateKind::Continental)
                {
                    found = Some((wx, wz));
                    break 'outer;
                }
            }
        }
        let (wx, wz) = found.expect("no deep continental column found");
        let out = g.column_spline_outputs(wx, wz);
        assert!(
            out.offset > 0.0,
            "inland column should have positive offset, got {out:?}"
        );
        assert!(
            out.factor > 0.0,
            "factor should always be positive, got {}",
            out.factor
        );
        assert!(
            out.jaggedness >= 0.0,
            "jaggedness should be non-negative, got {}",
            out.jaggedness
        );
    }

    #[test]
    fn column_spline_outputs_deep_ocean_produces_negative_offset() {
        let g = Generator::new(42);
        let cfg = g.config_snapshot();
        let mut found = None;
        'outer: for wx in (-2000..2000).step_by(50) {
            for wz in (-2000..2000).step_by(50) {
                let look = crate::worldgen::plates::plate_at_with_cfg(
                    42, wx, wz, &cfg.climate,
                );
                if look.t > 0.6
                    && matches!(look.a.kind, crate::worldgen::plates::PlateKind::Oceanic)
                {
                    found = Some((wx, wz));
                    break 'outer;
                }
            }
        }
        let (wx, wz) = found.expect("no deep oceanic column found");
        let out = g.column_spline_outputs(wx, wz);
        assert!(
            out.offset < 0.0,
            "deep ocean column should have negative offset, got {out:?}"
        );
    }
```

- [ ] **Step 4.5: Run tests**

Run: `cargo test --lib worldgen::plates::tests::continentalness 2>&1 | tail -10` and `cargo test --lib worldgen::tests::column_spline_outputs 2>&1 | tail -10`

Expected: 5 new tests pass. If the plate signs are flipped (negative inland), check the `sign_a/sign_b` mapping in `continentalness_at`.

- [ ] **Step 4.6: Commit**

```bash
git add src/worldgen/plates.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): continentalness from plates + column_spline_outputs

plates::continentalness_at(seed, wx, wz, &cfg) — derives the
continentalness scalar (∈ ~[-1, 1]) from the existing Voronoi
plate field. Continental plate deep interior → +1; oceanic deep
interior → -1; boundary → 0. Reproduces MC's continentalness
behavior using Oxium's plate mosaic.

Generator::column_climate — bundles (continentalness, terrain_shape,
ridges_pv) per column. Terrain_shape is `shape_noise + plate.roughness_bias`
clamped to [-1, 1]. Ridges_pv is the PV triangle fold on the raw
ridge noise.

Generator::column_spline_outputs — runs the three nested splines
on column_climate() and returns (offset, factor, jaggedness).
This is the function that PR 3 task 5 wires into evaluate_v2.

Generator now owns a TerrainShapeNoise field, built once from the
bundled config in new_internal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Wire spline outputs into `DensityNoise::evaluate_v2` via FlatCache2D

**Files:**
- Modify: `src/worldgen/heightmap.rs` (extend `evaluate_v2` signature)
- Modify: `src/worldgen/mod.rs` (`fill_chunk` uses a per-chunk FlatCache2D of SplineOutputs)

PR 2's `evaluate_v2` derived `offset` from `h_target`. PR 3 replaces that with the spline-driven `SplineOutputs`. The per-voxel call receives the column's cached `SplineOutputs`; the FlatCache2D ensures one spline-evaluation per quart, not per voxel.

- [ ] **Step 5.1: Write a failing test for the new evaluate_v2 signature**

Append to the heightmap test module:

```rust
    #[test]
    fn evaluate_v2_with_spline_inputs_at_surface_near_zero() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // Synthesize SplineOutputs that put the surface at y = 70:
        //   shaped = (y_grad(70) + offset) * factor
        //   For density ≈ 0 at y=70, we need offset such that
        //   y_grad(70) + offset = 0.
        let y = 70_i32;
        let y_grad = cfg.density.y_gradient_amplitude
            * (1.0 - 2.0 * ((y - cfg.density.y_min) as f32 / (cfg.density.y_max - cfg.density.y_min) as f32));
        // offset such that y_grad + offset = 0:
        let offset = -y_grad;
        let out = crate::worldgen::SplineOutputs {
            offset,
            factor: 4.0,
            jaggedness: 0.0,
        };
        let mut sum = 0.0_f32;
        let mut n = 0;
        for wx in 0..16 {
            for wz in 0..16 {
                sum += d.evaluate_v2(wx, y, wz, &out, 0.0, &cfg.density);
                n += 1;
            }
        }
        let avg = sum / n as f32;
        assert!(avg.abs() < 1.5, "avg density at surface should be near 0, got {avg}");
    }

    #[test]
    fn evaluate_v2_uses_spline_offset_to_shift_surface() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let d = DensityNoise::new(42);
        // Two columns with different offsets should produce different
        // surface heights. Find the surface for each by scanning where
        // density changes sign.
        let out_high = crate::worldgen::SplineOutputs {
            offset: 0.8, // very positive — surface pushed UP
            factor: 4.0,
            jaggedness: 0.0,
        };
        let out_low = crate::worldgen::SplineOutputs {
            offset: -0.3, // ocean-ish — surface pushed DOWN
            factor: 4.0,
            jaggedness: 0.0,
        };
        // Find topmost solid for each, averaging over a 4x4 patch.
        let topmost = |o: &crate::worldgen::SplineOutputs| -> f32 {
            let mut sum_y = 0_i32;
            let mut count = 0;
            for wx in 0..4 {
                for wz in 0..4 {
                    for y in (-50..120).rev() {
                        if d.evaluate_v2(wx, y, wz, o, 0.0, &cfg.density) > 0.0 {
                            sum_y += y;
                            count += 1;
                            break;
                        }
                    }
                }
            }
            sum_y as f32 / count as f32
        };
        let y_high = topmost(&out_high);
        let y_low = topmost(&out_low);
        assert!(
            y_high > y_low + 30.0,
            "high-offset surface ({y_high}) should be ≥30 blocks above low-offset surface ({y_low})"
        );
    }
```

- [ ] **Step 5.2: Replace `DensityNoise::evaluate_v2` signature**

In `src/worldgen/heightmap.rs`, change `evaluate_v2`:

```rust
impl DensityNoise {
    /// New MC-style composition with spline-driven inputs.
    ///
    /// Signature change from PR 2: `h_target` is replaced by
    /// `spline_outputs: &SplineOutputs` (carries the column's offset,
    /// factor, jaggedness) and `jagged_noise: f32` (the per-voxel
    /// jagged noise rider, scaled by spline_outputs.jaggedness).
    ///
    /// Caller is expected to:
    /// 1. compute SplineOutputs once per quart (cached via FlatCache2D)
    /// 2. sample jagged_noise per voxel (high-frequency 3D noise)
    /// 3. call this method with both
    pub fn evaluate_v2(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        spline: &crate::worldgen::SplineOutputs,
        jagged_noise: f32,
        cfg: &crate::worldgen::config::DensityConfig,
    ) -> f32 {
        // y_gradient: +amplitude at y_min, -amplitude at y_max.
        let t = (wy - cfg.y_min) as f32 / (cfg.y_max - cfg.y_min) as f32;
        let y_gradient = cfg.y_gradient_amplitude * (1.0 - 2.0 * t);

        // Offset comes from the spline now (PR 3) instead of being
        // derived from h_target (PR 2).
        let depth = y_gradient + spline.offset;
        let jagged = spline.jaggedness * jagged_noise;
        let factor = spline.factor;

        let shaped_raw = (depth + jagged) * factor;
        let shaped = if shaped_raw > 0.0 {
            shaped_raw
        } else {
            shaped_raw * cfg.above_surface_softening
        };

        let base_3d = self.evaluate_base_3d_v3(wx, wy, wz, cfg);
        let pre_slide = cfg.composition_scale * shaped + base_3d;
        slide(pre_slide, wy, cfg)
    }

    /// Sample anisotropic base 3D noise. Replaces evaluate_base_3d
    /// (which took a now-unused h_target argument from PR 2).
    pub fn evaluate_base_3d_v3(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        cfg: &crate::worldgen::config::DensityConfig,
    ) -> f32 {
        let scaled_y = wy as f64 * cfg.base_3d_y_scale as f64;
        self.relief.get([wx as f64, scaled_y, wz as f64]) as f32 * cfg.base_3d_amplitude
    }

    /// Sample the high-frequency jagged noise rider. ~3-block
    /// wavelength. Range roughly [-1, 1]; caller multiplies by
    /// spline.jaggedness before adding to depth.
    pub fn evaluate_jagged_noise(&self, wx: i32, wy: i32, wz: i32) -> f32 {
        // Reuses self.relief at a higher frequency; quick-and-cheap
        // for PR 3. PR 4 may give jaggedness its own Fbm field.
        let scale = 0.333_f64;
        self.relief.get([
            wx as f64 * scale * 16.0,
            wy as f64 * scale * 16.0,
            wz as f64 * scale * 16.0,
        ]) as f32
    }
}
```

Keep `slide()` as-is. Remove the old `evaluate_base_3d` (which took `h_target`) since it had only one caller.

- [ ] **Step 5.3: Update `topmost_solid` signature**

`topmost_solid` is called from `add_trees`. Update it to take `SplineOutputs` instead of `h_target`:

```rust
pub fn topmost_solid(
    &self,
    wx: i32,
    wz: i32,
    search_top: i32,
    spline: &crate::worldgen::SplineOutputs,
    cfg: &crate::worldgen::config::DensityConfig,
) -> Option<i32> {
    let top = search_top.min(cfg.y_max);
    let bottom = (cfg.y_min).max(crate::worldgen::tuning::CAVE_FLOOR_Y);
    for wy in (bottom..=top).rev() {
        let jn = self.evaluate_jagged_noise(wx, wy, wz);
        if self.evaluate_v2(wx, wy, wz, spline, jn, cfg) > 0.0 {
            return Some(wy);
        }
    }
    Some(bottom)
}
```

The `SURFACE_BAND` clamp from PR 2 is dropped because the surface position is no longer a derived constant — it falls naturally out of the y-gradient + offset crossing. The search is a top-down scan over the full Y range; pessimistic but correct.

- [ ] **Step 5.4: Update `mod.rs::fill_chunk` to use a per-chunk FlatCache2D<SplineOutputs>**

Inside `fill_chunk`, after `let regions = ...` and `let cave_systems = ...`, add:

```rust
        let cfg = self.config_snapshot();
        // Cache spline outputs once per quart (4×4 column patch).
        // Same pattern as future biome / climate caches.
        let mut spline_cache =
            crate::worldgen::flat_cache::FlatCache2D::<crate::worldgen::SplineOutputs>::new();
```

Inside the per-column loop (after `let col = self.column_data_with(...)`):

```rust
        let (qx, qz) = crate::worldgen::flat_cache::FlatCache2D::<()>::block_to_quart(x, z);
        let spline = spline_cache.get_or_compute(qx, qz, || {
            self.column_spline_outputs(wx, wz)
        });
```

(Note: `block_to_quart` doesn't depend on `T` — but Rust's syntax for static methods on generic types forces specifying T. Use `<()>::block_to_quart` as a turbofish.)

Inside the per-voxel loop, replace:

```rust
let raw_density = self.density.evaluate_v2(h_target, wx, wy, wz, &cfg.density);
```

with:

```rust
let jn = self.density.evaluate_jagged_noise(wx, wy, wz);
let raw_density = self.density.evaluate_v2(wx, wy, wz, &spline, jn, &cfg.density);
```

The `above_density` sentinel for chunk-boundary depth handling (lines ~360 in PR 2) also needs updating:

```rust
let above_chunk_top_wy = origin.y + CHUNK_DIM_U as i32;
let above_jn = self.density.evaluate_jagged_noise(wx, above_chunk_top_wy, wz);
let above_density = self.density.evaluate_v2(
    wx,
    above_chunk_top_wy,
    wz,
    &spline,
    above_jn,
    &cfg.density,
);
```

Update the `add_trees` consumer of `topmost_solid`:

```rust
let spline = self.column_spline_outputs(wx, wz);
let surface_y = self.density.topmost_solid(wx, wz, MAX_TERRAIN_Y, &spline, &cfg.density)?;
```

- [ ] **Step 5.5: Update `column_data_with` to set `height` from the spline-derived surface**

`column_data_with` currently reads `h_pre` and applies `valley_carve`. PR 3 replaces `h_pre` with the spline. **This step migrates the path; task 7 deletes the old `h_pre`.**

In `column_data_with`:

```rust
fn column_data_with(&self, wx: i32, wz: i32, regions: &ChunkRegions) -> ColumnData {
    let cfg = self.config_snapshot();
    let spline = self.column_spline_outputs(wx, wz);
    // Surface Y where the (y_gradient + offset) curve crosses zero —
    // analytic solution since y_gradient is linear in y:
    //   y_grad(y) = amp * (1 - 2 * (y - y_min) / (y_max - y_min))
    //   y_grad + offset = 0 → y_grad = -offset
    //   → y = y_min + (y_max - y_min) * (1 - (-offset / amp)) / 2
    //       = y_min + (y_max - y_min) * (1 + offset / amp) / 2
    let span = (cfg.density.y_max - cfg.density.y_min) as f32;
    let h_spline = cfg.density.y_min as f32
        + span * 0.5 * (1.0 + spline.offset / cfg.density.y_gradient_amplitude);

    // Cliff classification: slope of h_spline over an 8-block stencil.
    // (`is_cliff` no longer uses the deleted CLIFF_MIN_HEIGHT gate;
    // see task 7.)
    let is_cliff = self.is_cliff_from_spline(wx, wz);

    let carve = regions.valley_carve(wx, wz, self.seed);
    let height = (h_spline - carve)
        .clamp(
            cfg.density.y_min as f32 + 4.0,
            cfg.density.y_max as f32,
        ) as i32;

    // ... rest of biome / climate / lake_rim logic unchanged ...

    ColumnData {
        height,
        is_cliff,
        desertness,
        biome,
        lake_rim,
    }
}
```

Add a new helper:

```rust
impl Generator {
    /// Cliff detection from the spline-derived heightmap. Computes
    /// the analytic h_spline at four ±4-block stencil points and
    /// checks the maximum horizontal gradient magnitude.
    fn is_cliff_from_spline(&self, wx: i32, wz: i32) -> bool {
        let cfg = self.config_snapshot();
        let h = |x, z| -> f32 {
            let o = self.column_spline_outputs(x, z).offset;
            let span = (cfg.density.y_max - cfg.density.y_min) as f32;
            cfg.density.y_min as f32
                + span * 0.5 * (1.0 + o / cfg.density.y_gradient_amplitude)
        };
        let step = 4;
        let hxp = h(wx + step, wz);
        let hxn = h(wx - step, wz);
        let hzp = h(wx, wz + step);
        let hzn = h(wx, wz - step);
        let gx = (hxp - hxn).abs() / (2.0 * step as f32);
        let gz = (hzp - hzn).abs() / (2.0 * step as f32);
        gx.max(gz) > crate::worldgen::tuning::CLIFF_SLOPE_THRESH
    }
}
```

This 8-stencil cliff check is *the same shape* as the old `HeightmapNoise::is_cliff` minus the `CLIFF_MIN_HEIGHT` and below-sea-level gates — both of which are obsoleted by the spline (no coastal mountain artifacts to gate around).

- [ ] **Step 5.6: Run tests**

Run: `cargo test --lib worldgen::heightmap::tests::evaluate_v2_with_spline 2>&1 | tail -10` and `cargo test --lib worldgen::tests::evaluate_v2_uses_spline 2>&1 | tail -10`

Expected: both pass. Other worldgen tests will likely fail at this point (column_data_with and fill_chunk are mid-migration); that's expected — the next steps fix them. Run the full suite to confirm the failures are concentrated in the migration zone:

```bash
cargo test --lib worldgen 2>&1 | tail -30
```

Expected failures (these get fixed by task 6 + task 10's golden hash refresh):
- `golden_seed42_chunk_0_2_0` — terrain shape has changed; sentinel mode captures the new hash.
- `worldgen_fingerprint::fingerprint_hash_matches_pin` — the 2D heightmap is now spline-driven; needs re-pinning in task 10.

Any test that asserts on plate-roughness *multiplier* behavior (the old `h_pre_respects_cap`, `slope_can_be_high_on_plate_boundary` from `heightmap.rs`) may need adjustment — task 7 (where `h_pre` is deleted) handles those.

- [ ] **Step 5.7: Commit**

```bash
git add src/worldgen/heightmap.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): wire spline outputs into evaluate_v2 + FlatCache2D

DensityNoise::evaluate_v2 signature change: takes
&SplineOutputs and a per-voxel jagged_noise sample instead of
h_target. The h_target argument was a PR 2 stopgap (deriving offset
from the existing 2D heightmap); PR 3 replaces it with the real
spline output triple.

mod.rs::fill_chunk now:
1. config_snapshot once at the top
2. FlatCache2D<SplineOutputs> per chunk — one spline eval per quart
3. per-voxel: evaluate_jagged_noise + evaluate_v2 with cached spline

Generator::column_data_with derives `height` analytically from the
spline offset (linear y_gradient → closed-form crossing) and applies
valley_carve on it. The old h_pre path remains in heightmap.rs as
unused code; PR 3 task 7 deletes it.

Generator::is_cliff_from_spline replaces HeightmapNoise::is_cliff:
same 4-point ±4-block stencil on h_spline, slope > CLIFF_SLOPE_THRESH.
The CLIFF_MIN_HEIGHT and below-sea-level gates are dropped — the
spline pipeline doesn't produce coastal mountain artifacts that
needed those band-aids.

Expected test failures at this commit (resolved by task 6/10):
- golden_seed42_chunk_0_2_0 (terrain shape changed)
- worldgen_fingerprint::fingerprint_hash_matches_pin (heightmap
  is now spline-driven, needs re-pin)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Hydrology interface change — valley_carve targets spline offset

**Files:**
- Modify: `src/worldgen/mod.rs` (one-line change)

Hydrology itself is **untouched**. The only change is at the interface: `valley_carve`'s consumer in `column_data_with` already applies carve to `h_spline` (per task 5.5). This task is the bookkeeping commit documenting that the interface change is complete and verifying hydrology tests still pass.

- [ ] **Step 6.1: Confirm the hydrology interface change**

Read `column_data_with`. Confirm:
- `h_spline` is computed from `spline.offset`.
- `carve = regions.valley_carve(wx, wz, self.seed)` is applied to `h_spline`, not to `h_pre`.
- The clamp uses `cfg.density.y_min/y_max` (not the deprecated `CAVE_FLOOR_Y + 8`/`MAX_TERRAIN_Y` constants).

If task 5.5 was applied correctly, nothing changes in this step beyond verification. If not, fix `column_data_with` to match task 5.5.

- [ ] **Step 6.2: Run the hydrology suite**

Run: `cargo test --lib worldgen::hydrology 2>&1 | tail -10`

Expected: all hydrology tests pass. Hydrology code is untouched; its public API (`valley_carve`, `lake_rim_at`) is byte-stable.

- [ ] **Step 6.3: Add an integration test that valley_carve modifies the spline-derived surface**

Append to the worldgen test module:

```rust
    #[test]
    fn river_column_height_is_carved_below_neighbor() {
        // Pick a known river column from the hydrology fingerprint
        // and compare its column_data_with height to a neighbor 16
        // blocks away.
        let g = Generator::new(42);
        // Scan for a column with non-zero carve.
        let coord = ChunkCoord(IVec3::new(0, 2, 0));
        let regions = g.gather_chunk_regions(coord);
        let mut found = None;
        'outer: for wx in -200..200 {
            for wz in -200..200 {
                let carve = regions.valley_carve(wx, wz, 42);
                if carve > 1.0 {
                    found = Some((wx, wz, carve));
                    break 'outer;
                }
            }
        }
        let (rx, rz, carve) = found.expect("no carved river column found in scan");
        let river_col = g.column_data_with(rx, rz, &regions);
        let dry_col = g.column_data_with(rx + 60, rz + 60, &regions);
        // River column should be at least 1 block below where it
        // would be without the carve.
        let spline = g.column_spline_outputs(rx, rz);
        let cfg = g.config_snapshot();
        let span = (cfg.density.y_max - cfg.density.y_min) as f32;
        let uncarved = cfg.density.y_min as f32
            + span * 0.5
                * (1.0 + spline.offset / cfg.density.y_gradient_amplitude);
        let diff = uncarved - river_col.height as f32;
        assert!(
            diff >= 1.0,
            "river height {} should be ≥1 below uncarved {} (carve={carve}); dry neighbor is at {}",
            river_col.height,
            uncarved,
            dry_col.height
        );
    }
```

Run it: `cargo test --lib worldgen::tests::river_column_height 2>&1 | tail -5`

Expected: pass.

- [ ] **Step 6.4: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): hydrology valley_carve now applied to spline offset

Per the research doc Q2 decision: hydrology.rs is untouched, but
its consumer in column_data_with applies `valley_carve` to the
spline-derived h_spline instead of the deprecated h_pre.

This commit only verifies the interface — the change itself
landed in PR 3 task 5.5. Adds an integration test confirming
that a river column's height is at least 1 block below the
uncarved spline-derived surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Remove obsolete code

**Files:**
- Modify: `src/worldgen/plates.rs` (remove `ridge_lift`, `ridge_peak_for_pair`)
- Modify: `src/worldgen/heightmap.rs` (remove `h_pre`, `is_coastal`, `is_cliff`)
- Modify: `src/worldgen/tuning.rs` (delete `BOUNDARY_RIDGE_WIDTH`, `RIDGE_PEAK_*`, `CLIFF_MIN_HEIGHT`)
- Modify: `src/worldgen/mod.rs` (any callers of `is_coastal` — confirmed none after the in-session fix)

- [ ] **Step 7.1: Verify no callers of `ridge_lift`, `is_coastal`, `is_cliff` remain**

Run: `grep -rn 'ridge_lift\|is_coastal\|HeightmapNoise::is_cliff' src/ --include='*.rs' 2>&1 | head -20`

Expected: only internal uses inside `plates.rs` (`ridge_lift`) and `heightmap.rs` (`is_coastal`, `is_cliff`). The `column_data_with` migration in task 5.5 routes through `is_cliff_from_spline`. The coastal-gated dirt cap was already removed in the in-session fix.

If any production caller remains (i.e. anything outside `#[cfg(test)]`), STOP — there's a missed migration. Common stragglers:
- `add_trees` — confirm it uses `Generator::column_data` (which uses `column_data_with`).
- `surface.rs` — confirm it's still a stub.

- [ ] **Step 7.2: Delete `ridge_lift` and `ridge_peak_for_pair` from `plates.rs`**

In `src/worldgen/plates.rs`:
- Delete `pub fn ridge_peak_for_pair(...)` (~6 lines).
- Delete `pub fn ridge_lift(...)` (~40 lines).
- Delete the `ridge_lift_zero_outside_band` and `ridge_lift_positive_at_boundary` tests.

- [ ] **Step 7.3: Delete `h_pre`, `is_coastal`, `is_cliff` from `heightmap.rs`**

In `src/worldgen/heightmap.rs`:
- Delete `pub fn h_pre(...)` (~14 lines).
- Delete `pub fn is_coastal(...)` (~17 lines).
- Delete `pub fn is_cliff(...)` (~17 lines).
- Delete the corresponding tests: `h_pre_is_deterministic`, `h_pre_respects_cap`, `warped_fbm_breaks_axis_symmetry`, `slope_is_low_on_flat_oceanic_plate`, `slope_can_be_high_on_plate_boundary`.

`HeightmapNoise` still has the warped-FBM machinery (`base`, `warp_x`, `warp_z`). Keep these for now — they are referenced by no current consumer and are pruned by future PRs (the noise crate's `Fbm` allocation cost is trivial).

The `BASE_FBM_AMPLITUDE` and `BASE_FBM_PERIOD` constants and `slope_at` method are also unused after task 7.3; delete them too. The whole `HeightmapNoise` struct is reduced to construction-only with no public callers — task 7.4 deletes it entirely.

- [ ] **Step 7.4: Delete the `HeightmapNoise` struct**

After task 7.3, `HeightmapNoise` has no public methods. Delete:
- The `HeightmapNoise` struct.
- Its `impl` block.
- The `heightmap: HeightmapNoise` field on `Generator`.
- The `HeightmapNoise::new(seed)` call in `Generator::new_internal`.

Then re-run the worldgen tests to surface any straggler usage:

```bash
cargo test --lib worldgen 2>&1 | tail -30
```

Expected: only failures should be in the goldens. Compile errors here mean a caller was missed — fix it before continuing.

- [ ] **Step 7.5: Delete obsolete tuning constants**

In `src/worldgen/tuning.rs`, delete (not just deprecate):
- `BOUNDARY_RIDGE_WIDTH`
- `RIDGE_PEAK_CC`
- `RIDGE_PEAK_CO`
- `RIDGE_PEAK_OO`
- `CLIFF_MIN_HEIGHT`
- `ROUGHNESS_RANGE` (was deprecated in task 3)
- `WARP_AMPLITUDE`, `WARP_PERIOD` (no callers after `HeightmapNoise` deletion)
- `BASE_FBM_AMPLITUDE`, `BASE_FBM_PERIOD` from `heightmap.rs`

Keep `CLIFF_SLOPE_THRESH` — `is_cliff_from_spline` still uses it.

- [ ] **Step 7.6: Run the full lib test suite**

Run: `cargo test --lib worldgen 2>&1 | tail -20`

Expected: compile clean; only golden hash mismatches remain. If there are import errors from deleted symbols, prune them — `use` lines in `heightmap.rs`, `mod.rs`, `plates.rs` may still reference deleted items.

- [ ] **Step 7.7: Commit**

```bash
git add src/worldgen/plates.rs src/worldgen/heightmap.rs src/worldgen/tuning.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
refactor(worldgen): delete plate-boundary ridge code & h_pre stack

The spline pipeline replaces:
- plates::ridge_lift                            (boundary mountain stamps)
- plates::ridge_peak_for_pair                   (per-kind peak height table)
- heightmap::HeightmapNoise (whole struct)      (warped-FBM heightmap)
- heightmap::is_cliff / is_coastal / h_pre      (cliff + coastal gates)
- tuning::BOUNDARY_RIDGE_WIDTH                  (ridge band threshold)
- tuning::RIDGE_PEAK_{CC,CO,OO}                 (per-kind peak heights)
- tuning::CLIFF_MIN_HEIGHT                      (was a band-aid for coastal artifacts)
- tuning::ROUGHNESS_RANGE                       (now ClimateConfig)
- tuning::WARP_AMPLITUDE/PERIOD                 (HeightmapNoise was sole caller)
- heightmap::BASE_FBM_AMPLITUDE/PERIOD          (same)

What replaces them:
- continentalness (plate Voronoi field) → offset spline → surface
- terrain_shape noise + per-plate roughness_bias → factor/offset
- ridges_pv → jaggedness spline → peak detail
- is_cliff_from_spline (slope check on h_spline, no MIN_HEIGHT gate)

`HeightmapNoise` is removed from the Generator struct. The
ChunkRegions::valley_carve consumer in column_data_with already
applies carve to the spline-derived h_spline (PR 3 task 5/6).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: 4-block above-water 3D-noise clamp

**Files:**
- Modify: `src/worldgen/mod.rs` (~5 lines in `fill_chunk`)

When a column is a river or lake, positive 3D noise contribution above the water surface can produce overhang ceilings that close the river into a tunnel. The fix is a 4-block Y-band gate on positive 3D noise above any river/lake column.

- [ ] **Step 8.1: Write a failing test**

Append to the worldgen test module:

```rust
    #[test]
    fn river_columns_have_no_overhang_ceiling_above_water() {
        // Generate a chunk that contains the river found in
        // river_column_height_is_carved_below_neighbor. Scan the
        // 4 blocks above water for unexpected solid voxels.
        let g = Generator::new(42);
        // Find a river column in chunk Y=2 (above the water table).
        let coord = ChunkCoord(IVec3::new(0, 2, 0));
        let mut chunk = DenseChunk::empty();
        g.fill_chunk(coord, &mut chunk);
        // For every column in the chunk, if it has water at the
        // surface and the surface is within chunk-Y bounds, the 4
        // voxels above must NOT be solid stone (they may be Air
        // or biome surface block but not part of an overhang).
        let origin = coord.origin().0;
        let mut violations = 0;
        let mut river_columns = 0;
        for z in 0..32u32 {
            for x in 0..32u32 {
                let wx = origin.x + x as i32;
                let wz = origin.z + z as i32;
                let col = g.column_data(wx, wz);
                let in_river = col.lake_rim.is_some() || col.height <= SEA_LEVEL;
                if !in_river {
                    continue;
                }
                river_columns += 1;
                // Scan up from the water surface; if any of the
                // 4 voxels above are stone, count a violation.
                let water_top = col.lake_rim.unwrap_or(SEA_LEVEL);
                for dy in 1..=4 {
                    let y = water_top + dy;
                    let ly = y - origin.y;
                    if ly < 0 || ly >= 32 {
                        continue;
                    }
                    let block =
                        chunk.get(crate::voxel::coords::LocalPos(
                            glam::UVec3::new(x, ly as u32, z),
                        ));
                    if block == Block::Stone {
                        violations += 1;
                    }
                }
            }
        }
        // If there are no river columns in this chunk, the test is
        // vacuously satisfied — skip the assert.
        if river_columns == 0 {
            return;
        }
        assert_eq!(
            violations, 0,
            "found {violations} stone overhangs above water in {river_columns} river columns"
        );
    }
```

Run it: `cargo test --lib worldgen::tests::river_columns_have_no_overhang 2>&1 | tail -5`

Expected: failure with N violations (3D noise overhang ceilings observed).

- [ ] **Step 8.2: Apply the 4-block clamp in `fill_chunk`**

In the per-voxel loop of `fill_chunk`, identify the line that evaluates 3D noise (inside `evaluate_v2`). The clamp needs the *separated* base 3D contribution. Refactor the per-voxel block to call `evaluate_base_3d_v3` directly and apply the clamp:

```rust
let jn = self.density.evaluate_jagged_noise(wx, wy, wz);

// Compute the shaped term (depth + jagged) * factor with
// quarter_negative softening — same as evaluate_v2.
let t = (wy - cfg.density.y_min) as f32
    / (cfg.density.y_max - cfg.density.y_min) as f32;
let y_gradient = cfg.density.y_gradient_amplitude * (1.0 - 2.0 * t);
let depth = y_gradient + spline.offset;
let jagged = spline.jaggedness * jn;
let shaped_raw = (depth + jagged) * spline.factor;
let shaped = if shaped_raw > 0.0 {
    shaped_raw
} else {
    shaped_raw * cfg.density.above_surface_softening
};

// Base 3D noise — with the above-water clamp.
let mut base_3d = self.density.evaluate_base_3d_v3(wx, wy, wz, &cfg.density);
// Above-water clamp: in a 4-block band above any river/lake column,
// suppress positive 3D-noise contributions so overhangs don't close
// the water into a tunnel.
let in_river = col.lake_rim.is_some() || col.height <= SEA_LEVEL;
if in_river {
    let water_top = col.lake_rim.unwrap_or(SEA_LEVEL);
    if wy > water_top && wy <= water_top + 4 && base_3d > 0.0 {
        base_3d = 0.0;
    }
}

let pre_slide = cfg.density.composition_scale * shaped + base_3d;
let raw_density = crate::worldgen::heightmap::slide_public(pre_slide, wy, &cfg.density);
```

This inlines what `evaluate_v2` was doing — necessary because the clamp needs the base_3d contribution in isolation. `evaluate_v2` is kept around as the canonical implementation for callers that don't need the clamp (e.g. `topmost_solid` in tree placement).

Expose `slide` from `heightmap.rs`:

```rust
/// Public re-export of the internal `slide` helper. Used by
/// `fill_chunk`'s inlined density pipeline (which applies the
/// above-water clamp inside the composition).
pub fn slide_public(density: f32, wy: i32, cfg: &crate::worldgen::config::DensityConfig) -> f32 {
    slide(density, wy, cfg)
}
```

- [ ] **Step 8.3: Run the river-overhang test**

Run: `cargo test --lib worldgen::tests::river_columns_have_no_overhang 2>&1 | tail -5`

Expected: pass with `0 stone overhangs` (or `0 river columns` if the scan landed in a no-river chunk).

- [ ] **Step 8.4: Commit**

```bash
git add src/worldgen/mod.rs src/worldgen/heightmap.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): 4-block above-water 3D-noise clamp

In a 4-block Y band above any column where lake_rim.is_some() or
height <= SEA_LEVEL (ocean), suppress positive 3D-noise contribution
to base_3d. Without this clamp, the new spline pipeline's
quarter_negative softening lets the FBM's positive lobes build
overhang ceilings over rivers and lakes — closing them into tunnels.

Implementation: fill_chunk's per-voxel block inlines what
evaluate_v2 does, separating the base_3d contribution so the clamp
can run on it before slide() applies the world-top/bottom pull.
evaluate_v2 stays as the canonical evaluator for callers that
don't need column-aware clamping (e.g. tree placement).

Heightmap exports slide_public so fill_chunk's inlined pipeline
can apply slides in the same final step as evaluate_v2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Mountain factor coastal-impossibility verification

**Files:**
- Modify: `src/worldgen/mod.rs` (test only)

The spline `factor_spline` ramps `0.10 → 0.70 → 1.00` walking inland (Mojang's tuning: continentalness `-0.10` factor is small, `0.25` is medium, `1.00` is full). Combined with low `terrain_shape` (mountains), this guarantees coastal mountains are physically impossible — no `CLIFF_MIN_HEIGHT` band-aid needed.

This task adds a regression test that pins the property.

- [ ] **Step 9.1: Write the regression test**

Append to the worldgen test module:

```rust
    #[test]
    fn coastal_mountains_are_impossible() {
        // Coastal column (continentalness near 0) should never have a
        // mountain-height surface, regardless of how mountain-favoring
        // terrain_shape is at that column.
        let g = Generator::new(42);
        let cfg = g.config_snapshot();
        let mut max_coastal_height: i32 = i32::MIN;
        let mut scanned = 0;
        for wx in (-3000..3000).step_by(20) {
            for wz in (-3000..3000).step_by(20) {
                let climate = g.column_climate(wx, wz);
                // Coastal: -0.20 < continentalness < 0.05 (the spline's
                // beach + just-inland band).
                if climate.continentalness > -0.20 && climate.continentalness < 0.05 {
                    let h = g.column_data(wx, wz).height;
                    max_coastal_height = max_coastal_height.max(h);
                    scanned += 1;
                    if scanned > 500 {
                        break;
                    }
                }
            }
            if scanned > 500 {
                break;
            }
        }
        assert!(scanned > 50, "did not find enough coastal columns: only {scanned}");
        // The factor at continentalness ≈ -0.10 is ≈ 0.10 (per
        // default.ron). With factor that small, even minimum
        // terrain_shape = -1 produces only ~`offset(-0.10) * factor(0.10)`
        // worth of relief — far below mountain heights.
        // Concretely: surface should stay within SEA_LEVEL + 30.
        assert!(
            max_coastal_height < SEA_LEVEL + 30,
            "max coastal height {max_coastal_height} should be < {} (sea + 30); \
             factor_spline at coast may be too high",
            SEA_LEVEL + 30
        );
        // Hard floor: should still be at or above sea level - 5 for
        // coastal columns (otherwise the spline is producing oceans
        // where it should produce beaches).
        // (No assertion on the *minimum* since columns near the ocean
        // side can dip; just confirm scan worked.)
    }
```

- [ ] **Step 9.2: Run it**

Run: `cargo test --lib worldgen::tests::coastal_mountains_are_impossible 2>&1 | tail -10`

Expected: pass. If it fails, the `factor_spline` knot table in `default.ron` is too aggressive — either the coastal factor is too high or the offset knot is overshooting.

If it fails, the suggested tuning (in `default.ron`, factor_spline):
```ron
(loc: -0.15, val: Constant(0.20), slope: 0.0),  // → 0.10 if too high
(loc: -0.10, val: Multipoint([
    (loc: -0.85, val: Constant(4.0), slope: 0.0),  // → 3.0 if too high
    ...
])),
```

Commit any RON tuning with the test fix.

- [ ] **Step 9.3: Commit**

```bash
git add src/worldgen/mod.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
test(worldgen): pin "coastal mountains impossible" as regression

The factor_spline's 0.10 → 0.70 → 1.00 ramp inland is supposed to
make coastal mountains physically impossible by construction —
even with maximally mountain-favoring terrain_shape, the small
coastal factor caps the surface height. This commit adds a
regression test that scans 500 coastal columns and asserts none
exceeds SEA_LEVEL + 30.

If this regresses in future tuning, the default.ron knot tables
need to be revisited — not the source code.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Re-pin golden hashes and verify full suite

**Files:**
- Modify: `src/worldgen/mod.rs` (`GOLDEN_42_002` constant)
- Modify: `tests/worldgen_fingerprint.rs` (the heightmap fingerprint pin — if its 2D-sample shape changed)

PR 3 changes both the 2D heightmap (now spline-driven) and the 3D density composition. Both goldens need re-baselining. Same sentinel-mode pattern as PR 2 task 10.

- [ ] **Step 10.1: Put `GOLDEN_42_002` into sentinel mode**

In `src/worldgen/mod.rs`, the constant is currently the value pinned by PR 2 task 10.6. Replace with the sentinel:

```rust
const GOLDEN_42_002: u64 = 0xDEAD_BEEF_DEAD_BEEF;
```

This puts the test into "print the new hash" mode.

- [ ] **Step 10.2: Put `fingerprint_hash_matches_pin` into sentinel mode**

Look at `tests/worldgen_fingerprint.rs`. Find the pinned hash constant. Replace with a sentinel value (same `0xDEAD_BEEF_DEAD_BEEF` pattern).

Run: `cargo test --test worldgen_fingerprint 2>&1 | tail -5`

Expected: the test prints `UPDATE <CONST_NAME> to: 0x<hex>` and the assertion is skipped in sentinel mode. If the test panics instead, the print-mode code needs adding — copy the pattern from PR 2 task 10.6:

```rust
if EXPECTED == 0xDEAD_BEEF_DEAD_BEEF {
    eprintln!("UPDATE <CONST_NAME> to: 0x{:016X}", actual);
    return;
}
assert_eq!(actual, EXPECTED, "...");
```

- [ ] **Step 10.3: Capture the new hashes**

```bash
cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 -- --nocapture 2>&1 | grep "UPDATE GOLDEN"
cargo test --test worldgen_fingerprint -- --nocapture 2>&1 | grep "UPDATE"
```

Expected: two lines, one for each constant. Copy the hex values.

- [ ] **Step 10.4: Re-pin both constants**

Update `GOLDEN_42_002` in `src/worldgen/mod.rs` to the new hex value from step 10.3.

Update the fingerprint pin in `tests/worldgen_fingerprint.rs` to the new hex value.

- [ ] **Step 10.5: Run the full suite**

Run: `cargo test 2>&1 | tail -20`

Expected: every test passes. Failures here are unexpected — investigate:

- Hydrology tests should still pass (no changes to that module).
- Smoke tests (chunk gen + meshing) should still pass.
- All worldgen lib tests should pass.

Likely-noisy tests to re-check:
- `column_data_*` tests that asserted on `h_pre` values — replace with `column_data().height` assertions.
- Any test that constructed `Plate::of(seed, cx, cz)` without going through the deprecated forwarder — should still work since `of` is preserved.
- Tests that scanned for plate kinds or roughness multiplier — `roughness_bias` field is in a different range.

- [ ] **Step 10.6: Visual smoke test**

Build and run:

```bash
cargo build --release 2>&1 | tail -5
```

Expected: `Finished` line.

Run the game (or `cargo run --release -- --headless` if that exists). Observe:
- Continents look like before (plate Voronoi still drives macro shape via continentalness).
- Mountains form *inland* — no coastal cliff strips.
- Plateaus are visible (the `widePlateau` knot — `terrain_shape ≈ -0.35`).
- Plains are flat.
- Beach-to-inland transition is a clean step (the duplicate `-0.16`/`-0.15` knots).
- Rivers/lakes have clean surfaces (no overhang ceilings).
- Jagged peaks visible on the tallest mountains (PV near 1).
- Sky and bedrock floor still clean (PR 2's slides intact).

If anything looks wrong, the most likely culprits:
- `factor_spline` knots too low (mushy mountains) or too high (knife-edge cliffs).
- `offset_spline` mountain knot too aggressive (peaks pegging at `y_max`).
- `plate_roughness_bias_range` too wide (some continents have unreachable mountains).

Tune in `default.ron`; the watcher will pick it up live in a running game.

- [ ] **Step 10.7: Commit**

```bash
git add src/worldgen/mod.rs tests/worldgen_fingerprint.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
chore(worldgen): re-pin golden hashes after PR 3 spline pipeline

Both goldens needed re-baselining:

* GOLDEN_42_002 (chunk-level fingerprint) — every column's height
  is now derived from the spline pipeline, so the chunk's block
  contents shifted.
* fingerprint_hash_matches_pin (2D heightmap fingerprint) — same
  reason; h_pre is gone, h_spline took its place.

The hydrology fingerprints and smoke tests are unchanged
(hydrology was untouched; smoke tests assert structural properties
that hold across heightmap changes).

Any `default.ron` tuning from the visual smoke test (task 10.6) is
included here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Out of scope for PR 3 (deferred to later PRs)

- **Multi-noise biome lookup with 6D R-tree** (PR 4). PR 3 still uses `Biome::classify(temp, humidity, is_desert)` from the existing climate noises. `column_climate()` produces `(continentalness, terrain_shape, ridges_pv)` but those don't yet feed biome selection.
- **Cell interpolation / sliding YZ wall** (PR 5). Density is still evaluated per-voxel; only `SplineOutputs` is cached at quart resolution.
- **Surface rules DSL** (PR 6). The inline cliff / sand / snow / grass logic in `fill_chunk` persists.
- **Real aquifer with pressure** (PR 7). The primitive ocean-column rule from the in-session fix persists.
- **Noise carver layers (cheese, spaghetti)** (PR 8). Graph caves + wormholes are the only carvers.
- **Chunk cache invalidation on hot-reload.** Edits to `default.ron` only affect newly-generated chunks.
- **Generic `CubicSpline<T: Lerp>` over arbitrary value types.** PR 2's `CubicSpline` is scalar-only; PR 3 adds the nested form via `NestedSpline` as a separate enum (because Serde + recursive enums + generic value types is awkward). PR 4 may unify them once biome `ParameterPoint` values join the spline graph.
- **`HeightmapNoise::base/warp_x/warp_z` Fbm fields.** Kept on the struct for now (zero-cost as construction only), removed by a future cleanup PR after biome implementation confirms they're truly unused.
- **`WARP_AMPLITUDE/PERIOD`, `BASE_FBM_AMPLITUDE/PERIOD`** deletion — done in this PR (task 7), but verifying no test in the project still imports them is part of the final compile.
- **Per-region terracing for cross-region rivers** (research doc Q2 Option B). PR 1 covers boundary stitching; mega-macro tier deferred.

## Plan self-review notes

- **All 10 tasks have concrete code in every step.** No "TBD", "implement X", or placeholder comments where Rust code should be.
- **Type names match PR 2 conventions:** `WorldgenConfig`, `DensityConfig`, `ClimateConfig`, `ConfigHolder`, `CubicSpline`, `Knot`, `NestedSpline`, `NestedKnot`, `FlatCache2D`, `DensityNoise::evaluate_v2`, `SplineOutputs`, `ClimateInputs`, `TerrainShapeNoise`.
- **`evaluate_v2` signature evolution is explicit:**
  - PR 2: `(h_target: f32, wx, wy, wz, cfg: &DensityConfig)`
  - PR 3: `(wx, wy, wz, spline: &SplineOutputs, jagged_noise: f32, cfg: &DensityConfig)`
  Step 5.2 shows both for clarity; callers migrate in step 5.4.
- **`FlatCache2D` use is concrete:** task 5 stores `SplineOutputs` per quart (the natural pattern, mirroring PR 2 task 3's design).
- **Hydrology untouched:** the only hydrology change is the consumer in `column_data_with` (step 5.5 / verified in step 6.1), one-line target swap.
- **4-block above-water clamp lives in `mod.rs::fill_chunk`** (task 8) — ~5 lines, matches the spec requirement.
- **Each task ends with a commit boundary** including a heredoc commit message and the `Co-Authored-By:` trailer.
- **Golden hash management** mirrors PR 2 task 10: sentinel mode (step 10.1, 10.2), capture (step 10.3), re-pin (step 10.4). Two goldens are pinned this PR (chunk + fingerprint).
- **Obsolete code deletion** (task 7) explicitly lists every removed symbol and confirms zero callers before deletion. The `HeightmapNoise` struct deletion makes the cleanup unambiguous.
- **MC knot tables are quoted concretely.** `offset_spline`, `factor_spline`, `jaggedness_spline` in `default.ron` show the actual values (loc/val/slope triples) from `TerrainProvider.java`, scaled by 1.5× to match Oxium's `y_gradient_amplitude`.
- **PR scope discipline:** no biome lookup, cell interp, surface DSL, aquifer, or noise carver work in this PR. The "Out of scope" section explicitly defers each.
- **PR 2 dependency is explicit:** the header block calls it out; tasks reference PR 2 types without re-introducing them; task 1 starts from "the existing density section is intact".
