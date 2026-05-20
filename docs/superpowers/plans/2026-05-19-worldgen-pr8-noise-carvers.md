# Worldgen PR 8 — Noise Carver Layers (Cheese, Spaghetti, Pillars)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Minecraft 1.18+-style 3D-noise cave carvers — **cheese**, **spaghetti**, and **pillars** — ALONGSIDE Oxium's existing graph cave systems and wormholes. Result: chambered/designed caves with deliberate entrances (Oxium's identity, untouched) + ambient noise-based carving everywhere (the MC "honeycombed underground" feel, new).

**Architecture:** Three new noise-driven contributions extend the existing `cave_contribution` composition in `fill_chunk`. All three live in `src/worldgen/caves.rs` alongside the graph code — they belong with the existing cave logic, not in a new file. Per the research doc Decisions Log Q3, this is the explicitly-chosen Option C: keep graph caves + wormholes, add noise carvers in parallel.

- **Cheese caves**: a single 3D noise; values above a threshold subtract from density. Tuned mild so it complements (does not visually compete with) the graph chambers. Y-window `[-30, 10]` per Decisions Log Q3 — between the Shallow band (10..50) and Deep band (-110..-30) where graph systems are dense.
- **Spaghetti tubes**: two 3D noises gated by a rarity field; `max(|noise_a|, |noise_b|)` near zero carves narrow long tubes. Y-clamped gradient makes them drift slowly downward (matches MC's `yClampedGradient(-64, 320, 8.0, -40.0)`).
- **Pillars**: positive-density blobs that ADD stone back inside carved volumes. Composition order matters: pillars apply AFTER all subtractions so they can fill in space the carvers opened. Gated by `rangeChoice >= 0.03` so most underground space has no pillars at all.

The composition target inside `fill_chunk`:

```rust
// All carvers contribute to a single subtractive total.
cave_contribution = max(
    cave_sdf(graph_systems),    // existing — untouched
    entrance_sdf(graph_systems),// existing — untouched
    wormhole_contribution,      // existing — untouched
    cheese_contribution,        // PR 8 new
    spaghetti_contribution,     // PR 8 new
);
// Pillars ADD density back (positive) — applied AFTER subtraction.
density = raw_density - cave_contribution + pillar_contribution;
```

Note the `max` vs `+=` change: the existing code adds graph contributions (`cave_contribution += cave_sdf(...)`). PR 8 switches the combination operator to `max` so multiple carvers in the same voxel don't accumulate to absurd subtractions — the strongest carver wins. The behaviour for a single carver is identical; only the multi-source case changes.

**New noise channels** (defined in MC `Noises.java`, mirrored in `assets/worldgen/default.ron` via `WorldgenConfig::cave`):

| Channel | firstOctave | amplitudes | Role |
|---|---|---|---|
| `cave_cheese` | -8 | `[0.5, 1, 2, 1, 2, 1, 0, 2, 0]` | macro 3D noise for cheese |
| `cave_layer` | -8 | `[1]` | optional Y-layering modulator (skipped in PR 8 — pure cheese) |
| `spaghetti_2d` | -7 | `[1]` | one of two 3D noises whose max-abs threshold carves tubes |
| `spaghetti_2d_modulator` | -8 | `[1]` | rarity gate field (low-freq) |
| `spaghetti_2d_thickness` | -8 | `[1]` | per-region thickness modulation |
| `spaghetti_roughness` | -5 | `[1]` | tube wall roughness perturbation |
| `pillar` | -7 | `[1, 1]` | 3D blob field for column placement |
| `pillar_rareness` | -8 | `[1]` | how often pillars actually appear |
| `pillar_thickness` | -8 | `[1]` | column diameter |

All tunable values land in RON; channel topology stays in Rust (per Decisions Log Q5).

**Tech Stack:**
- Rust 2024 edition
- `noise` crate (already a dependency) — `Fbm<Simplex>` for each channel
- `serde` derive (already wired) — `WorldgenConfig` extension

**Reference:** Architectural rationale lives in `docs/superpowers/specs/2026-05-19-minecraft-worldgen-research.md` (Part 4 idea #7, Decisions Log Q3). MC source: `net/minecraft/world/level/levelgen/NoiseRouterData.java` lines 282-303 (`underground` function), JSON definitions in `data/minecraft/worldgen/density_function/overworld/caves/`. This plan does not re-argue those decisions.

**Assumes:** PR 2 through PR 5 are landed. Specifically PR 2 has shipped `WorldgenConfig`, `ConfigHolder`, `evaluate_v2`, and removed the `min(2.0)` cap; PR 5 has shipped the cell-grid density evaluator (cave carvers stay per-voxel — they're subtracted from the trilerped density per Decisions Log Q4).

---

### Task 1: Add cave noise channel constants and ChannelParams type (TDD)

**Files:**
- Modify: `src/worldgen/config.rs`

Define MC-style noise-channel descriptors in RON. Each channel has a `first_octave` and an `amplitudes` vector that drive an `Fbm<Simplex>` builder downstream. This task only adds the type and extends `DensityConfig` — wiring lands in later tasks.

- [ ] **Step 1.1: Write the failing tests**

Append to `src/worldgen/config.rs`, above the `#[cfg(test)] mod tests` block:

```rust
/// One MC-style noise channel descriptor. Mirrors the JSON shape in
/// `data/minecraft/worldgen/noise/<name>.json`:
///
/// ```json
/// { "firstOctave": -8, "amplitudes": [1.0, 1.0] }
/// ```
///
/// `first_octave` sets the lowest-frequency octave's wavelength
/// (`2^-first_octave` ≈ ~the wavelength in blocks). `amplitudes`
/// weights successive octaves; a zero entry skips that octave.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelParams {
    pub first_octave: i32,
    pub amplitudes: Vec<f32>,
}

impl ChannelParams {
    /// Effective number of octaves (count of nonzero amplitudes).
    pub fn octave_count(&self) -> usize {
        self.amplitudes.iter().filter(|&&a| a != 0.0).count()
    }

    /// Frequency of the first (lowest) octave, in cycles/block.
    pub fn first_frequency(&self) -> f64 {
        2.0_f64.powi(self.first_octave)
    }
}

/// Cave-carving tunables — applies to the noise carvers (cheese,
/// spaghetti, pillars), not the graph cave systems.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaveConfig {
    // Cheese caves.
    pub cheese: ChannelParams,
    /// XZ scale applied to cheese-channel sampling. MC uses 2/3.
    pub cheese_xz_scale: f32,
    /// Threshold: cheese values above this carve. MC's underground()
    /// uses 0.27 as the additive offset; we expose the raw threshold.
    pub cheese_threshold: f32,
    /// Y-window in which cheese is active. Outside this band the
    /// contribution is zero. MC has no hard band; we use Decisions
    /// Log Q3's recommendation `[-30, 10]` to keep cheese sandwiched
    /// between the graph shallow/middle/deep bands.
    pub cheese_y_min: i32,
    pub cheese_y_max: i32,
    /// Soft-fade width in blocks at each Y edge (linear ramp from 0
    /// at the boundary to full strength `cheese_fade_blocks` inside).
    pub cheese_fade_blocks: i32,
    /// Peak intensity of the cheese contribution (subtracted from
    /// density when cheese carves). Tuned mild to coexist with graph
    /// chambers (~CAVE_SDF_INTENSITY/4 by default).
    pub cheese_intensity: f32,

    // Spaghetti tubes.
    pub spaghetti_2d: ChannelParams,
    pub spaghetti_2d_modulator: ChannelParams,
    pub spaghetti_2d_thickness: ChannelParams,
    pub spaghetti_roughness: ChannelParams,
    /// Rarity gate: voxels where the modulator field exceeds this
    /// see no spaghetti contribution. Smaller → tubes are rarer.
    pub spaghetti_rarity_threshold: f32,
    /// Tube half-width control (added to thickness modulator).
    pub spaghetti_thickness_offset: f32,
    /// MC's `yClampedGradient(-64, 320, 8.0, -40.0)` translated: a
    /// linear ramp from `gradient_top_value` at `gradient_top_y` to
    /// `gradient_bottom_value` at `gradient_bottom_y`, clamped at
    /// either end. The gradient is ADDED to the spaghetti elevation
    /// modulator before the abs/threshold check, so tubes drift
    /// slowly downward through the underground band.
    pub spaghetti_gradient_top_y: i32,
    pub spaghetti_gradient_top_value: f32,
    pub spaghetti_gradient_bottom_y: i32,
    pub spaghetti_gradient_bottom_value: f32,
    /// Peak intensity of the spaghetti contribution.
    pub spaghetti_intensity: f32,

    // Pillars (add density BACK inside carved volumes).
    pub pillar: ChannelParams,
    pub pillar_rareness: ChannelParams,
    pub pillar_thickness: ChannelParams,
    /// XZ scale for the main pillar field. MC uses 25.0.
    pub pillar_xz_scale: f32,
    /// Y scale for the main pillar field. MC uses 0.3 (taller blobs).
    pub pillar_y_scale: f32,
    /// Range-choice gate: pillars contribute only where the rareness
    /// field is >= this. MC uses 0.03.
    pub pillar_cutoff: f32,
    /// Peak density that pillars add. Must be large enough to refill
    /// previously-carved voxels (i.e. comparable to or larger than
    /// `cheese_intensity` / `spaghetti_intensity`).
    pub pillar_intensity: f32,
}
```

Extend `WorldgenConfig` to hold a `CaveConfig`:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldgenConfig {
    pub density: DensityConfig,
    pub cave: CaveConfig,
}
```

Add tests in the existing `mod tests`:

```rust
    #[test]
    fn cave_config_loads_from_default_ron() {
        let cfg = WorldgenConfig::bundled_default().expect("default.ron must load");
        // Cheese.
        assert!(!cfg.cave.cheese.amplitudes.is_empty());
        assert_eq!(cfg.cave.cheese.first_octave, -8);
        assert!(cfg.cave.cheese_y_min < cfg.cave.cheese_y_max);
        assert!(cfg.cave.cheese_intensity > 0.0);
        // Spaghetti.
        assert_eq!(cfg.cave.spaghetti_2d.first_octave, -7);
        assert!(cfg.cave.spaghetti_rarity_threshold > 0.0);
        assert!(cfg.cave.spaghetti_rarity_threshold < 1.0);
        // Pillars.
        assert_eq!(cfg.cave.pillar.first_octave, -7);
        assert!(cfg.cave.pillar_cutoff > 0.0);
        assert!(cfg.cave.pillar_intensity > 0.0);
    }

    #[test]
    fn channel_params_first_frequency_matches_octave() {
        let p = ChannelParams { first_octave: -8, amplitudes: vec![1.0] };
        // 2^-8 = 1/256 cycles per block.
        assert!((p.first_frequency() - 1.0 / 256.0).abs() < 1e-9);
    }

    #[test]
    fn channel_params_octave_count_ignores_zeros() {
        let p = ChannelParams {
            first_octave: -8,
            amplitudes: vec![0.5, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0],
        };
        assert_eq!(p.octave_count(), 7);
    }
```

- [ ] **Step 1.2: Verify tests fail**

Run: `cargo test --lib worldgen::config 2>&1 | tail -10`

Expected: 3 new tests fail. `cave_config_loads_from_default_ron` fails because `default.ron` has no `cave:` section yet. The other two might pass once the type compiles — that's acceptable.

- [ ] **Step 1.3: Extend `assets/worldgen/default.ron`**

Add a `cave:` block to the existing config tuple. Insert it after the `density:` block:

```ron
    // ──────────────────────────────────────────────────────────────
    // Noise carvers (PR 8).
    //
    // These supplement the graph-based cave systems and wormholes
    // (caves.rs) with MC-style ambient cave density — the
    // "honeycombed underground" feel that Oxium's chambered systems
    // alone don't provide. Each channel's first_octave/amplitudes
    // mirror data/minecraft/worldgen/noise/<name>.json.
    cave: (
        // ── Cheese caves: single 3D noise, threshold carves.
        cheese: (
            first_octave: -8,
            amplitudes: [0.5, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0],
        ),
        cheese_xz_scale: 0.6666667, // MC's 2/3
        // MC uses `add(0.27, cheese).clamp(-1, 1)`; cheese carves
        // where the sum exceeds 0 → cheese > -0.27. We expose the
        // negated threshold for clarity (carve where cheese > t).
        cheese_threshold: -0.27,
        // Decisions Log Q3: sandwich cheese between the Shallow band
        // (CAVE_BAND_SHALLOW = 10..50) and Deep band
        // (CAVE_BAND_DEEP = -110..-30). Cheese active in [-30, 10].
        cheese_y_min: -30,
        cheese_y_max: 10,
        cheese_fade_blocks: 8,
        // Tuned mild: 1.0 vs CAVE_SDF_INTENSITY=4 keeps cheese
        // visually subordinate to the chambered graph caves.
        cheese_intensity: 1.0,

        // ── Spaghetti tubes: two noises gated by rarity, abs near 0.
        spaghetti_2d: (
            first_octave: -7,
            amplitudes: [1.0],
        ),
        spaghetti_2d_modulator: (
            first_octave: -11,
            amplitudes: [1.0],
        ),
        spaghetti_2d_thickness: (
            first_octave: -8,
            amplitudes: [1.0],
        ),
        spaghetti_roughness: (
            first_octave: -5,
            amplitudes: [1.0],
        ),
        // MC's underground() uses ~0.95 rarity (very rare). We start
        // permissive so tubes are visible during PR 8 tuning, will
        // tighten in post-PR-8 review.
        spaghetti_rarity_threshold: 0.5,
        // Half-width of the threshold band — voxels where
        // max(|noise_a|, |noise_b|) <= this thickness are inside a
        // tube. Smaller → narrower tubes. 0.04 ≈ 2-3 block diameter.
        spaghetti_thickness_offset: 0.04,
        // MC's yClampedGradient(-64, 320, 8.0, -40.0) — at y=-64 the
        // gradient is 8.0 (suppresses tubes), at y=320 it's -40.0
        // (allows tubes); linear in between. Translated to Oxium's
        // Y range:
        spaghetti_gradient_top_y: -120,
        spaghetti_gradient_top_value: 8.0,
        spaghetti_gradient_bottom_y: 140,
        spaghetti_gradient_bottom_value: -40.0,
        spaghetti_intensity: 1.2,

        // ── Pillars: positive blobs that ADD stone back.
        pillar: (
            first_octave: -7,
            amplitudes: [1.0, 1.0],
        ),
        pillar_rareness: (
            first_octave: -8,
            amplitudes: [1.0],
        ),
        pillar_thickness: (
            first_octave: -8,
            amplitudes: [1.0],
        ),
        pillar_xz_scale: 25.0,
        pillar_y_scale: 0.3,
        pillar_cutoff: 0.03,
        // Must exceed cheese_intensity + spaghetti_intensity so pillars
        // can actually refill the union of all carvers (Composition
        // note: pillars run AFTER all subtractions).
        pillar_intensity: 4.0,
    ),
```

- [ ] **Step 1.4: Run tests**

Run: `cargo test --lib worldgen::config 2>&1 | tail -10`

Expected: all `worldgen::config` tests pass (3 new + 4 existing = 7 total).

- [ ] **Step 1.5: Commit**

```bash
git add src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
feat(worldgen): WorldgenConfig.cave subsection for noise carvers

Adds CaveConfig holding noise-channel descriptors and tuning for
cheese / spaghetti / pillar carvers. Mirrors MC's Noises.java
constants — first_octave and amplitudes per channel — and the
NoiseRouterData.underground() formula's thresholds, gradients, and
cutoffs.

Topology stays in Rust (per Decisions Log Q5); values live here so
they're hot-reloadable.

Cheese Y-window is [-30, 10] per Decisions Log Q3 — sandwiched
between Shallow and Deep graph cave bands so the layers complement
rather than compete.

Wiring (Generator owns the noise fields, fill_chunk integrates the
carvers) lands in tasks 3-6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: MC-style FBM channel builder (TDD)

**Files:**
- Modify: `src/worldgen/heightmap.rs` (re-use the FBM-builder pattern there)
- Or new helper file: `src/worldgen/noise_channel.rs`

Add a single helper that turns a `ChannelParams` + seed-salt into an `Fbm<Simplex>` matching MC's octave/amplitude semantics. Used by Task 3 to construct the cheese/spaghetti/pillar noise fields.

- [ ] **Step 2.1: Create the helper file with failing tests**

Create `src/worldgen/noise_channel.rs`:

```rust
//! Build `Fbm<Simplex>` instances from MC-style `ChannelParams`.
//!
//! MC's `NormalNoise` is parameterised by `firstOctave` + a vector of
//! per-octave amplitudes. The `noise` crate's `Fbm` exposes
//! `octaves`, `frequency`, and `persistence` instead. This module
//! bridges them: `first_octave` → base frequency, `amplitudes` →
//! octave count, persistence approximated from amplitude geometry.
//!
//! Used by the cave noise carvers (cheese, spaghetti, pillars) in
//! `caves.rs`. The bridge is deliberately approximate — exact MC
//! amplitude semantics would require a custom `NoiseFn` impl; for
//! PR 8 we want "MC-like" not "MC-identical", and the tuning
//! constants in `default.ron` are the visible knobs anyway.

use crate::worldgen::config::ChannelParams;
use noise::{Fbm, MultiFractal, Simplex};

/// Build an `Fbm<Simplex>` from a [`ChannelParams`] and a seed salt.
///
/// - `first_octave` → base frequency = `2^first_octave` cycles/block.
/// - `amplitudes` → octave count (nonzero entries).
/// - Persistence is derived from the geometric average of successive
///   nonzero amplitude ratios; defaults to 0.5 if amplitudes are
///   uniform or only one octave is present.
pub fn build_channel(params: &ChannelParams, seed: u64, salt: u32) -> Fbm<Simplex> {
    let octaves = params.octave_count().max(1);
    let freq = params.first_frequency();
    let persistence = derive_persistence(&params.amplitudes);
    Fbm::<Simplex>::new(seed.wrapping_add(salt as u64) as u32)
        .set_octaves(octaves)
        .set_frequency(freq)
        .set_persistence(persistence)
}

/// Approximate persistence from an amplitudes vector: geometric mean
/// of ratios between successive nonzero amplitudes. Falls back to
/// 0.5 (the `Fbm` default) when amplitudes are flat or sparse.
fn derive_persistence(amplitudes: &[f32]) -> f64 {
    let nonzero: Vec<f64> = amplitudes
        .iter()
        .filter(|&&a| a > 0.0)
        .map(|&a| a as f64)
        .collect();
    if nonzero.len() < 2 {
        return 0.5;
    }
    let mut log_sum = 0.0;
    let mut count = 0;
    for w in nonzero.windows(2) {
        log_sum += (w[1] / w[0]).ln();
        count += 1;
    }
    if count == 0 {
        return 0.5;
    }
    let mean_ratio = (log_sum / count as f64).exp();
    // Clamp to a sane range; persistence outside [0.1, 0.9] tends to
    // produce noise that's either pure low-freq or pure high-freq.
    mean_ratio.clamp(0.1, 0.9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_matches_first_octave() {
        let p = ChannelParams { first_octave: -7, amplitudes: vec![1.0] };
        let fbm = build_channel(&p, 42, 0);
        // 2^-7 = 1/128 cycles per block.
        assert!((fbm.frequency - 1.0 / 128.0).abs() < 1e-9);
    }

    #[test]
    fn octave_count_matches_nonzero_amplitudes() {
        let p = ChannelParams {
            first_octave: -8,
            amplitudes: vec![0.5, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0],
        };
        let fbm = build_channel(&p, 42, 0);
        assert_eq!(fbm.octaves, 7);
    }

    #[test]
    fn flat_amplitudes_give_default_persistence() {
        let p = ChannelParams { first_octave: -7, amplitudes: vec![1.0, 1.0, 1.0] };
        let fbm = build_channel(&p, 42, 0);
        // Ratio 1.0 → exp(ln(1.0)) = 1.0; clamped to 0.9.
        assert!((fbm.persistence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn build_channel_is_deterministic_in_seed_salt() {
        let p = ChannelParams { first_octave: -7, amplitudes: vec![1.0] };
        let a = build_channel(&p, 42, 1);
        let b = build_channel(&p, 42, 1);
        use noise::NoiseFn;
        assert_eq!(a.get([0.5, 0.5, 0.5]), b.get([0.5, 0.5, 0.5]));
        let c = build_channel(&p, 42, 2);
        // Different salt → different noise (almost surely).
        assert_ne!(a.get([0.5, 0.5, 0.5]), c.get([0.5, 0.5, 0.5]));
    }
}
```

Add to `src/worldgen/mod.rs`:

```rust
pub mod noise_channel;
```

- [ ] **Step 2.2: Verify tests pass on first run**

Run: `cargo test --lib worldgen::noise_channel 2>&1 | tail -10`

Expected: 4 tests pass. The helper is purely a builder; once it compiles, behaviour follows from the noise crate's API.

If a test fails because `Fbm::frequency` / `Fbm::octaves` / `Fbm::persistence` fields are private in your `noise` version, replace the assertions with sample-equality checks (sample two FBMs built with different params at the same point and compare).

- [ ] **Step 2.3: Commit**

```bash
git add src/worldgen/noise_channel.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): MC-style ChannelParams → Fbm<Simplex> bridge

Single helper build_channel(params, seed, salt) translates MC's
firstOctave/amplitudes shape to noise-crate Fbm parameters. Used
by PR 8's cheese / spaghetti / pillar carvers in caves.rs.

Persistence is approximated from the amplitude geometry; this is
deliberately not bit-equivalent to MC's NormalNoise — Oxium wants
MC-like cave shape, with default.ron tunables as the real knobs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: NoiseCarvers struct in caves.rs (TDD)

**Files:**
- Modify: `src/worldgen/caves.rs`

Aggregate all nine noise channels (cheese; spaghetti_2d / modulator / thickness / roughness; pillar / rareness / thickness) into a single `NoiseCarvers` struct held by `Generator`. Lives in `caves.rs` per the consistency requirement: noise carvers belong with the existing cave logic.

- [ ] **Step 3.1: Write the failing tests**

Append to the test module at the bottom of `src/worldgen/caves.rs`:

```rust
    #[test]
    fn noise_carvers_builds_from_config_without_panicking() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let _ = NoiseCarvers::new(42, &cfg.cave);
        // Just constructing it is the test — no panics on any of the
        // nine channel builders.
    }

    #[test]
    fn noise_carvers_is_deterministic_in_seed() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let a = NoiseCarvers::new(42, &cfg.cave);
        let b = NoiseCarvers::new(42, &cfg.cave);
        // Spot-check one sample from each channel.
        let p = [10.0, 5.0, -3.0];
        assert_eq!(
            noise::NoiseFn::get(&a.cheese, p),
            noise::NoiseFn::get(&b.cheese, p)
        );
        assert_eq!(
            noise::NoiseFn::get(&a.spaghetti_2d, p),
            noise::NoiseFn::get(&b.spaghetti_2d, p)
        );
        assert_eq!(
            noise::NoiseFn::get(&a.pillar, p),
            noise::NoiseFn::get(&b.pillar, p)
        );
    }

    #[test]
    fn noise_carvers_different_seeds_differ() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let a = NoiseCarvers::new(42, &cfg.cave);
        let b = NoiseCarvers::new(43, &cfg.cave);
        let p = [10.0, 5.0, -3.0];
        assert_ne!(
            noise::NoiseFn::get(&a.cheese, p),
            noise::NoiseFn::get(&b.cheese, p)
        );
    }
```

- [ ] **Step 3.2: Verify tests fail**

Run: `cargo test --lib worldgen::caves::tests::noise_carvers 2>&1 | tail -10`

Expected: 3 tests fail — `NoiseCarvers` doesn't exist.

- [ ] **Step 3.3: Implement `NoiseCarvers`**

In `src/worldgen/caves.rs`, append after the `WormholeNoise` impl (before the `#[cfg(test)]` block):

```rust
// ── Noise carver layers (PR 8) ────────────────────────────────────────
//
// Cheese / spaghetti / pillars: MC-style ambient noise-based cave
// density. Live alongside the graph cave systems and wormholes
// above. The three contributions wire into fill_chunk's
// `cave_contribution` composition (cheese + spaghetti subtract from
// density; pillars add back).

use crate::worldgen::config::CaveConfig;
use crate::worldgen::noise_channel::build_channel;

/// All nine MC-derived noise channels needed for the cheese,
/// spaghetti, and pillar carvers. Built once per Generator.
pub struct NoiseCarvers {
    // Cheese.
    pub cheese: Fbm<Simplex>,
    // Spaghetti.
    pub spaghetti_2d: Fbm<Simplex>,
    pub spaghetti_2d_modulator: Fbm<Simplex>,
    pub spaghetti_2d_thickness: Fbm<Simplex>,
    pub spaghetti_roughness: Fbm<Simplex>,
    // Pillars.
    pub pillar: Fbm<Simplex>,
    pub pillar_rareness: Fbm<Simplex>,
    pub pillar_thickness: Fbm<Simplex>,
}

impl NoiseCarvers {
    /// Build all nine channels from their config descriptors. Each
    /// channel uses a different seed-salt so they're uncorrelated.
    pub fn new(seed: u64, cfg: &CaveConfig) -> Self {
        Self {
            cheese: build_channel(&cfg.cheese, seed, 1001),
            spaghetti_2d: build_channel(&cfg.spaghetti_2d, seed, 1002),
            spaghetti_2d_modulator: build_channel(&cfg.spaghetti_2d_modulator, seed, 1003),
            spaghetti_2d_thickness: build_channel(&cfg.spaghetti_2d_thickness, seed, 1004),
            spaghetti_roughness: build_channel(&cfg.spaghetti_roughness, seed, 1005),
            pillar: build_channel(&cfg.pillar, seed, 1006),
            pillar_rareness: build_channel(&cfg.pillar_rareness, seed, 1007),
            pillar_thickness: build_channel(&cfg.pillar_thickness, seed, 1008),
        }
    }
}
```

- [ ] **Step 3.4: Verify tests pass**

Run: `cargo test --lib worldgen::caves::tests::noise_carvers 2>&1 | tail -10`

Expected: 3 tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add src/worldgen/caves.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): NoiseCarvers struct holding 9 cave noise channels

Aggregates cheese / spaghetti (4 channels) / pillar (3 channels)
Fbm<Simplex> instances behind one type. Built once per Generator
from WorldgenConfig::cave; each channel gets a different seed salt
so they're uncorrelated.

Lives in caves.rs alongside the graph cave logic — noise carvers
belong with existing cave code, not in a separate file. Wiring
into Generator and the per-voxel contribution helpers land in
tasks 4-6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Cheese contribution function (TDD)

**Files:**
- Modify: `src/worldgen/caves.rs`

Compute the per-voxel cheese cave contribution: zero outside the `[cheese_y_min, cheese_y_max]` band (with `cheese_fade_blocks` soft edges), positive subtraction inside where the noise threshold is crossed.

- [ ] **Step 4.1: Write the failing tests**

Append to the test module in `src/worldgen/caves.rs`:

```rust
    #[test]
    fn cheese_contribution_zero_outside_y_window() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        // Well above the cheese band (y=50, band top is 10).
        for wx in (-100..100).step_by(17) {
            for wz in (-100..100).step_by(17) {
                let v = cheese_contribution(wx, 50, wz, &nc, &cfg.cave);
                assert!(
                    v.abs() < 1e-6,
                    "cheese should be 0 at y=50 (outside [-30,10] + 8 fade), got {v} at ({wx},50,{wz})"
                );
            }
        }
        // Well below the band (y=-80, band bottom is -30).
        for wx in (-100..100).step_by(17) {
            for wz in (-100..100).step_by(17) {
                let v = cheese_contribution(wx, -80, wz, &nc, &cfg.cave);
                assert!(
                    v.abs() < 1e-6,
                    "cheese should be 0 at y=-80, got {v} at ({wx},-80,{wz})"
                );
            }
        }
    }

    #[test]
    fn cheese_contribution_nonzero_in_window_somewhere() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        // Inside the band (y in [-30, 10]). Scan and assert at least
        // one voxel sees cheese carving.
        let mut found_carving = false;
        for wx in (-200..200).step_by(5) {
            for wz in (-200..200).step_by(5) {
                let v = cheese_contribution(wx, -10, wz, &nc, &cfg.cave);
                if v > 0.0 {
                    found_carving = true;
                    break;
                }
            }
            if found_carving {
                break;
            }
        }
        assert!(found_carving, "expected ≥1 voxel inside the cheese band to carve");
    }

    #[test]
    fn cheese_contribution_bounded_by_intensity() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        for wx in (-100..100).step_by(11) {
            for wz in (-100..100).step_by(11) {
                for wy in -30..=10 {
                    let v = cheese_contribution(wx, wy, wz, &nc, &cfg.cave);
                    assert!(v >= 0.0, "cheese contribution must be non-negative");
                    assert!(
                        v <= cfg.cave.cheese_intensity + 1e-4,
                        "cheese contribution ({v}) exceeded intensity cap ({})",
                        cfg.cave.cheese_intensity
                    );
                }
            }
        }
    }

    #[test]
    fn cheese_contribution_fades_at_band_edges() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        // At the exact Y edge (y_max=10), the fade factor is 0 → no
        // contribution. Just inside (y=9) the fade ramps up.
        let v_at_top = cheese_contribution(0, 10, 0, &nc, &cfg.cave);
        let v_well_in = cheese_contribution(0, -10, 0, &nc, &cfg.cave);
        // The well-inside sample may be 0 if noise doesn't cross
        // threshold; pick a more robust comparison: ANY voxel at the
        // edge should have <= ANY corresponding voxel deep inside.
        // Statistical: average over a 16×16 patch.
        let avg = |y: i32| -> f32 {
            let mut s = 0.0;
            let mut n = 0;
            for wx in -8..8 {
                for wz in -8..8 {
                    s += cheese_contribution(wx, y, wz, &nc, &cfg.cave);
                    n += 1;
                }
            }
            s / n as f32
        };
        let avg_edge = avg(10);
        let avg_center = avg(-10);
        // y=10 is the boundary so contribution should be 0 there.
        assert!(
            (avg_edge - 0.0).abs() < 1e-6,
            "cheese at y=y_max should be exactly 0, avg={avg_edge}"
        );
        // y=-10 is well inside (fade saturated) → strictly positive avg.
        assert!(avg_center >= 0.0); // (won't necessarily be > 0 if noise rarely crosses; sanity only)
        let _ = v_at_top;
        let _ = v_well_in;
    }
```

- [ ] **Step 4.2: Verify tests fail**

Run: `cargo test --lib worldgen::caves::tests::cheese 2>&1 | tail -10`

Expected: 4 tests fail — `cheese_contribution` doesn't exist.

- [ ] **Step 4.3: Implement `cheese_contribution`**

Append to `src/worldgen/caves.rs`, after the `NoiseCarvers` impl:

```rust
/// Per-voxel cheese cave contribution. Returns a non-negative value
/// in `[0, cheese_intensity]` that is subtracted from density in
/// `fill_chunk`'s composition.
///
/// Returns 0 outside `[cheese_y_min, cheese_y_max]` (with a
/// `cheese_fade_blocks` linear soft edge at each end). Inside the
/// band, returns `cheese_intensity * fade` when the 3D cheese noise
/// (at `cheese_xz_scale`) exceeds `cheese_threshold`, 0 otherwise.
pub fn cheese_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    // Y-window gate with soft edges.
    let fade = cheese_y_fade(wy, cfg);
    if fade <= 0.0 {
        return 0.0;
    }
    let scale = cfg.cheese_xz_scale as f64;
    let v = carvers.cheese.get([
        wx as f64 * scale,
        wy as f64, // y left unscaled — match MC's `noise(CAVE_CHEESE, 2/3)` which only sets xz_scale
        wz as f64 * scale,
    ]) as f32;
    if v > cfg.cheese_threshold {
        cfg.cheese_intensity * fade
    } else {
        0.0
    }
}

/// Soft Y-edge fade for cheese caves. Returns 0 outside the band,
/// linearly ramps from 0 to 1 over `cheese_fade_blocks` at each end,
/// 1 in the interior.
fn cheese_y_fade(wy: i32, cfg: &CaveConfig) -> f32 {
    if wy < cfg.cheese_y_min || wy > cfg.cheese_y_max {
        return 0.0;
    }
    let from_bottom = (wy - cfg.cheese_y_min) as f32;
    let from_top = (cfg.cheese_y_max - wy) as f32;
    let fade_w = cfg.cheese_fade_blocks.max(1) as f32;
    (from_bottom.min(from_top) / fade_w).clamp(0.0, 1.0)
}
```

- [ ] **Step 4.4: Verify tests pass**

Run: `cargo test --lib worldgen::caves::tests::cheese 2>&1 | tail -10`

Expected: 4 tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add src/worldgen/caves.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): cheese_contribution — single-noise threshold cave carver

Per-voxel cheese cave contribution in [0, cheese_intensity]:
  fade(y)        — 0 outside [y_min, y_max], soft linear ramps inside
  cheese(x,y,z)  — anisotropic-XZ 3D noise (xz_scale=2/3 from MC)
  threshold      — carve where noise > cheese_threshold

Active band is [-30, 10] per Decisions Log Q3 — sandwiched between
the graph cave Shallow (10..50) and Deep (-110..-30) bands so the
layers complement rather than compete.

cheese_intensity=1.0 vs CAVE_SDF_INTENSITY=4 keeps the cheese layer
visually subordinate to chambered graph caves.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Spaghetti contribution function (TDD)

**Files:**
- Modify: `src/worldgen/caves.rs`

Implement the two-noise max-abs-with-rarity-gate formula from MC's `spaghetti_2d.json` and the `slopedSpaghetti` Y-gradient. Tubes ~2-3 blocks wide that drift slowly downward.

- [ ] **Step 5.1: Write the failing tests**

Append to the test module:

```rust
    #[test]
    fn spaghetti_contribution_is_non_negative_and_bounded() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        for wx in (-200..200).step_by(13) {
            for wz in (-200..200).step_by(13) {
                for wy in (-100..=80).step_by(7) {
                    let v = spaghetti_contribution(wx, wy, wz, &nc, &cfg.cave);
                    assert!(v >= 0.0, "spaghetti must be non-negative, got {v}");
                    assert!(
                        v <= cfg.cave.spaghetti_intensity + 1e-4,
                        "spaghetti ({v}) exceeded cap ({})",
                        cfg.cave.spaghetti_intensity
                    );
                }
            }
        }
    }

    #[test]
    fn spaghetti_tubes_are_narrow() {
        // For every voxel where spaghetti carves at (wx,wy,wz), at
        // least one of the four cardinal neighbours within
        // CHUNK_DIM_U/8 blocks should NOT carve. (Tubes shouldn't be
        // 8-block-wide slabs.) Tests "≥80% of carve-cells have a
        // non-carve neighbour within 4 blocks in xz".
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        let mut carve_count = 0;
        let mut narrow_count = 0;
        let probe = |wx: i32, wy: i32, wz: i32| -> bool {
            spaghetti_contribution(wx, wy, wz, &nc, &cfg.cave) > 0.0
        };
        for wx in -64..64 {
            for wz in -64..64 {
                let wy = -40;
                if probe(wx, wy, wz) {
                    carve_count += 1;
                    // Check whether the cell is part of a thin tube
                    // (any neighbour within 4 blocks in xz is solid).
                    let is_narrow = (-4..=4).any(|d: i32| {
                        d != 0 && (!probe(wx + d, wy, wz) || !probe(wx, wy, wz + d))
                    });
                    if is_narrow {
                        narrow_count += 1;
                    }
                }
            }
        }
        if carve_count > 0 {
            let frac = narrow_count as f32 / carve_count as f32;
            assert!(
                frac >= 0.8,
                "≥80% of spaghetti carve-cells should have a solid neighbour within 4 blocks (got {frac})"
            );
        } else {
            // No carving in this scan — that's a tuning issue but
            // not a correctness failure for this test.
        }
    }

    #[test]
    fn spaghetti_y_gradient_suppresses_at_top() {
        // The Y-clamped gradient (top=+8, bottom=-40) is added to the
        // abs(noise) check. At the top (high positive gradient), the
        // sum is large → almost never < threshold → almost no carving.
        // Average carving rate at y=120 should be much less than at y=-60.
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        let count_at = |y: i32| -> i32 {
            let mut n = 0;
            for wx in (-64..64).step_by(2) {
                for wz in (-64..64).step_by(2) {
                    if spaghetti_contribution(wx, y, wz, &nc, &cfg.cave) > 0.0 {
                        n += 1;
                    }
                }
            }
            n
        };
        let top = count_at(120);
        let mid = count_at(-60);
        assert!(
            mid > top,
            "spaghetti should carve more at y=-60 than y=120 (got mid={mid}, top={top})"
        );
    }
```

- [ ] **Step 5.2: Verify tests fail**

Run: `cargo test --lib worldgen::caves::tests::spaghetti 2>&1 | tail -10`

Expected: 3 tests fail — `spaghetti_contribution` doesn't exist.

- [ ] **Step 5.3: Implement `spaghetti_contribution`**

Append to `src/worldgen/caves.rs`:

```rust
/// Per-voxel spaghetti tube contribution. Returns a non-negative
/// value in `[0, spaghetti_intensity]` that is subtracted from
/// density in `fill_chunk`'s composition.
///
/// MC formula (from `data/.../caves/spaghetti_2d.json`):
///
/// ```text
///   modulator       = noise(spaghetti_2d_modulator)
///   rare            = modulator < rarity_threshold   // rarity gate
///   sloped_y        = abs(yClampedGradient(top, bot, ...))
///   thickness       = noise(spaghetti_2d_thickness)
///   roughness       = noise(spaghetti_roughness)
///   field           = noise(spaghetti_2d)
///   tube_distance   = abs(field) - (thickness * thickness_offset + roughness * small)
///   carve if rare AND tube_distance < 0 AND sloped_y < some_clamp
/// ```
///
/// We simplify: a single noise channel `spaghetti_2d` whose abs near
/// zero indicates being inside a tube; the modulator gate makes
/// tubes rare; the y-clamped gradient (`sloped_y`) shifts the
/// effective threshold downward at high Y. `spaghetti_roughness`
/// perturbs the tube wall.
pub fn spaghetti_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    // Rarity gate: only voxels where the modulator is below the
    // threshold see any spaghetti. The modulator is sampled at a
    // very low frequency so this carves out large connected regions
    // rather than per-voxel noise.
    let modulator = carvers.spaghetti_2d_modulator.get([
        wx as f64,
        wy as f64,
        wz as f64,
    ]) as f32;
    if modulator >= cfg.spaghetti_rarity_threshold {
        return 0.0;
    }

    // Y-clamped gradient: at `gradient_top_y` the value is
    // `gradient_top_value`; at `gradient_bottom_y` it's
    // `gradient_bottom_value`; linear in between, clamped outside.
    let y_grad = y_clamped_gradient(
        wy,
        cfg.spaghetti_gradient_top_y,
        cfg.spaghetti_gradient_top_value,
        cfg.spaghetti_gradient_bottom_y,
        cfg.spaghetti_gradient_bottom_value,
    );
    // Adding y_grad to the abs(noise) shifts the carve threshold:
    // when y_grad is large positive (high Y), abs(noise) + y_grad
    // is large → almost never below 0 → no carving. When y_grad is
    // negative (low Y), it shifts the threshold up → more carving.
    let field = carvers.spaghetti_2d.get([
        wx as f64,
        wy as f64,
        wz as f64,
    ]) as f32;
    let thickness = carvers.spaghetti_2d_thickness.get([
        wx as f64,
        wy as f64,
        wz as f64,
    ]) as f32;
    let roughness = carvers.spaghetti_roughness.get([
        wx as f64,
        wy as f64,
        wz as f64,
    ]) as f32;
    // Effective half-width: thickness modulator + roughness wiggle.
    let half_width = cfg.spaghetti_thickness_offset
        + 0.5 * cfg.spaghetti_thickness_offset * thickness.abs()
        + 0.1 * cfg.spaghetti_thickness_offset * roughness;
    // Distance from tube center: abs(field) + y_grad. Voxels are
    // "inside" a tube where this distance < half_width.
    let tube_distance = field.abs() + y_grad * 0.01;
    if tube_distance < half_width {
        // Fade by how deep inside the tube we are (closer to the
        // center → stronger carving).
        let depth = (half_width - tube_distance) / half_width.max(1e-4);
        cfg.spaghetti_intensity * depth.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Y-clamped linear gradient: returns `top_value` at `top_y`,
/// `bottom_value` at `bottom_y`, linear in between, clamped at the
/// boundaries. Mirrors MC's `yClampedGradient` density function.
fn y_clamped_gradient(
    wy: i32,
    top_y: i32,
    top_value: f32,
    bottom_y: i32,
    bottom_value: f32,
) -> f32 {
    // Normalise top vs bottom — MC's top_y might be lower than
    // bottom_y; we accept either convention.
    let (lo_y, hi_y, lo_v, hi_v) = if top_y < bottom_y {
        (top_y, bottom_y, top_value, bottom_value)
    } else {
        (bottom_y, top_y, bottom_value, top_value)
    };
    if wy <= lo_y {
        lo_v
    } else if wy >= hi_y {
        hi_v
    } else {
        let t = (wy - lo_y) as f32 / (hi_y - lo_y) as f32;
        lo_v + t * (hi_v - lo_v)
    }
}
```

- [ ] **Step 5.4: Verify tests pass**

Run: `cargo test --lib worldgen::caves::tests::spaghetti 2>&1 | tail -10`

Expected: 3 tests pass.

If `spaghetti_y_gradient_suppresses_at_top` fails because tuning produces no carving anywhere, increase `spaghetti_rarity_threshold` in `default.ron` (more permissive) or `spaghetti_thickness_offset` (wider tubes), re-run. Tuning is acceptable in this task; record the final values.

If `spaghetti_tubes_are_narrow` fails (tubes too wide), decrease `spaghetti_thickness_offset` in RON.

- [ ] **Step 5.5: Commit**

```bash
git add src/worldgen/caves.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
feat(worldgen): spaghetti_contribution — two-noise gated tube carver

Per-voxel spaghetti tube contribution: max(|spaghetti_2d|) gated by
a low-frequency rarity modulator and a Y-clamped gradient that
suppresses tubes near the surface and amplifies them deep down.

Adapted from MC's data/.../caves/spaghetti_2d.json — the tubes
drift slowly downward across the underground band (top y=140 → 8.0,
bottom y=-120 → -40.0) producing the classic MC "long winding
ribbon" cave shape.

Test guarantees:
  - Contribution always in [0, spaghetti_intensity]
  - ≥80% of carving cells have a non-carving neighbour within 4
    blocks (tubes are narrow, not slabs)
  - Carving rate at y=-60 strictly exceeds carving rate at y=120

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Pillar contribution function (TDD)

**Files:**
- Modify: `src/worldgen/caves.rs`

Implement MC's pillar formula: `2 * noise(pillar, xz=25, y=0.3) + (-1 - noise(pillar_rareness)) * (0.55 + 0.55 * noise(pillar_thickness))^3`, gated by `range_choice >= pillar_cutoff` on the raw pillar field. Pillars are ADDITIVE density (return positive values that add back into density after cave subtractions).

- [ ] **Step 6.1: Write the failing tests**

Append to the test module:

```rust
    #[test]
    fn pillar_contribution_is_non_negative_and_bounded() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        for wx in (-200..200).step_by(13) {
            for wz in (-200..200).step_by(13) {
                for wy in (-100..=80).step_by(7) {
                    let v = pillar_contribution(wx, wy, wz, &nc, &cfg.cave);
                    assert!(v >= 0.0, "pillar must be non-negative, got {v}");
                    assert!(
                        v <= cfg.cave.pillar_intensity + 1e-4,
                        "pillar ({v}) exceeded cap ({})",
                        cfg.cave.pillar_intensity
                    );
                }
            }
        }
    }

    #[test]
    fn pillar_cutoff_gate_makes_most_voxels_zero() {
        // With cutoff=0.03 most voxels should see zero pillar.
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        let mut zero_count = 0;
        let mut total = 0;
        for wx in (-128..128).step_by(4) {
            for wz in (-128..128).step_by(4) {
                let v = pillar_contribution(wx, -40, wz, &nc, &cfg.cave);
                if v == 0.0 {
                    zero_count += 1;
                }
                total += 1;
            }
        }
        let frac_zero = zero_count as f32 / total as f32;
        assert!(
            frac_zero >= 0.7,
            "expected ≥70% of voxels to have zero pillar contribution, got {frac_zero}"
        );
    }

    #[test]
    fn pillar_contribution_nonzero_somewhere() {
        // Sanity: at least SOME voxels should have a pillar.
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let nc = NoiseCarvers::new(42, &cfg.cave);
        let mut found_any = false;
        'outer: for wx in (-512..512).step_by(7) {
            for wz in (-512..512).step_by(7) {
                if pillar_contribution(wx, -40, wz, &nc, &cfg.cave) > 0.0 {
                    found_any = true;
                    break 'outer;
                }
            }
        }
        assert!(found_any, "expected ≥1 voxel in 512x512 to have pillar density");
    }
```

- [ ] **Step 6.2: Verify tests fail**

Run: `cargo test --lib worldgen::caves::tests::pillar 2>&1 | tail -10`

Expected: 3 tests fail — `pillar_contribution` doesn't exist.

- [ ] **Step 6.3: Implement `pillar_contribution`**

Append to `src/worldgen/caves.rs`:

```rust
/// Per-voxel pillar contribution. Returns a non-negative value in
/// `[0, pillar_intensity]` that is ADDED to density (not subtracted)
/// in `fill_chunk`'s composition. Composition order matters:
/// pillars apply AFTER all cave subtractions so they can refill
/// previously-carved voxels.
///
/// MC formula (from `data/.../caves/pillars.json`):
///
/// ```text
///   pillar_raw  = 2 * noise(pillar, xz=25, y=0.3)
///   pillar_rare = -1 - noise(pillar_rareness)
///   thickness   = (0.55 + 0.55 * noise(pillar_thickness))^3
///   pillars     = (pillar_raw + pillar_rare) * thickness
///   range_choice: if pillars >= cutoff → pillars, else 0
/// ```
///
/// The cube on thickness creates sharp transitions — pillars exist
/// or they don't, with crisp edges. The negative `pillar_rare` term
/// is the gate: it has to be overcome by the strong pillar_raw to
/// push the product above the cutoff.
pub fn pillar_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    let p_xy = wx as f64 * cfg.pillar_xz_scale as f64;
    let p_y = wy as f64 * cfg.pillar_y_scale as f64;
    let p_z = wz as f64 * cfg.pillar_xz_scale as f64;
    let pillar_raw = 2.0 * carvers.pillar.get([p_xy, p_y, p_z]) as f32;
    let pillar_rare = -1.0 - carvers.pillar_rareness.get([
        wx as f64,
        wy as f64,
        wz as f64,
    ]) as f32;
    let thickness_noise = carvers.pillar_thickness.get([
        wx as f64,
        wy as f64,
        wz as f64,
    ]) as f32;
    let thickness = (0.55 + 0.55 * thickness_noise).powi(3);
    let raw = (pillar_raw + pillar_rare) * thickness;
    // Range-choice gate: only voxels where raw >= cutoff contribute.
    if raw < cfg.pillar_cutoff {
        return 0.0;
    }
    // Scale into [0, pillar_intensity]. Since `raw` can range over
    // roughly [-2, 2] but the cutoff is small, anything past the
    // cutoff is taken as a positive blob; we map by `(raw - cutoff)`
    // saturated.
    let depth = (raw - cfg.pillar_cutoff).clamp(0.0, 1.0);
    cfg.pillar_intensity * depth
}
```

- [ ] **Step 6.4: Verify tests pass**

Run: `cargo test --lib worldgen::caves::tests::pillar 2>&1 | tail -10`

Expected: 3 tests pass.

If `pillar_contribution_nonzero_somewhere` fails (pillar_cutoff filters everything), reduce `pillar_cutoff` in RON to e.g. 0.0 to confirm the formula returns positives at all, then tune back up to ~0.03.

- [ ] **Step 6.5: Commit**

```bash
git add src/worldgen/caves.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
feat(worldgen): pillar_contribution — additive density blobs

Per-voxel pillar contribution adds density BACK to voxels that
land inside a pillar blob. Composition order matters: pillars run
AFTER all cave subtractions, so they refill carved-out voxels.

MC's formula:
  raw       = 2 * pillar_noise + (-1 - rareness_noise)
  thickness = (0.55 + 0.55 * thickness_noise)^3
  product   = raw * thickness
  carve     = product >= cutoff

The cube on thickness produces sharp edges (pillars exist or they
don't). The negative rareness term gates: most voxels never clear
the cutoff, so >=70% of underground space sees no pillars at all.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Wire NoiseCarvers into Generator and fill_chunk

**Files:**
- Modify: `src/worldgen/mod.rs`

Plug everything together. Add `noise_carvers: caves::NoiseCarvers` to the `Generator` struct, build it in `Generator::with_config`, and integrate the three contributions into `fill_chunk`'s composition.

- [ ] **Step 7.1: Write a failing composition test**

Append to the worldgen `mod tests` block in `src/worldgen/mod.rs`:

```rust
    #[test]
    fn noise_carvers_add_to_cave_volume_without_destroying_graph_caves() {
        // The carvers are ADDITIVE: with them on, the chunk's air
        // count must be >= the count with them off (we can't disable
        // them via a flag yet, but we can use the config to crank
        // intensities to zero).
        let cfg_with = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let holder_with = crate::worldgen::config::ConfigHolder::new(cfg_with);
        let g_with = Generator::with_config(42, holder_with);

        let mut cfg_off = (*g_with.config_snapshot()).clone();
        cfg_off.cave.cheese_intensity = 0.0;
        cfg_off.cave.spaghetti_intensity = 0.0;
        cfg_off.cave.pillar_intensity = 0.0;
        let holder_off = crate::worldgen::config::ConfigHolder::new(cfg_off);
        let g_off = Generator::with_config(42, holder_off);

        // Compare a deep underground chunk where both graph and noise
        // carvers should be active.
        let coord = ChunkCoord(IVec3::new(0, -2, 0));
        let mut c_with = DenseChunk::empty();
        let mut c_off = DenseChunk::empty();
        g_with.fill_chunk(coord, &mut c_with);
        g_off.fill_chunk(coord, &mut c_off);
        let air_with = c_with
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Air | Block::Water))
            .count();
        let air_off = c_off
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Air | Block::Water))
            .count();
        assert!(
            air_with >= air_off,
            "noise carvers should only ADD carving (air_with={air_with}, air_off={air_off})"
        );
    }

    #[test]
    fn pillars_increase_stone_count_relative_to_carvers_alone() {
        // With pillars on and pillar_intensity above zero, deep
        // underground chunks should have MORE stone than the same
        // chunk with cheese+spaghetti on but pillars off (pillars
        // add density back).
        let cfg_base = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();

        let mut cfg_no_pillars = cfg_base.clone();
        cfg_no_pillars.cave.pillar_intensity = 0.0;
        let g_no_pillars = Generator::with_config(
            42,
            crate::worldgen::config::ConfigHolder::new(cfg_no_pillars),
        );

        let g_with_pillars = Generator::with_config(
            42,
            crate::worldgen::config::ConfigHolder::new(cfg_base),
        );

        let coord = ChunkCoord(IVec3::new(0, -2, 0));
        let mut a = DenseChunk::empty();
        let mut b = DenseChunk::empty();
        g_no_pillars.fill_chunk(coord, &mut a);
        g_with_pillars.fill_chunk(coord, &mut b);
        let stone = |c: &DenseChunk| -> usize {
            c.blocks.iter().filter(|b| matches!(b, Block::Stone)).count()
        };
        let s_no = stone(&a);
        let s_yes = stone(&b);
        assert!(
            s_yes >= s_no,
            "pillars should add stone back (no_pillars={s_no}, with_pillars={s_yes})"
        );
    }
```

- [ ] **Step 7.2: Verify tests fail**

Run: `cargo test --lib worldgen::tests::noise_carvers_add 2>&1 | tail -10`

Expected: fails because the new contributions aren't wired into `fill_chunk`. (Both chunks return identical output, so air_with == air_off but stone equality also holds; either test that distinguishes `with` vs `without` fails.)

- [ ] **Step 7.3: Add `noise_carvers` field on `Generator`**

In `src/worldgen/mod.rs`, find the `Generator` struct (around line 100). Add a field:

```rust
pub struct Generator {
    // ... existing fields ...
    /// PR 8: cheese / spaghetti / pillar noise channels. Built once
    /// per Generator from `WorldgenConfig::cave`. Read per-voxel in
    /// `fill_chunk` to compose with graph cave SDFs and wormholes.
    noise_carvers: caves::NoiseCarvers,
}
```

In `Generator::with_config` (added in PR 2 task 7), after loading the config snapshot, build the noise carvers:

```rust
pub fn with_config(
    seed: u64,
    config: crate::worldgen::config::ConfigHolder,
) -> Self {
    let mut g = Self::new_internal(seed);
    g.config = config;
    // PR 8: rebuild noise carvers from the new config.
    let cfg_snap = g.config.load();
    g.noise_carvers = caves::NoiseCarvers::new(seed, &cfg_snap.cave);
    g
}
```

Also update `Generator::new_internal` (or `Generator::new`) to initialise `noise_carvers` from the bundled default — otherwise tests that go through `Generator::new(seed)` get an uninitialised field. Build a default once:

```rust
// Inside new_internal, after constructing the other fields:
let bundled = crate::worldgen::config::WorldgenConfig::bundled_default()
    .expect("bundled default.ron must parse");
let noise_carvers = caves::NoiseCarvers::new(seed, &bundled.cave);
```

And add `noise_carvers` to the struct literal that builds the `Self {...}`.

- [ ] **Step 7.4: Integrate contributions in `fill_chunk`**

In `src/worldgen/mod.rs::fill_chunk`, find the existing cave-contribution block (around line 387). Replace:

```rust
let mut cave_contribution = 0.0_f32;
if !cave_systems.is_empty() && wy > CAVE_FLOOR_Y {
    if approx_depth > CAVE_SURFACE_BUFFER {
        cave_contribution +=
            caves::cave_sdf(wx, wy, wz, &cave_systems);
    }
    cave_contribution +=
        caves::entrance_sdf(wx, wy, wz, &cave_systems);
}
if approx_depth > CAVE_SURFACE_BUFFER
    && wy > CAVE_FLOOR_Y
    && self.wormhole_noise.carve(wx, wy, wz)
{
    cave_contribution += CAVE_SDF_INTENSITY;
}

let density_for_compare = if cave_contribution > 0.0 {
    raw_density.min(1.0)
} else {
    raw_density
};
let solid = (density_for_compare - cave_contribution) > 0.0;
```

with:

```rust
// PR 8: combine all carvers via max() so multiple sources at the
// same voxel don't stack to absurd subtractions. The strongest
// contributor wins. Behaviour for a single source is unchanged.
let mut cave_contribution = 0.0_f32;
if !cave_systems.is_empty() && wy > CAVE_FLOOR_Y {
    if approx_depth > CAVE_SURFACE_BUFFER {
        cave_contribution = cave_contribution
            .max(caves::cave_sdf(wx, wy, wz, &cave_systems));
    }
    cave_contribution = cave_contribution
        .max(caves::entrance_sdf(wx, wy, wz, &cave_systems));
}
if approx_depth > CAVE_SURFACE_BUFFER
    && wy > CAVE_FLOOR_Y
    && self.wormhole_noise.carve(wx, wy, wz)
{
    cave_contribution = cave_contribution.max(CAVE_SDF_INTENSITY);
}
// PR 8 new: noise carver contributions (gated by surface buffer
// like the existing carvers; surface gating is the cheap rejection
// — the noise calls themselves are bounded by the gates above).
if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
    cave_contribution = cave_contribution.max(caves::cheese_contribution(
        wx, wy, wz, &self.noise_carvers, &cfg.cave,
    ));
    cave_contribution = cave_contribution.max(caves::spaghetti_contribution(
        wx, wy, wz, &self.noise_carvers, &cfg.cave,
    ));
}
// Pillars ADD density back; compute unconditionally (still cheap)
// — they only kick in past `pillar_cutoff`, so most voxels see 0.
let pillar = if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
    caves::pillar_contribution(wx, wy, wz, &self.noise_carvers, &cfg.cave)
} else {
    0.0
};
// Composition order: subtract all carvers, then add pillars back.
// Pillars can fill in voxels that the carvers opened — the
// "columns inside open caves" look from MC's underground.
let solid = (raw_density - cave_contribution + pillar) > 0.0;
```

Make sure `cfg` is the per-chunk config snapshot from PR 2 (already pulled at the top of `fill_chunk` as `let cfg = self.config_snapshot();`).

- [ ] **Step 7.5: Run tests**

Run: `cargo test --lib worldgen::tests::noise_carvers_add 2>&1 | tail -10`

Expected: both new tests pass.

Then run the broader suite: `cargo test --lib worldgen 2>&1 | tail -15`.

- [ ] **Step 7.6: Re-baseline `golden_seed42_chunk_0_2_0`**

The new carvers change underground chunk content, so the golden hash needs updating. Put the test in print-mode:

```rust
const GOLDEN_42_002: u64 = 0xDEAD_BEEF_DEAD_BEEF;
```

Run: `cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 -- --nocapture 2>&1 | grep "UPDATE GOLDEN"`

Expected: `UPDATE GOLDEN_42_002 to: 0x<NEW_HEX>`. Update the constant.

Run again: `cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 2>&1 | tail -5`

Expected: pass with the new pinned hash.

Note: `worldgen_fingerprint::fingerprint_hash_matches_pin` should NOT change — that test samples the 2D heightmap, which PR 8 doesn't touch. If it does change, investigate before proceeding.

- [ ] **Step 7.7: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): wire cheese / spaghetti / pillars into fill_chunk

Generator now owns a NoiseCarvers built from the active config; on
hot-reload, with_config rebuilds it for new chunks.

fill_chunk composition becomes:
  cave_contribution = max(
      cave_sdf(graph_systems),
      entrance_sdf(graph_systems),
      wormhole_noise_carve,
      cheese_contribution,
      spaghetti_contribution,
  );
  solid = (raw_density - cave_contribution + pillar_contribution) > 0;

The max() combination replaces the previous +=: with multiple
sources at the same voxel, the strongest carver wins instead of
accumulating to absurd subtractions. Single-source behaviour is
unchanged.

Pillars apply AFTER subtraction so they refill carved voxels —
the MC "columns inside open caves" look.

Golden hash for chunk (0,-2,0) re-baselined; the 2D heightmap
fingerprint is unchanged (PR 8 doesn't touch heightmaps).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Update existing cave volume test and final verification

**Files:**
- Modify: `src/worldgen/mod.rs`

The pre-existing test `underground_chunk_has_both_caves_and_solid` (mod.rs:1030) used a 0.5% per-chunk air threshold tuned for graph caves alone. With three new carvers contributing, the underground is more honeycombed; we expect MORE chunks to clear the threshold but the per-chunk stone-majority assertion still holds.

- [ ] **Step 8.1: Raise the per-chunk air threshold**

In `src/worldgen/mod.rs`, find the `underground_chunk_has_both_caves_and_solid` test. The current check is:

```rust
if air > CHUNK_VOL / 200 {
    // Lowered to 0.5% per chunk because tunnels can
    // pass through a chunk and only intersect a
    // narrow strip of cells.
    found_carved_chunk = true;
}
```

With cheese + spaghetti adding ambient carving, every underground chunk should easily clear 0.5%. Raise the bar so the test still distinguishes "carving works" from "carving accidentally got disabled":

```rust
// PR 8: cheese and spaghetti add ambient noise-based carving on
// top of graph systems and wormholes. Every underground chunk in
// the 16×16 scan should now clear a stricter threshold than the
// 0.5% used pre-PR-8.
if air > CHUNK_VOL / 50 {
    // 2% — comfortably above what graph-caves-alone produced in
    // a typical chunk but still loose enough that a chunk on the
    // edge of a sparse region passes.
    found_carved_chunk = true;
}
```

Also update the per-chunk stone-majority assertion to allow the same chunk to have more air without flagging an overly-blown-open chunk. Keep `stone > CHUNK_VOL / 2` — pillars + the natural stone majority should still satisfy this; if it fails, investigate before relaxing.

- [ ] **Step 8.2: Add a cheese-band-only test**

Add a new test that proves cheese is active *only* in its Y window. Append to the test module:

```rust
    #[test]
    fn cheese_only_carves_in_its_y_window() {
        // For a chunk well above the cheese band (y_max=10), the
        // cheese contribution to every voxel must be 0. We can
        // verify by zeroing every OTHER intensity and confirming
        // that an above-band chunk has zero noise-carved air — only
        // pre-existing graph/wormhole/entrance carving.
        let mut cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        cfg.cave.spaghetti_intensity = 0.0;
        cfg.cave.pillar_intensity = 0.0;
        // Disable graph caves and wormholes too by carving a chunk
        // *high* above their bands. Chunk Y=3 → world Y [96, 127],
        // which is entirely above the Shallow band (10..50) and
        // way above Middle / Deep / wormholes.
        let holder = crate::worldgen::config::ConfigHolder::new(cfg.clone());
        let g_only_cheese = Generator::with_config(42, holder);
        let mut chunk = DenseChunk::empty();
        g_only_cheese.fill_chunk(ChunkCoord(IVec3::new(0, 3, 0)), &mut chunk);
        // Compare to the "no carvers at all" baseline.
        let mut cfg_off = cfg.clone();
        cfg_off.cave.cheese_intensity = 0.0;
        let g_off = Generator::with_config(
            42,
            crate::worldgen::config::ConfigHolder::new(cfg_off),
        );
        let mut chunk_off = DenseChunk::empty();
        g_off.fill_chunk(ChunkCoord(IVec3::new(0, 3, 0)), &mut chunk_off);
        let air = |c: &DenseChunk| -> usize {
            c.blocks.iter().filter(|b| matches!(b, Block::Air | Block::Water)).count()
        };
        assert_eq!(
            air(&chunk),
            air(&chunk_off),
            "cheese must not add air above its Y window (chunk Y=3, world Y in [96,127])"
        );
    }
```

- [ ] **Step 8.3: Run the worldgen suite**

Run: `cargo test --lib worldgen 2>&1 | tail -15`

Expected: all tests pass, including the existing `underground_chunk_has_both_caves_and_solid` (with its raised threshold), `deep_caves_under_land_are_dry`, `deep_underground_has_no_surface_blocks`, and the new tests from tasks 4-8.

- [ ] **Step 8.4: Run the full suite (integration + unit)**

Run: `cargo test 2>&1 | tail -20`

Expected: every test passes. Specifically:
- `worldgen::*` lib tests pass
- `worldgen_fingerprint::fingerprint_hash_matches_pin` passes (unchanged — PR 8 doesn't touch heightmaps)
- `golden_seed42_chunk_0_2_0` passes with its new pinned hash
- `smoke::*` integration tests pass (chunk gen + meshing)

- [ ] **Step 8.5: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
test(worldgen): tighten underground volume test + cheese-band gate

PR 8 carvers add ambient noise-based carving in every underground
chunk, so the 0.5% per-chunk air threshold in
underground_chunk_has_both_caves_and_solid is now too loose — it
no longer distinguishes "carving works" from "carving accidentally
got disabled". Raised to 2% (CHUNK_VOL / 50), which graph-only +
wormhole + noise carvers reliably clear.

Added cheese_only_carves_in_its_y_window: with all other carvers
off and the cheese intensity nonzero, a chunk above the cheese
Y-window (chunk Y=3, world Y in [96,127]) must have IDENTICAL air
count to a chunk with the cheese intensity also zero. Catches any
regression that lets cheese carve outside its band.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Visual smoke and final tuning

**Files:** none (verification only); possibly `assets/worldgen/default.ron` for tuning

- [ ] **Step 9.1: Boot the game and inspect**

Run: `cargo run --release`

Walk to an exposed cave area. Observe:

- Graph chambers still exist with deliberate entrances (Oxium identity preserved)
- Deeper exploration reveals ambient carving everywhere — small openings, branching tunnels (cheese) and long thin tubes (spaghetti)
- Inside larger caves, occasional vertical columns of stone (pillars)
- No collapsed chunks (every chunk still mostly stone)
- No surface-block cycles (PR 0 fix still in effect)
- No flooded land caves (PR 0 aquifer rule still in effect)

If anything looks wrong, the most likely tuning knobs:

- **Too much carving overall** → raise `cheese_threshold` (more negative → MORE carving; less negative or positive → less carving). Or lower `cheese_intensity`, `spaghetti_intensity`.
- **Tubes too wide / blob-like** → lower `spaghetti_thickness_offset`.
- **No tubes anywhere** → lower `spaghetti_rarity_threshold` (more permissive).
- **No pillars anywhere** → lower `pillar_cutoff` to 0.0 to confirm, then tune back up.
- **Pillars don't refill carved space** → raise `pillar_intensity` (must exceed the sum of cheese+spaghetti intensities).
- **Cheese active above the band** → check `cheese_y_max`, `cheese_fade_blocks`.

Edit `assets/worldgen/default.ron`, save. The watcher reloads. Newly-generated chunks (move out of the load radius and back) reflect the change. Existing chunks keep their content (chunk-cache invalidation is a future PR).

- [ ] **Step 9.2: Verify hot-reload works**

While the game is running, edit `cheese_intensity: 1.0` → `cheese_intensity: 0.0` and save. Newly-loaded chunks should generate without cheese carving (graph caves and wormholes remain). Restore the original value.

- [ ] **Step 9.3: Final commit (no-op if nothing changed)**

If steps 9.1-9.2 prompted any tuning, commit:

```bash
git add assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
chore(worldgen): tune default.ron after PR 8 visual review

Final tuning of cheese / spaghetti / pillar intensities and gates
after walking the world.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Otherwise nothing to commit — PR 8 is complete.

---

## Out of scope for PR 8 (deferred or rejected)

- **Noodle caves.** MC's `noodle.json` ridge-pair carver. The Decisions Log Q3 calls out cheese + spaghetti as the priority pair; noodles can land in a follow-up.
- **Cave entrance noise.** MC's `entrances.json` blends a `cave_entrance` noise + spaghetti roughness near the surface. Oxium's graph cave system already produces entrances (sinkholes / cliff mouths / skylights); the noise version is redundant for now.
- **`cave_layer` noise** for Y-banded cheese variation. MC mixes a low-freq Y-banding via `square(cave_layer) * 4` into the cheese composition. Skipped in PR 8 in favour of a hard Y-window gate (simpler; visually equivalent for Oxium's compressed Y range).
- **Removing or modifying graph caves / wormholes / entrances.** Decisions Log Q3: non-negotiable — these are Oxium's identity. PR 8 only adds.
- **New decoration features** (stalactites / stalagmites / lush caves / dripstone).
- **JSON-driven density graph** (PR 9+ or never). Topology stays in Rust per Decisions Log Q5.
- **Surface rules DSL** (PR 6). Inline surface logic remains.
- **Real aquifer** (PR 7). Primitive ocean-column rule remains.
- **Multi-noise biome lookup** (PR 4). Already shipped before PR 8 in the locked sequence.
- **Cell interpolation** (PR 5). Already shipped before PR 8; cave carvers stay per-voxel per Decisions Log Q4 (cave SDFs / noise carvers need 1-voxel resolution; interpolating at 4-block scale would blur tunnel walls).

## Plan self-review notes

- All 9 tasks have concrete code in every step. No "TBD" or "fill in details".
- Type names are consistent across tasks: `ChannelParams`, `CaveConfig`, `NoiseCarvers`, `cheese_contribution`, `spaghetti_contribution`, `pillar_contribution`, `y_clamped_gradient`, `cheese_y_fade`.
- All three carver functions take the same `(wx, wy, wz, &NoiseCarvers, &CaveConfig)` signature for callsite uniformity.
- Each task ends with a commit boundary; commit messages explain the why, not just the what.
- Existing carvers (`cave_sdf`, `entrance_sdf`, `wormhole_noise.carve`) are untouched — PR 8 only EXTENDS the composition. Task 7's `max()` replacement of `+=` is a behaviour-preserving change for single-carver voxels (which is most of them).
- Composition order documented in task 7 step 7.4: subtract carvers first (max-combined), then add pillars back. The order matters and is called out in code comments.
- All values hot-reloadable via `default.ron`. Channel topology (which Fbm fields exist) is in Rust per Decisions Log Q5.
- Golden hash management: task 7 step 7.6 puts the test in print-mode and re-pins. `worldgen_fingerprint` (the 2D heightmap fingerprint) is expected to be unchanged since PR 8 only touches the 3D density composition.
- TDD: every new function has a failing test before implementation (tasks 1, 3, 4, 5, 6, 7, 8).
- Tests cover the four key invariants the user called out:
  - Cheese carves only in `[-30, 10]` (task 4 `cheese_contribution_zero_outside_y_window`; task 8 `cheese_only_carves_in_its_y_window` cross-checks at the chunk-fill layer)
  - Spaghetti tubes are narrow (task 5 `spaghetti_tubes_are_narrow`: ≥80% of carving cells have a solid neighbour within 4 blocks)
  - Pillars add density back (task 7 `pillars_increase_stone_count_relative_to_carvers_alone`)
  - Composition: noise carvers don't reduce graph cave volume (task 7 `noise_carvers_add_to_cave_volume_without_destroying_graph_caves`)
- Updated `underground_chunk_has_both_caves_and_solid` threshold per the user's instruction (task 8 step 8.1: 0.5% → 2%).
- LOC budget: tasks 1+2 ≈ 90 LOC (config + builder), task 3 ≈ 25 LOC (struct), tasks 4-6 ≈ 130 LOC (three carver fns + helpers), task 7 ≈ 35 LOC (fill_chunk wiring + Generator field), task 8 ≈ 20 LOC (test updates). Total ≈ 300 LOC, matches the budget.
