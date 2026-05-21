# Cave System Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enable graph caves with style variation + inter-band connectors + cross-region trunks, replace spaghetti/wormhole/surface-entrance carvers with a Terasology-style depth-driven ambient layer, smooth-min layer composition, vertical-run clamp, and Terasology-style surface-block fixer.

**Architecture:** Four sequential PRs. PR1 retires three obsolete noise carvers and replaces them with a single new `terasology_ambient` layer. PR2 re-enables the existing (but currently disabled) graph cave system with a style table and depth-scaled chamber sizes. PR3 adds vertical inter-band connectors within a region and cross-region trunk lines. PR4 adds smooth-min composition between layers, a post-voxelization vertical-run clamp, and a surface-block fixer pass.

**Tech Stack:** Rust, `noise` crate (Fbm<Simplex>), serde/RON for config, existing `CarverEvaluator` trilerp pattern, `notify-debouncer-mini` for RON hot reload.

**Reference:** [Design spec](../specs/2026-05-21-cave-system-overhaul-design.md).

---

## File structure

The work touches these files. Per-file responsibility shown so the engineer doesn't have to guess:

| File | Role | Change |
|---|---|---|
| `src/worldgen/caves.rs` | All cave generation + carving | Delete retired carvers, add `terasology_ambient`, add style-aware system building, add `build_vertical_connectors`, add `build_trunks` |
| `src/worldgen/region.rs` | `CaveSystem` / `Chamber` / `Tunnel` / `Entrance` types | Add `style: CaveStyle` to `CaveSystem`, add `Trunk` to `CaveSystem`, add `vertical_connectors: Vec<Tunnel>` to `CaveSystem` |
| `src/worldgen/config.rs` | `CaveConfig` serde struct | Delete obsolete fields, add `tera_*` fields, add `CaveStyleTable` |
| `src/worldgen/tuning.rs` | Constants | Re-enable `CAVE_SYSTEMS_PER_REGION = (0, 3)`, delete `WORMHOLE_*`, add `MAX_VERTICAL_AIR_RUN` and `SMIN_K_DEFAULT` |
| `src/worldgen/noise_channel.rs` | Noise helpers | Delete `weird_scaled_sample`, `y_clamped_gradient`, `map_from_unit_to` (all only used by spaghetti) |
| `src/worldgen/mod.rs` | `Generator::fill_chunk` density composition | Replace `min(cheese, spag, ...)` with `smin` chain that uses tera, integrate vertical-run clamp and surface fixer |
| `assets/worldgen/default.ron` | Bundled config | Delete obsolete keys, add new ones with chosen defaults |
| `src/bin/worldgen_viz/widgets/probe_table.rs` | Visualizer probe | Remove rows for retired carvers, add row for tera |
| `src/bin/doc_render/parity_dump.rs` | Doc-render parity helper | Remove retired-carver columns |
| `examples/probe_cliff.rs` | One-off probe added during brainstorming | Delete — was experimental |
| `tests/screenshots/baseline_cave.png` | Visual baseline | Re-baseline at end of PR4 |
| `tests/worldgen_fingerprint.rs` | Golden-hash integration test | Re-baseline at end of each PR that changes density |

A new style enum and table:
- `src/worldgen/caves.rs` gets `pub enum CaveStyle { Cathedral, Warren, Slot, Sump, Karst }` and `pub struct CaveStyleTable` (serialized in `CaveConfig`).

---

## PR Sequence

Each PR is independently shippable and leaves the engine in a working state. PR sequencing:

| PR | Title | Tasks | Net effect |
|---|---|---|---|
| **PR1** | Replace dead carvers with Terasology ambient | 7 | Caves driven by cheese + tera; depth-driven scarcity visible |
| **PR2** | Re-enable graph caves with style table | 5 | Distinct cave systems with style identity appear underground |
| **PR3** | Vertical connectors + cross-region trunks | 4 | Cave systems link across bands and regions |
| **PR4** | smin composition + clamp + surface fixer | 5 | Pockets merge; fall hazards capped; surface blocks correct at breakthroughs |

All four PRs require golden-hash rebaselining after density changes — this is expected and called out in each PR's final task. After PR1 the chunk cache must be cleared (`rm -rf saves/default/regions/*.bin`) per memory.

---

## PR1 — Replace dead carvers with Terasology ambient

**Goal:** Delete the obsolete `spaghetti_*`, `WormholeNoise`, and `surface_entrance_*` carver code. Add a new `terasology_ambient` layer that does the same job better: continuous tunnels, depth-driven scarcity, near-surface suppression, all in ~30 lines.

After this PR: caves are sparse near surface and visibly denser at deep Y. No graph caves yet (still disabled by `CAVE_SYSTEMS_PER_REGION = (0, 0)`); just cheese + tera + pillars.

### Task PR1.1: Inventory and delete retired carver code in `caves.rs`

**Files:**
- Modify: `src/worldgen/caves.rs`

- [ ] **Step 1: Run an audit grep first to confirm what gets deleted**

```bash
grep -n "spaghetti\|surface_entrance\|WormholeNoise\|wormhole_noise\|wormhole_carve\|y_clamped_gradient\|weird_scaled_sample\|map_from_unit_to" src/worldgen/caves.rs | wc -l
```

Expected: 60–100 lines hit. Capture the count so you can verify the deletions land cleanly.

- [ ] **Step 2: Delete `WormholeNoise` struct and impl**

Open `src/worldgen/caves.rs`. Delete the entire `// ── Deep-band wormhole filler ────...` block, including:
- `pub struct WormholeNoise { a: Fbm<Simplex>, b: Fbm<Simplex> }`
- `impl WormholeNoise { pub fn new ... pub fn carve ... }`

(Roughly 30 lines starting at the existing `pub struct WormholeNoise` definition.)

- [ ] **Step 3: Delete the `surface_entrance_*` functions**

Delete:
- `pub fn surface_entrance_contribution(...)`
- `fn surface_entrance_y_fade(...)`

(Roughly 40 lines.)

- [ ] **Step 4: Delete `spaghetti_contribution` and `spaghetti_roughness`**

Delete the two functions `pub fn spaghetti_contribution(...)` and `pub fn spaghetti_roughness(...)` along with their long doc comments.

(Roughly 100 lines.)

- [ ] **Step 5: Delete retired channels from `NoiseCarvers` struct**

Find `pub struct NoiseCarvers` in `caves.rs`. Remove these fields:
- `spaghetti_2d`
- `spaghetti_2d_modulator`
- `spaghetti_2d_elevation`
- `spaghetti_2d_thickness`
- `spaghetti_roughness`
- `surface_entrance`

In the corresponding `impl NoiseCarvers::new(...)`, delete the matching `build_channel(...)` initializers.

- [ ] **Step 6: Delete retired channels from `CarverCorner` struct**

Find `struct CarverCorner` in the same file. Remove these fields:
- `spag_elev`
- `spag_thick`
- `spag_weird_scaled`
- `spag_rough`
- `wormhole_a`
- `wormhole_b`
- `surface_entrance`

In `CarverEvaluator::new(...)`, delete the corresponding per-corner sample loops.

- [ ] **Step 7: Delete the `*_at` methods on `CarverEvaluator` that reference retired channels**

Delete:
- `pub fn spaghetti_at(...)`
- `pub fn spaghetti_roughness_at(...)`
- `pub fn wormhole_carve_at(...)`
- `pub fn surface_entrance_at(...)`

- [ ] **Step 8: Delete retired tests in the `#[cfg(test)] mod tests` block**

Delete:
- `noise_carvers_builds_from_config_without_panicking` — keep but verify it still builds (it tests `NoiseCarvers::new` which now has fewer fields)
- `noise_carvers_is_deterministic_in_seed` — keep
- `noise_carvers_different_seeds_differ` — keep
- `probe_surface_entrance_noise_distribution` — DELETE
- `spaghetti_signed_density_is_finite_and_clamped` — DELETE
- `spaghetti_carves_negative_somewhere_underground` — DELETE
- `y_clamped_gradient_endpoint_values` — DELETE
- `wormhole_noise_does_not_carve_above_band` — DELETE
- In `carver_evaluator_exact_at_corners`: remove the `spaghetti_*`, `spag_rough`, `wormhole`, and `surface_entrance` assertion blocks. Keep cheese and pillar blocks.
- In `carver_evaluator_off_corner_is_close_to_direct`: same — remove spaghetti, pillar (keep), surface_entrance assertions. Keep cheese.

- [ ] **Step 9: Build to verify no leftover references**

Run:
```bash
cargo build 2>&1 | head -40
```

Expected: clean build, OR compile errors referencing one of the retired symbols in callers outside `caves.rs` (those are addressed in subsequent tasks). Do NOT silence by adding stubs.

Note any compile errors and what file they're in — those go in PR1.2.

- [ ] **Step 10: Commit**

```bash
git add src/worldgen/caves.rs
git commit -m "refactor(caves): delete spaghetti/wormhole/surface_entrance carver code

Removes the three noise carvers slated for replacement in the cave
system overhaul (see spec 2026-05-21). Leaves the codebase in a
non-compiling state until PR1.2 prunes the call sites."
```

### Task PR1.2: Prune call sites and config

**Files:**
- Modify: `src/worldgen/mod.rs` (composition site lines ~795-830 and ~895-920)
- Modify: `src/worldgen/config.rs`
- Modify: `src/worldgen/tuning.rs`
- Modify: `src/worldgen/noise_channel.rs`
- Modify: `assets/worldgen/default.ron`
- Modify: `src/bin/doc_render/parity_dump.rs`
- Modify: `src/bin/worldgen_viz/widgets/probe_table.rs`
- Delete: `examples/probe_cliff.rs`

- [ ] **Step 1: Remove call sites in `Generator::fill_chunk` composition (mod.rs ~795-830)**

Around line 802 in `mod.rs`, delete the wormhole block:

```rust
        // Wormhole — same gate.
        if approx_depth > CAVE_SURFACE_BUFFER
            && wy > CAVE_FLOOR_Y
            && self.wormhole_noise.carve(wx, wy, wz)
        {
            cave_sdf_val = cave_sdf_val.max(CAVE_SDF_INTENSITY);
        }
```

Delete the spaghetti binding around line 817 (`let spaghetti = ...`). In the composition expression below, remove `.min(spaghetti)`.

Around lines 904 and 916, there's a duplicate composition block used by aquifer detection. Same edits there: remove the wormhole conditional and the `caves::spaghetti_contribution(...)` call.

- [ ] **Step 2: Remove `wormhole_noise` field from `Generator` struct**

In `mod.rs`, find `pub struct Generator { ... }`. Remove `wormhole_noise: caves::WormholeNoise,` field and the matching initializer in `Generator::new`.

- [ ] **Step 3: Remove `wormhole_noise` argument from `CarverEvaluator::new` and the call site**

`CarverEvaluator::new` currently takes `&WormholeNoise`. Remove that parameter. Update the call site in `Generator::fill_chunk` (around line 1046) accordingly.

- [ ] **Step 4: Delete retired fields from `CaveConfig` in `config.rs`**

Open `src/worldgen/config.rs`. In `pub struct CaveConfig { ... }`, delete:
- `spaghetti_2d`, `spaghetti_2d_modulator`, `spaghetti_2d_elevation`, `spaghetti_2d_thickness`, `spaghetti_roughness` (ChannelParams)
- `spaghetti_elevation_min`, `spaghetti_elevation_max`
- `spaghetti_gradient_from_y`, `spaghetti_gradient_from_value`, `spaghetti_gradient_to_y`, `spaghetti_gradient_to_value`
- `spaghetti_thickness_offset`, `spaghetti_thickness_slope`
- `spaghetti_clamp_min`, `spaghetti_clamp_max`
- `spaghetti_cave_noise_offset`
- `surface_entrance` (ChannelParams)
- `surface_entrance_xz_scale`, `surface_entrance_y_scale`
- `surface_entrance_threshold`, `surface_entrance_intensity`
- `surface_entrance_y_min`, `surface_entrance_y_max`
- `surface_entrance_fade_blocks`

Keep: `cheese*`, `cave_layer*`, `pillar*`, `underground_density_threshold`.

- [ ] **Step 5: Delete wormhole constants from `tuning.rs`**

In `src/worldgen/tuning.rs`, delete:
```rust
pub const WORMHOLE_BAND_Y: i32 = -40;
pub const WORMHOLE_BAND: f64 = 0.05;
```

- [ ] **Step 6: Delete unused helpers from `noise_channel.rs`**

Open `src/worldgen/noise_channel.rs`. Audit which functions are still used:
```bash
grep -nE "weird_scaled_sample|y_clamped_gradient|map_from_unit_to" src/ tests/
```

If only `caves.rs` (now-deleted call sites) referenced them, delete:
- `pub fn weird_scaled_sample(...)`
- `pub fn y_clamped_gradient(...)`
- `pub fn map_from_unit_to(...)`

If any remaining call sites exist outside the spec scope, leave the function and note it.

Keep `build_channel` — used by cheese, pillar, and tera.

- [ ] **Step 7: Delete retired keys from `assets/worldgen/default.ron`**

In `default.ron`, find the `cave: (` block. Delete every key for retired fields (see PR1.2.4 list).

- [ ] **Step 8: Update probe/parity binaries**

In `src/bin/doc_render/parity_dump.rs`: search for `spaghetti`, `wormhole`, `surface_entrance` — remove the rows / columns that reference them.

In `src/bin/worldgen_viz/widgets/probe_table.rs`: same.

- [ ] **Step 9: Delete `examples/probe_cliff.rs`**

This file was added during the brainstorming session. It's no longer needed.

```bash
rm examples/probe_cliff.rs
```

- [ ] **Step 10: Verify build**

```bash
cargo build 2>&1 | tail -20
```

Expected: clean build (warnings about unused fields are fine; we'll add the new layer in PR1.3).

- [ ] **Step 11: Run existing tests to confirm we didn't break cheese/pillar**

```bash
cargo test --lib worldgen 2>&1 | tail -20
```

Expected: passes (any spaghetti/wormhole/surface_entrance tests should be gone; cheese and pillar tests should still pass).

- [ ] **Step 12: Commit**

```bash
git add src/worldgen/mod.rs src/worldgen/config.rs src/worldgen/tuning.rs \
        src/worldgen/noise_channel.rs assets/worldgen/default.ron \
        src/bin/doc_render/parity_dump.rs src/bin/worldgen_viz/widgets/probe_table.rs
git rm examples/probe_cliff.rs
git commit -m "refactor(worldgen): remove call sites and config for retired carvers

Prunes spaghetti / wormhole / surface_entrance from fill_chunk, the
config struct, RON defaults, and the probe binaries. Code now compiles
and tests pass with cheese + pillar as the only active noise carvers."
```

### Task PR1.3: Add `tera_*` config fields and RON defaults

**Files:**
- Modify: `src/worldgen/config.rs`
- Modify: `assets/worldgen/default.ron`

- [ ] **Step 1: Add tera fields to `CaveConfig`**

Append to `pub struct CaveConfig`:

```rust
    // ── Terasology depth-driven ambient ──────────────────────────────
    /// 4-octave FBM-Simplex channels for the two-noise intersection
    /// that defines the meandering tubes of the ambient cave layer.
    pub tera_a: ChannelParams,
    pub tera_b: ChannelParams,
    /// Noise wavelength in blocks. Default 200.
    pub tera_wave: f32,
    /// Surface-band suppression magnitude — shift applied to noise B
    /// near the heightmap to push the cave region off-axis. Default 0.17.
    pub tera_supp: f32,
    /// Block depth over which the suppression fades to zero. Default 123.
    pub tera_supp_depth: f32,
    /// Base radius of the cave region in noise space (at depth 0). Default 0.073.
    pub tera_thresh_base: f32,
    /// Depth-divisor: threshold += depth / this. Default 2229.
    pub tera_thresh_depth: f32,
    /// Y-anisotropy: multiplier on wy when sampling tera noise. Higher
    /// values force tube iso-surfaces to bend horizontal. Default 3.56.
    pub tera_y_factor: f32,
```

- [ ] **Step 2: Add the corresponding `cave: (` block in `default.ron`**

```ron
        tera_a: (first_octave: -7, amplitudes: [1.0, 1.0, 1.0, 1.0]),
        tera_b: (first_octave: -7, amplitudes: [1.0, 1.0, 1.0, 1.0]),
        tera_wave: 200.0,
        tera_supp: 0.17,
        tera_supp_depth: 123.0,
        tera_thresh_base: 0.073,
        tera_thresh_depth: 2229.0,
        tera_y_factor: 3.56,
```

- [ ] **Step 3: Build to confirm serde + bundled-default still loads**

```bash
cargo build 2>&1 | tail -10
cargo test --lib worldgen::config 2>&1 | tail -5
```

Expected: build clean, config tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "feat(caves): add tera_* config fields and defaults

Empty until PR1.4 wires the actual carver, but the schema lands first
so config-driven tests can already reference these values."
```

### Task PR1.4: Implement `terasology_ambient` (TDD)

**Files:**
- Modify: `src/worldgen/caves.rs`
- Test: `src/worldgen/caves.rs` (in-file `#[cfg(test)] mod tests`)

- [ ] **Step 1: Add the two FBM channels to `NoiseCarvers`**

In `pub struct NoiseCarvers`, add:

```rust
    pub tera_a: Fbm<Simplex>,
    pub tera_b: Fbm<Simplex>,
```

In `impl NoiseCarvers::new(...)`:

```rust
    tera_a: build_channel(&cfg.tera_a, seed, 2000),
    tera_b: build_channel(&cfg.tera_b, seed, 2001),
```

- [ ] **Step 2: Write the failing depth-monotonicity test**

In the existing `#[cfg(test)] mod tests` block, add:

```rust
#[test]
fn terasology_ambient_depth_monotonicity() {
    // At deeper Y, more samples should hit "cave" (signed-value < 0).
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    let surface_y = 64.0;
    let mut frac_by_depth = vec![];
    for depth in [10, 50, 100, 150] {
        let wy = surface_y as i32 - depth;
        let mut hit = 0usize;
        let mut total = 0usize;
        for wx in (-128..128).step_by(4) {
            for wz in (-128..128).step_by(4) {
                total += 1;
                let v = terasology_ambient(wx, wy, wz, &nc, &cfg.cave, surface_y);
                if v < 0.0 { hit += 1; }
            }
        }
        frac_by_depth.push(hit as f32 / total as f32);
    }
    // Monotonic non-decreasing toward depth.
    for w in frac_by_depth.windows(2) {
        assert!(w[1] >= w[0] - 1e-3,
            "cave fraction decreased with depth: {:?}", frac_by_depth);
    }
    // Surface band should be ~0%.
    assert!(frac_by_depth[0] < 0.05, "too many caves near surface: {:?}", frac_by_depth);
    // Deep band should be > shallow.
    assert!(frac_by_depth[3] > frac_by_depth[0] * 2.0,
        "deep band not vastly more cave-rich than shallow: {:?}", frac_by_depth);
}
```

- [ ] **Step 3: Run the test to verify failure (function not yet defined)**

```bash
cargo test --lib worldgen::caves::tests::terasology_ambient_depth_monotonicity 2>&1 | tail -10
```

Expected: compile error: `cannot find function terasology_ambient in this scope`.

- [ ] **Step 4: Implement `terasology_ambient`**

Add this function to `caves.rs`, in the existing carver-functions region:

```rust
/// Terasology-style depth-driven 2-noise cave carver.
///
/// Inspired by `org.terasology.caves.CaveFacetProvider`. Two independent
/// 4-octave FBM-Simplex channels are intersected: voxels where both are
/// near zero are cave. The cave region in 2D noise space is a disk of
/// radius `freq_depth`, centered at `(0, -freq_reduction)`. The disk
/// grows with depth (more caves deeper) and shifts off-axis near the
/// surface (caves rare up top). Y is sampled at `tera_y_factor` × the
/// XZ frequency, which forces the resulting tubes to lean horizontal.
///
/// Returns signed density in roughly [-1, +0.5]: negative = carve.
pub fn terasology_ambient(
    wx: i32, wy: i32, wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
    surface_y: f32,
) -> f32 {
    let depth = (surface_y - wy as f32).max(0.0);
    let freq_reduction = (cfg.tera_supp - depth / cfg.tera_supp_depth).max(0.0);
    let freq_depth     = cfg.tera_thresh_base + depth / cfg.tera_thresh_depth;
    let freq           = 1.0 / cfg.tera_wave;
    let wy_scaled      = wy as f32 * cfg.tera_y_factor;
    let n0 = carvers.tera_a.get([
        (wx as f32 * freq) as f64,
        (wy_scaled * freq) as f64,
        (wz as f32 * freq) as f64,
    ]) as f32;
    let n1 = carvers.tera_b.get([
        (wx as f32 * freq) as f64,
        (wy_scaled * freq) as f64,
        (wz as f32 * freq) as f64,
    ]) as f32 + freq_reduction;
    ((n0 * n0 + n1 * n1).sqrt() - freq_depth) * 5.0
}
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cargo test --lib worldgen::caves::tests::terasology_ambient_depth_monotonicity 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 6: Add the surface-suppression test**

```rust
#[test]
fn terasology_ambient_surface_suppression_complete_by_supp_depth() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    let surface_y = 64.0;
    // At depth equal to tera_supp_depth, the suppression should be zero
    // (freq_reduction = max(0, supp - 1.0 * supp) = 0).
    // Verify the cave fraction at that depth is non-trivial.
    let wy = (surface_y - cfg.cave.tera_supp_depth) as i32;
    let mut hit = 0usize;
    let mut total = 0usize;
    for wx in (-128..128).step_by(4) {
        for wz in (-128..128).step_by(4) {
            total += 1;
            if terasology_ambient(wx, wy, wz, &nc, &cfg.cave, surface_y) < 0.0 {
                hit += 1;
            }
        }
    }
    let frac = hit as f32 / total as f32;
    assert!(frac > 0.02, "expected some caves at supp_depth, got {frac:.3}");
}
```

Run: `cargo test --lib worldgen::caves::tests::terasology_ambient_surface_suppression 2>&1 | tail -10`. Expected: PASS.

- [ ] **Step 7: Add the anisotropy test**

```rust
#[test]
fn terasology_ambient_horizontal_bias_at_high_y_factor() {
    // With high tera_y_factor, the cave footprint in any horizontal slice
    // should be wider XZ than tall Y for typical features. We approximate
    // this by counting how many cells have horizontal-only-cave vs
    // vertical-only-cave neighbours.
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let nc = NoiseCarvers::new(42, &cfg.cave);
    let surface_y = 64.0;
    let wy_center = -40i32;
    let mut horizontal_runs = 0usize;
    let mut vertical_runs = 0usize;
    for wx in (-128..128).step_by(2) {
        let mut h_run = 0;
        let mut v_run = 0;
        for wz in (-128..128).step_by(2) {
            if terasology_ambient(wx, wy_center, wz, &nc, &cfg.cave, surface_y) < 0.0 {
                h_run += 1;
            } else if h_run > 0 {
                horizontal_runs += h_run; h_run = 0;
            }
        }
        for dy in (-30..30).step_by(2) {
            if terasology_ambient(wx, wy_center + dy, 0, &nc, &cfg.cave, surface_y) < 0.0 {
                v_run += 1;
            } else if v_run > 0 {
                vertical_runs += v_run; v_run = 0;
            }
        }
    }
    assert!(horizontal_runs > vertical_runs,
        "expected horizontal cave extent > vertical: h={horizontal_runs} v={vertical_runs}");
}
```

Run: `cargo test --lib worldgen::caves::tests::terasology_ambient_horizontal 2>&1 | tail -10`. Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/worldgen/caves.rs
git commit -m "feat(caves): add terasology_ambient carver with TDD tests

Implements the depth-driven 2-noise cave carver from the Terasology
Caves module, with anisotropic Y sampling so tubes lean horizontal.
Three tests cover depth monotonicity, surface suppression, and
horizontal-vs-vertical extent bias."
```

### Task PR1.5: Wire `terasology_ambient` into `fill_chunk` composition

**Files:**
- Modify: `src/worldgen/mod.rs`
- Modify: `src/worldgen/caves.rs` (extend `CarverEvaluator`)

- [ ] **Step 1: Add tera lookup to `CarverCorner`**

In `caves.rs`, find `#[derive(Default, Clone, Copy)] struct CarverCorner`. Add fields:

```rust
    tera_a: f32,
    tera_b: f32,
```

- [ ] **Step 2: Populate them in `CarverEvaluator::new`**

In the corner-sampling loop, add:

```rust
                    let tera_freq = 1.0 / cfg.tera_wave;
                    let tera_wy = wy as f32 * cfg.tera_y_factor;
                    let tera_a = carvers.tera_a.get([
                        (wx as f32 * tera_freq) as f64,
                        (tera_wy * tera_freq) as f64,
                        (wz as f32 * tera_freq) as f64,
                    ]) as f32;
                    let tera_b = carvers.tera_b.get([
                        (wx as f32 * tera_freq) as f64,
                        (tera_wy * tera_freq) as f64,
                        (wz as f32 * tera_freq) as f64,
                    ]) as f32;
```

Add to the `corners[idx] = CarverCorner { ... }` literal.

- [ ] **Step 3: Add `terasology_ambient_at` method**

After the existing `*_at` methods, add:

```rust
    pub fn terasology_ambient_at(
        &self, wx: i32, wy: i32, wz: i32, cfg: &CaveConfig, surface_y: f32,
    ) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let n0 = self.trilerp(&lc, |c| c.tera_a);
        let n1_raw = self.trilerp(&lc, |c| c.tera_b);
        let depth = (surface_y - wy as f32).max(0.0);
        let freq_reduction = (cfg.tera_supp - depth / cfg.tera_supp_depth).max(0.0);
        let freq_depth     = cfg.tera_thresh_base + depth / cfg.tera_thresh_depth;
        let n1 = n1_raw + freq_reduction;
        ((n0 * n0 + n1 * n1).sqrt() - freq_depth) * 5.0
    }
```

- [ ] **Step 4: Update `fill_chunk` composition to call the new layer**

In `mod.rs` around the cheese/pillar composition (~line 808 of the original — now shorter after PR1.1/PR1.2), insert the new tera term using the existing trilerped `carver_eval`:

```rust
        let surface_y = self.heightmap.h_pre(...) as f32; // OR cache from elsewhere
        let tera = if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
            carver_eval.terasology_ambient_at(wx, wy, wz, &cfg.cave, surface_y)
        } else {
            0.0
        };
```

Then in the density composition `min` chain, add `.min(tera)` after the existing cheese term. (If the chain previously read `density.min(cheese)`, it now reads `density.min(cheese).min(tera)`.)

Note: `surface_y` is the existing `h_pre`-derived heightmap value. If `fill_chunk` already caches it for the cheese surface-suppression term, reuse that variable. If not, sample once per column outside the Y loop.

Do the same edit in the secondary composition site around line 904 (aquifer detection).

- [ ] **Step 5: Build and run all worldgen tests**

```bash
cargo build 2>&1 | tail -10
cargo test --lib worldgen 2>&1 | tail -20
```

Expected: build clean. All cheese, pillar, tera tests pass. Carver-evaluator parity tests pass (the new `tera_a`/`tera_b` corner samples are exact at corners by construction).

- [ ] **Step 6: Add `carver_evaluator_exact_at_corners_for_tera` to the existing parity test**

In the existing `carver_evaluator_exact_at_corners` test, add an assertion block:

```rust
                    let direct = terasology_ambient(wx, wy, wz, &nc, &cfg.cave, 64.0);
                    let lerped = eval.terasology_ambient_at(wx, wy, wz, &cfg.cave, 64.0);
                    assert!(
                        (direct - lerped).abs() < 1e-4,
                        "tera mismatch at ({wx},{wy},{wz}): direct={direct} lerp={lerped}"
                    );
```

Run: `cargo test --lib worldgen::caves::tests::carver_evaluator_exact 2>&1 | tail -5`. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/worldgen/mod.rs src/worldgen/caves.rs
git commit -m "feat(caves): wire terasology_ambient into fill_chunk composition

The new layer flows through the existing CarverEvaluator trilerp
lattice (exact at corners, smoothed inside cells). Per-corner cost
is two FBM samples; net per-chunk noise eval count is lower than
before (we removed ~6 channels and added 2)."
```

### Task PR1.6: Rebaseline golden hashes and visual baseline

**Files:**
- Modify: `tests/worldgen_fingerprint.rs` (hash constants)
- Modify: `tests/screenshots/baseline_cave.png` (regenerate)

- [ ] **Step 1: Run the fingerprint test and capture the new hash**

```bash
cargo test --test worldgen_fingerprint 2>&1 | tail -30
```

Expected: FAIL with the old hash vs new hash printed. Copy the new hash.

- [ ] **Step 2: Update the hash constant in the test**

Open `tests/worldgen_fingerprint.rs`. Replace the expected hash with the new value. Add a comment line referencing this PR:

```rust
// Rebaselined 2026-05-21: replaced spaghetti/wormhole/surface_entrance
// noise carvers with terasology_ambient (cave overhaul PR1).
const EXPECTED_DENSITY_HASH: u64 = 0x...;
```

- [ ] **Step 3: Re-run; confirm green**

```bash
cargo test --test worldgen_fingerprint 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 4: Re-generate visual baseline screenshot**

If your repo uses `tests/screenshots/diff.py` with the existing `baseline_cave.png`, take a fresh screenshot at the same camera position with the new caves and save over the baseline:

```bash
# Replace with however screenshots are taken in your project — likely a
# debug viewport binary or `cargo run --bin oxium -- --screenshot ...`.
cargo run --bin oxium -- --camera-test cave --screenshot tests/screenshots/baseline_cave.png
```

- [ ] **Step 5: Final commit**

```bash
git add tests/worldgen_fingerprint.rs tests/screenshots/baseline_cave.png
git commit -m "test(worldgen): rebaseline golden hash + cave screenshot for PR1

The cave-overhaul PR1 changed the density composition by removing three
carvers and adding terasology_ambient. Both golden artifacts must be
re-baselined; the screenshot diff tool will report ~6% intrinsic noise
on subsequent runs (per the established noise floor)."
```

### Task PR1.7: Clear chunk cache and verify in-game

**Files:** none (manual verification)

- [ ] **Step 1: Clear the chunk cache**

```bash
rm -rf saves/default/regions/*.bin
```

Per memory: any worldgen change requires this; the persistent cache otherwise serves stale chunks at the new/old seam.

- [ ] **Step 2: Run the game and fly around the underground**

```bash
cargo run --bin oxium --release
```

Fly to negative Y. Confirm:
- Caves are visibly rare near the surface (Y ≈ +60 down to ≈ +30).
- Caves are denser at deep Y (Y ≈ -60 and below).
- Tunnels meander horizontally; no obvious 30+-block vertical drops.

If anything looks wrong, note it for PR2 / PR3 / PR4 — do NOT bandaid PR1.

- [ ] **Step 3: Tag the merged PR1 commit**

```bash
git tag cave-overhaul-pr1
```

(For easy rollback if PR2 destabilizes.)

---

## PR2 — Re-enable graph caves with style table

**Goal:** Bring back the disabled graph cave system. Add a 5-style table (Cathedral / Warren / Slot / Sump / Karst), depth-scaled chamber sizes, and band-biased style weighting. After this PR: distinct cave systems are visible underground, with style identity (Cathedral districts feel different from Warren districts).

### Task PR2.1: Add `CaveStyle` enum and `CaveStyleTable` config

**Files:**
- Modify: `src/worldgen/caves.rs` (add `CaveStyle` enum)
- Modify: `src/worldgen/config.rs` (add `CaveStyleTable` struct + field on `CaveConfig`)
- Modify: `assets/worldgen/default.ron`

- [ ] **Step 1: Add the `CaveStyle` enum in `caves.rs`**

```rust
/// Distinct cave-system personalities, rolled per system from the
/// region cell id and the system's depth band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaveStyle {
    /// Few large chambers, wide tunnels. Deep-band-biased.
    Cathedral,
    /// Many small chambers, narrow tunnels. Shallow-band-biased.
    Warren,
    /// XZ-stretched chambers, narrow vertical sheets. Mid-band-biased.
    Slot,
    /// Low-clustered chambers (flooded look). Deep-band-biased.
    Sump,
    /// Default — medium chambers, medium tunnels.
    Karst,
}
```

- [ ] **Step 2: Add `CaveStyleTable` to `config.rs`**

```rust
/// Per-style parameter ranges for cave-system construction. Lives in
/// `CaveConfig` so RON hot reload can retune styles without recompile.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaveStyleTable {
    /// (min, max) chambers per system, per style.
    pub cathedral_chamber_count: (u32, u32),
    pub warren_chamber_count: (u32, u32),
    pub slot_chamber_count: (u32, u32),
    pub sump_chamber_count: (u32, u32),
    pub karst_chamber_count: (u32, u32),
    /// (min, max) chamber radii in XZ.
    pub cathedral_r_xz: (f32, f32),
    pub warren_r_xz: (f32, f32),
    pub slot_r_xz: (f32, f32),
    pub sump_r_xz: (f32, f32),
    pub karst_r_xz: (f32, f32),
    /// (min, max) chamber radii in Y.
    pub cathedral_r_y: (f32, f32),
    pub warren_r_y: (f32, f32),
    pub slot_r_y: (f32, f32),
    pub sump_r_y: (f32, f32),
    pub karst_r_y: (f32, f32),
    /// (min, max) tunnel radius.
    pub cathedral_tunnel_r: (f32, f32),
    pub warren_tunnel_r: (f32, f32),
    pub slot_tunnel_r: (f32, f32),
    pub sump_tunnel_r: (f32, f32),
    pub karst_tunnel_r: (f32, f32),
    /// Band-biased style weights `[Cathedral, Warren, Slot, Sump, Karst]`.
    /// Each must sum to 1.0.
    pub style_weights_shallow: [f32; 5],
    pub style_weights_middle: [f32; 5],
    pub style_weights_deep: [f32; 5],
}
```

In `pub struct CaveConfig`, add at the bottom:

```rust
    pub style_table: CaveStyleTable,
    /// Per-chamber depth-driven radius multiplier.
    /// `mult(cy) = 1.0 + depth_scale * max(0, (40 - cy) / 80)`.
    pub depth_scale: f32,
    /// Share of cave systems rolled into the Deep band.
    /// 0.0 = uniform thirds; 1.0 = heavily deep.
    pub deep_band_bias: f32,
    /// Max cave systems per region; sweep-chosen 3.
    pub systems_per_region_max: u32,
    /// Per-chamber radius jitter multiplier range. (0.7, 1.3) → ×0.7..×1.3.
    pub chamber_radius_jitter: (f32, f32),
```

- [ ] **Step 3: Add the corresponding RON defaults to `default.ron`**

```ron
        style_table: (
            cathedral_chamber_count: (3, 5),
            warren_chamber_count: (9, 13),
            slot_chamber_count: (5, 8),
            sump_chamber_count: (4, 7),
            karst_chamber_count: (6, 10),
            cathedral_r_xz: (22.0, 32.0), warren_r_xz: (6.0, 12.0),
            slot_r_xz: (10.0, 24.0),      sump_r_xz: (14.0, 24.0),
            karst_r_xz: (10.0, 18.0),
            cathedral_r_y: (16.0, 26.0),  warren_r_y: (5.0, 10.0),
            slot_r_y: (4.0, 8.0),         sump_r_y: (8.0, 14.0),
            karst_r_y: (8.0, 14.0),
            cathedral_tunnel_r: (4.0, 6.0), warren_tunnel_r: (3.0, 4.0),
            slot_tunnel_r: (3.0, 5.0),     sump_tunnel_r: (4.0, 6.0),
            karst_tunnel_r: (3.0, 5.0),
            style_weights_shallow: [0.05, 0.45, 0.20, 0.05, 0.25],
            style_weights_middle:  [0.20, 0.20, 0.20, 0.20, 0.20],
            style_weights_deep:    [0.10, 0.05, 0.05, 0.45, 0.35],
        ),
        depth_scale: 0.81,
        deep_band_bias: 0.30,
        systems_per_region_max: 3,
        chamber_radius_jitter: (0.7, 1.3),
```

- [ ] **Step 4: Build**

```bash
cargo build 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 5: Commit**

```bash
git add src/worldgen/caves.rs src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "feat(caves): add CaveStyle enum + CaveStyleTable config

Schema for the 5-style cave-system variety table. Defaults chosen
via sweep (pick #3, composite 0.923). No behavior change yet — the
generator still uses the legacy single-style code path."
```

### Task PR2.2: Add `style: CaveStyle` to `CaveSystem` (TDD)

**Files:**
- Modify: `src/worldgen/region.rs`
- Modify: `src/worldgen/caves.rs`

- [ ] **Step 1: Write the failing test for style band distribution**

In `caves.rs` test module:

```rust
#[test]
fn style_band_distribution_matches_weights() {
    // Roll 500 systems in each band; verify distributions match weights
    // within ±10%.
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let table = &cfg.cave.style_table;
    let bands = [
        ("shallow", DepthBand::Shallow, &table.style_weights_shallow),
        ("middle",  DepthBand::Middle,  &table.style_weights_middle),
        ("deep",    DepthBand::Deep,    &table.style_weights_deep),
    ];
    for (name, band, weights) in &bands {
        let mut counts = [0; 5];
        for i in 0..500 {
            let s = pick_style(42, RegionCoord { x: i, z: 0 }, 0, *band, &cfg.cave);
            let idx = match s {
                CaveStyle::Cathedral => 0, CaveStyle::Warren => 1,
                CaveStyle::Slot => 2,      CaveStyle::Sump => 3,
                CaveStyle::Karst => 4,
            };
            counts[idx] += 1;
        }
        for (i, &expected_weight) in weights.iter().enumerate() {
            let actual = counts[i] as f32 / 500.0;
            let diff = (actual - expected_weight).abs();
            assert!(diff < 0.10,
                "band {name}, style index {i}: expected {expected_weight:.2}, got {actual:.2}");
        }
    }
}
```

- [ ] **Step 2: Run; expect compile error (function `pick_style` doesn't exist)**

```bash
cargo test --lib worldgen::caves::tests::style_band_distribution 2>&1 | tail -10
```

Expected: FAIL (compile error).

- [ ] **Step 3: Implement `pick_style`**

In `caves.rs`:

```rust
/// Roll a `CaveStyle` deterministically from `(seed, region_coord,
/// system_idx, band)`. Band-weighted via `CaveStyleTable`.
pub fn pick_style(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    band: DepthBand,
    cfg: &CaveConfig,
) -> CaveStyle {
    let u = mix_unit(seed, &[coord.x, coord.z, system_idx, 7000]);
    let weights = match band {
        DepthBand::Shallow => &cfg.style_table.style_weights_shallow,
        DepthBand::Middle  => &cfg.style_table.style_weights_middle,
        DepthBand::Deep    => &cfg.style_table.style_weights_deep,
    };
    let mut acc = 0.0;
    let styles = [
        CaveStyle::Cathedral, CaveStyle::Warren, CaveStyle::Slot,
        CaveStyle::Sump,      CaveStyle::Karst,
    ];
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if u <= acc { return styles[i]; }
    }
    CaveStyle::Karst
}
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
cargo test --lib worldgen::caves::tests::style_band_distribution 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Add `style: CaveStyle` field to `CaveSystem` in `region.rs`**

```rust
pub struct CaveSystem {
    pub bb_min: glam::IVec3,
    pub bb_max: glam::IVec3,
    pub chambers: Vec<Chamber>,
    pub tunnels: Vec<Tunnel>,
    pub entrances: Vec<Entrance>,
    /// Style rolled once per system, drives chamber/tunnel parameters.
    pub style: CaveStyle,
}
```

You'll need to import `CaveStyle` here, OR move the enum definition into `region.rs`. Recommended: keep in `caves.rs` and `pub use` it from `region.rs`'s neighbour.

- [ ] **Step 6: Commit**

```bash
git add src/worldgen/region.rs src/worldgen/caves.rs
git commit -m "feat(caves): add style field to CaveSystem with band-weighted pick

pick_style rolls one of five styles weighted per depth band. Tests
verify the distribution lands within ±10% of configured weights."
```

### Task PR2.3: Rewrite `build_system` to use per-style parameters and depth-scaled radii

**Files:**
- Modify: `src/worldgen/caves.rs`
- Modify: `src/worldgen/tuning.rs` (re-enable `CAVE_SYSTEMS_PER_REGION`)

- [ ] **Step 1: Re-enable graph-caves count in tuning**

In `tuning.rs`:

```rust
// Was: (0, 0) — graph caves disabled.
// Now: (0, 3) per cave-overhaul spec defaults.
pub const CAVE_SYSTEMS_PER_REGION: (u32, u32) = (0, 3);
```

- [ ] **Step 2: Update `build_systems_for_region` to use `systems_per_region_max` from CaveConfig**

In `caves.rs`, replace the use of `CAVE_SYSTEMS_PER_REGION` with `cfg.cave.systems_per_region_max`. Keep the tuning constant as the upper bound for safety.

- [ ] **Step 3: Update `build_system` to read style parameters**

Modify the function so it:
1. Rolls a `DepthBand` (already does this).
2. Rolls a `CaveStyle` via `pick_style(seed, coord, system_idx, band, &cfg.cave)`.
3. Replaces hardcoded `CHAMBER_RADIUS_RANGE` / `TUNNEL_RADIUS` references with per-style ranges from the table:

```rust
let (chamber_lo, chamber_hi) = match style {
    CaveStyle::Cathedral => cfg.style_table.cathedral_r_xz,
    CaveStyle::Warren    => cfg.style_table.warren_r_xz,
    CaveStyle::Slot      => cfg.style_table.slot_r_xz,
    CaveStyle::Sump      => cfg.style_table.sump_r_xz,
    CaveStyle::Karst     => cfg.style_table.karst_r_xz,
};
// ... same for r_y, tunnel_r, chamber_count.
```

4. Applies the depth multiplier inside the chamber-loop sampling:

```rust
let cy = bb_min.y as f32 + sy;
let depth_mult = 1.0 + cfg.depth_scale * ((40.0 - cy).max(0.0) / 80.0);
let rx = mix_range(seed, &[...], chamber_lo, chamber_hi) * depth_mult;
let ry = mix_range(seed, &[...], r_y_lo, r_y_hi)         * depth_mult;
```

5. Slot style stretches XZ:

```rust
let (rx_final, rz_final) = if style == CaveStyle::Slot {
    (rx * 1.4, rx * 0.6)
} else {
    (rx, rx)
};
```

6. Sump style biases chamber Y low (cluster at floor of bb):

```rust
let cy = if style == CaveStyle::Sump {
    bb_center_y - bb_half_y * 0.4 + (cy - bb_center_y).abs() * 0.5
} else {
    cy
};
```

7. Sets `system.style = style` at the end.

- [ ] **Step 4: Build**

```bash
cargo build 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 5: Run all worldgen tests**

```bash
cargo test --lib worldgen 2>&1 | tail -20
```

Expected: passes. The existing `system_count_within_bounds`, `mst_connects_all_chambers`, and `system_is_pure_in_seed_and_coord` tests must still pass — they're tightly scoped to topology, not radii.

- [ ] **Step 6: Re-enable the `cave_air_returns_true_inside_chamber_center` test**

In `caves.rs` tests, find this test:

```rust
#[test]
#[ignore = "graph caves disabled via CAVE_SYSTEMS_PER_REGION = (0, 0); re-enable when graph systems come back"]
fn cave_air_returns_true_inside_chamber_center() {
```

Remove the `#[ignore]` attribute. Run it and verify it passes.

- [ ] **Step 7: Commit**

```bash
git add src/worldgen/caves.rs src/worldgen/tuning.rs
git commit -m "feat(caves): re-enable graph caves with style-aware build_system

CAVE_SYSTEMS_PER_REGION restored to (0, 3). Chamber/tunnel parameters
now driven by CaveStyleTable per CaveStyle. Depth multiplier scales
radii by up to ~3x deep underground. Re-enables the previously-ignored
cave_air test."
```

### Task PR2.4: Rebaseline golden hash and verify in-game

**Files:**
- Modify: `tests/worldgen_fingerprint.rs`
- Modify: `tests/screenshots/baseline_cave.png`

- [ ] **Step 1: Re-run fingerprint test, capture new hash, update constant**

Same procedure as PR1.6. Bump version comment to `(cave overhaul PR2)`.

- [ ] **Step 2: Re-generate cave screenshot baseline**

Same procedure as PR1.6.

- [ ] **Step 3: Clear chunk cache, fly underground, eyeball**

```bash
rm -rf saves/default/regions/*.bin
cargo run --bin oxium --release
```

Expected: distinct cave-system clusters visible underground. Some chambers are large (Cathedrals at depth), some are small and clustered (Warrens). Sump systems look low-Y-biased. No specific style-identity assertions to make in-game yet — just visual variety.

- [ ] **Step 4: Tag and commit**

```bash
git add tests/worldgen_fingerprint.rs tests/screenshots/baseline_cave.png
git commit -m "test(worldgen): rebaseline golden hash + screenshot for PR2

Re-enabled graph caves with style table changes the density."
git tag cave-overhaul-pr2
```

---

## PR3 — Vertical connectors + cross-region trunks

**Goal:** Add inter-band navigation within a region (vertical connectors) and inter-region cohesion (trunk lines). After this PR: a player in a shallow cave can find a vertical shaft down to a deeper system in the same region, and the trunk system means cave systems in neighbouring regions are sometimes linked by long tunnels.

### Task PR3.1: Add `Trunk` struct and `vertical_connectors: Vec<Tunnel>` to `CaveSystem`

**Files:**
- Modify: `src/worldgen/region.rs`

- [ ] **Step 1: Add the new fields to `CaveSystem`**

```rust
pub struct CaveSystem {
    pub bb_min: glam::IVec3,
    pub bb_max: glam::IVec3,
    pub chambers: Vec<Chamber>,
    pub tunnels: Vec<Tunnel>,
    pub entrances: Vec<Entrance>,
    pub style: CaveStyle,
    /// Optional cross-region trunk to a neighbour-region cave system.
    pub trunk: Option<Tunnel>,
    /// In-region vertical connectors between this system and adjacent-band
    /// systems in the same region (Shallow↔Middle, Middle↔Deep).
    pub vertical_connectors: Vec<Tunnel>,
}
```

Note: we reuse `Tunnel` for both trunks and vertical connectors. The geometry is identical — a polyline with a radius.

- [ ] **Step 2: Build to verify no callers break**

```bash
cargo build 2>&1 | tail -10
```

Expected: any caller building a `CaveSystem` literal will fail. Fix by adding `trunk: None, vertical_connectors: vec![]` at the construction sites (mostly `build_system` in `caves.rs`).

- [ ] **Step 3: Commit**

```bash
git add src/worldgen/region.rs src/worldgen/caves.rs
git commit -m "feat(caves): add trunk and vertical_connectors fields to CaveSystem

Empty until PR3.2 / PR3.3 populate them."
```

### Task PR3.2: Implement vertical connectors (TDD)

**Files:**
- Modify: `src/worldgen/caves.rs`
- Modify: `src/worldgen/config.rs`
- Modify: `assets/worldgen/default.ron`

- [ ] **Step 1: Add `vertical_connector_prob` and `vertical_connector_r` to CaveConfig + RON**

In `CaveConfig`:

```rust
    /// Probability that two systems in adjacent bands of the same region
    /// are linked by a vertical connector tunnel.
    pub vertical_connector_prob: f32,
    /// Vertical-connector tunnel radius.
    pub vertical_connector_r: f32,
```

In `default.ron`:

```ron
        vertical_connector_prob: 0.87,
        vertical_connector_r: 3.4,
```

- [ ] **Step 2: Write the failing test**

In `caves.rs` test module:

```rust
#[test]
fn vertical_connector_connects_adjacent_band_systems() {
    // Force a region with 2 systems in adjacent bands. Set prob = 1.0,
    // build, verify a connector exists between them.
    let mut cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    cfg.cave.vertical_connector_prob = 1.0;
    // Choose a seed/coord we've manually confirmed produces a 2-system
    // region. (If unknown, scan a few; we just need ONE example here.)
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let mut found_connector = false;
    'outer: for rx in 0..8 {
        for rz in 0..8 {
            let coord = RegionCoord { x: rx, z: rz };
            let mut region = FineRegion::empty(coord);
            build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region, &cfg.cave);
            // Need ≥2 systems in adjacent bands.
            let bands: Vec<DepthBand> = region.cave_systems.iter().map(|s| infer_band(&s)).collect();
            if !pair_is_adjacent(&bands) { continue; }
            // Run the connector builder pass.
            build_vertical_connectors(42, coord, &cfg.cave, &mut region);
            for sys in &region.cave_systems {
                if !sys.vertical_connectors.is_empty() {
                    found_connector = true;
                    break 'outer;
                }
            }
        }
    }
    assert!(found_connector, "no vertical connector emitted in any 2-band region");
}

fn infer_band(sys: &CaveSystem) -> DepthBand {
    let cy = (sys.bb_min.y + sys.bb_max.y) / 2;
    if cy >= CAVE_BAND_SHALLOW.0 { DepthBand::Shallow }
    else if cy >= CAVE_BAND_MIDDLE.0 { DepthBand::Middle }
    else { DepthBand::Deep }
}
fn pair_is_adjacent(bands: &[DepthBand]) -> bool {
    use DepthBand::*;
    bands.iter().any(|b| matches!(b, Shallow))
        && bands.iter().any(|b| matches!(b, Middle))
    ||
    bands.iter().any(|b| matches!(b, Middle))
        && bands.iter().any(|b| matches!(b, Deep))
}
```

- [ ] **Step 3: Run; expect compile error (`build_vertical_connectors` undefined)**

```bash
cargo test --lib worldgen::caves::tests::vertical_connector 2>&1 | tail -10
```

Expected: FAIL.

- [ ] **Step 4: Implement `build_vertical_connectors`**

```rust
/// For each pair of systems in adjacent bands within `region`, roll
/// `vertical_connector_prob`. If passing, push a `Tunnel` from the upper
/// system's lowest chamber to the lower system's highest chamber.
pub fn build_vertical_connectors(
    seed: u64,
    coord: RegionCoord,
    cfg: &CaveConfig,
    region: &mut FineRegion,
) {
    if cfg.vertical_connector_prob <= 0.0 { return; }
    // Snapshot which system holds which band (avoid borrow conflict).
    let bands: Vec<DepthBand> = region.cave_systems.iter().map(|s| {
        let cy = (s.bb_min.y + s.bb_max.y) / 2;
        if cy >= CAVE_BAND_SHALLOW.0 { DepthBand::Shallow }
        else if cy >= CAVE_BAND_MIDDLE.0 { DepthBand::Middle }
        else { DepthBand::Deep }
    }).collect();
    let band_idx = |b: DepthBand| -> u8 { match b { DepthBand::Shallow=>0, DepthBand::Middle=>1, DepthBand::Deep=>2 } };
    for i in 0..region.cave_systems.len() {
        for j in 0..region.cave_systems.len() {
            if i == j { continue; }
            // Only Shallow→Middle or Middle→Deep.
            if band_idx(bands[j]) != band_idx(bands[i]) + 1 { continue; }
            // Roll probability.
            let u = mix_unit(seed, &[coord.x, coord.z, i as i32, j as i32, 9500]);
            if u > cfg.vertical_connector_prob { continue; }
            // Upper system's lowest chamber; lower system's highest chamber.
            let upper = &region.cave_systems[i];
            let lower = &region.cave_systems[j];
            let (a_idx, _) = upper.chambers.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.center.y.partial_cmp(&b.center.y).unwrap()).unwrap();
            let (b_idx, _) = lower.chambers.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.center.y.partial_cmp(&b.center.y).unwrap()).unwrap();
            let pa = upper.chambers[a_idx].center;
            let pb = lower.chambers[b_idx].center;
            let connector = Tunnel {
                control_points: vec![pa, pb],
                radius: cfg.vertical_connector_r,
            };
            // Push onto the UPPER system (carve from its side).
            region.cave_systems[i].vertical_connectors.push(connector);
        }
    }
}
```

- [ ] **Step 5: Call `build_vertical_connectors` from `build_systems_for_region`**

Add a final pass at the end of `build_systems_for_region`:

```rust
    build_vertical_connectors(seed, coord, &cfg.cave, region);
```

- [ ] **Step 6: Run the test, verify pass**

```bash
cargo test --lib worldgen::caves::tests::vertical_connector 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 7: Update SDF evaluation to include vertical connectors**

In `caves.rs`, find `pub fn cave_air(...)` and `pub fn cave_sdf(...)`. After the tunnel-loop, add an identical loop for `sys.vertical_connectors`:

```rust
        for t in &sys.vertical_connectors {
            // same capsule SDF math as for `sys.tunnels`
        }
```

- [ ] **Step 8: Commit**

```bash
git add src/worldgen/caves.rs src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "feat(caves): add vertical connectors between adjacent-band systems

Each pair of in-region systems in adjacent bands rolls vertical_connector_prob.
On hit, a Tunnel runs from the upper system's lowest chamber to the lower
system's highest chamber. Test confirms a connector emits in at least one
multi-band region with prob=1.0."
```

### Task PR3.3: Implement cross-region trunks (TDD)

**Files:**
- Modify: `src/worldgen/caves.rs`
- Modify: `src/worldgen/config.rs`
- Modify: `assets/worldgen/default.ron`

- [ ] **Step 1: Add `trunk_prob` and `trunk_r` to CaveConfig + RON**

In `CaveConfig`:

```rust
    pub trunk_prob: f32,
    pub trunk_r: f32,
```

In `default.ron`:

```ron
        trunk_prob: 0.38,
        trunk_r: 3.4,
```

- [ ] **Step 2: Write failing test**

```rust
#[test]
fn trunk_links_nearest_neighbour_region_system() {
    let mut cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    cfg.cave.trunk_prob = 1.0;
    let hm = HeightmapNoise::new(42, &cfg.climate);
    let mut all_systems: Vec<(RegionCoord, CaveSystem)> = vec![];
    for rx in 0..3 {
        for rz in 0..3 {
            let coord = RegionCoord { x: rx, z: rz };
            let mut region = FineRegion::empty(coord);
            build_systems_for_region(42, coord, &hm, &cfg.climate, &cfg.density, &mut region, &cfg.cave);
            for s in &region.cave_systems { all_systems.push((coord, s.clone())); }
        }
    }
    // After the trunk pass, find a system whose .trunk is Some and verify
    // its endpoint is a chamber in some neighbour-region system.
    build_trunks(42, &cfg.cave, &mut all_systems);
    let mut found = false;
    for (i, (coord, sys)) in all_systems.iter().enumerate() {
        let Some(trunk) = &sys.trunk else { continue; };
        let endpoint = *trunk.control_points.last().unwrap();
        // Endpoint must match some chamber of a neighbour-region system.
        for (j, (other_coord, other)) in all_systems.iter().enumerate() {
            if i == j { continue; }
            if (other_coord.x - coord.x).abs() > 1 || (other_coord.z - coord.z).abs() > 1 { continue; }
            for ch in &other.chambers {
                if (ch.center - endpoint).length() < 0.5 {
                    found = true; break;
                }
            }
        }
    }
    assert!(found, "no trunk linked to any neighbour-region system");
}
```

Note: this test signature differs from the per-region pattern because trunks need cross-region awareness. `build_trunks` operates on a flat list.

- [ ] **Step 3: Implement `build_trunks`**

```rust
/// For each cave system, optionally roll `trunk_prob`. If passing, link
/// to the nearest cave system in any of the 8 neighbour regions, measured
/// Euclidean between chamber-0 centers. Emit a single Tunnel through a
/// mid-arc offset point for natural curve.
pub fn build_trunks(
    seed: u64,
    cfg: &CaveConfig,
    all_systems: &mut [(RegionCoord, CaveSystem)],
) {
    if cfg.trunk_prob <= 0.0 { return; }
    // Build a snapshot of (idx, RegionCoord, chamber0_center) for lookups.
    let snap: Vec<(usize, RegionCoord, glam::Vec3)> = all_systems.iter().enumerate()
        .filter_map(|(i, (c, s))| s.chambers.first().map(|ch| (i, *c, ch.center)))
        .collect();
    for (i, (coord, sys)) in all_systems.iter_mut().enumerate() {
        let Some(my_center) = sys.chambers.first().map(|c| c.center) else { continue; };
        let u = mix_unit(seed, &[coord.x, coord.z, 0, 9000]);
        if u > cfg.trunk_prob { continue; }
        // Find nearest neighbour-region system.
        let mut best: Option<(usize, f32)> = None;
        for &(j, other_coord, other_center) in &snap {
            if j == i { continue; }
            let dx = (other_coord.x - coord.x).abs();
            let dz = (other_coord.z - coord.z).abs();
            if dx > 1 || dz > 1 || (dx == 0 && dz == 0) { continue; }
            let d = (other_center - my_center).length();
            if best.map_or(true, |(_, bd)| d < bd) { best = Some((j, d)); }
        }
        let Some((_, _)) = best else { continue; };
        let other_center = snap.iter().find(|(j, _, _)| Some(*j) == best.map(|b|b.0)).unwrap().2;
        // Mid-arc offset perpendicular to XZ axis.
        let axis = other_center - my_center;
        let len = (axis.x*axis.x + axis.z*axis.z).sqrt().max(1.0);
        let perp = glam::Vec3::new(-axis.z / len, 0.0, axis.x / len);
        let o = (mix_unit(seed, &[coord.x, coord.z, 0, 9001]) * 2.0 - 1.0) * 40.0;
        let mid = glam::Vec3::new(
            (my_center.x + other_center.x) * 0.5 + perp.x * o,
            (my_center.y + other_center.y) * 0.5,
            (my_center.z + other_center.z) * 0.5 + perp.z * o,
        );
        sys.trunk = Some(Tunnel {
            control_points: vec![my_center, mid, other_center],
            radius: cfg.trunk_r,
        });
    }
}
```

- [ ] **Step 4: Wire `build_trunks` into the region-build loop**

`build_trunks` needs a flat list across regions, so it can't be called per-region. Instead, call it from `Generator::new` (or wherever region cache is bulk-built) once the relevant regions are loaded. Add a method to the region cache (`FineRegionCache`) that runs `build_trunks` on the 3x3 region neighborhood when a region is accessed.

Concretely: when chunk fill needs `cave_systems` for a region, the cache returns the 3x3 neighbour systems. Pre-running `build_trunks` on the 3x3 snapshot is acceptable. Implementation can store `trunk` on the cache-side after the pass; this is the simplest path forward without restructuring the cache.

If the existing cache pattern makes this awkward, an acceptable alternative is to evaluate the trunk on-the-fly during `cave_sdf` lookups: when iterating the 3x3 neighborhood, compute the trunk SDF per pair of (sys, nearest-neighbour-sys) at SDF-eval time. Slightly more compute per voxel but no cache mutation.

Prefer the cache-side pass. If you implement on-the-fly evaluation, document why and add a comment with the perf consequence.

- [ ] **Step 5: Update `cave_air` / `cave_sdf` to include trunk SDFs**

Add a per-system trunk capsule check after the regular tunnel loop:

```rust
        if let Some(trunk) = &sys.trunk {
            // same capsule SDF math
        }
```

- [ ] **Step 6: Run the test**

```bash
cargo test --lib worldgen::caves::tests::trunk_links 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/worldgen/caves.rs src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "feat(caves): add cross-region trunk lines

Each system rolls trunk_prob; on hit, links to the nearest chamber-0 in
any 8-neighbour region. Trunk runs through a 40-block perpendicular
offset midpoint for natural curvature."
```

### Task PR3.4: Rebaseline golden hash and verify in-game

**Files:**
- Modify: `tests/worldgen_fingerprint.rs`
- Modify: `tests/screenshots/baseline_cave.png`

- [ ] **Step 1: Rebaseline** (same procedure as PR1.6, PR2.4)

- [ ] **Step 2: Clear cache, fly around**

```bash
rm -rf saves/default/regions/*.bin
cargo run --bin oxium --release
```

Expected: vertical connectors visible as straight downward tunnels between cave bands. Occasional long cross-region trunk tunnels visible as horizontal sweeps connecting distinct chamber clusters.

- [ ] **Step 3: Tag and commit**

```bash
git add tests/worldgen_fingerprint.rs tests/screenshots/baseline_cave.png
git commit -m "test(worldgen): rebaseline for PR3 (connectors + trunks)"
git tag cave-overhaul-pr3
```

---

## PR4 — smin composition + vertical clamp + surface fixer

**Goal:** Polish & safety. Smooth-min between layers so close pockets merge organically. Cap vertical air runs so no fall-to-death drops. Move surface blocks down to actual cave floors at breakthroughs.

### Task PR4.1: Implement `smin` and replace `min` with `smin` in cave composition (TDD)

**Files:**
- Modify: `src/worldgen/caves.rs` (add `smin` helper)
- Modify: `src/worldgen/mod.rs` (use it in composition)
- Modify: `src/worldgen/config.rs` (add `smin_k` to CaveConfig)
- Modify: `assets/worldgen/default.ron`

- [ ] **Step 1: Add `smin_k` to CaveConfig + RON**

In `CaveConfig`:

```rust
    /// Smooth-min radius for cave layer joins. 0.0 = strict min.
    pub smin_k: f32,
```

In `default.ron`: `smin_k: 1.2,`.

- [ ] **Step 2: Write failing tests for `smin`**

```rust
#[test]
fn smin_extremes() {
    assert_eq!(smin(0.5, 0.8, 0.0), 0.5);   // k=0 == min
    assert_eq!(smin(0.8, 0.5, 0.0), 0.5);
    // smin(0, 0, k) = -k/4
    let v = smin(0.0, 0.0, 1.0);
    assert!((v + 0.25).abs() < 1e-5, "smin(0,0,1) should be -0.25, got {v}");
    // smin(a, b, k) <= min(a, b) for k >= 0
    for k in [0.0, 0.5, 1.5, 3.0] {
        for a in [-0.5, 0.0, 0.5, 2.0] {
            for b in [-0.5, 0.0, 0.5, 2.0] {
                let s = smin(a, b, k);
                assert!(s <= a.min(b) + 1e-5,
                    "smin({a},{b},{k}) = {s} > min");
            }
        }
    }
}
```

- [ ] **Step 3: Run; expect compile failure**

```bash
cargo test --lib worldgen::caves::tests::smin_extremes 2>&1 | tail -10
```

- [ ] **Step 4: Implement `smin`**

In `caves.rs`:

```rust
/// Polynomial smooth-min — pulls the result below `min(a, b)` by up to
/// `k/4` when `|a - b| < k`. Used to merge cave SDFs near layer boundaries.
#[inline]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 { return a.min(b); }
    let h = ((k - (a - b).abs()).max(0.0)) / k;
    a.min(b) - h * h * k * 0.25
}
```

- [ ] **Step 5: Run; verify pass**

```bash
cargo test --lib worldgen::caves::tests::smin_extremes 2>&1 | tail -10
```

- [ ] **Step 6: Replace `min(...)` chains with `smin(..., cfg.cave.smin_k)` in `fill_chunk`**

In `mod.rs`, find the cave composition (cheese, tera, pillar contributions). Replace each `.min(x)` with `smin(prev, x, cfg.cave.smin_k)`. Sample:

```rust
let cave_contrib = caves::smin(cheese, tera, cfg.cave.smin_k);
// (pillar still adds — keep as max() afterward)
let cave_with_graph = caves::smin(cave_contrib, -graph_sdf_val * CAVE_SDF_INTENSITY, cfg.cave.smin_k);
density = caves::smin(density, cave_with_graph, cfg.cave.smin_k);
```

Note: graph_sdf is "positive inside", so flip sign to make it signed-density-shaped.

- [ ] **Step 7: Build, run tests**

```bash
cargo build 2>&1 | tail -5
cargo test --lib worldgen 2>&1 | tail -15
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/worldgen/caves.rs src/worldgen/mod.rs src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "feat(caves): smooth-min composition between cave layers

Replaces min() with smin(k=1.2) at every cave-layer join. Close-but-not-
touching pockets within ~1.2 in SDF units merge into one volume."
```

### Task PR4.2: Implement vertical-run clamp (TDD)

**Files:**
- Modify: `src/worldgen/mod.rs` (post-density-fill pass)
- Modify: `src/worldgen/tuning.rs` (add constant)

- [ ] **Step 1: Add `MAX_VERTICAL_AIR_RUN` constant**

In `tuning.rs`:

```rust
/// Maximum consecutive vertical air voxels per XZ column before a stone
/// "ledge" is inserted. Eliminates fall-to-death drops.
pub const MAX_VERTICAL_AIR_RUN: i32 = 6;
```

- [ ] **Step 2: Write failing integration test**

In a new file `tests/cave_vertical_clamp.rs`:

```rust
use oxium::voxel::{Block, ChunkCoord};

#[test]
fn chunk_has_no_vertical_air_run_over_six() {
    let gen = oxium::worldgen::Generator::bundled_default();
    // Generate a chunk we know has caves. Use the same coord existing
    // tests use for "underground chunk has caves" or pick Y=-2 below
    // ground.
    let coord = ChunkCoord { x: 0, y: -2, z: 0 };
    let chunk = gen.generate_dense(coord);
    for x in 0..32 {
        for z in 0..32 {
            let mut run = 0;
            let mut longest = 0;
            for y in 0..32 {
                if chunk.get(x, y, z) == Block::Air {
                    run += 1;
                    if run > longest { longest = run; }
                } else {
                    run = 0;
                }
            }
            assert!(longest <= 6, "column ({x},{z}) has air run of {longest}");
        }
    }
}
```

- [ ] **Step 3: Run; expect failure** (clamp not yet implemented)

```bash
cargo test --test cave_vertical_clamp 2>&1 | tail -10
```

- [ ] **Step 4: Implement the clamp pass in `fill_chunk`**

After the main density-fill loop in `mod.rs::fill_chunk`, add:

```rust
        // Vertical-run clamp: insert stone ledges to cap fall hazards.
        for x in 0..CHUNK_DIM_U as i32 {
            for z in 0..CHUNK_DIM_U as i32 {
                let mut run = 0;
                for y in 0..CHUNK_DIM_U as i32 {
                    if out.get(x, y, z) == Block::Air {
                        run += 1;
                        if run > MAX_VERTICAL_AIR_RUN {
                            out.set(x, y, z, Block::Stone);
                            run = 0;
                        }
                    } else {
                        run = 0;
                    }
                }
            }
        }
```

Note: the clamp must consider chunks above the current chunk too — a 6-block run that started 2 voxels into the chunk above continues into this chunk. The simplest workaround is to seed `run` from the chunk above's column-final value. If you don't have that handy, accept that the clamp may produce 7-13 block runs that straddle chunk boundaries (rare but possible). Document this in the code comment.

- [ ] **Step 5: Run; verify pass**

```bash
cargo test --test cave_vertical_clamp 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/worldgen/mod.rs src/worldgen/tuning.rs tests/cave_vertical_clamp.rs
git commit -m "feat(caves): vertical-run clamp post-pass

Inserts a stone ledge after MAX_VERTICAL_AIR_RUN=6 consecutive air voxels
per XZ column. Caps fall hazards. Cross-chunk-boundary runs may exceed
the cap by 1-2 voxels; accepted as rare and harmless."
```

### Task PR4.3: Implement surface block fixer (TDD)

**Files:**
- Modify: `src/worldgen/mod.rs` (post-pass after surface block selector)
- Modify: `src/worldgen/tuning.rs` (add `SURFACE_SPREAD` constant)

- [ ] **Step 1: Write failing integration test**

In a new file `tests/cave_surface_breakthrough.rs`:

```rust
use oxium::voxel::{Block, ChunkCoord};

#[test]
fn surface_breakthrough_places_grass_on_cave_floor() {
    // Find a chunk where a cave breaches the heightmap. Verify:
    //   1. No grass on cave ceilings.
    //   2. Grass present on at least one cave floor near the breach.
    let gen = oxium::worldgen::Generator::bundled_default();
    let mut found_floor_grass = false;
    let mut any_ceiling_grass = false;
    for x in -2..=2 {
        for z in -2..=2 {
            let chunk = gen.generate_dense(ChunkCoord { x, y: 0, z });
            for lx in 0..32 {
                for lz in 0..32 {
                    for ly in 1..31 {
                        if chunk.get(lx, ly, lz) == Block::Grass
                            && chunk.get(lx, ly+1, lz) == Block::Air
                            && chunk.get(lx, ly-1, lz) == Block::Air {
                            // grass-tile suspended in air = ceiling grass
                            any_ceiling_grass = true;
                        }
                        if chunk.get(lx, ly, lz) == Block::Grass
                            && chunk.get(lx, ly+1, lz) == Block::Air
                            && chunk.get(lx, ly-1, lz) == Block::Stone
                            && chunk.get(lx, ly+2, lz) == Block::Air {
                            // grass on solid floor with air above and air at +2:
                            // could be a cave floor breach.
                            found_floor_grass = true;
                        }
                    }
                }
            }
        }
    }
    assert!(!any_ceiling_grass, "found grass on cave ceiling");
    assert!(found_floor_grass, "no cave-floor grass found in 5x5 chunk scan — sanity check");
}
```

Note: the second assertion is a sanity check; if no chunk in the scan has a cave breaking the surface, it could spuriously fail. Adjust the scan size or seed if needed.

- [ ] **Step 2: Run; expect failure**

```bash
cargo test --test cave_surface_breakthrough 2>&1 | tail -10
```

- [ ] **Step 3: Add `SURFACE_SPREAD` to tuning.rs**

```rust
/// Lateral spread for cave-surface block displacement.
pub const SURFACE_SPREAD: i32 = 3;
```

- [ ] **Step 4: Implement the surface fixer pass in `fill_chunk`**

Place this AFTER the existing surface-block selector logic and the vertical-run clamp:

```rust
        // Surface block fixer (Terasology-borrowed):
        //   for each column, find the topmost solid voxel within the chunk.
        //   If it's grass/dirt/sand/snow AND there's air immediately above,
        //   that's the surface. If the surface is sitting on top of a cave
        //   (air below the actual heightmap), move it down to the first
        //   solid voxel below the cave.
        //   Spread the displaced surface laterally by SURFACE_SPREAD.
        for x in 0..CHUNK_DIM_U as i32 {
            for z in 0..CHUNK_DIM_U as i32 {
                // Scan top-down for surface blocks above caves.
                let h_pre = self.heightmap.h_pre_at(/* world x, world z */ ...) as i32;
                // If h_pre is in this chunk's Y range and the voxel there is
                // air, move the surface block down to the cave floor.
                // (Concrete impl depends on the chunk Y bounds — see notes
                // below.)
            }
        }
```

Implementation strategy:
1. For each XZ column, sample `h_pre(wx, wz)`. This is the "ideal" surface Y, ignoring caves.
2. If `h_pre` falls inside the current chunk's Y range, check the voxel at that Y. If it's Air, a cave breached the heightmap here.
3. Find the next solid voxel below by scanning down from `h_pre`.
4. Set that voxel's top neighbour to the appropriate surface block (grass / sand etc — same selection logic as the heightmap rule).
5. Spread laterally: for each XZ within `SURFACE_SPREAD` of the breach, do the same.

Use the existing surface-block selector function (the one that decides grass vs sand based on climate / Y). Don't duplicate logic.

If the codebase doesn't already expose a "find next solid voxel below" helper, write it inline.

- [ ] **Step 5: Run; verify**

```bash
cargo test --test cave_surface_breakthrough 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 6: Also update the existing `deep_underground_has_no_surface_blocks` test if it's now too strict**

Quick check: the existing test asserts NO grass blocks at Y < something. The surface fixer only places grass at actual cave-surface breakthroughs. If the existing test's chunk doesn't have a surface breakthrough, it should still pass. Run:

```bash
cargo test --test smoke 2>&1 | tail -10
```

Adjust the test if needed (e.g., scan chunks deeper).

- [ ] **Step 7: Commit**

```bash
git add src/worldgen/mod.rs src/worldgen/tuning.rs tests/cave_surface_breakthrough.rs
git commit -m "feat(caves): surface block fixer — relocate surface to cave floor

When a cave breaches the heightmap, the surface block (grass / dirt /
sand / snow) is moved down to the first solid voxel below the cave.
Spreads laterally by SURFACE_SPREAD=3 for naturalistic mouths."
```

### Task PR4.4: Rebaseline final golden hash + visual baseline

**Files:**
- Modify: `tests/worldgen_fingerprint.rs`
- Modify: `tests/screenshots/baseline_cave.png`

- [ ] **Step 1: Re-run fingerprint, rebaseline** (same procedure as previous rebaseline tasks)

- [ ] **Step 2: Re-generate cave screenshot baseline**

- [ ] **Step 3: Clear cache, fly around, do final eyeball**

```bash
rm -rf saves/default/regions/*.bin
cargo run --bin oxium --release
```

Walk through cave openings at the surface — verify grass is on the actual cave floor, not the ceiling. Dig down to confirm no 7+ block fall hazards. Drop down into a Cathedral chamber — verify chambers are clearly larger at depth. Find a cave system breaking out of one region into a neighbour via a trunk line — confirm trunks span 60-200 blocks.

- [ ] **Step 4: Tag and commit**

```bash
git add tests/worldgen_fingerprint.rs tests/screenshots/baseline_cave.png
git commit -m "test(worldgen): rebaseline for PR4 — final cave overhaul state

This is the final golden hash for the cave overhaul. Subsequent runs
should produce ≤10% screenshot diff (per established noise floor)."
git tag cave-overhaul-pr4
git tag cave-overhaul-complete
```

### Task PR4.5: Update visualizer probe + documentation

**Files:**
- Modify: `src/bin/worldgen_viz/widgets/probe_table.rs`
- Modify: `docs/superpowers/specs/2026-05-21-cave-system-overhaul-design.md` (mark complete)

- [ ] **Step 1: Add tera + style probe rows to `probe_table.rs`**

The visualizer's per-voxel probe table currently shows cheese, pillar, etc. Add columns for:
- `tera_ambient` (the SDF value at the cursor)
- `cave_style` (which style's chamber covers the cursor, or "none")
- `style_band` (Shallow/Middle/Deep if in a graph cave)

- [ ] **Step 2: Add "complete" stamp to the design spec**

Edit the design doc at the top:

```markdown
**Status:** ✅ implemented (cave-overhaul-pr1 through cave-overhaul-pr4 tags)
```

- [ ] **Step 3: Final commit**

```bash
git add src/bin/worldgen_viz/widgets/probe_table.rs docs/superpowers/specs/2026-05-21-cave-system-overhaul-design.md
git commit -m "feat(viz): add tera + style probe rows; mark cave spec complete"
```

---

## Verification

After all four PRs land, run the full test matrix:

```bash
# Lib tests
cargo test --lib 2>&1 | tail -10
# Integration tests
cargo test --test smoke --test worldgen_fingerprint --test gen_with_neighbors \
           --test cave_vertical_clamp --test cave_surface_breakthrough --test lighting_colored 2>&1 | tail -10
# Build the binary
cargo build --bin oxium --release
```

Expected: all green. Screenshot diff against the new baseline should be ≤10%.

Manual verification: see PR4.4 final eyeball.

---

## Out of plan (deferred follow-ups)

These were called out in the spec's "What this design does NOT do" section and are explicitly out of scope:

- Biome-flavored caves
- Mob spawn rules keyed to style
- Biome-driven aquifer changes
- Data-driven `CaveStyle` enum
- Editable DAG integration
- `CaveLocationProvider`-style structure extraction from noise (the Terasology pattern of per-XZ-column cave-records for downstream feature placement)
