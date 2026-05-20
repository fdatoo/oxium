# Worldgen viz redesign — PR 2: Pipeline overlays + column probe (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface every pipeline stage the worldgen pipeline already computes (plate, climate, hydrology, biome, aquifer) as a 2D pan/zoom map overlay, and add a click-to-pin column probe that shows the full pipeline trace for one column.

**Architecture:** Library adds a `probe` module with `ColumnProbe`, `Stage`, `DensityBreakdown` plus three additive `Generator` methods (`probe_column`, `sample_stage`, `evaluate_density_breakdown`). Viz adds `overlays/` (MapView + per-stage samplers + colormap palettes) and a `probe` module wiring click-on-map → pinned column → field table in the right panel.

**Tech Stack:** Same as PR 1 (Rust, wgpu, egui, glam). No new crates.

**Spec:** [`docs/superpowers/specs/2026-05-19-worldgen-viz-redesign-design.md`](../specs/2026-05-19-worldgen-viz-redesign-design.md), sections "Worldgen API additions", "Pipeline overlays", "Column probe".

**Prior PR:** PR 1 (streaming skeleton) is shipped on this branch. The right SidePanel is currently a "Probe" placeholder labelled `(PR 2)`. This plan replaces that placeholder.

**Worldgen state:** This worktree was rebased onto main at `7995bfc` after PR 1 landed, picking up PR 6 (surface rules DSL), PR 7 (MC aquifer + `Block::Lava`), PR 8 (cheese/spaghetti/pillar carvers). The `Stage` enum in this PR includes aquifer stages.

---

## Stage list (v1)

| Stage | Sampler | Colormap |
|---|---|---|
| `Continentalness` | `plate_at(seed, wx, wz).signed_continentalness` | divergent ocean↔continent |
| `PlateId` | `plate_at(...).primary.id` | categorical (hashed→hue) |
| `Temperature` | climate noise | viridis |
| `Humidity` | climate noise | viridis |
| `Desertness` | desert-mask noise | viridis |
| `Weirdness` | weirdness noise | viridis |
| `HPre` | heightmap pre-carve | terrain ramp |
| `ValleyCarve` | `valley_carve(wx, wz, &region)` | hot (depth) |
| `HTarget` | `column_data.height` | terrain ramp |
| `FlowAccum` | `region.flow_accum_at(wx, wz)` | log blues |
| `BiomeId` | `column_data.biome` | categorical |
| `AquiferY` | aquifer cell `y_top` at this column | divergent around sea level |
| `AquiferSubstance` | `Water`=blue / `Lava`=orange | binary |

13 stages total. Each implements the same `StageSampler` trait so adding more in PR 3 (cave-coverage, density-at-Y, etc.) is mechanical.

---

## File structure (delta)

**Create (library):**
- `src/worldgen/probe.rs` — `ColumnProbe`, `Stage`, `DensityBreakdown` types + impls.

**Modify (library):**
- `src/worldgen/mod.rs` — `pub mod probe;` + three new methods on `impl Generator`.

**Create (viz):**
- `src/bin/worldgen_viz/overlays/mod.rs` — `MapView` + module exports.
- `src/bin/worldgen_viz/overlays/stages.rs` — `StageSampler` trait + 13 impls.
- `src/bin/worldgen_viz/overlays/colormap.rs` — gradient palettes.
- `src/bin/worldgen_viz/probe.rs` — `Probe` UI state (pinned coords + last `ColumnProbe`).
- `src/bin/worldgen_viz/widgets/probe_table.rs` — formatted field-value list.

**Modify (viz):**
- `src/bin/worldgen_viz/main.rs` — register `mod overlays; mod probe;`.
- `src/bin/worldgen_viz/widgets/mod.rs` — add `pub mod probe_table;`.
- `src/bin/worldgen_viz/session.rs` — add `pub probe: Probe` field; init in `Session::new`.
- `src/bin/worldgen_viz/layout.rs` — replace right panel placeholder with live probe + map sections.

**Tests:**
- `#[cfg(test)] mod tests` in `src/worldgen/probe.rs` and each new viz module that has pure logic.

---

## Library API additions

```rust
// In oxium::worldgen::probe

pub struct ColumnProbe {
    pub wx: i32, pub wz: i32,
    // Geometry
    pub plate: PlateLookup,
    pub continentalness: f32,
    pub h_pre: f32,
    pub valley_carve: f32,
    pub h_target: i32,
    pub is_cliff: bool,
    pub slope: f32,
    // Climate
    pub temperature: f32,
    pub humidity: f32,
    pub desertness: f32,
    pub weirdness: f32,
    pub biome: Biome,
    // Hydrology
    pub flow_accum: u32,
    pub lake_rim: Option<i32>,
    // Aquifer
    pub aquifer_y_top: i32,
    pub aquifer_substance: aquifer::Substance,
    // Cave systems intersecting this column's chunk
    pub cave_systems_count: usize,
}

pub struct DensityBreakdown {
    pub wx: i32, pub wy: i32, pub wz: i32,
    pub bias: f32,
    pub base_3d: f32,
    pub cave_sdf: f32,
    pub cheese: f32,
    pub spaghetti: f32,
    pub pillar: f32,
    pub final_density: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Continentalness, PlateId,
    Temperature, Humidity, Desertness, Weirdness,
    HPre, ValleyCarve, HTarget, FlowAccum,
    BiomeId,
    AquiferY, AquiferSubstance,
}

impl Stage {
    pub const ALL: &'static [Stage] = &[/* 13 variants */];
    pub fn label(self) -> &'static str { /* "Continentalness", etc. */ }
    pub fn is_categorical(self) -> bool { /* PlateId, BiomeId, AquiferSubstance */ }
}

// New methods on impl Generator (in mod.rs)
pub fn probe_column(&self, wx: i32, wz: i32) -> ColumnProbe;
pub fn sample_stage(&self, stage: Stage, wx: i32, wz: i32) -> f32;
pub fn evaluate_density_breakdown(&self, wx: i32, wy: i32, wz: i32) -> DensityBreakdown;
```

All additive; no signature changes to existing methods.

---

## Tasks

### Task 1: Library — `ColumnProbe`, `Stage`, `DensityBreakdown` types

**Files:**
- Create: `src/worldgen/probe.rs`
- Modify: `src/worldgen/mod.rs` (add `pub mod probe;`)

This task introduces the TYPES only — no Generator methods yet. Empty structs/enums with all the fields. Subsequent tasks fill in the impls.

- [ ] **Step 1: Write `src/worldgen/probe.rs`**

```rust
//! Inspection types for the worldgen visualizer (PR 2).
//!
//! `ColumnProbe` is a snapshot of every value the pipeline computes
//! for one (wx, wz) column. `Stage` enumerates the per-column scalar
//! stages exposed as 2D overlays. `DensityBreakdown` is the
//! per-voxel density decomposition surfaced in the probe panel's
//! "sliding y" section.

use crate::voxel::block::Block;
use crate::worldgen::aquifer;
use crate::worldgen::plates::PlateLookup;
use crate::worldgen::Biome;

/// Full pipeline trace for one (wx, wz) column.
#[derive(Debug, Clone)]
pub struct ColumnProbe {
    pub wx: i32,
    pub wz: i32,
    // Geometry
    pub plate: PlateLookup,
    pub continentalness: f32,
    pub h_pre: f32,
    pub valley_carve: f32,
    pub h_target: i32,
    pub is_cliff: bool,
    pub slope: f32,
    // Climate
    pub temperature: f32,
    pub humidity: f32,
    pub desertness: f32,
    pub weirdness: f32,
    pub biome: Biome,
    // Hydrology
    pub flow_accum: u32,
    pub lake_rim: Option<i32>,
    // Aquifer
    pub aquifer_y_top: i32,
    pub aquifer_substance: aquifer::Substance,
    // Cave systems whose bbox intersects this column's region
    pub cave_systems_count: usize,
}

/// Per-voxel density decomposition. Filled lazily as the user drags
/// the y-slider in the probe panel.
#[derive(Debug, Clone, Copy)]
pub struct DensityBreakdown {
    pub wx: i32,
    pub wy: i32,
    pub wz: i32,
    /// Bias from `(h_target - wy) / FALLOFF`.
    pub bias: f32,
    /// 3D base noise contribution at this voxel.
    pub base_3d: f32,
    /// Graph cave / entrance / wormhole combined SDF.
    pub cave_sdf: f32,
    /// Cheese carver contribution.
    pub cheese: f32,
    /// Spaghetti carver contribution.
    pub spaghetti: f32,
    /// Pillar contribution (adds back density inside carved volumes).
    pub pillar: f32,
    /// Composed density after all contributions (post-slide).
    pub final_density: f32,
    /// The block this voxel would resolve to. Computed by re-running
    /// the same selection logic `fill_chunk` uses.
    pub block: Block,
}

/// Per-column scalar stages the overlay map can render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Continentalness,
    PlateId,
    Temperature,
    Humidity,
    Desertness,
    Weirdness,
    HPre,
    ValleyCarve,
    HTarget,
    FlowAccum,
    BiomeId,
    AquiferY,
    AquiferSubstance,
}

impl Stage {
    pub const ALL: &'static [Stage] = &[
        Stage::Continentalness,
        Stage::PlateId,
        Stage::Temperature,
        Stage::Humidity,
        Stage::Desertness,
        Stage::Weirdness,
        Stage::HPre,
        Stage::ValleyCarve,
        Stage::HTarget,
        Stage::FlowAccum,
        Stage::BiomeId,
        Stage::AquiferY,
        Stage::AquiferSubstance,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Stage::Continentalness => "Continentalness",
            Stage::PlateId => "Plate ID",
            Stage::Temperature => "Temperature",
            Stage::Humidity => "Humidity",
            Stage::Desertness => "Desertness",
            Stage::Weirdness => "Weirdness",
            Stage::HPre => "h_pre",
            Stage::ValleyCarve => "Valley carve",
            Stage::HTarget => "h_target",
            Stage::FlowAccum => "Flow accumulation",
            Stage::BiomeId => "Biome",
            Stage::AquiferY => "Aquifer Y",
            Stage::AquiferSubstance => "Aquifer substance",
        }
    }

    /// Categorical stages need a discrete colormap (PlateId, BiomeId,
    /// AquiferSubstance); the rest are scalar.
    pub fn is_categorical(self) -> bool {
        matches!(self, Stage::PlateId | Stage::BiomeId | Stage::AquiferSubstance)
    }
}
```

- [ ] **Step 2: Add `pub mod probe;` to `src/worldgen/mod.rs`** at the same position as the other `pub mod` declarations (alphabetical, after `plates`).

- [ ] **Step 3: Build**

Run: `cargo check --bin worldgen_viz && cargo check --lib`. Expected: clean (the types are unused in this task; warnings are fine).

- [ ] **Step 4: Commit**

```bash
git add src/worldgen/probe.rs src/worldgen/mod.rs
git commit -m "worldgen: ColumnProbe + Stage + DensityBreakdown types (viz PR 2)"
```

---

### Task 2: Library — `Generator::probe_column` impl

**Files:**
- Modify: `src/worldgen/mod.rs`

Adds `pub fn probe_column(&self, wx: i32, wz: i32) -> ColumnProbe`. Internally reuses `column_data(wx, wz)` for the values it already computes; samples `plate_at`, climate noise, aquifer system, flow_accum directly for the rest.

- [ ] **Step 1: Read what's available**

You'll need to call (in `mod.rs`):
- `plates::plate_at(self.seed, wx, wz)` → `PlateLookup` (gives `primary`, `secondary`, `boundary_t`, plus `signed_continentalness`).
- `self.climate.temperature_at(wx, wz)` / `humidity_at` / `desertness_at` / `weirdness_at` — confirm these exist in `climate::Climate` or via `Generator::climate`. If named differently, find the actual accessors. The `column_data` method already calls them so the signatures are visible there.
- `self.aquifer.cell_at(wx, wz)` or similar — read `aquifer::AquiferSystem`'s public surface in `src/worldgen/aquifer.rs:165`. If the right accessor doesn't exist, add a `pub fn lookup_column(&self, wx: i32, wz: i32) -> AquiferCell` to AquiferSystem.
- `self.column_data(wx, wz)` for `is_cliff`, `desertness`, `biome`, `lake_rim`, `h_target = height`.
- For `h_pre` and `valley_carve`, find them via `column_data_with` or call `heightmap::h_pre(...)` and `hydrology::valley_carve(...)` directly. Read `mod.rs::column_data_with` for the exact calls.
- For `flow_accum`, fetch the fine region via the cache and read `region.flow_accum_at(wx, wz)` or similar.
- For `cave_systems_count`, gather systems intersecting the chunk containing (wx, h_target, wz) via `caves::build_systems_for_region` keyed by the column's region coord.
- For `slope`, use the existing `h_pre` gradient approximation that the cliff detection uses; if it's not exposed publicly, factor it out from `column_data_with` into a private helper that `probe_column` can also call.

- [ ] **Step 2: Add to `impl Generator`** (in `mod.rs`):

```rust
/// Snapshot every pipeline value computed for this column. Used by
/// the viz column probe. Read-only, byte-stable per (seed, wx, wz).
pub fn probe_column(&self, wx: i32, wz: i32) -> probe::ColumnProbe {
    use crate::worldgen::probe::ColumnProbe;
    // Reuse column_data for the values it already produces.
    let col = self.column_data(wx, wz);

    // Plate
    let plate = crate::worldgen::plates::plate_at(self.seed, wx, wz);
    let continentalness = plate.signed_continentalness();

    // Heightmap pre-carve (read straight from the heightmap noise).
    let h_pre = self.heightmap.h_pre(self.seed, wx, wz);

    // Valley carve at this column.
    let regions = self.chunk_regions_for(wx, wz);
    let valley_carve = crate::worldgen::hydrology::valley_carve(wx, wz, &regions);

    // Slope (factor the cliff-detection's gradient out of column_data_with
    // into a helper if it isn't already public).
    let slope = self.slope_at(wx, wz);

    // Climate
    let cfg_arc = self.config_snapshot();
    let cfg = &*cfg_arc;
    let temperature = self.climate.temperature_at(wx, wz);
    let humidity = self.climate.humidity_at(wx, wz);
    let weirdness = self.weirdness_noise.sample(wx, wz);

    // Hydrology
    let flow_accum = regions
        .fine_region(wx, wz)
        .map(|r| r.flow_accum_at(wx, wz))
        .unwrap_or(0);

    // Aquifer
    let acell = self.aquifer.cell_for_column(wx, wz);

    // Cave systems intersecting this column's region.
    let cave_systems = self.cave_systems_for_column(wx, wz);

    ColumnProbe {
        wx, wz,
        plate,
        continentalness,
        h_pre,
        valley_carve,
        h_target: col.height,
        is_cliff: col.is_cliff,
        slope,
        temperature,
        humidity,
        desertness: col.desertness,
        weirdness,
        biome: col.biome,
        flow_accum,
        lake_rim: col.lake_rim,
        aquifer_y_top: acell.y_top,
        aquifer_substance: acell.substance,
        cave_systems_count: cave_systems.len(),
    }
}
```

The actual accessor names above (`plate.signed_continentalness()`, `self.climate.temperature_at`, `self.heightmap.h_pre`, `region.flow_accum_at`, `self.aquifer.cell_for_column`, `self.cave_systems_for_column`, `self.slope_at`, `self.chunk_regions_for`, `regions.fine_region`) probably DON'T all exist with those exact names. Open `src/worldgen/mod.rs::column_data_with` and `src/worldgen/heightmap.rs`/`aquifer.rs`/`climate.rs` to find the actual function shapes used internally; either call them directly or add thin public wrappers. **Where the existing helper is private and you need a public version, add it.** Goal: probe_column should compile and produce a `ColumnProbe` where every field matches a value the existing pipeline already computes.

- [ ] **Step 3: Add the test**

In `src/worldgen/probe.rs`, append:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::Generator;

    #[test]
    fn probe_column_is_deterministic() {
        let gen = Generator::new(42);
        let a = gen.probe_column(100, 200);
        let b = gen.probe_column(100, 200);
        assert_eq!(a.h_target, b.h_target);
        assert_eq!(a.biome, b.biome);
        assert!((a.continentalness - b.continentalness).abs() < 1e-6);
        assert!((a.temperature - b.temperature).abs() < 1e-6);
    }

    #[test]
    fn probe_column_height_matches_column_data() {
        let gen = Generator::new(42);
        let p = gen.probe_column(100, 200);
        let c = gen.column_data(100, 200);
        assert_eq!(p.h_target, c.height);
        assert_eq!(p.biome, c.biome);
        assert_eq!(p.is_cliff, c.is_cliff);
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --lib worldgen::probe::tests`. Expected: 2 pass.

- [ ] **Step 5: Commit**

```bash
git add src/worldgen/
git commit -m "worldgen: Generator::probe_column — full pipeline trace per column (viz PR 2)"
```

---

### Task 3: Library — `Generator::sample_stage`

**Files:**
- Modify: `src/worldgen/mod.rs`

Adds `pub fn sample_stage(&self, stage: Stage, wx: i32, wz: i32) -> f32`. Internally a match over `Stage`:

- Continentalness → `plates::plate_at(seed, wx, wz).signed_continentalness()`
- PlateId → hash of `plate.primary.id` cast to f32 (overlays use it for categorical hue keying)
- Temperature / Humidity / Desertness / Weirdness → climate sampler
- HPre → heightmap pre-carve
- ValleyCarve → `valley_carve` for the column
- HTarget → `column_data(wx, wz).height as f32`
- FlowAccum → `flow_accum` for the column, cast to f32
- BiomeId → biome variant as f32 (categorical)
- AquiferY → aquifer cell's `y_top as f32`
- AquiferSubstance → 0.0 for Water, 1.0 for Lava (categorical)

- [ ] **Step 1: Add to `impl Generator`**

```rust
pub fn sample_stage(&self, stage: probe::Stage, wx: i32, wz: i32) -> f32 {
    use crate::worldgen::probe::Stage;
    match stage {
        Stage::Continentalness => {
            crate::worldgen::plates::plate_at(self.seed, wx, wz).signed_continentalness()
        }
        Stage::PlateId => {
            let p = crate::worldgen::plates::plate_at(self.seed, wx, wz);
            // Stable hash → [0, 1] for categorical hue keying.
            (crate::worldgen::hash::mix2(self.seed, p.primary.id as i64) & 0xFFFF) as f32 / 65535.0
        }
        Stage::Temperature => self.climate.temperature_at(wx, wz),
        Stage::Humidity => self.climate.humidity_at(wx, wz),
        Stage::Desertness => self.column_data(wx, wz).desertness,
        Stage::Weirdness => self.weirdness_noise.sample(wx, wz),
        Stage::HPre => self.heightmap.h_pre(self.seed, wx, wz),
        Stage::ValleyCarve => {
            let regions = self.chunk_regions_for(wx, wz);
            crate::worldgen::hydrology::valley_carve(wx, wz, &regions)
        }
        Stage::HTarget => self.column_data(wx, wz).height as f32,
        Stage::FlowAccum => {
            let regions = self.chunk_regions_for(wx, wz);
            regions.fine_region(wx, wz).map(|r| r.flow_accum_at(wx, wz) as f32).unwrap_or(0.0)
        }
        Stage::BiomeId => self.column_data(wx, wz).biome as i32 as f32,
        Stage::AquiferY => self.aquifer.cell_for_column(wx, wz).y_top as f32,
        Stage::AquiferSubstance => match self.aquifer.cell_for_column(wx, wz).substance {
            crate::worldgen::aquifer::Substance::Water => 0.0,
            crate::worldgen::aquifer::Substance::Lava => 1.0,
        },
    }
}
```

Same caveat as Task 2: many of these accessors may not exist by these exact names. Use the same actual functions Task 2 wired up; if you added thin public wrappers there, they apply here too.

- [ ] **Step 2: Tests** — append to `src/worldgen/probe.rs::tests`:

```rust
#[test]
fn sample_stage_is_deterministic() {
    let gen = Generator::new(42);
    for &stage in Stage::ALL {
        let a = gen.sample_stage(stage, 200, 300);
        let b = gen.sample_stage(stage, 200, 300);
        assert_eq!(a.to_bits(), b.to_bits(), "stage {:?} not byte-stable", stage);
    }
}

#[test]
fn sample_stage_matches_probe_for_scalar_stages() {
    let gen = Generator::new(42);
    let p = gen.probe_column(200, 300);
    let cont = gen.sample_stage(Stage::Continentalness, 200, 300);
    assert!((cont - p.continentalness).abs() < 1e-5);
    let h = gen.sample_stage(Stage::HTarget, 200, 300);
    assert!((h - p.h_target as f32).abs() < 1e-5);
}
```

- [ ] **Step 3: Run** `cargo test --lib worldgen::probe::tests`. All 4 tests in the module pass.

- [ ] **Step 4: Commit**

```bash
git add src/worldgen/
git commit -m "worldgen: Generator::sample_stage — per-stage scalar sampler (viz PR 2)"
```

---

### Task 4: Library — `Generator::evaluate_density_breakdown`

**Files:**
- Modify: `src/worldgen/mod.rs`

Adds the per-voxel breakdown used by the probe's "sliding y" section. The existing `fill_chunk` evaluates all these contributions already — extract them into a method that returns the values rather than reducing them to one bit (solid/air).

- [ ] **Step 1: Read `fill_chunk`'s density composition**

`src/worldgen/mod.rs::fill_chunk` contains the actual density assembly: `evaluator.evaluate(wx, wy, wz)` → `slide(...)` → subtract `cave_contribution` (max-composed from cave_sdf + entrance_sdf + wormhole + cheese + spaghetti). The breakdown method should:

1. Compute `bias` from `(h_target - wy) / DENSITY_FALLOFF` (or read the actual formula in `heightmap.rs::slide`).
2. Compute `base_3d` from the same `evaluator.evaluate` call but separated from the bias.
3. Compute each carver contribution individually (cave_sdf, cheese_contribution, spaghetti_contribution, pillar_contribution).
4. Recompose to `final_density`.
5. Use `final_density > 0` + the surface-rules tree to resolve `block` (call `self.surface_system.surface_block(&ctx)` for solids, fall to Water/Lava/Air for non-solids based on aquifer cell).

If the evaluator currently composes too tightly to expose `bias` separately, add a small helper on the evaluator or pass an "include bias / not" flag. Don't redo the worldgen math — call the same functions, just observe the intermediate values.

- [ ] **Step 2: Add the method**

```rust
pub fn evaluate_density_breakdown(
    &self,
    wx: i32, wy: i32, wz: i32,
) -> probe::DensityBreakdown {
    // Reuse the same calls fill_chunk makes. Sketch:
    let cfg_arc = self.config_snapshot();
    let cfg = &*cfg_arc;
    let evaluator = self.density.evaluator(self.seed, &cfg.density);
    let col = self.column_data(wx, wz);
    let h_target = col.height;

    let bias = (h_target as f32 - wy as f32) / cfg.density.density_falloff();
    let raw = evaluator.evaluate(wx, wy, wz);
    let base_3d = raw - bias;
    let post_slide = crate::worldgen::heightmap::slide(raw, wy, &cfg.density);

    let cave_systems = self.cave_systems_for_column(wx, wz);
    let cave_sdf = if !cave_systems.is_empty() {
        crate::worldgen::caves::cave_sdf(wx, wy, wz, &cave_systems)
            .max(crate::worldgen::caves::entrance_sdf(wx, wy, wz, &cave_systems))
    } else { 0.0 };
    let cheese = crate::worldgen::caves::cheese_contribution(
        wx, wy, wz, &self.noise_carvers, &cfg.cave,
    );
    let spaghetti = crate::worldgen::caves::spaghetti_contribution(
        wx, wy, wz, &self.noise_carvers, &cfg.cave,
    );
    let pillar = crate::worldgen::caves::pillar_contribution(
        wx, wy, wz, &self.noise_carvers, &cfg.cave,
    );
    let final_density = post_slide - cave_sdf.max(cheese).max(spaghetti) + pillar;

    // Block resolution — for PR 2 a minimal version is OK:
    let block = if final_density > 0.0 {
        // Walk surface rules; if none match, default Stone.
        let ctx = crate::worldgen::surface::SurfaceContext { /* … */ };
        self.surface_system.surface_block(&ctx).unwrap_or(crate::voxel::block::Block::Stone)
    } else {
        // Air / Water / Lava based on aquifer cell.
        let cell = self.aquifer.cell_for_column(wx, wz);
        if wy <= cell.y_top {
            match cell.substance {
                crate::worldgen::aquifer::Substance::Water => crate::voxel::block::Block::Water,
                crate::worldgen::aquifer::Substance::Lava  => crate::voxel::block::Block::Lava,
            }
        } else { crate::voxel::block::Block::Air }
    };

    probe::DensityBreakdown {
        wx, wy, wz,
        bias, base_3d, cave_sdf, cheese, spaghetti, pillar,
        final_density, block,
    }
}
```

Function/method names above are approximations — find the actual ones via the existing `fill_chunk` body. The plan's structure is fixed; the call sites adapt to the actual API.

- [ ] **Step 3: Tests** — append to `src/worldgen/probe.rs::tests`:

```rust
#[test]
fn density_breakdown_is_deterministic() {
    let gen = Generator::new(42);
    let a = gen.evaluate_density_breakdown(0, 70, 0);
    let b = gen.evaluate_density_breakdown(0, 70, 0);
    assert_eq!(a.final_density.to_bits(), b.final_density.to_bits());
    assert_eq!(a.block, b.block);
}

#[test]
fn density_above_terrain_is_air_or_water() {
    let gen = Generator::new(42);
    // y=200 is well above any reasonable surface — should be air or
    // water (no aquifer surface that high in default config).
    let b = gen.evaluate_density_breakdown(0, 200, 0);
    assert!(matches!(
        b.block,
        crate::voxel::block::Block::Air | crate::voxel::block::Block::Water
    ));
}
```

- [ ] **Step 4: Run** `cargo test --lib worldgen::probe::tests`. All 6 module tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/worldgen/
git commit -m "worldgen: Generator::evaluate_density_breakdown — per-voxel composition (viz PR 2)"
```

---

### Task 5: viz — `overlays/colormap.rs`

**Files:**
- Create: `src/bin/worldgen_viz/overlays/colormap.rs`
- Create: `src/bin/worldgen_viz/overlays/mod.rs` (declares `pub mod colormap; pub mod stages;` plus the MapView struct that lands in Task 7; Tasks 5-6 can leave MapView as a stub).

Colormaps:

```rust
//! Named gradient palettes for stage overlays. Each maps a normalised
//! f32 in `[0, 1]` to an RGBA byte tuple.

/// Categorical → hue cycle around HSV. For PlateId / BiomeId etc.
pub fn categorical(value_01: f32) -> [u8; 4] {
    let hue = (value_01.fract() * 360.0).abs();
    hsv_to_rgba(hue, 0.65, 0.85)
}

/// Cool blue → green → warm tan. For h_pre / h_target.
pub fn terrain_ramp(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t / 0.5;
        (lerp(40.0, 80.0, s), lerp(80.0, 180.0, s), lerp(140.0, 100.0, s))
    } else {
        let s = (t - 0.5) / 0.5;
        (lerp(80.0, 220.0, s), lerp(180.0, 200.0, s), lerp(100.0, 140.0, s))
    };
    [r as u8, g as u8, b as u8, 255]
}

/// Approximate viridis (perceptually uniform). For temperature / humidity.
pub fn viridis(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    // 5-stop linear gradient: dark purple → blue → teal → green → yellow.
    let stops: [(f32, [f32; 3]); 5] = [
        (0.00, [ 68.0,   1.0,  84.0]),
        (0.25, [ 59.0,  82.0, 139.0]),
        (0.50, [ 33.0, 144.0, 140.0]),
        (0.75, [ 94.0, 201.0,  98.0]),
        (1.00, [253.0, 231.0,  37.0]),
    ];
    let (a, b) = stops.windows(2).find(|w| t >= w[0].0 && t <= w[1].0).unwrap().split_at(1);
    let (lo, hi) = (&a[0], &b[0]);
    let s = (t - lo.0) / (hi.0 - lo.0);
    let r = [
        lerp(lo.1[0], hi.1[0], s),
        lerp(lo.1[1], hi.1[1], s),
        lerp(lo.1[2], hi.1[2], s),
    ];
    [r[0] as u8, r[1] as u8, r[2] as u8, 255]
}

/// Divergent: red below 0.5, blue above. For continentalness etc.
pub fn divergent(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        let s = (0.5 - t) * 2.0;
        [(220.0 * s + 35.0) as u8, ((1.0 - s) * 200.0) as u8, ((1.0 - s) * 200.0) as u8, 255]
    } else {
        let s = (t - 0.5) * 2.0;
        [((1.0 - s) * 200.0) as u8, ((1.0 - s) * 200.0) as u8, (220.0 * s + 35.0) as u8, 255]
    }
}

/// Hot (intensity) — black → red → yellow → white. For valley_carve depth.
pub fn hot(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.33 {
        let s = t / 0.33;
        (255.0 * s, 0.0, 0.0)
    } else if t < 0.66 {
        let s = (t - 0.33) / 0.33;
        (255.0, 255.0 * s, 0.0)
    } else {
        let s = (t - 0.66) / 0.34;
        (255.0, 255.0, 255.0 * s)
    };
    [r as u8, g as u8, b as u8, 255]
}

/// Binary (0 → A, 1 → B). For AquiferSubstance.
pub fn binary(t: f32, a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
    if t < 0.5 { a } else { b }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

fn hsv_to_rgba(h: f32, s: f32, v: f32) -> [u8; 4] {
    let c = v * s;
    let h_p = h / 60.0;
    let x = c * (1.0 - (h_p.rem_euclid(2.0) - 1.0).abs());
    let (r, g, b) = match h_p as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
        255,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_endpoints_match_stops() {
        assert_eq!(viridis(0.0), [68, 1, 84, 255]);
        assert_eq!(viridis(1.0), [253, 231, 37, 255]);
    }

    #[test]
    fn divergent_midpoint_is_neutral() {
        let mid = divergent(0.5);
        // Both red and blue components should be near zero at the midpoint.
        assert!(mid[0] < 50 && mid[2] < 50);
    }

    #[test]
    fn terrain_ramp_low_is_water_blue_ish() {
        let low = terrain_ramp(0.05);
        assert!(low[2] > low[0]); // more blue than red at the low end
    }
}
```

Then `src/bin/worldgen_viz/overlays/mod.rs`:

```rust
//! 2D pan/zoom map of pipeline stages. Click-to-probe.

pub mod colormap;
pub mod stages;

// MapView lands in Task 7.
```

- [ ] **Step 1: Write both files.**
- [ ] **Step 2: Register `mod overlays;` in `main.rs`** at the same depth as `mod world;` etc.
- [ ] **Step 3: Run** `cargo test --bin worldgen_viz overlays::colormap`. All 3 tests pass.
- [ ] **Step 4: Commit**

```bash
git add src/bin/worldgen_viz/overlays/ src/bin/worldgen_viz/main.rs
git commit -m "viz: colormap palettes + overlays module skeleton (PR 2)"
```

---

### Task 6: viz — `overlays/stages.rs`

**Files:**
- Create: `src/bin/worldgen_viz/overlays/stages.rs`

Pure mapping layer: Stage → (sampler, normalisation range, colormap). The sampler returns f32 from the Generator; normalisation maps the f32 into `[0, 1]` for the colormap.

- [ ] **Step 1: Write the file**

```rust
//! Per-Stage sampler + colormap dispatch. Pure layer over
//! `Generator::sample_stage` — no rendering happens here; `MapView`
//! consumes this module's `render_pixel` to fill its texture.

use crate::overlays::colormap;
use oxium::worldgen::probe::Stage;
use oxium::worldgen::Generator;

/// Sensible value range per stage for normalisation into `[0, 1]`.
/// Returning `None` means the stage is categorical and the colormap
/// keys on the raw value rather than normalising.
pub fn range(stage: Stage) -> Option<(f32, f32)> {
    match stage {
        Stage::Continentalness    => Some((-1.0, 1.0)),
        Stage::Temperature        => Some((-1.0, 1.0)),
        Stage::Humidity           => Some((-1.0, 1.0)),
        Stage::Desertness         => Some((-1.0, 1.0)),
        Stage::Weirdness          => Some((-1.0, 1.0)),
        Stage::HPre               => Some((40.0, 160.0)),
        Stage::ValleyCarve        => Some((0.0, 16.0)),
        Stage::HTarget            => Some((40.0, 160.0)),
        Stage::FlowAccum          => Some((0.0, 4096.0)),
        Stage::AquiferY           => Some((-64.0, 96.0)),
        Stage::PlateId | Stage::BiomeId | Stage::AquiferSubstance => None,
    }
}

/// Color one pixel given the raw sampler output for that stage.
pub fn pixel(stage: Stage, raw: f32) -> [u8; 4] {
    if let Some((lo, hi)) = range(stage) {
        let t = ((raw - lo) / (hi - lo)).clamp(0.0, 1.0);
        match stage {
            Stage::Continentalness => colormap::divergent(t),
            Stage::Temperature
            | Stage::Humidity
            | Stage::Desertness
            | Stage::Weirdness     => colormap::viridis(t),
            Stage::HPre | Stage::HTarget => colormap::terrain_ramp(t),
            Stage::ValleyCarve     => colormap::hot(t),
            Stage::FlowAccum       => {
                // log-scale flow accumulation before colormap
                let t = (raw.max(1.0).ln() / 4096_f32.ln()).clamp(0.0, 1.0);
                colormap::viridis(t)
            }
            Stage::AquiferY        => colormap::divergent(t),
            _ => [0, 0, 0, 255],
        }
    } else {
        match stage {
            Stage::PlateId | Stage::BiomeId => colormap::categorical(raw),
            Stage::AquiferSubstance => colormap::binary(
                raw, [38, 99, 200, 255], [220, 110, 30, 255],
            ),
            _ => [0, 0, 0, 255],
        }
    }
}

/// Render one pixel by sampling the generator and colouring.
pub fn render_pixel(generator: &Generator, stage: Stage, wx: i32, wz: i32) -> [u8; 4] {
    let raw = generator.sample_stage(stage, wx, wz);
    pixel(stage, raw)
}
```

- [ ] **Step 2: Tests** — append a `#[cfg(test)] mod tests` block:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use oxium::worldgen::Generator;

    #[test]
    fn render_pixel_is_deterministic() {
        let g = Generator::new(42);
        let a = render_pixel(&g, Stage::HTarget, 0, 0);
        let b = render_pixel(&g, Stage::HTarget, 0, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn categorical_stages_have_no_range() {
        assert!(range(Stage::PlateId).is_none());
        assert!(range(Stage::BiomeId).is_none());
        assert!(range(Stage::AquiferSubstance).is_none());
    }

    #[test]
    fn scalar_stages_have_range() {
        assert!(range(Stage::HTarget).is_some());
        assert!(range(Stage::Continentalness).is_some());
    }
}
```

- [ ] **Step 3: Run** `cargo test --bin worldgen_viz overlays::stages::tests`. 3 pass.

- [ ] **Step 4: Commit**

```bash
git add src/bin/worldgen_viz/overlays/stages.rs
git commit -m "viz: per-stage samplers + colormap dispatch (PR 2)"
```

---

### Task 7: viz — `MapView` (pan/zoom 2D map)

**Files:**
- Modify: `src/bin/worldgen_viz/overlays/mod.rs` (add the MapView struct + impl).

`MapView` holds: center (in world coords), `blocks_per_pixel`, current `Stage`, the most recent egui `TextureHandle` + size. Public API:

- `MapView::new(...)` — defaults: 256×256 px, 4 blocks/px, center at origin, Stage::HTarget.
- `MapView::regenerate(generator, egui_ctx)` — rebuilds the texture by sampling `render_pixel` over the 256×256 grid.
- `MapView::show(ui, generator) -> Option<(i32, i32)>` — egui widget; returns `Some((wx, wz))` if the user clicked.
- `MapView::pan(dx_px, dy_px)`, `MapView::zoom_in()`, `MapView::zoom_out()`.

- [ ] **Step 1: Write**

```rust
//! 2D pan/zoom map of pipeline stages. Click-to-probe.

pub mod colormap;
pub mod stages;

use egui::{ColorImage, TextureHandle, TextureOptions, Ui};
use oxium::worldgen::probe::Stage;
use oxium::worldgen::Generator;

pub const MAP_SIZE_PX: usize = 256;

pub struct MapView {
    pub stage: Stage,
    pub center_wx: f32,
    pub center_wz: f32,
    pub blocks_per_pixel: f32,
    texture: Option<TextureHandle>,
    last_render_key: Option<RenderKey>,
}

#[derive(PartialEq)]
struct RenderKey {
    stage: Stage,
    center_wx: i32,
    center_wz: i32,
    blocks_per_pixel: i32,
    config_revision: u64,
}

impl MapView {
    pub fn new() -> Self {
        Self {
            stage: Stage::HTarget,
            center_wx: 0.0,
            center_wz: 0.0,
            blocks_per_pixel: 4.0,
            texture: None,
            last_render_key: None,
        }
    }

    pub fn set_stage(&mut self, stage: Stage) {
        if self.stage != stage {
            self.stage = stage;
            self.last_render_key = None;
        }
    }

    /// Translate `(px, py)` of the map into world `(wx, wz)`.
    pub fn pixel_to_world(&self, px: f32, py: f32) -> (i32, i32) {
        let half = MAP_SIZE_PX as f32 * 0.5;
        let wx = self.center_wx + (px - half) * self.blocks_per_pixel;
        let wz = self.center_wz + (py - half) * self.blocks_per_pixel;
        (wx.round() as i32, wz.round() as i32)
    }

    /// Translate world `(wx, wz)` into map pixel coords, if visible.
    pub fn world_to_pixel(&self, wx: i32, wz: i32) -> Option<(f32, f32)> {
        let half = MAP_SIZE_PX as f32 * 0.5;
        let px = half + (wx as f32 - self.center_wx) / self.blocks_per_pixel;
        let py = half + (wz as f32 - self.center_wz) / self.blocks_per_pixel;
        if px < 0.0 || px >= MAP_SIZE_PX as f32 || py < 0.0 || py >= MAP_SIZE_PX as f32 {
            None
        } else {
            Some((px, py))
        }
    }

    pub fn pan(&mut self, dx_px: f32, dy_px: f32) {
        self.center_wx -= dx_px * self.blocks_per_pixel;
        self.center_wz -= dy_px * self.blocks_per_pixel;
        self.last_render_key = None;
    }

    pub fn zoom(&mut self, factor: f32) {
        self.blocks_per_pixel = (self.blocks_per_pixel * factor).clamp(0.5, 64.0);
        self.last_render_key = None;
    }

    fn regenerate(&mut self, generator: &Generator, ctx: &egui::Context, revision: u64) {
        let key = RenderKey {
            stage: self.stage,
            center_wx: self.center_wx as i32,
            center_wz: self.center_wz as i32,
            blocks_per_pixel: self.blocks_per_pixel as i32,
            config_revision: revision,
        };
        if self.last_render_key.as_ref() == Some(&key) && self.texture.is_some() {
            return;
        }
        let mut pixels = vec![egui::Color32::TRANSPARENT; MAP_SIZE_PX * MAP_SIZE_PX];
        for py in 0..MAP_SIZE_PX {
            for px in 0..MAP_SIZE_PX {
                let (wx, wz) = self.pixel_to_world(px as f32, py as f32);
                let rgba = stages::render_pixel(generator, self.stage, wx, wz);
                pixels[py * MAP_SIZE_PX + px] = egui::Color32::from_rgba_premultiplied(
                    rgba[0], rgba[1], rgba[2], rgba[3],
                );
            }
        }
        let img = ColorImage { size: [MAP_SIZE_PX, MAP_SIZE_PX], pixels };
        let tex = ctx.load_texture("viz_overlay_map", img, TextureOptions::NEAREST);
        self.texture = Some(tex);
        self.last_render_key = Some(key);
    }

    /// Render the map widget. Returns `Some((wx, wz))` if the user
    /// clicked. Pan with drag (Middle button); zoom with scroll.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        generator: &Generator,
        revision: u64,
    ) -> Option<(i32, i32)> {
        self.regenerate(generator, ui.ctx(), revision);

        // Stage dropdown
        egui::ComboBox::from_id_source("stage_combo")
            .selected_text(self.stage.label())
            .show_ui(ui, |ui| {
                for &s in Stage::ALL {
                    if ui.selectable_label(self.stage == s, s.label()).clicked() {
                        self.set_stage(s);
                    }
                }
            });

        let tex = self.texture.clone();
        let mut clicked = None;
        if let Some(tex) = tex {
            let size = egui::vec2(MAP_SIZE_PX as f32, MAP_SIZE_PX as f32);
            let resp = ui.add(egui::Image::new((tex.id(), size)).sense(egui::Sense::click_and_drag()));
            if resp.clicked() {
                if let Some(pos) = resp.interact_pointer_pos() {
                    let local = pos - resp.rect.left_top();
                    clicked = Some(self.pixel_to_world(local.x, local.y));
                }
            }
            if resp.dragged_by(egui::PointerButton::Middle) {
                let d = resp.drag_delta();
                self.pan(d.x, d.y);
            }
            if resp.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.1 {
                    let factor = if scroll > 0.0 { 0.9 } else { 1.1 };
                    self.zoom(factor);
                }
            }
        }
        ui.label(format!(
            "center ({:.0}, {:.0})  bpp {:.1}",
            self.center_wx, self.center_wz, self.blocks_per_pixel,
        ));
        clicked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_to_world_centers_at_center_wx() {
        let m = MapView { center_wx: 100.0, center_wz: 200.0, blocks_per_pixel: 4.0, ..MapView::new() };
        let (wx, wz) = m.pixel_to_world((MAP_SIZE_PX as f32) / 2.0, (MAP_SIZE_PX as f32) / 2.0);
        assert_eq!(wx, 100);
        assert_eq!(wz, 200);
    }

    #[test]
    fn world_to_pixel_inverse_of_pixel_to_world() {
        let m = MapView { center_wx: 32.0, center_wz: -64.0, blocks_per_pixel: 4.0, ..MapView::new() };
        let (wx, wz) = m.pixel_to_world(64.0, 64.0);
        let (px, py) = m.world_to_pixel(wx, wz).unwrap();
        assert!((px - 64.0).abs() < 4.0);
        assert!((py - 64.0).abs() < 4.0);
    }

    #[test]
    fn pan_shifts_center_inversely_to_drag() {
        let mut m = MapView::new();
        m.center_wx = 0.0;
        m.pan(10.0, 0.0); // dragging right by 10 px → world center moves left
        assert!(m.center_wx < 0.0);
    }
}
```

`MapView { ..MapView::new() }` syntax: that's invalid Rust because `MapView` has private fields. Use `MapView::new()` then mutate fields in tests. Adjust the tests accordingly:

```rust
let mut m = MapView::new();
m.center_wx = 100.0; m.center_wz = 200.0;
let (wx, wz) = m.pixel_to_world(/*half*/ 128.0, 128.0);
assert_eq!(wx, 100); assert_eq!(wz, 200);
```

- [ ] **Step 2: Run** `cargo test --bin worldgen_viz overlays::tests`. 3 pass.

- [ ] **Step 3: Commit**

```bash
git add src/bin/worldgen_viz/overlays/mod.rs
git commit -m "viz: MapView — pan/zoom 2D stage overlay with click → world coords (PR 2)"
```

---

### Task 8: viz — `probe.rs` (pinned column state)

**Files:**
- Create: `src/bin/worldgen_viz/probe.rs`

Tracks the currently-pinned column. When set, holds the last `ColumnProbe` snapshot. Single Y-slider for the density-breakdown section.

- [ ] **Step 1: Write**

```rust
//! Column probe state: pinned (wx, wz), last `ColumnProbe` snapshot,
//! and the Y at which the user is currently inspecting density.

use oxium::worldgen::probe::{ColumnProbe, DensityBreakdown};
use oxium::worldgen::Generator;

pub struct Probe {
    pub pinned: Option<(i32, i32)>,
    pub snapshot: Option<ColumnProbe>,
    pub probe_y: i32,
    pub breakdown: Option<DensityBreakdown>,
}

impl Probe {
    pub fn new() -> Self {
        Self { pinned: None, snapshot: None, probe_y: 70, breakdown: None }
    }

    /// Pin a new column and refresh its snapshot.
    pub fn pin(&mut self, generator: &Generator, wx: i32, wz: i32) {
        let snapshot = generator.probe_column(wx, wz);
        self.probe_y = snapshot.h_target;
        self.snapshot = Some(snapshot.clone());
        self.pinned = Some((wx, wz));
        self.breakdown = Some(generator.evaluate_density_breakdown(wx, self.probe_y, wz));
    }

    pub fn unpin(&mut self) {
        self.pinned = None;
        self.snapshot = None;
        self.breakdown = None;
    }

    /// Refresh the snapshot + breakdown for the currently pinned
    /// column. Used when the config changes.
    pub fn refresh(&mut self, generator: &Generator) {
        if let Some((wx, wz)) = self.pinned {
            let snapshot = generator.probe_column(wx, wz);
            self.breakdown = Some(generator.evaluate_density_breakdown(wx, self.probe_y, wz));
            self.snapshot = Some(snapshot);
        }
    }

    /// Update the Y slider; refreshes the density breakdown only.
    pub fn set_y(&mut self, generator: &Generator, y: i32) {
        if y != self.probe_y {
            self.probe_y = y;
            if let Some((wx, wz)) = self.pinned {
                self.breakdown = Some(generator.evaluate_density_breakdown(wx, y, wz));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxium::worldgen::Generator;

    #[test]
    fn pin_populates_snapshot_and_breakdown() {
        let g = Generator::new(42);
        let mut p = Probe::new();
        p.pin(&g, 50, 50);
        assert_eq!(p.pinned, Some((50, 50)));
        assert!(p.snapshot.is_some());
        assert!(p.breakdown.is_some());
        assert_eq!(p.probe_y, p.snapshot.as_ref().unwrap().h_target);
    }

    #[test]
    fn unpin_clears_state() {
        let g = Generator::new(42);
        let mut p = Probe::new();
        p.pin(&g, 0, 0);
        p.unpin();
        assert!(p.pinned.is_none());
        assert!(p.snapshot.is_none());
        assert!(p.breakdown.is_none());
    }

    #[test]
    fn set_y_refreshes_breakdown_only() {
        let g = Generator::new(42);
        let mut p = Probe::new();
        p.pin(&g, 0, 0);
        let initial_breakdown = p.breakdown.unwrap();
        p.set_y(&g, p.probe_y + 20);
        let new_breakdown = p.breakdown.unwrap();
        // Y should have changed; breakdown's bias should differ.
        assert_ne!(initial_breakdown.bias.to_bits(), new_breakdown.bias.to_bits());
    }
}
```

- [ ] **Step 2: Register `mod probe;` in `main.rs`.**

- [ ] **Step 3: Run** `cargo test --bin worldgen_viz probe::tests`. 3 pass.

- [ ] **Step 4: Commit**

```bash
git add src/bin/worldgen_viz/probe.rs src/bin/worldgen_viz/main.rs
git commit -m "viz: Probe state — pin, unpin, refresh, set_y (PR 2)"
```

---

### Task 9: viz — `widgets/probe_table.rs`

**Files:**
- Create: `src/bin/worldgen_viz/widgets/probe_table.rs`
- Modify: `src/bin/worldgen_viz/widgets/mod.rs` (`pub mod probe_table;`)

Pure rendering: given a `&ColumnProbe` and optionally a `&DensityBreakdown`, draw the labelled field table from the spec (Geometry / Climate / Hydrology / Density / Caves).

- [ ] **Step 1: Write**

```rust
//! Field-value table for the probe panel.

use egui::Ui;
use oxium::worldgen::aquifer::Substance;
use oxium::worldgen::probe::{ColumnProbe, DensityBreakdown};

pub fn show(ui: &mut Ui, snapshot: &ColumnProbe, breakdown: Option<&DensityBreakdown>, probe_y: i32) {
    ui.heading(format!("Column ({}, {})", snapshot.wx, snapshot.wz));

    ui.collapsing("Geometry", |ui| {
        kv(ui, "plate primary", format!("{:?}", snapshot.plate.primary));
        kv(ui, "plate secondary", format!("{:?}", snapshot.plate.secondary));
        kv(ui, "boundary_t", fmt(snapshot.plate.boundary_t));
        kv(ui, "continentalness", fmt(snapshot.continentalness));
        kv(ui, "h_pre", fmt(snapshot.h_pre));
        kv(ui, "valley_carve", fmt(snapshot.valley_carve));
        kv(ui, "h_target", snapshot.h_target.to_string());
        kv(ui, "is_cliff", snapshot.is_cliff.to_string());
        kv(ui, "slope", fmt(snapshot.slope));
    });

    ui.collapsing("Climate", |ui| {
        kv(ui, "temperature", fmt(snapshot.temperature));
        kv(ui, "humidity", fmt(snapshot.humidity));
        kv(ui, "desertness", fmt(snapshot.desertness));
        kv(ui, "weirdness", fmt(snapshot.weirdness));
        kv(ui, "biome", format!("{:?}", snapshot.biome));
    });

    ui.collapsing("Hydrology", |ui| {
        kv(ui, "flow_accum", snapshot.flow_accum.to_string());
        kv(ui, "lake_rim", match snapshot.lake_rim {
            Some(y) => y.to_string(),
            None => "—".to_string(),
        });
    });

    ui.collapsing("Aquifer", |ui| {
        kv(ui, "y_top", snapshot.aquifer_y_top.to_string());
        kv(ui, "substance", match snapshot.aquifer_substance {
            Substance::Water => "Water".to_string(),
            Substance::Lava => "Lava".to_string(),
        });
    });

    ui.collapsing("Caves", |ui| {
        kv(ui, "intersecting systems", snapshot.cave_systems_count.to_string());
    });

    if let Some(b) = breakdown {
        ui.collapsing(format!("Density @ y={probe_y}"), |ui| {
            kv(ui, "bias", fmt(b.bias));
            kv(ui, "base_3d", fmt(b.base_3d));
            kv(ui, "cave_sdf", fmt(b.cave_sdf));
            kv(ui, "cheese", fmt(b.cheese));
            kv(ui, "spaghetti", fmt(b.spaghetti));
            kv(ui, "pillar", fmt(b.pillar));
            kv(ui, "final_density", fmt(b.final_density));
            kv(ui, "block", format!("{:?}", b.block));
        });
    }
}

fn kv(ui: &mut Ui, label: &str, value: String) {
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));
        ui.label(value);
    });
}

fn fmt(v: f32) -> String {
    if v.abs() < 1e-3 || v.abs() > 1e4 {
        format!("{:.3e}", v)
    } else {
        format!("{:.3}", v)
    }
}
```

No tests for this module — it's pure egui drawing.

- [ ] **Step 2: Add to widgets/mod.rs**: `pub mod probe_table;`

- [ ] **Step 3: Build check.** `cargo check --bin worldgen_viz` clean.

- [ ] **Step 4: Commit**

```bash
git add src/bin/worldgen_viz/widgets/probe_table.rs src/bin/worldgen_viz/widgets/mod.rs
git commit -m "viz: probe_table widget — labelled field display (PR 2)"
```

---

### Task 10: viz — wire map + probe into dashboard

**Files:**
- Modify: `src/bin/worldgen_viz/session.rs` (add `pub probe: Probe` field; init in `Session::new`).
- Modify: `src/bin/worldgen_viz/layout.rs` (replace right-panel placeholder with the actual probe + map sections).
- Modify: `src/bin/worldgen_viz/main.rs` (refresh the probe when the config changes; expose a `revision()` counter on `Invalidator` for the MapView's cache key).

- [ ] **Step 1: Extend Invalidator with a public revision counter**

`src/bin/worldgen_viz/world/invalidate.rs` — add:

```rust
/// Current revision counter. Bumped on `bump()`; consumed by
/// `take_pending`. Used as a cache key by overlays that want to
/// re-render exactly once per config change.
pub fn revision(&self) -> u64 {
    self.current
}
```

- [ ] **Step 2: Extend Session**

In `src/bin/worldgen_viz/session.rs`, add `pub probe: Probe` to the struct + init in `new()` + import.

- [ ] **Step 3: Extend Session with a `MapView`**

```rust
pub struct Session {
    // … existing fields
    pub probe: crate::probe::Probe,
    pub map: crate::overlays::MapView,
}
```

Init both in `Session::new`.

- [ ] **Step 4: Rework `layout.rs`'s right panel**

Replace the existing right-panel block with:

```rust
egui::SidePanel::right("probe_panel")
    .resizable(true)
    .default_width(360.0)
    .show(ctx, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            // Map section
            ui.heading("Overlay map");
            let revision = app.session.invalidator.revision();
            let clicked = app.session.map.show(ui, &app.session.generator, revision);
            if let Some((wx, wz)) = clicked {
                app.session.probe.pin(&app.session.generator, wx, wz);
            }
            ui.separator();

            // Probe section
            ui.heading("Probe");
            if let Some(snap) = app.session.probe.snapshot.clone() {
                ui.horizontal(|ui| {
                    if ui.button("Unpin").clicked() {
                        app.session.probe.unpin();
                    }
                });
                let probe_y_before = app.session.probe.probe_y;
                let mut y_local = probe_y_before;
                ui.add(egui::Slider::new(&mut y_local, -64..=256).text("probe y"));
                if y_local != probe_y_before {
                    app.session.probe.set_y(&app.session.generator, y_local);
                }
                crate::widgets::probe_table::show(
                    ui,
                    &snap,
                    app.session.probe.breakdown.as_ref(),
                    app.session.probe.probe_y,
                );
            } else {
                ui.label("Click the map to pin a column.");
            }
        });
    });
```

- [ ] **Step 5: Refresh probe on config change**

In `main.rs`, where the invalidator wipe happens (`take_pending()`), also call `self.state.session.probe.refresh(&self.state.session.generator);` so the pinned-column display tracks edits.

- [ ] **Step 6: Verify**

```
cargo build --bin worldgen_viz
cargo test --bin worldgen_viz
cargo run --bin worldgen_viz -- --check
```

All must succeed.

- [ ] **Step 7: Commit**

```bash
git add src/bin/worldgen_viz/
git commit -m "viz: wire MapView + Probe into dashboard right panel (PR 2)"
```

---

### Task 11: viz — final cleanup + spec table tick

- [ ] **Step 1: Smoke run**

`cargo run --bin worldgen_viz -- --seed 42` — verify the right panel renders a stage overlay (default Stage::HTarget), clicking the map pins a column, the field table populates with sensible numbers, dragging the Y slider updates the breakdown.

If this fails to run, document the failure and BLOCK with diagnostic info.

- [ ] **Step 2: Tick the spec table**

In `docs/superpowers/specs/2026-05-19-worldgen-viz-redesign-design.md`, locate the row beginning `| **2** | viz: pipeline overlays + column probe |` and prepend `✅` after `**2**`.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-19-worldgen-viz-redesign-design.md
git commit -m "viz: PR 2 (overlays + probe) complete — mark in spec"
```

---

## Self-Review

**Spec coverage:**
- 2D pan/zoom map with all stages → Tasks 5, 6, 7 ✓
- Click-to-probe pinning a column → Task 7 + Task 10 ✓
- Probe panel with full field list → Task 9 + Task 10 ✓
- Worldgen API additions (`probe_column`, `sample_stage`, `evaluate_density_breakdown`, DensityBreakdown, Stage, ColumnProbe) → Tasks 1-4 ✓
- Aquifer / noise carver visibility through stages + breakdown ✓

**Placeholder scan:** Code in Tasks 2 and 4 references accessor methods that may not exist by those exact names (`heightmap.h_pre`, `climate.temperature_at`, `region.flow_accum_at`, `aquifer.cell_for_column`, `cave_systems_for_column`, `slope_at`, `chunk_regions_for`). This is intentional — the actual call sites are visible in the existing `column_data_with` / `fill_chunk` paths and the implementer adapts. Where a needed accessor is private, the implementer adds a thin public wrapper. Each task description calls this out explicitly.

**Type consistency:**
- `ColumnProbe`, `Stage`, `DensityBreakdown` defined in Task 1, used in Tasks 2-9 ✓
- `Probe` defined in Task 8, used in Task 10 ✓
- `MapView` defined in Task 7, used in Task 10 ✓
- `Invalidator::revision()` added in Task 10 step 1; used by `MapView::regenerate` from Task 7 (the cache key field's value comes from Invalidator) ✓

**Scope check:** 11 tasks. Each is one file's worth of work + tests. Each ends in a commit. PR 2 produces working, testable software on its own.

**Ambiguity check:** the lone real ambiguity is the exact set of new public accessors the library task adds. Documented as a deliberate "use the existing call sites" pattern.
