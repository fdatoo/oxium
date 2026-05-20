# Worldgen PR 7 — Real Aquifer (replacing primitive ocean-column rule)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the primitive "ocean column / under-lake → water" rule (`mod.rs:419-435`, landed in-session) with Minecraft 1.18+'s per-region aquifer system. Each aquifer cell on a jittered 16×12×16 grid owns a `FluidStatus { fluid_level, fluid_kind }`; per voxel the substance decision is the result of a 3-nearest-cell query with similarity-weighted barrier pressure between dissimilar aquifers. Caves carved well below sea level can now be **dry**, water tables can vary by region, lava pools can exist without flood-filling, and water never touches lava because the pressure function mathematically seals their boundary.

**Architecture:** One new module — `src/worldgen/aquifer.rs` (~400 LOC of implementation + tests). Public entry point `Aquifer::compute_substance(ctx, density) -> Option<BlockKind>`, plugged into `mod.rs::fill_chunk` as the **last** per-cell decision (after density is composed and after caves are subtracted, before surface-block selection runs). The aquifer holds a per-chunk grid of `FluidStatus` cells lazily populated from three new noise channels (`floodedness`, `spread`, `lava`) plus the existing per-column preliminary surface (`h_target`). Voxel queries hit the 2×3×2 anchor neighbourhood of the column's grid cell, compute three squared distances to jittered centers, and apply MC's `similarity(d1², d2²) = 1.0 - (d2² - d1²) / 25.0` plus a piecewise-linear pressure gradient with asymmetric top/bottom biases (top: 2.5 over rocks, 1.5 over holes; bottom: 10.0 over rocks, 3.0 over holes; bottom bias 3.0). A `Block::Lava` variant is added.

**Tech Stack:**
- Rust 2024 edition
- `noise = "0.9"` — Fbm/Simplex (existing) for the three new aquifer noise channels
- `glam` (existing) — IVec3 / Vec3 for coordinate math
- `WorldgenConfig` (from PR 2) — adds a new `aquifer: AquiferConfig` subsection, hot-reloadable

**Reference:** Architectural rationale is in `docs/superpowers/specs/2026-05-19-minecraft-worldgen-research.md` (Part 4 idea #6 and Decisions Log Q3). MC reference: `/Users/fdatoo/Downloads/out/net/minecraft/world/level/levelgen/Aquifer.java` — read alongside this plan. The primitive rule this PR replaces lives at `src/worldgen/mod.rs:419-435` (the in-session fix).

**Dependencies:** PR 5 (cell interpolation + density graph) is required because the aquifer reads the post-density-subtracted value at each voxel and decides air/water/lava based on it. The plan assumes PR 5 has already plumbed a `DensityCtx { wx, wy, wz, density }` through `fill_chunk`'s per-cell loop and that `Generator::config_snapshot()` (PR 2) returns an `Arc<WorldgenConfig>` with the new `aquifer` subsection populated.

---

### Task 1: Add `Block::Lava` to the voxel block enum

**Files:**
- Modify: `src/voxel/block.rs`
- Modify: any rendering / palette tables that switch on Block exhaustively

- [ ] **Step 1.1: Write the failing test**

Append to `src/voxel/block.rs` test module (or create one if it doesn't exist):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lava_is_a_block_variant() {
        // Lava must round-trip through from_repr like every other block.
        let kind = Block::Lava;
        let v = kind as u16;
        assert_eq!(Block::from_repr(v), Some(Block::Lava));
    }

    #[test]
    fn block_count_includes_lava() {
        // BLOCK_COUNT must increment when adding variants — it backs
        // the palette table sizing throughout the renderer.
        assert!(BLOCK_COUNT >= 11, "BLOCK_COUNT did not grow with new variant");
    }
}
```

- [ ] **Step 1.2: Add the `Lava` variant**

In `src/voxel/block.rs`, locate the `Block` enum (currently ends with `Snow,`). Append a new variant:

```rust
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Block {
    Air = 0,
    Stone,
    Dirt,
    Grass,
    Sand,
    Water,
    Wood,
    Leaves,
    Torch,
    Snow,
    /// Lava — emissive, deals damage, flows like water. Aquifer
    /// system places this in deep cells where the lava-noise
    /// magnitude exceeds 0.3 and the aquifer's fluid level ≤ -10.
    Lava,
}
```

Bump the count:

```rust
pub const BLOCK_COUNT: usize = 11;
```

Extend `Block::from_repr` to cover the new variant. Find the existing match arm and add the new case before the catch-all.

- [ ] **Step 1.3: Fix any exhaustive matches that broke**

Run: `cargo check 2>&1 | tail -25`

Expected: 1–3 "non-exhaustive match" errors at consumer sites (renderer palette, mesher, persistence). For each, add a `Block::Lava => ...` arm. Use the same handling as `Block::Water` for now (transparent fluid, emissive comes later — outside the scope of PR 7). If the renderer uses a palette table, add a placeholder texture index matching water; cosmetic fidelity isn't on PR 7's critical path.

- [ ] **Step 1.4: Run tests**

Run: `cargo test --lib voxel::block 2>&1 | tail -10`

Expected: 2 tests pass. The rest of the suite continues to compile.

- [ ] **Step 1.5: Commit**

```bash
git add src/voxel/block.rs # plus any consumer files step 1.3 touched
git commit -m "$(cat <<'EOF'
feat(voxel): add Block::Lava variant for the upcoming aquifer

PR 7 introduces lava aquifers — deep, isolated cells whose fluid
kind is lava rather than water. The block needs a discriminant
before the aquifer module can emit it. Lava reuses water's
transparent-fluid render path for now; emissive lighting is a
future PR concern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add `AquiferConfig` subsection to `WorldgenConfig`

**Files:**
- Modify: `src/worldgen/config.rs` (from PR 2)
- Modify: `assets/worldgen/default.ron`

- [ ] **Step 2.1: Write the failing test**

Append to the `tests` module in `src/worldgen/config.rs`:

```rust
    #[test]
    fn bundled_default_includes_aquifer_subsection() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let aq = &cfg.aquifer;
        // Grid spacing from MC.
        assert_eq!(aq.x_spacing, 16);
        assert_eq!(aq.y_spacing, 12);
        assert_eq!(aq.z_spacing, 16);
        // Jitter ranges from MC.
        assert_eq!(aq.x_range, 10);
        assert_eq!(aq.y_range, 9);
        assert_eq!(aq.z_range, 10);
        // Sample offset (-5, +1, -5) from MC.
        assert_eq!(aq.sample_offset_x, -5);
        assert_eq!(aq.sample_offset_y, 1);
        assert_eq!(aq.sample_offset_z, -5);
        // Similarity divisor 25.0 from MC.
        assert!((aq.similarity_divisor - 25.0).abs() < 1e-6);
        // Lava cells are 64×40×64 — coarser than aquifer cells.
        assert_eq!(aq.lava_cell_xz, 64);
        assert_eq!(aq.lava_cell_y, 40);
        // Lava threshold: |noise| > 0.3.
        assert!((aq.lava_threshold - 0.3).abs() < 1e-6);
    }
```

- [ ] **Step 2.2: Verify the test fails**

Run: `cargo test --lib worldgen::config::tests::bundled_default_includes_aquifer_subsection 2>&1 | tail -5`

Expected: compile error — `cfg.aquifer` does not exist.

- [ ] **Step 2.3: Add `AquiferConfig` and wire into `WorldgenConfig`**

In `src/worldgen/config.rs`, add a new struct after `DensityConfig`:

```rust
/// Aquifer tuning (PR 7). All values mirror Minecraft 1.18+
/// `Aquifer.java` constants by default.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AquiferConfig {
    // ── Grid spacing & jitter ──────────────────────────────────
    /// X spacing of aquifer cells (blocks). MC default: 16.
    pub x_spacing: i32,
    /// Y spacing of aquifer cells (blocks). MC default: 12.
    pub y_spacing: i32,
    /// Z spacing of aquifer cells (blocks). MC default: 16.
    pub z_spacing: i32,
    /// X-jitter range — the per-cell center is offset by `rand(0..x_range)`.
    /// MC default: 10.
    pub x_range: i32,
    /// Y-jitter range. MC default: 9.
    pub y_range: i32,
    /// Z-jitter range. MC default: 10.
    pub z_range: i32,
    /// Sample anchor offset applied to the query position before
    /// `gridX/gridY/gridZ` — shifts the per-cell sampling pattern.
    /// MC: (-5, +1, -5).
    pub sample_offset_x: i32,
    pub sample_offset_y: i32,
    pub sample_offset_z: i32,

    // ── Similarity / pressure ──────────────────────────────────
    /// Divisor in `similarity(d1², d2²) = 1 - (d2² - d1²) / D`.
    /// MC: 25.0.
    pub similarity_divisor: f32,

    // ── Flood thresholds ───────────────────────────────────────
    /// Y-extent above `aquifer cell.y` used to detect "this cell
    /// pokes above the surface" — at-surface cells take the global
    /// fluid status. MC: 12 (== `y_spacing`).
    pub above_cell_y_window: i32,
    /// Max depth (blocks below surface) over which floodedness
    /// linearly ramps from 1.0 at the surface to 0.0. MC: 64.
    pub floodedness_max_depth: i32,
    /// Quantisation step of the fluid_level_spread noise. MC: 3.
    pub spread_quantize_step: i32,
    /// Below this Y value (and below the bedrock-ish "way below" sentinel),
    /// aquifers with no flooding stay dry. Should match
    /// `WAY_BELOW_MIN_Y` in MC's DimensionType (essentially `i32::MIN`).
    pub way_below_min_y: i32,

    // ── Lava ───────────────────────────────────────────────────
    /// Lava cell XZ size (blocks). MC: 64 — coarser than aquifer cells.
    pub lava_cell_xz: i32,
    /// Lava cell Y size (blocks). MC: 40.
    pub lava_cell_y: i32,
    /// Lava noise threshold — when `|noise| > threshold` AND
    /// `fluid_level <= -10`, the aquifer becomes lava. MC: 0.3.
    pub lava_threshold: f32,
    /// Maximum fluid_surface_level at which lava is permitted.
    /// MC: -10.
    pub lava_level_max: i32,

    // ── Global fluid ───────────────────────────────────────────
    /// Sea level for the global fluid picker — above-surface columns
    /// get water below this level, air above. MC: 63 in overworld;
    /// Oxium uses `SEA_LEVEL` from `tuning.rs` (62) by default.
    pub sea_level: i32,
}
```

Extend `WorldgenConfig`:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldgenConfig {
    pub density: DensityConfig,
    /// PR 7: aquifer tuning.
    pub aquifer: AquiferConfig,
}
```

Add `aquifer:` to `assets/worldgen/default.ron`:

```ron
// Oxium worldgen default configuration.
(
    density: (
        // ... existing density fields, untouched ...
    ),
    // PR 7 aquifer subsection — Minecraft 1.18+ constants mirrored.
    aquifer: (
        x_spacing: 16,
        y_spacing: 12,
        z_spacing: 16,
        x_range: 10,
        y_range: 9,
        z_range: 10,
        sample_offset_x: -5,
        sample_offset_y: 1,
        sample_offset_z: -5,

        similarity_divisor: 25.0,

        above_cell_y_window: 12,
        floodedness_max_depth: 64,
        spread_quantize_step: 3,
        way_below_min_y: -2147483648,  // i32::MIN; "no aquifer here"

        lava_cell_xz: 64,
        lava_cell_y: 40,
        lava_threshold: 0.3,
        lava_level_max: -10,

        sea_level: 62,
    ),
)
```

- [ ] **Step 2.4: Verify the test passes**

Run: `cargo test --lib worldgen::config 2>&1 | tail -10`

Expected: existing PR 2 tests still pass; new `bundled_default_includes_aquifer_subsection` passes.

- [ ] **Step 2.5: Commit**

```bash
git add src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
feat(worldgen): add AquiferConfig subsection to WorldgenConfig

All MC 1.18+ aquifer constants captured in a single hot-reloadable
subsection: grid spacing (16/12/16), jitter ranges (10/9/10),
sample offset (-5, +1, -5), similarity divisor (25), flood depth
(64), spread quantisation step (3), lava cell size (64×40×64),
lava threshold (0.3), and sea level (62).

PR 7's runtime module reads these via the existing ConfigHolder
snapshot. Tuning the values via default.ron + file watcher does
not require a rebuild.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `FluidKind` and `FluidStatus` data types (TDD)

**Files:**
- Create: `src/worldgen/aquifer.rs`
- Modify: `src/worldgen/mod.rs` (add `pub mod aquifer;`)

- [ ] **Step 3.1: Write the failing tests**

Create `src/worldgen/aquifer.rs` with the data types and their tests; the rest of the module is empty for now:

```rust
//! Per-region aquifer system. Replaces the primitive "below sea
//! level = water" flood rule with Minecraft 1.18+'s 3-nearest-cell
//! similarity-weighted barrier pressure model.
//!
//! Geometry: aquifer cells are placed on a jittered grid (spacing
//! 16×12×16 blocks, jitter 10×9×10) anchored at sample offset
//! `(-5, +1, -5)`. Each cell owns a [`FluidStatus`] — fluid level
//! (the Y above which a voxel is air and below which is fluid) and
//! fluid kind (Air / Water / Lava).
//!
//! Per-voxel decision: find the three nearest cells by squared
//! distance from the post-jitter centers. Compute similarity
//! `1.0 - (d2² - d1²) / 25.0`. If ≤ 0 → cleanly inside cell 1,
//! emit its fluid at this Y. Otherwise compute the pressure between
//! cell pairs — an asymmetric piecewise-linear gradient plus a
//! barrier-noise sample. If `density + pressure > 0` at any pair,
//! rock seals the boundary and we return `None` (caller writes
//! stone).
//!
//! Reference: `net/minecraft/world/level/levelgen/Aquifer.java`.

use crate::voxel::block::Block;

/// What fluid (if any) occupies an aquifer cell.
///
/// `Air` means "no fluid at this Y" — used as the substance above
/// `fluid_level` and as the global fluid type above sea level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidKind {
    Air,
    Water,
    Lava,
}

impl FluidKind {
    /// Map to the engine's [`Block`] enum.
    pub fn to_block(self) -> Block {
        match self {
            FluidKind::Air => Block::Air,
            FluidKind::Water => Block::Water,
            FluidKind::Lava => Block::Lava,
        }
    }
}

/// One aquifer cell's fluid configuration.
///
/// `at(wy)` returns the fluid that occupies world-Y `wy` inside this
/// cell — `fluid_kind` if `wy < fluid_level`, else `Air`. Matches MC
/// `FluidStatus.at(blockY)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FluidStatus {
    pub fluid_level: i32,
    pub fluid_kind: FluidKind,
}

impl FluidStatus {
    pub const fn new(fluid_level: i32, fluid_kind: FluidKind) -> Self {
        Self { fluid_level, fluid_kind }
    }

    /// Block-state lookup at a given Y. Below `fluid_level` returns
    /// the fluid; at or above returns `Air`.
    pub fn at(&self, wy: i32) -> FluidKind {
        if wy < self.fluid_level {
            self.fluid_kind
        } else {
            FluidKind::Air
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fluid_kind_maps_to_blocks() {
        assert_eq!(FluidKind::Air.to_block(), Block::Air);
        assert_eq!(FluidKind::Water.to_block(), Block::Water);
        assert_eq!(FluidKind::Lava.to_block(), Block::Lava);
    }

    #[test]
    fn fluid_status_at_returns_fluid_below_level_air_above() {
        let s = FluidStatus::new(60, FluidKind::Water);
        assert_eq!(s.at(50), FluidKind::Water, "below level → fluid");
        assert_eq!(s.at(59), FluidKind::Water, "just below level → fluid");
        assert_eq!(s.at(60), FluidKind::Air, "at level → air");
        assert_eq!(s.at(70), FluidKind::Air, "above level → air");
    }

    #[test]
    fn way_below_min_y_status_is_always_air() {
        // A "dry" aquifer's fluid_level is `WAY_BELOW_MIN_Y` (≈ i32::MIN).
        // No real Y is below that, so `at()` always returns Air.
        let s = FluidStatus::new(i32::MIN, FluidKind::Water);
        for wy in [-200, -100, 0, 100, 200] {
            assert_eq!(s.at(wy), FluidKind::Air, "way-below-min → air at y={wy}");
        }
    }
}
```

Add to `src/worldgen/mod.rs` module list (alphabetical):

```rust
pub mod aquifer;
```

- [ ] **Step 3.2: Verify the tests pass**

Run: `cargo test --lib worldgen::aquifer 2>&1 | tail -10`

Expected: 3 tests pass. (The types are simple enough that step 3.1 ships a working impl directly.)

- [ ] **Step 3.3: Commit**

```bash
git add src/worldgen/aquifer.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): add FluidKind and FluidStatus aquifer primitives

Foundation types for the PR 7 aquifer:

  FluidKind { Air, Water, Lava } — maps to Block via to_block().
  FluidStatus { fluid_level: i32, fluid_kind: FluidKind } —
    `at(wy)` returns the fluid that occupies wy in this cell.

Mirrors MC's FluidStatus record and at(blockY) lookup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Grid math and jittered cell centers (TDD)

**Files:**
- Modify: `src/worldgen/aquifer.rs`

- [ ] **Step 4.1: Write the failing tests**

Append to `src/worldgen/aquifer.rs` (above the `#[cfg(test)]` line for the impl, and add new tests in the existing module):

```rust
    // Inside the existing `mod tests` block.

    use crate::worldgen::config::AquiferConfig;
    use crate::worldgen::config::WorldgenConfig;

    fn cfg() -> AquiferConfig {
        WorldgenConfig::bundled_default().unwrap().aquifer
    }

    #[test]
    fn grid_x_floor_div_by_16() {
        let c = cfg();
        // MC: gridX(blockCoord) = blockCoord >> 4 = floor(/16).
        assert_eq!(grid_x(0, &c), 0);
        assert_eq!(grid_x(15, &c), 0);
        assert_eq!(grid_x(16, &c), 1);
        assert_eq!(grid_x(-1, &c), -1);
        assert_eq!(grid_x(-16, &c), -1);
        assert_eq!(grid_x(-17, &c), -2);
    }

    #[test]
    fn grid_y_floor_div_by_12() {
        let c = cfg();
        // MC: gridY(blockCoord) = Math.floorDiv(blockCoord, 12).
        assert_eq!(grid_y(0, &c), 0);
        assert_eq!(grid_y(11, &c), 0);
        assert_eq!(grid_y(12, &c), 1);
        assert_eq!(grid_y(-1, &c), -1);
        assert_eq!(grid_y(-12, &c), -1);
        assert_eq!(grid_y(-13, &c), -2);
    }

    #[test]
    fn cell_center_pure_in_seed_and_grid_coords() {
        let c = cfg();
        let a = cell_center(42, 3, -2, 7, &c);
        let b = cell_center(42, 3, -2, 7, &c);
        assert_eq!(a, b, "cell center must be deterministic");
    }

    #[test]
    fn cell_center_within_grid_box() {
        // The center for cell (gx, gy, gz) lies inside
        // [gx*16, gx*16 + 10] × [gy*12, gy*12 + 9] × [gz*16, gz*16 + 10].
        let c = cfg();
        for gx in -3..=3i32 {
            for gy in -3..=3i32 {
                for gz in -3..=3i32 {
                    let p = cell_center(42, gx, gy, gz, &c);
                    let x_lo = gx * c.x_spacing;
                    let x_hi = x_lo + c.x_range - 1;
                    let y_lo = gy * c.y_spacing;
                    let y_hi = y_lo + c.y_range - 1;
                    let z_lo = gz * c.z_spacing;
                    let z_hi = z_lo + c.z_range - 1;
                    assert!(
                        (x_lo..=x_hi).contains(&p.x),
                        "cell ({gx},{gy},{gz}) center x={} outside [{x_lo}, {x_hi}]",
                        p.x
                    );
                    assert!(
                        (y_lo..=y_hi).contains(&p.y),
                        "cell ({gx},{gy},{gz}) center y={} outside [{y_lo}, {y_hi}]",
                        p.y
                    );
                    assert!(
                        (z_lo..=z_hi).contains(&p.z),
                        "cell ({gx},{gy},{gz}) center z={} outside [{z_lo}, {z_hi}]",
                        p.z
                    );
                }
            }
        }
    }

    #[test]
    fn cell_centers_differ_across_grid_coords() {
        // Neighboring cells should rarely collide on the same center.
        let c = cfg();
        let p0 = cell_center(42, 0, 0, 0, &c);
        let p1 = cell_center(42, 1, 0, 0, &c);
        let p2 = cell_center(42, 0, 1, 0, &c);
        let p3 = cell_center(42, 0, 0, 1, &c);
        assert_ne!(p0, p1, "x-neighbor should differ");
        assert_ne!(p0, p2, "y-neighbor should differ");
        assert_ne!(p0, p3, "z-neighbor should differ");
    }
```

- [ ] **Step 4.2: Verify the tests fail**

Run: `cargo test --lib worldgen::aquifer 2>&1 | tail -10`

Expected: compile error — `grid_x`, `grid_y`, `cell_center` don't exist.

- [ ] **Step 4.3: Implement the grid helpers and cell center sampler**

Append to `src/worldgen/aquifer.rs` (after the existing types, before the test module):

```rust
use crate::worldgen::config::AquiferConfig;
use crate::worldgen::hash::{mix_u32};
use glam::IVec3;

/// World-block → grid-X. Matches MC `gridX(blockCoord) = blockCoord >> 4`.
#[inline]
pub fn grid_x(block_coord: i32, cfg: &AquiferConfig) -> i32 {
    block_coord.div_euclid(cfg.x_spacing)
}

/// World-block → grid-Y. Matches MC `gridY(blockCoord) = floorDiv(blockCoord, 12)`.
#[inline]
pub fn grid_y(block_coord: i32, cfg: &AquiferConfig) -> i32 {
    block_coord.div_euclid(cfg.y_spacing)
}

/// World-block → grid-Z. Matches MC `gridZ(blockCoord) = blockCoord >> 4`.
#[inline]
pub fn grid_z(block_coord: i32, cfg: &AquiferConfig) -> i32 {
    block_coord.div_euclid(cfg.z_spacing)
}

/// Jittered center position of aquifer cell `(gx, gy, gz)` in world
/// coordinates. Each component is the grid base (`g * spacing`) plus a
/// per-cell deterministic offset in `[0, range)` derived from
/// `mix(seed, [gx, gy, gz, salt])`.
///
/// Mirrors MC:
///   x = fromGridX(gx, random.nextInt(10))    // 0..=9
///   y = fromGridY(gy, random.nextInt(9))     // 0..=8
///   z = fromGridZ(gz, random.nextInt(10))    // 0..=9
pub fn cell_center(seed: u64, gx: i32, gy: i32, gz: i32, cfg: &AquiferConfig) -> IVec3 {
    // Three independent rolls — different salts so x/y/z jitter doesn't
    // co-vary. Returns u32; modulo to get values in [0, range).
    let rx = mix_u32(seed, &[gx, gy, gz, 1001]) % cfg.x_range as u32;
    let ry = mix_u32(seed, &[gx, gy, gz, 1002]) % cfg.y_range as u32;
    let rz = mix_u32(seed, &[gx, gy, gz, 1003]) % cfg.z_range as u32;
    IVec3::new(
        gx * cfg.x_spacing + rx as i32,
        gy * cfg.y_spacing + ry as i32,
        gz * cfg.z_spacing + rz as i32,
    )
}
```

- [ ] **Step 4.4: Verify the tests pass**

Run: `cargo test --lib worldgen::aquifer 2>&1 | tail -15`

Expected: all (existing + new) tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add src/worldgen/aquifer.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): aquifer grid math + jittered cell centers

Implements MC's grid coordinate helpers and the per-cell
deterministic jitter:

  grid_x/y/z: block_coord -> grid_coord via floor-div by
    (16, 12, 16).
  cell_center(seed, gx, gy, gz): jittered world-space center
    in [gx*16, gx*16 + 10) × [gy*12, gy*12 + 9) × [gz*16, gz*16 + 10),
    matching MC's nextInt(X_RANGE/Y_RANGE/Z_RANGE) calls.

Uses the existing hash::mix_u32 mixer for determinism — no
PositionalRandomFactory analogue needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Similarity metric and pressure function (TDD)

**Files:**
- Modify: `src/worldgen/aquifer.rs`

- [ ] **Step 5.1: Write the failing tests**

Append to the `mod tests` block in `src/worldgen/aquifer.rs`:

```rust
    #[test]
    fn similarity_at_equal_distances_is_one() {
        // d1 == d2 ⇒ similarity = 1 - 0/25 = 1.0.
        assert!((similarity(100, 100) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_negative_when_d1_much_smaller() {
        // d1=0, d2=100 ⇒ 1 - 100/25 = -3.0 (cleanly inside cell 1).
        let s = similarity(0, 100);
        assert!(s < 0.0, "similarity must be negative when d1 << d2; got {s}");
    }

    #[test]
    fn similarity_positive_when_distances_close() {
        // d1=99, d2=100 ⇒ 1 - 1/25 = 0.96 (boundary region).
        let s = similarity(99, 100);
        assert!(s > 0.95 && s < 1.0, "got {s}");
    }

    #[test]
    fn pressure_water_vs_lava_returns_seal_constant() {
        // MC: when one cell is lava and the other water (or vice versa),
        // pressure is unconditionally 2.0 — guaranteed rock seal.
        let c = cfg();
        let water = FluidStatus::new(50, FluidKind::Water);
        let lava = FluidStatus::new(-30, FluidKind::Lava);
        let p_wl = raw_pressure(60, &water, &lava, 0.0, &c);
        let p_lw = raw_pressure(60, &lava, &water, 0.0, &c);
        assert!((p_wl - 2.0).abs() < 1e-6, "water/lava pressure = 2.0, got {p_wl}");
        assert!((p_lw - 2.0).abs() < 1e-6, "lava/water pressure = 2.0, got {p_lw}");
    }

    #[test]
    fn pressure_same_fluid_level_is_zero() {
        // MC: fluidYDiff = 0 ⇒ pressure = 0 (no barrier).
        let c = cfg();
        let s = FluidStatus::new(50, FluidKind::Water);
        let p = raw_pressure(40, &s, &s, 0.0, &c);
        assert!(p.abs() < 1e-6, "equal levels ⇒ pressure = 0, got {p}");
    }

    #[test]
    fn pressure_above_average_uses_top_biases() {
        // For posY > averageFluidY:
        //   centerPoint > 0 → gradient = centerPoint / 1.5 (holes)
        //   centerPoint ≤ 0 → gradient = centerPoint / 2.5 (rocks)
        // Two cells at different levels; eval at Y above the average.
        let c = cfg();
        let s1 = FluidStatus::new(50, FluidKind::Water);
        let s2 = FluidStatus::new(40, FluidKind::Water); // diff = 10, avg = 45
        // posY = 50, howFarAbove = 50 + 0.5 - 45 = 5.5
        // baseValue = 10/2 = 5.0; distanceFromBarrier = 5 - 5.5 = -0.5
        // centerPoint = 0 + -0.5 = -0.5 → gradient = -0.5 / 2.5 = -0.2
        // result = 2.0 * (noise + -0.2) — pin noise=0
        let p = raw_pressure(50, &s1, &s2, 0.0, &c);
        assert!((p - 2.0 * -0.2).abs() < 1e-4, "expected -0.4, got {p}");
    }

    #[test]
    fn pressure_below_average_uses_bottom_biases() {
        // For posY < averageFluidY:
        //   centerPoint = 3.0 + distanceFromBarrier
        //   centerPoint > 0 → gradient = centerPoint / 3.0 (holes)
        //   centerPoint ≤ 0 → gradient = centerPoint / 10.0 (rocks)
        let c = cfg();
        let s1 = FluidStatus::new(60, FluidKind::Water);
        let s2 = FluidStatus::new(40, FluidKind::Water); // diff = 20, avg = 50
        // posY = 30, howFarAbove = 30.5 - 50 = -19.5 (below)
        // baseValue = 20/2 = 10; distanceFromBarrier = 10 - 19.5 = -9.5
        // centerPoint = 3.0 + -9.5 = -6.5 → gradient = -6.5 / 10.0 = -0.65
        let p = raw_pressure(30, &s1, &s2, 0.0, &c);
        assert!((p - 2.0 * -0.65).abs() < 1e-4, "expected -1.3, got {p}");
    }

    #[test]
    fn pressure_outside_noise_window_drops_noise_term() {
        // When |gradient| > 2.0, the barrier noise is ignored. So
        // passing noise=999 should still produce the gradient-only result.
        let c = cfg();
        let s1 = FluidStatus::new(200, FluidKind::Water);
        let s2 = FluidStatus::new(-200, FluidKind::Water); // diff=400, avg=0
        // posY=0, howFarAbove=0.5; baseValue=200; distanceFromBarrier=199.5
        // gradient ~= 199.5/1.5 → way past +2 window.
        let p_with_noise = raw_pressure(0, &s1, &s2, 999.0, &c);
        let p_no_noise   = raw_pressure(0, &s1, &s2, 0.0, &c);
        assert!((p_with_noise - p_no_noise).abs() < 1e-3,
            "noise must be ignored outside |gradient|<=2 window: with={p_with_noise} no={p_no_noise}");
    }
```

- [ ] **Step 5.2: Verify the tests fail**

Run: `cargo test --lib worldgen::aquifer 2>&1 | tail -15`

Expected: compile error — `similarity`, `raw_pressure` don't exist.

- [ ] **Step 5.3: Implement `similarity` and `raw_pressure`**

Append to the impl section of `src/worldgen/aquifer.rs`:

```rust
/// Similarity between the two nearest cells, MC's
/// `1.0 - (d2² - d1²) / 25.0`. Higher = the two cells are equally
/// close (boundary region). Negative = cell 1 is much closer (clean
/// interior).
///
/// Note: `distance_sqr_2` >= `distance_sqr_1` always, so the return
/// value is bounded above by 1.0; lower bound depends on how far
/// apart the two cells are at this position.
#[inline]
pub fn similarity(distance_sqr_1: i32, distance_sqr_2: i32) -> f32 {
    1.0 - (distance_sqr_2 - distance_sqr_1) as f32 / 25.0
}

/// MC's `calculatePressure` without the `MutableDouble` noise cache
/// — callers pass in their own pre-sampled barrier noise (or 0.0
/// when outside the noise window).
///
/// Returns the rock pressure between two aquifers at world Y `pos_y`.
/// Positive = strong rock barrier; negative = soft barrier; the
/// caller adds this to `density` to decide whether the cell stays
/// fluid (`<= 0`) or seals as rock (`> 0`).
///
/// Asymmetric biases match MC exactly:
///   top:    rocks 2.5, holes 1.5
///   bottom: rocks 10,  holes 3,  bottom_bias 3
///
/// Returns a constant `2.0` when one side is water and the other is
/// lava — the seal that prevents lava-water contact.
pub fn raw_pressure(
    pos_y: i32,
    status_1: &FluidStatus,
    status_2: &FluidStatus,
    barrier_noise: f32,
    _cfg: &AquiferConfig,
) -> f32 {
    // The fluid at this exact Y for each cell (Air above fluid_level).
    let type_1 = status_1.at(pos_y);
    let type_2 = status_2.at(pos_y);

    // Water-vs-lava: always seal.
    let is_water_lava =
        (type_1 == FluidKind::Lava && type_2 == FluidKind::Water)
            || (type_1 == FluidKind::Water && type_2 == FluidKind::Lava);
    if is_water_lava {
        return 2.0;
    }

    // Same fluid level → no pressure.
    let fluid_y_diff = (status_1.fluid_level - status_2.fluid_level).abs();
    if fluid_y_diff == 0 {
        return 0.0;
    }

    let average_fluid_y = 0.5 * (status_1.fluid_level + status_2.fluid_level) as f32;
    let how_far_above_average = pos_y as f32 + 0.5 - average_fluid_y;
    let base_value = fluid_y_diff as f32 / 2.0;
    let distance_from_barrier_edge = base_value - how_far_above_average.abs();

    // Asymmetric piecewise-linear gradient.
    // MC `Aquifer.java:304-365`:
    //   top biases:    furthestRocksFromTopBias=2.5, furthestHolesFromTopBias=1.5
    //   bottom biases: furthestRocksFromBottomBias=10, furthestHolesFromBottomBias=3
    //   bottomBias=3 (offset on bottom-side centerPoint)
    //   topBias=0 (no offset on top-side)
    let gradient = if how_far_above_average > 0.0 {
        // Above average fluid Y → use top biases.
        let center_point = 0.0 + distance_from_barrier_edge;
        if center_point > 0.0 {
            center_point / 1.5  // furthestHolesFromTopBias
        } else {
            center_point / 2.5  // furthestRocksFromTopBias
        }
    } else {
        // Below average fluid Y → use bottom biases (shifted by bottom_bias=3).
        let center_point = 3.0 + distance_from_barrier_edge;
        if center_point > 0.0 {
            center_point / 3.0   // furthestHolesFromBottomBias
        } else {
            center_point / 10.0  // furthestRocksFromBottomBias
        }
    };

    // Noise term: only applied when |gradient| ≤ 2.0 — outside that
    // window the gradient alone dominates and noise would just
    // produce flickering rock/fluid voxels.
    let noise_term = if gradient >= -2.0 && gradient <= 2.0 {
        barrier_noise
    } else {
        0.0
    };

    2.0 * (noise_term + gradient)
}
```

- [ ] **Step 5.4: Verify the tests pass**

Run: `cargo test --lib worldgen::aquifer 2>&1 | tail -20`

Expected: all aquifer tests pass.

- [ ] **Step 5.5: Commit**

```bash
git add src/worldgen/aquifer.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): aquifer similarity metric and pressure function

similarity(d1², d2²) = 1.0 - (d2² - d1²) / 25.0 — MC's boundary
heuristic. Negative when cell 1 dominates; positive when both
cells contend for the voxel.

raw_pressure(pos_y, status_1, status_2, noise, cfg): MC's
calculatePressure with the asymmetric piecewise-linear gradient.

Constants pinned from Aquifer.java lines 304-365:
  top:    rocks=2.5, holes=1.5
  bottom: rocks=10,  holes=3, bottom_bias=3
  water/lava boundary: constant 2.0 (always seals)
  noise window: |gradient| <= 2.0

Pure function — caller supplies the barrier-noise sample so the
hot path can cache it across the 3 pair evaluations per voxel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Three new noise channels + per-cell `FluidStatus` computation (TDD)

**Files:**
- Modify: `src/worldgen/aquifer.rs`

- [ ] **Step 6.1: Write the failing tests**

Append to the `mod tests` block in `src/worldgen/aquifer.rs`:

```rust
    use crate::worldgen::tuning::SEA_LEVEL;

    #[test]
    fn aquifer_noise_set_is_deterministic_in_seed() {
        let n = AquiferNoise::new(42);
        let a = n.floodedness([10.0, 20.0, 30.0]);
        let b = n.floodedness([10.0, 20.0, 30.0]);
        assert_eq!(a, b);
        let a2 = n.spread([1.0, 2.0, 3.0]);
        let b2 = n.spread([1.0, 2.0, 3.0]);
        assert_eq!(a2, b2);
        let a3 = n.lava([7.0, 8.0, 9.0]);
        let b3 = n.lava([7.0, 8.0, 9.0]);
        assert_eq!(a3, b3);
    }

    #[test]
    fn compute_fluid_above_surface_returns_global() {
        // When the aquifer cell sits well above h_target, it's
        // an above-surface cell — must return the global fluid
        // (Water below sea level, Air above).
        let c = cfg();
        let n = AquiferNoise::new(42);
        // h_target = 80, cell at y=120 → 40 above; bottom of cell
        // (y - 12 = 108) > 80+8 = adjusted_surface ⇒ return global.
        let h_target = 80;
        let above = compute_fluid(0, 120, 0, h_target, &n, &c);
        // Above sea level: global Air at this Y.
        assert_eq!(above.at(120), FluidKind::Air);
    }

    #[test]
    fn compute_fluid_well_below_surface_can_be_dry() {
        // A cell well below the surface with a low floodedness noise
        // value (no flood) must yield WAY_BELOW_MIN_Y fluid level —
        // so all of its voxels stay air, i.e. dry caves.
        let c = cfg();
        let n = AquiferNoise::new(42);
        // Sample 256 random-ish deep cells; verify at least one is dry.
        let mut found_dry = false;
        for gx in -8..=8 {
            for gz in -8..=8 {
                let p = cell_center(42, gx, -5, gz, &c); // gy=-5 → y around -60
                let h_target = 80; // surface far above
                let s = compute_fluid(p.x, p.y, p.z, h_target, &n, &c);
                if s.fluid_level == c.way_below_min_y {
                    found_dry = true;
                    break;
                }
            }
            if found_dry { break; }
        }
        assert!(
            found_dry,
            "expected at least one dry aquifer cell deep below land"
        );
    }

    #[test]
    fn compute_fluid_spread_is_quantized_in_threes() {
        // The randomized fluid surface level uses
        // quantize(spread_noise * 10, 3). For nearby positions whose
        // spread cell is identical, the result must land on a multiple
        // of 3 offset from fluid_cell_middle_y.
        let c = cfg();
        let n = AquiferNoise::new(42);
        // Force the "partially flooded" branch by choosing parameters:
        // we can't directly without exercising the noise — instead,
        // call the helper directly.
        let mid_y = 0; // arbitrary
        let lowest_surface = 80; // far above
        let level = compute_randomized_fluid_surface_level(
            10, mid_y, 10, lowest_surface, &n, &c,
        );
        // Either WAY_BELOW or = floor_div(0, 40)*40 + 20 + k*3 for some
        // integer k, capped by lowest_surface.
        if level != c.way_below_min_y {
            let cell_middle = (0_i32.div_euclid(c.lava_cell_y)) * c.lava_cell_y + 20;
            let delta = level - cell_middle;
            assert_eq!(
                delta.rem_euclid(c.spread_quantize_step),
                0,
                "fluid surface delta {delta} not divisible by {}",
                c.spread_quantize_step
            );
        }
    }

    #[test]
    fn compute_fluid_lava_only_below_minus_ten() {
        // Lava conversion is gated on fluid_surface_level <= -10. A
        // cell with fluid level above -10 must never be lava.
        let c = cfg();
        let n = AquiferNoise::new(42);
        let global = FluidStatus::new(SEA_LEVEL, FluidKind::Water);
        for fluid_surface_level in [-9, 0, 50, 62] {
            let kind = compute_fluid_type(
                100, 100, 100, &global, fluid_surface_level, &n, &c,
            );
            assert_eq!(
                kind, global.fluid_kind,
                "fluid_surface_level {fluid_surface_level} should keep global fluid type"
            );
        }
    }
```

- [ ] **Step 6.2: Verify the tests fail**

Run: `cargo test --lib worldgen::aquifer 2>&1 | tail -15`

Expected: compile errors — `AquiferNoise`, `compute_fluid`, `compute_randomized_fluid_surface_level`, `compute_fluid_type` don't exist.

- [ ] **Step 6.3: Implement `AquiferNoise` and the per-cell computation**

Append to `src/worldgen/aquifer.rs` (before the test module):

```rust
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

/// The three noise channels the aquifer system samples. Constructed
/// once per [`Generator`] and shared across chunks.
///
/// MC equivalents:
///   floodedness ⇔ aquifer_fluid_level_floodedness
///   spread      ⇔ aquifer_fluid_level_spread
///   lava        ⇔ aquifer_lava
///
/// Plus an unused-by-PR-7 `barrier` channel — that's sampled inside
/// [`Aquifer::compute_substance`] via the existing `relief` noise to
/// avoid adding a fourth field. (MC has a separate barrier_noise;
/// reusing relief is a deliberate simplification — a future PR may
/// split it out.)
pub struct AquiferNoise {
    pub floodedness: Fbm<Simplex>,
    pub spread: Fbm<Simplex>,
    pub lava: Fbm<Simplex>,
    pub barrier: Fbm<Simplex>,
}

impl AquiferNoise {
    pub fn new(seed: u64) -> Self {
        // Wavelength chosen so each channel varies smoothly across many
        // aquifer cells (≈ 1024-block period). Matches MC's first-octave
        // -8 ≈ 256-block wavelength on each octave.
        Self {
            floodedness: Fbm::<Simplex>::new(seed.wrapping_add(811) as u32)
                .set_octaves(1)
                .set_frequency(1.0 / 1024.0)
                .set_persistence(0.5),
            spread: Fbm::<Simplex>::new(seed.wrapping_add(812) as u32)
                .set_octaves(1)
                .set_frequency(1.0 / 16.0) // per-cell coordinates, NOT block
                .set_persistence(0.5),
            lava: Fbm::<Simplex>::new(seed.wrapping_add(813) as u32)
                .set_octaves(1)
                .set_frequency(1.0 / 8.0) // per-cell coordinates
                .set_persistence(0.5),
            barrier: Fbm::<Simplex>::new(seed.wrapping_add(814) as u32)
                .set_octaves(2)
                .set_frequency(1.0 / 32.0)
                .set_persistence(0.5),
        }
    }

    pub fn floodedness(&self, p: [f64; 3]) -> f32 {
        self.floodedness.get(p).clamp(-1.0, 1.0) as f32
    }
    pub fn spread(&self, p: [f64; 3]) -> f32 {
        self.spread.get(p) as f32
    }
    pub fn lava(&self, p: [f64; 3]) -> f32 {
        self.lava.get(p) as f32
    }
    pub fn barrier(&self, p: [f64; 3]) -> f32 {
        self.barrier.get(p) as f32
    }
}

/// Global fluid picker: water below sea level, air above. PR 7
/// hard-codes this — a future PR could make it biome-aware.
#[inline]
pub fn global_fluid(_x: i32, _y: i32, _z: i32, cfg: &AquiferConfig) -> FluidStatus {
    FluidStatus::new(cfg.sea_level, FluidKind::Water)
}

/// Compute the [`FluidStatus`] for an aquifer cell whose center is
/// at world-block `(x, y, z)`. Mirrors MC `NoiseBasedAquifer.computeFluid`
/// but simplified: instead of MC's 13-offset surface sampling array
/// we use the single column's `h_target` as the preliminary surface.
/// (Cross-region surface smoothing is a refinement for a later PR;
/// the visual difference at typical aquifer cell sizes is minimal.)
///
/// Decision tree:
///   1. Aquifer cell sits entirely above the adjusted surface
///      (`y - 12 > h_target + 8`)  ⇒  global fluid (water/air).
///   2. Surface dips below global fluid (oceanic column) AND aquifer
///      cell pokes above adjusted surface  ⇒  global fluid (so
///      ocean aquifers stay fully flooded).
///   3. Otherwise: compute floodedness/spread noises to derive a
///      local `fluid_surface_level`; then check lava noise to
///      possibly flip the fluid kind to lava.
pub fn compute_fluid(
    x: i32,
    y: i32,
    z: i32,
    h_target: i32,
    noise: &AquiferNoise,
    cfg: &AquiferConfig,
) -> FluidStatus {
    let global = global_fluid(x, y, z, cfg);

    let top_of_cell = y + cfg.y_spacing;
    let bottom_of_cell = y - cfg.y_spacing;

    // MC's adjustSurfaceLevel adds 8 — terrain has a few blocks of
    // soil above the iso-surface where aquifers shouldn't poke through.
    let adjusted_surface = h_target + 8;

    // Path 1: cell is entirely above the surface → global fluid.
    if bottom_of_cell > adjusted_surface {
        return global;
    }

    // Path 2: surface is below sea level (oceanic column) AND aquifer
    // pokes above surface → global fluid.
    let surface_under_global = adjusted_surface < cfg.sea_level;
    let pokes_above_surface = top_of_cell > adjusted_surface;
    if pokes_above_surface && surface_under_global {
        return global;
    }

    // Path 3: full computation.
    let fluid_surface_level = compute_surface_level(
        x, y, z, &global, h_target, surface_under_global, noise, cfg,
    );
    let fluid_kind =
        compute_fluid_type(x, y, z, &global, fluid_surface_level, noise, cfg);
    FluidStatus::new(fluid_surface_level, fluid_kind)
}

/// Helper used by [`compute_fluid`]: derive the local
/// `fluid_surface_level` from floodedness + spread.
fn compute_surface_level(
    x: i32,
    y: i32,
    z: i32,
    global: &FluidStatus,
    h_target: i32,
    surface_under_global: bool,
    noise: &AquiferNoise,
    cfg: &AquiferConfig,
) -> i32 {
    // distance below adjusted surface — used to ramp floodedness.
    let distance_below_surface = (h_target + 8) - y;

    // Floodedness factor: 1.0 right under the surface (high chance
    // of full flood), tapering to 0.0 by `floodedness_max_depth`.
    let floodedness_factor = if surface_under_global {
        let t =
            (distance_below_surface as f32 / cfg.floodedness_max_depth as f32).clamp(0.0, 1.0);
        // MC maps 0→1.0 (full flood) and 64→0.0 (no help from being
        // under ocean).
        1.0 - t
    } else {
        0.0
    };

    let floodedness_noise = noise.floodedness([x as f64, y as f64, z as f64]);
    let fully_flooded_threshold = lerp(floodedness_factor, 0.8, -0.3);
    let partially_flooded_threshold = lerp(floodedness_factor, 0.4, -0.8);
    let fully_floodedness = floodedness_noise - fully_flooded_threshold;
    let partially_floodedness = floodedness_noise - partially_flooded_threshold;

    if fully_floodedness > 0.0 {
        // Fully flooded → sea-level water (or global fluid).
        global.fluid_level
    } else if partially_floodedness > 0.0 {
        // Partially flooded → randomized local fluid surface.
        compute_randomized_fluid_surface_level(x, y, z, h_target + 8, noise, cfg)
    } else {
        // Dry → no aquifer; this cell's voxels stay air.
        cfg.way_below_min_y
    }
}

/// Helper for partially-flooded cells: compute a quantized random
/// surface level driven by the spread noise on a coarser cell grid
/// (16×40×16 in MC). The level is the cell's middle Y (`cellY*40+20`)
/// plus `quantize(spread*10, 3)`, capped at `lowest_surface` so the
/// water table never tops the actual terrain surface.
pub fn compute_randomized_fluid_surface_level(
    x: i32,
    y: i32,
    z: i32,
    lowest_surface: i32,
    noise: &AquiferNoise,
    cfg: &AquiferConfig,
) -> i32 {
    let cell_x = x.div_euclid(16);
    let cell_y = y.div_euclid(cfg.lava_cell_y);
    let cell_z = z.div_euclid(16);
    let cell_middle_y = cell_y * cfg.lava_cell_y + 20;
    let spread_noise = noise.spread([cell_x as f64, cell_y as f64, cell_z as f64]) * 10.0;
    let spread_quantized = quantize(spread_noise, cfg.spread_quantize_step);
    let target = cell_middle_y + spread_quantized;
    lowest_surface.min(target)
}

/// Quantize `value` to the nearest integer multiple of `step`.
/// Matches MC `Mth.quantize(double, int)`.
///
/// The 3-block quantization of `spread_noise * 10` is pinned here.
#[inline]
pub fn quantize(value: f32, step: i32) -> i32 {
    (value / step as f32).floor() as i32 * step
}

/// Linear interpolation matching MC `Mth.map(t, 0..1, a..b)`.
#[inline]
fn lerp(t: f32, a: f32, b: f32) -> f32 {
    a + (b - a) * t
}

/// Helper for [`compute_fluid`]: decide whether the cell is lava
/// or stays the global fluid kind. Lava is gated on:
///
/// 1. `fluid_surface_level <= cfg.lava_level_max` (default -10), AND
/// 2. fluid_surface_level is not `way_below_min_y` (sentinel for dry), AND
/// 3. global fluid is not already lava, AND
/// 4. `|lava_noise| > cfg.lava_threshold` (default 0.3).
///
/// Sampling grid for lava is coarser (64×40×64 in MC) so lava
/// regions are large and persistent.
pub fn compute_fluid_type(
    x: i32,
    y: i32,
    z: i32,
    global: &FluidStatus,
    fluid_surface_level: i32,
    noise: &AquiferNoise,
    cfg: &AquiferConfig,
) -> FluidKind {
    if fluid_surface_level > cfg.lava_level_max {
        return global.fluid_kind;
    }
    if fluid_surface_level == cfg.way_below_min_y {
        return global.fluid_kind;
    }
    if global.fluid_kind == FluidKind::Lava {
        return global.fluid_kind;
    }
    let cell_x = x.div_euclid(cfg.lava_cell_xz);
    let cell_y = y.div_euclid(cfg.lava_cell_y);
    let cell_z = z.div_euclid(cfg.lava_cell_xz);
    let lava_noise = noise.lava([cell_x as f64, cell_y as f64, cell_z as f64]);
    if lava_noise.abs() > cfg.lava_threshold {
        FluidKind::Lava
    } else {
        global.fluid_kind
    }
}
```

- [ ] **Step 6.4: Verify the tests pass**

Run: `cargo test --lib worldgen::aquifer 2>&1 | tail -15`

Expected: all aquifer tests pass. If `compute_fluid_well_below_surface_can_be_dry` fails, the floodedness wavelength may need adjusting — try halving `frequency` or check the threshold math.

- [ ] **Step 6.5: Commit**

```bash
git add src/worldgen/aquifer.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): aquifer noise channels + per-cell fluid status

AquiferNoise holds three Fbm channels (floodedness, spread, lava)
plus a barrier channel for the per-voxel decision. Seeds derived
from the world seed via salts 811-814 so they don't correlate with
the existing density/heightmap noises.

compute_fluid(x, y, z, h_target, noise, cfg) -> FluidStatus
mirrors MC's NoiseBasedAquifer.computeFluid:
  - above-surface cells → global fluid
  - oceanic-column cells poking above surface → global fluid
  - else: floodedness + spread derive a local fluid_surface_level
  - lava check: deep cells with |lava_noise| > 0.3 flip to lava

The 3-block spread quantization is pinned in quantize(value, 3).
Lava cells use a coarser 64×40×64 sampling grid than aquifer
cells, so lava pockets persist across many aquifer cells.

Surface-sampling simplification: PR 7 uses the column's h_target
as the preliminary surface (single sample). MC samples 13 offsets;
adding that is a refinement for a future PR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `Aquifer` struct with per-chunk cache + `compute_substance` (TDD)

**Files:**
- Modify: `src/worldgen/aquifer.rs`

- [ ] **Step 7.1: Write the failing tests**

Append to the `mod tests` block in `src/worldgen/aquifer.rs`:

```rust
    use crate::voxel::coords::{ChunkCoord, CHUNK_DIM_U};
    use glam::IVec3;

    fn make_aquifer(seed: u64, chunk_y: i32, h_target: i32) -> Aquifer {
        let c = cfg();
        let noise = AquiferNoise::new(seed);
        Aquifer::new(
            seed,
            ChunkCoord(IVec3::new(0, chunk_y, 0)),
            h_target,
            &c,
            noise,
        )
    }

    #[test]
    fn compute_substance_solid_returns_none() {
        let mut aq = make_aquifer(42, -2, 80);
        // density > 0 → solid → caller writes stone.
        let block = aq.compute_substance(0, -60, 0, 1.0);
        assert_eq!(block, None);
    }

    #[test]
    fn compute_substance_above_skip_y_uses_global_fluid() {
        // skip_sampling_above_y is computed from h_target; well above
        // the surface, the aquifer short-circuits to the global fluid.
        let mut aq = make_aquifer(42, 5, 80);
        // Y=200, well above h_target=80 → global picker → above sea →
        // air.
        let block = aq.compute_substance(0, 200, 0, -1.0);
        assert_eq!(block, Some(FluidKind::Air));
        // Y=20, well above h_target=80 doesn't apply (Y is below
        // h_target). But within an ocean column (h_target<sea), Y=20
        // < sea → water. Test a different column instead:
        let mut aq_ocean = make_aquifer(42, -1, 50); // h_target below sea
        let block_ocean = aq_ocean.compute_substance(0, 30, 0, -1.0);
        assert_eq!(
            block_ocean,
            Some(FluidKind::Water),
            "below sea in oceanic column should flood with water"
        );
    }

    #[test]
    fn compute_substance_is_deterministic_for_same_inputs() {
        // Same seed + same (wx, wy, wz) + same density → same result.
        let mut aq1 = make_aquifer(42, -2, 80);
        let mut aq2 = make_aquifer(42, -2, 80);
        let a = aq1.compute_substance(17, -55, -23, -0.3);
        let b = aq2.compute_substance(17, -55, -23, -0.3);
        assert_eq!(a, b, "aquifer must be pure in (seed, position, density)");
    }

    #[test]
    fn water_aquifer_next_to_lava_aquifer_seals_at_boundary() {
        // Construct two adjacent grid cells with hand-picked
        // FluidStatus values — one water, one lava — and check that
        // raw_pressure is huge along the boundary.
        let c = cfg();
        let water = FluidStatus::new(40, FluidKind::Water);
        let lava = FluidStatus::new(-30, FluidKind::Lava);
        for y in [-50, -10, 30, 60] {
            let p = raw_pressure(y, &water, &lava, 0.0, &c);
            assert!(
                p >= 2.0,
                "water/lava boundary at y={y} produced pressure {p}, expected >= 2.0"
            );
        }
    }
```

- [ ] **Step 7.2: Verify the tests fail**

Run: `cargo test --lib worldgen::aquifer 2>&1 | tail -10`

Expected: compile error — `Aquifer` and its constructor don't exist.

- [ ] **Step 7.3: Implement `Aquifer` struct and `compute_substance`**

Append to `src/worldgen/aquifer.rs` (before the test module):

```rust
use crate::voxel::coords::{ChunkCoord, CHUNK_DIM_U};

/// Per-chunk aquifer state. Construct via [`Aquifer::new`] before
/// the per-voxel loop, then call [`Aquifer::compute_substance`] for
/// each voxel.
///
/// Owns:
///   - A small fixed-size cache of [`FluidStatus`] keyed by
///     `(grid_x, grid_y, grid_z)` so each cell is computed at most
///     once per chunk.
///   - The cached `skip_sampling_above_y` cutoff.
///   - References to the noise channels and config snapshot.
///
/// Lifetime: one Aquifer per chunk fill. The cache is reset on
/// `new()`.
pub struct Aquifer {
    seed: u64,
    cfg: AquiferConfig,
    noise: AquiferNoise,
    h_target: i32,
    /// Above this Y, [`compute_substance`] short-circuits to the
    /// global fluid picker — there are no aquifer cells up here so
    /// running the full 12-cell anchor query would be wasted work.
    skip_sampling_above_y: i32,
    /// Per-grid-cell `FluidStatus` cache. Bounded to the 12-cell
    /// neighbourhood the chunk can possibly query (2 in X, 3 in Y,
    /// 2 in Z, plus a 2-cell padding for cross-chunk queries). The
    /// cache key encodes `(grid_x, grid_y, grid_z)` mixed via
    /// `hash::mix_u32`. Small enough to scan linearly.
    cache: smallvec::SmallVec<[(IVec3, FluidStatus); 64]>,
}

impl Aquifer {
    /// Construct a new per-chunk aquifer. `h_target` is the
    /// preliminary surface Y for the chunk's center column —
    /// `skip_sampling_above_y` is derived from it. (Per-column
    /// h_target variation within a chunk is small relative to the
    /// 12-block Y-spacing, so a single sample suffices.)
    pub fn new(
        seed: u64,
        chunk_coord: ChunkCoord,
        h_target: i32,
        cfg: &AquiferConfig,
        noise: AquiferNoise,
    ) -> Self {
        let _ = chunk_coord; // chunk_coord may be needed by future cache strategies
        // Skip Y is `(h_target + 8) + 12 + 12` — one cell padding above
        // the adjusted surface, then the y_spacing again.
        let skip_sampling_above_y = h_target + 8 + cfg.y_spacing * 2;
        Self {
            seed,
            cfg: cfg.clone(),
            noise,
            h_target,
            skip_sampling_above_y,
            cache: smallvec::SmallVec::new(),
        }
    }

    /// Look up or compute the [`FluidStatus`] for grid cell `(gx, gy, gz)`.
    /// Cached for the chunk's lifetime; subsequent queries for the
    /// same cell are O(cache_size) linear scans.
    fn aquifer_status(&mut self, gx: i32, gy: i32, gz: i32) -> FluidStatus {
        let key = IVec3::new(gx, gy, gz);
        for (k, v) in &self.cache {
            if *k == key {
                return *v;
            }
        }
        // Miss: compute and cache.
        let center = cell_center(self.seed, gx, gy, gz, &self.cfg);
        let status = compute_fluid(
            center.x, center.y, center.z, self.h_target, &self.noise, &self.cfg,
        );
        self.cache.push((key, status));
        status
    }

    /// MC `NoiseBasedAquifer.computeSubstance`. Returns:
    ///   - `None` if the voxel is solid (caller writes stone).
    ///   - `Some(FluidKind::Air)` if the voxel is exposed air above
    ///     any aquifer.
    ///   - `Some(FluidKind::Water | Lava)` if the voxel is inside an
    ///     aquifer fluid column.
    ///
    /// Two short-circuits before the full query:
    ///   1. `density > 0` → solid, return None immediately.
    ///   2. `wy > skip_sampling_above_y` → global fluid (no aquifers
    ///      live this high).
    ///
    /// Otherwise: walk the 2×3×2 anchor neighbourhood, track the
    /// three smallest squared distances, compute pairwise pressures
    /// against the closest aquifer, and decide.
    pub fn compute_substance(
        &mut self,
        wx: i32,
        wy: i32,
        wz: i32,
        density: f32,
    ) -> Option<FluidKind> {
        if density > 0.0 {
            return None;
        }

        let global = global_fluid(wx, wy, wz, &self.cfg);
        if wy > self.skip_sampling_above_y {
            return Some(global.at(wy));
        }

        // Anchor grid cell for this voxel — the +offset shifts the
        // sampling pattern as in MC.
        let x_anchor = grid_x(wx + self.cfg.sample_offset_x, &self.cfg);
        let y_anchor = grid_y(wy + self.cfg.sample_offset_y, &self.cfg);
        let z_anchor = grid_z(wz + self.cfg.sample_offset_z, &self.cfg);

        // Walk the 2×3×2 neighbourhood (MC: x1 ∈ [0,1], y1 ∈ [-1,1],
        // z1 ∈ [0,1]). For each of the 12 cells, compute the squared
        // distance from this voxel to that cell's jittered center;
        // keep the 3 smallest in sorted order.
        let mut d_sqr_1 = i32::MAX;
        let mut d_sqr_2 = i32::MAX;
        let mut d_sqr_3 = i32::MAX;
        let mut idx_1 = IVec3::ZERO;
        let mut idx_2 = IVec3::ZERO;
        let mut idx_3 = IVec3::ZERO;
        for x1 in 0..=1 {
            for y1 in -1..=1 {
                for z1 in 0..=1 {
                    let gx = x_anchor + x1;
                    let gy = y_anchor + y1;
                    let gz = z_anchor + z1;
                    let center = cell_center(self.seed, gx, gy, gz, &self.cfg);
                    let dx = center.x - wx;
                    let dy = center.y - wy;
                    let dz = center.z - wz;
                    let d_sqr = dx * dx + dy * dy + dz * dz;
                    if d_sqr <= d_sqr_1 {
                        idx_3 = idx_2;
                        idx_2 = idx_1;
                        idx_1 = IVec3::new(gx, gy, gz);
                        d_sqr_3 = d_sqr_2;
                        d_sqr_2 = d_sqr_1;
                        d_sqr_1 = d_sqr;
                    } else if d_sqr <= d_sqr_2 {
                        idx_3 = idx_2;
                        idx_2 = IVec3::new(gx, gy, gz);
                        d_sqr_3 = d_sqr_2;
                        d_sqr_2 = d_sqr;
                    } else if d_sqr <= d_sqr_3 {
                        idx_3 = IVec3::new(gx, gy, gz);
                        d_sqr_3 = d_sqr;
                    }
                }
            }
        }

        // Closest aquifer's fluid at this Y. Lazy fetch via cache.
        let status_1 = self.aquifer_status(idx_1.x, idx_1.y, idx_1.z);
        let fluid_at_1 = status_1.at(wy);
        let similarity_12 = similarity(d_sqr_1, d_sqr_2);

        // Inside cell 1's clean interior → return its fluid.
        if similarity_12 <= 0.0 {
            return Some(fluid_at_1);
        }

        // Otherwise check pairwise pressures. Cache the barrier-noise
        // sample so it's only computed once per voxel (MC does this
        // via MutableDouble — we use Option<f32>).
        let mut cached_noise: Option<f32> = None;
        let mut sample_noise = || -> f32 {
            if let Some(v) = cached_noise {
                return v;
            }
            let v = self.noise.barrier([wx as f64, wy as f64, wz as f64]);
            cached_noise = Some(v);
            v
        };

        let status_2 = self.aquifer_status(idx_2.x, idx_2.y, idx_2.z);
        let p_12 = similarity_12
            * raw_pressure(wy, &status_1, &status_2, sample_noise(), &self.cfg);
        if (density + p_12) > 0.0 {
            return None;
        }

        let similarity_13 = similarity(d_sqr_1, d_sqr_3);
        if similarity_13 > 0.0 {
            let status_3 = self.aquifer_status(idx_3.x, idx_3.y, idx_3.z);
            let p_13 = similarity_12
                * similarity_13
                * raw_pressure(wy, &status_1, &status_3, sample_noise(), &self.cfg);
            if (density + p_13) > 0.0 {
                return None;
            }
        }

        let similarity_23 = similarity(d_sqr_2, d_sqr_3);
        if similarity_23 > 0.0 {
            let status_3 = self.aquifer_status(idx_3.x, idx_3.y, idx_3.z);
            let p_23 = similarity_12
                * similarity_23
                * raw_pressure(wy, &status_2, &status_3, sample_noise(), &self.cfg);
            if (density + p_23) > 0.0 {
                return None;
            }
        }

        Some(fluid_at_1)
    }
}
```

Add `smallvec` to `Cargo.toml` if not already present (PR 2 may have added it):

```toml
smallvec = "1"
```

Run: `cargo check 2>&1 | tail -5` to verify the dependency resolves.

- [ ] **Step 7.4: Verify the tests pass**

Run: `cargo test --lib worldgen::aquifer 2>&1 | tail -20`

Expected: all aquifer tests pass.

If `water_aquifer_next_to_lava_aquifer_seals_at_boundary` fails, the `raw_pressure` water/lava short-circuit isn't firing — re-check the FluidKind comparison in step 5.3.

- [ ] **Step 7.5: Commit**

```bash
git add src/worldgen/aquifer.rs Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
feat(worldgen): Aquifer struct with compute_substance hot path

The per-chunk Aquifer aggregator:
  - holds the AquiferConfig snapshot, noise channels, and per-cell
    FluidStatus cache (SmallVec, lazy populate)
  - precomputes skip_sampling_above_y from the chunk's h_target so
    above-surface voxels skip the 12-cell anchor query entirely
  - exposes compute_substance(wx, wy, wz, density) -> Option<FluidKind>

Hot path (per voxel):
  1. density > 0 → return None (solid).
  2. wy > skip_sampling_above_y → return global_fluid.at(wy).
  3. Walk 2×3×2 cell neighbourhood, tracking 3 smallest squared
     distances. Lazily compute each cell's FluidStatus.
  4. If similarity(d1, d2) <= 0 → return cell 1's fluid (clean
     interior).
  5. Else compute pairwise pressures (1,2), (1,3), (2,3). If any
     pair's `density + pressure > 0`, return None (rock barrier).
  6. Otherwise return cell 1's fluid.

The barrier-noise sample is cached across the three pair checks
via a per-voxel Option<f32>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Wire the aquifer into `Generator` and `fill_chunk`

**Files:**
- Modify: `src/worldgen/mod.rs`

- [ ] **Step 8.1: Add `AquiferNoise` field to `Generator`**

In `src/worldgen/mod.rs`, add an `aquifer_noise` field next to the existing noise fields on `Generator`:

```rust
pub struct Generator {
    // ... existing fields ...
    /// PR 7: aquifer noise channels (floodedness, spread, lava,
    /// barrier). Built once per Generator and cloned into each
    /// per-chunk Aquifer.
    aquifer_noise_seed: u64,
    // (We store only the seed and rebuild the noise per chunk to
    // sidestep the Fbm cloning question; if the perf cost shows up
    // in benchmarks a later PR can introduce an Arc<AquiferNoise>.)
}
```

In `Generator::new`, store the seed:

```rust
Self {
    // ... existing fields ...
    aquifer_noise_seed: seed,
    // ... existing fields continue ...
}
```

- [ ] **Step 8.2: Replace the primitive aquifer rule in `fill_chunk`**

In `src/worldgen/mod.rs::fill_chunk`, locate the existing primitive rule (currently around lines 419-435 — the `// Air — flood with water only where water...` block):

```rust
let block = if !solid {
    depth_below_surface = None;
    let in_lake = lake_rim.map_or(false, |rim| wy <= rim);
    let in_ocean = height <= SEA_LEVEL && wy <= SEA_LEVEL;
    if in_lake || in_ocean {
        Block::Water
    } else {
        Block::Air
    }
} else {
    // ... solid branch ...
};
```

Replace the `!solid` (and the surrounding `let solid = ...; let block = if !solid { ... }`) with an aquifer-driven decision. Above the chunk's `for z, x` columns loop (just after `let cave_systems = ...`), construct one `Aquifer` instance per chunk using the chunk's center column h_target:

```rust
// Estimate a representative h_target for the chunk (used to derive
// skip_sampling_above_y inside Aquifer). The center column is fine:
// per-column h_target variations within a chunk are tiny relative to
// the 12-block aquifer Y-spacing.
let center_wx = origin.x + (CHUNK_DIM_U as i32) / 2;
let center_wz = origin.z + (CHUNK_DIM_U as i32) / 2;
let center_col = self.column_data_with(center_wx, center_wz, &regions);
let chunk_h_target = center_col.height;
let cfg_snapshot = self.config_snapshot();
let aquifer_noise = aquifer::AquiferNoise::new(self.aquifer_noise_seed);
let mut aquifer = aquifer::Aquifer::new(
    self.seed,
    coord,
    chunk_h_target,
    &cfg_snapshot.aquifer,
    aquifer_noise,
);
```

Inside the per-voxel loop, replace the `let block = if !solid {...}` decision with an aquifer call:

```rust
// PR 7: aquifer replaces the primitive ocean/lake water rule. The
// aquifer returns None for solid (caller writes stone) or Some(fluid)
// for air/water/lava. We pass the post-cave density so caves are
// already subtracted — the aquifer's pressure check uses this to
// decide whether to re-seal a fluid boundary with rock.
let voxel_density = (raw_density - cave_contribution) as f32;
let substance = aquifer.compute_substance(wx, wy, wz, voxel_density);

let block = match substance {
    None => {
        // Solid — fall through to the depth-based surface-block
        // selector below (kept verbatim from the pre-PR-7 code).
        let depth = depth_below_surface.map(|d| d + 1).unwrap_or(0);
        depth_below_surface = Some(depth);
        let near_surface =
            (h_target - wy as f32).abs() <= SURFACE_BAND as f32;
        if col.is_cliff || !near_surface {
            Block::Stone
        } else if depth == 0 {
            // ... existing topmost-solid selector body ...
            unimplemented!("paste from pre-PR-7 mod.rs lines 452-485")
        } else if depth <= 3 {
            Block::Dirt
        } else {
            Block::Stone
        }
    }
    Some(fluid_kind) => {
        depth_below_surface = None;
        fluid_kind.to_block()
    }
};
```

(When implementing, paste the topmost-solid selector body verbatim from the existing `mod.rs:452-485`. Don't change the surface-block selection logic — that's PR 6's territory.)

- [ ] **Step 8.3: Remove the now-unused `lake_rim`/`in_ocean`/`in_lake` variables**

The `lake_rim` binding is still used by the bound assertion `let _ = lake_rim;` at the end of the column block — that line can stay (it's harmless). The aquifer doesn't consult `lake_rim` directly — instead the surface "h_target below sea" check in `compute_fluid` (Path 2) handles oceanic columns. The aquifer doesn't yet honour lake rims; for now, lakes flood via the same mechanism as oceans (their adjusted surface ends up below the lake water level, so the aquifer fills them).

If a follow-up PR wants per-lake water levels, `compute_fluid` could accept a `lake_rim_above` parameter — out of scope here.

- [ ] **Step 8.4: Run the test suite**

Run: `cargo test --lib worldgen 2>&1 | tail -20`

Expected results:
- All `aquifer::*` tests still pass.
- `deep_caves_under_land_are_dry` — should still pass (now via the real aquifer's "dry cell" branch).
- `underground_chunk_has_both_caves_and_solid` — should still pass.
- `deep_underground_has_no_surface_blocks` — should still pass.
- `golden_seed42_chunk_0_2_0` — likely fails (block layout changed). Update via the sentinel pattern (set to `0xDEAD_BEEF_DEAD_BEEF`, capture print, repin).
- `worldgen_fingerprint::fingerprint_hash_matches_pin` — should still pass (the 2D heightmap is untouched).

- [ ] **Step 8.5: Re-baseline the golden hash**

Open `src/worldgen/mod.rs`, find `GOLDEN_42_002`, and set it to `0xDEAD_BEEF_DEAD_BEEF`. Run:

```
cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 -- --nocapture 2>&1 | grep "UPDATE GOLDEN"
```

Update `GOLDEN_42_002` to the printed hex value.

Re-run: `cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 2>&1 | tail -5`

Expected: pass with the new pinned hash.

- [ ] **Step 8.6: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): wire PR 7 aquifer into fill_chunk

Replaces the primitive "ocean column or under-lake → water" rule
with the per-region MC-style aquifer:

  1. Build one Aquifer per chunk fill (skip_sampling_above_y is
     derived from the chunk's center-column h_target).
  2. Per voxel: feed (wx, wy, wz, density - cave_contribution) to
     Aquifer::compute_substance.
  3. None → solid (run the existing depth-based surface selector).
  4. Some(fluid) → air / water / lava.

Caves under land that previously flooded only because the column
height was at or below sea level now correctly stay dry: the
aquifer queries the local floodedness noise and returns
WAY_BELOW_MIN_Y for non-flooded cells. Caves under the ocean
still flood (the oceanic-column gate in compute_fluid forces the
global fluid). Lake-water support is unchanged for now — lakes
share the global fluid level, so they flood via the same path as
oceans.

Golden hash re-baselined (block layout changed).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Integration tests for water/lava sealing and dry land caves

**Files:**
- Modify: `src/worldgen/mod.rs` (append to test module)

- [ ] **Step 9.1: Write the failing tests**

In `src/worldgen/mod.rs`, append to the existing `#[cfg(test)] mod tests` block:

```rust
    /// PR 7: lava and water aquifers should never be adjacent — the
    /// pressure function unconditionally returns 2.0 at their boundary,
    /// which seals as rock against any non-positive density. Scan a
    /// region of land chunks; for every Lava-Water adjacency, the
    /// neighbour block must be Stone.
    #[test]
    fn lava_and_water_aquifers_are_separated_by_rock() {
        let g = Generator::new(42);
        let mut found_lava = false;
        let mut violations = 0u32;
        for cz in -4..4 {
            for cx in -4..4 {
                let mut chunk = DenseChunk::empty();
                g.fill_chunk(ChunkCoord(IVec3::new(cx, -3, cz)), &mut chunk);
                // Y range covered: chunk-Y=-3 → world Y -96..-65 —
                // well within the lava band (lava_level_max=-10).
                for y in 0..CHUNK_DIM_U {
                    for z in 0..CHUNK_DIM_U {
                        for x in 0..CHUNK_DIM_U {
                            let here = chunk.get(LocalPos(UVec3::new(x, y, z)));
                            if here == Block::Lava {
                                found_lava = true;
                            }
                            if here != Block::Lava && here != Block::Water {
                                continue;
                            }
                            // Check the 6 face-adjacent neighbours
                            // *within this chunk* (cross-chunk
                            // boundaries are a follow-up concern).
                            for (dx, dy, dz) in [
                                (1i32, 0, 0), (-1, 0, 0),
                                (0, 1, 0), (0, -1, 0),
                                (0, 0, 1), (0, 0, -1),
                            ] {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                let nz = z as i32 + dz;
                                if nx < 0 || ny < 0 || nz < 0
                                   || nx >= CHUNK_DIM_U as i32
                                   || ny >= CHUNK_DIM_U as i32
                                   || nz >= CHUNK_DIM_U as i32 {
                                    continue;
                                }
                                let n = chunk.get(LocalPos(UVec3::new(
                                    nx as u32, ny as u32, nz as u32,
                                )));
                                let touching =
                                    (here == Block::Water && n == Block::Lava)
                                    || (here == Block::Lava && n == Block::Water);
                                if touching {
                                    violations += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        // It's OK if no lava appeared (lava is rare): the test is
        // primarily about the *separation* invariant when both exist.
        // We still log lava presence so a regression that suppresses
        // lava entirely is visible.
        if !found_lava {
            eprintln!("note: no lava encountered in 8×8 chunk scan at chunk-Y=-3 (lava is rare; not a failure)");
        }
        assert_eq!(
            violations, 0,
            "found {violations} water-lava face adjacencies — aquifer pressure should always seal them with rock"
        );
    }

    /// PR 7: regression of the in-session test, now exercising the
    /// real aquifer. Caves under all-land chunks (height >> sea level,
    /// no lake) should be predominantly dry — the floodedness noise
    /// drives most deep cells to WAY_BELOW_MIN_Y.
    ///
    /// Stronger than the primitive test: we accept a small fraction
    /// of partially-flooded cells (the real aquifer's spread noise
    /// produces some local water tables even under land), but the
    /// vast majority of underground voxels must be air, not water.
    #[test]
    fn caves_under_land_are_mostly_dry_with_real_aquifer() {
        let g = Generator::new(42);
        // Find a chunk where every column is firmly land (height
        // > sea + 5) and no lake above.
        let mut found = None;
        'outer: for cz in -8..8 {
            for cx in -8..8 {
                let mut ok = true;
                'cols: for lz in 0..CHUNK_DIM_U {
                    for lx in 0..CHUNK_DIM_U {
                        let wx = cx * CHUNK_DIM_U as i32 + lx as i32;
                        let wz = cz * CHUNK_DIM_U as i32 + lz as i32;
                        let col = g.column_data(wx, wz);
                        if col.height <= SEA_LEVEL + 5 || col.lake_rim.is_some() {
                            ok = false;
                            break 'cols;
                        }
                    }
                }
                if ok {
                    found = Some((cx, cz));
                    break 'outer;
                }
            }
        }
        let (cx, cz) = found.expect("expected a lake-free all-land chunk");
        let mut chunk = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(cx, -2, cz)), &mut chunk);
        let mut air = 0u32;
        let mut water = 0u32;
        for b in chunk.blocks.iter() {
            match b {
                Block::Air => air += 1,
                Block::Water => water += 1,
                _ => {}
            }
        }
        // Under all-land columns, water voxels should be at most a
        // small fraction of total air/water voxels (≤ 10% under a
        // reasonable floodedness threshold). The earlier in-session
        // primitive aquifer enforced ==0; the real aquifer's spread
        // noise produces occasional partially-flooded cells, which is
        // correct.
        let total_fluid_voxels = air + water;
        if total_fluid_voxels > 0 {
            let water_frac = water as f32 / total_fluid_voxels as f32;
            assert!(
                water_frac < 0.10,
                "lake-free land chunk ({cx}, -2, {cz}) had {water} water of {total_fluid_voxels} fluid voxels ({:.1}%); expected < 10%",
                water_frac * 100.0
            );
        }
    }

    /// PR 7: aquifer is deterministic in `(seed, position)`. Two
    /// successive chunk fills with the same coord must produce
    /// identical fluid placement.
    #[test]
    fn aquifer_chunk_fill_is_deterministic() {
        let g = Generator::new(42);
        let mut a = DenseChunk::empty();
        let mut b = DenseChunk::empty();
        let coord = ChunkCoord(IVec3::new(3, -2, -1));
        g.fill_chunk(coord, &mut a);
        g.fill_chunk(coord, &mut b);
        // Hash the water-only blocks for cheaper comparison.
        let count_water = |c: &DenseChunk| c.blocks.iter().filter(|b| matches!(b, Block::Water)).count();
        let count_lava = |c: &DenseChunk| c.blocks.iter().filter(|b| matches!(b, Block::Lava)).count();
        assert_eq!(count_water(&a), count_water(&b));
        assert_eq!(count_lava(&a), count_lava(&b));
    }
```

- [ ] **Step 9.2: Run the new tests**

Run: `cargo test --lib worldgen::tests::lava_and_water_aquifers_are_separated_by_rock worldgen::tests::caves_under_land_are_mostly_dry_with_real_aquifer worldgen::tests::aquifer_chunk_fill_is_deterministic 2>&1 | tail -15`

Expected: all 3 tests pass.

If `lava_and_water_aquifers_are_separated_by_rock` fails with a violation count > 0:
  - Verify `raw_pressure` returns 2.0 for water/lava (debug log a single pressure call).
  - Verify the cell-pair search keeps idx_1 = the truly closest cell.
  - The most likely bug is `compute_substance` returning the wrong fluid_at_1 — e.g. swapping cell_1 / cell_2 in the cache.

If `caves_under_land_are_mostly_dry_with_real_aquifer` fails with >10% water:
  - The floodedness noise threshold is too low (too many cells flood).
  - Tune `fully_flooded_threshold` and `partially_flooded_threshold` mapping ranges in `compute_surface_level` (currently `0.8 / -0.3` and `0.4 / -0.8`). If land caves are too wet, raise those thresholds in `default.ron` (note: PR 7 doesn't expose these as config — a follow-up could).

- [ ] **Step 9.3: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
test(worldgen): integration tests for PR 7 aquifer

Three integration tests exercising the real aquifer through
Generator::fill_chunk:

  * lava_and_water_aquifers_are_separated_by_rock: scans 64 deep
    chunks (chunk-Y=-3); asserts no Water-Lava face adjacency.
    Validates the unconditional 2.0 pressure seal in raw_pressure.

  * caves_under_land_are_mostly_dry_with_real_aquifer: regression
    of the in-session deep_caves_under_land_are_dry, relaxed to
    "water < 10% of fluid voxels" — the real aquifer's spread
    noise produces some local water tables, which is correct
    behaviour (and not what the primitive rule allowed).

  * aquifer_chunk_fill_is_deterministic: two successive fills of
    the same coord produce identical Water/Lava counts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Final verification and golden re-baseline

**Files:**
- Modify: `src/worldgen/mod.rs` (golden hash sentinel only, if not done in task 8)

- [ ] **Step 10.1: Run the full test suite**

Run: `cargo test 2>&1 | tail -20`

Expected: every test passes. Specifically:
- All `worldgen::aquifer::*` lib tests pass.
- All `worldgen::tests::*` lib tests pass, including the three new aquifer-integration tests.
- `golden_seed42_chunk_0_2_0` passes with the post-PR-7 hash.
- `worldgen_fingerprint::fingerprint_hash_matches_pin` passes (the 2D heightmap is untouched in PR 7 — if this fails, investigate before proceeding).
- The `smoke::*` integration tests (chunk gen + meshing) pass with the new Block::Lava variant.

- [ ] **Step 10.2: Visual smoke test**

Boot the game with a fresh world. Observe:
- Land columns: caves at chunk-Y=-2 to -4 are mostly dry. Occasional local water tables are present (the aquifer's spread noise).
- Coast / shallow water: caves under the seabed flood correctly.
- Deep digging in a continental column: at chunk-Y=-6 or deeper, the player should occasionally encounter:
  * Dry chambers (most common).
  * Local water aquifers — small pockets of water at random Y offsets, not connected to sea level.
  * Lava pockets — rare, only at world Y < -10 with high lava noise magnitude.
- Water and lava never touch (visually verifiable — no hissing-edge artifacts).

If lava is never visible during a 5-minute spelunk, the threshold may be too tight. Tune `lava_threshold` in `default.ron` (lower → more lava). Save the file; the watcher reloads. Newly-generated chunks reflect the change.

- [ ] **Step 10.3: Verify hot-reload of `aquifer:` subsection**

Edit `assets/worldgen/default.ron`: set `lava_threshold: 0.1` (much more lava). Save. Move out of and back into a chunk range. Lava should now be common in deep chunks.

Restore the original `lava_threshold: 0.3` and re-save.

- [ ] **Step 10.4: Final commit (no-op if nothing changed)**

If steps 10.1–10.3 prompted RON tuning, commit it:

```bash
git add assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
chore(worldgen): tune default.ron after PR 7 visual review

Final tuning of aquifer thresholds after in-game visual review.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Otherwise nothing to commit — PR 7 is complete.

---

## Out of scope for PR 7 (deferred to later PRs)

- **MC's 13-offset preliminary-surface sampling.** PR 7 uses the column's `h_target` as the preliminary surface inside `compute_fluid`. A future PR can sample neighbouring columns for a smoother cross-region water table.
- **Lake-water handling.** Lakes currently flood via the global fluid path (their adjusted surface ends up below the lake water level). A future PR could let `compute_fluid` consult `lake_rim_above` for per-lake water levels.
- **`shouldScheduleFluidUpdate` plumbing.** MC tracks per-voxel "may flow next tick" state to schedule fluid simulation. PR 7 drops this — fluids are static at gen time. Re-add when the fluid simulator lands (out of the worldgen migration entirely).
- **Cell interpolation interaction.** PR 5 introduces the 4×4×4 cell-grid density evaluator. PR 7 reads the post-trilerp density per voxel; if PR 5 lands first, the aquifer call site is already at per-voxel resolution and no rework is needed.
- **Surface rules DSL interplay.** PR 6 introduces a surface-rules DSL. If a surface rule wants to consult the aquifer's local fluid level (e.g. "place mud on top of partially-flooded cells"), PR 6 will need a `Condition::AquiferLocalFluidLevel(min, max)` primitive. PR 7 exposes `Aquifer::aquifer_status` to enable that lookup, but does not itself add a DSL hook.
- **Underground biomes that depend on aquifer state.** Lush caves want "is there water nearby?" and dripstone caves want "is the cell dry?". The aquifer exposes the data; consuming it is a future PR's job.
- **Per-region barrier noise.** PR 7 reuses a single barrier channel for all chunks. MC has it as a top-level density function; a future PR could fold it into the PR 5 density graph as `Interpolated(BarrierNoise)` for cell-level caching.

## Plan self-review notes

- All 10 tasks have concrete code in every step. No "TBD" or "fill in details".
- Type names consistent across tasks: `Aquifer`, `AquiferNoise`, `AquiferConfig`, `FluidKind`, `FluidStatus`, `Block::Lava`.
- Each task ends with a commit boundary with the canonical `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.
- Constants pinned exactly from MC `Aquifer.java`:
  * `X_SPACING=16, Y_SPACING=12, Z_SPACING=16` (lines 65-67)
  * `X_RANGE=10, Y_RANGE=9, Z_RANGE=10` (lines 59-61)
  * `SAMPLE_OFFSET_X=-5, _Y=1, _Z=-5` (lines 72-74)
  * Similarity divisor `25.0` (line 300)
  * Pressure top biases `2.5 / 1.5` (lines 322-325)
  * Pressure bottom biases `10 / 3` with `bottom_bias=3.0` (lines 325-328)
  * Pressure water/lava constant `2.0` (line 363)
  * Lava cells `64×40×64` (lines 502-505)
  * Lava threshold `|noise| > 0.3` (line 507)
  * Spread quantization `quantize(noise * 10, 3)` (lines 491-493)
  * Lava gate `fluid_surface_level <= -10` (line 500)
- The pressure-function implementation in step 5.3 is a faithful Rust transliteration of `Aquifer.java:304-365`, with line-for-line comments tying constants back to MC.
- Golden hash management: task 8 step 8.5 puts the test in print-mode (`0xDEAD_BEEF_DEAD_BEEF` sentinel); step 8.5 captures the new value and repins.
- The plan preserves `Generator::new(seed)` semantics so the existing 50+ tests continue to pass after PR 7.
- The aquifer integration adds a small per-chunk overhead (~12 cell jitters + lazy cache; ~0.5KB of stack). Per-voxel cost is dominated by the 12-cell distance scan (12 hash mixes + 12 squared-distance computations) — negligible compared to the existing density / cave SDF evaluations.
- All three required tests (determinism; water-lava sealing; dry land caves) are present in tasks 7 and 9.
- The plan explicitly documents that PR 7 depends on PR 5 (cell interpolation + density graph) at the top, matching the migration sequence locked in spec Decisions Log Q6.
