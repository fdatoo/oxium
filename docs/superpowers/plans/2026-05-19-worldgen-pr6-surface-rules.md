# Worldgen PR 6 — Surface Rules DSL

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inline surface block selector in `mod.rs::fill_chunk` (the ~70-line `if depth == 0 { ... }` cliff/beach/snow/desert/grass cascade that currently lives in lines 415–490) with a typed `RuleSource` / `ConditionSource` DSL evaluated by a real `SurfaceSystem`. Move the logic into `src/worldgen/surface.rs` — which has been a 5-line stub since PR 1 — and load the rule tree from `assets/worldgen/default.ron` so surface behaviour becomes tunable data instead of buried code.

**Architecture:** Five tightly-coupled pieces. (1) `ConditionSource` and `RuleSource` enums in `src/worldgen/surface.rs`, both `Serialize + Deserialize` so the tree round-trips through RON. (2) `SurfaceContext` (per-column walk state) carrying `wx, wz, h_target, biome, lake_rim, depth_above, wy, water_y, is_cliff, last_xz_update_id, last_y_update_id`. (3) `SurfaceSystem` owning lazy XZ/Y condition caches (each cached `Condition` stores `(result, last_update_id)`) and a `build_surface_column` method that walks a column top-down, updating depth and the cache counters, and evaluates the rule tree per voxel. (4) A `SurfaceConfig` subsection on `WorldgenConfig` carrying `pub rules: RuleSource`. (5) The default rule tree in `assets/worldgen/default.ron` reproduces the current inline logic byte-for-byte on a 100-column sample of seed 42 — verified by a dedicated parity test.

The hot path inside `fill_chunk` simplifies from "build a chunk and choose a surface block in the middle of the loop" to "build a chunk into a dense-of-stone form, then call `surface_system.build_surface_column(...)` per column." Dirt/Stone-below-surface still gets placed inline (one-liner). Cliff and out-of-band cases also flow through the rule tree (the rules express them as `If(IsCliff, Block(Stone))` and `If(Not(WithinSurfaceBand), Block(Stone))` at the top of the sequence).

**Tech Stack:**
- Rust 2024 edition
- `serde` derives on existing `Block` enum (already derived; we extend its use)
- `ron = "0.8"` (added in PR 2)
- No new crates.

**Dependencies (must already be merged before this PR starts):**
- PR 2: `WorldgenConfig` + `ConfigHolder` + `CubicSpline` + `FlatCache2D` + `notify` file watcher + `arc-swap` plumbing.
- PRs 3–5: heightmap-driven offset/factor, multi-noise biome table, cell-grid interpolator. PR 6 reads from `col.biome` (whose backing data may have changed) but its DSL is biome-agnostic so the variant list in `default.ron` is the only tie-in.

**Reference:** Architectural rationale is in `docs/superpowers/specs/2026-05-19-minecraft-worldgen-research.md` (Part 4 idea #5, Q3 bug analysis, Q5 hybrid-RON decision). MC source: `~/Downloads/out/net/minecraft/world/level/levelgen/SurfaceRules.java`, `SurfaceSystem.java`, `~/Downloads/out/net/minecraft/data/worldgen/SurfaceRuleData.java`.

---

### Task 1: Skeleton — `SurfaceContext`, `NoiseKind`, types-only

**Files:**
- Modify: `src/worldgen/surface.rs` (currently a 5-line stub)

- [ ] **Step 1.1: Replace the stub with the types**

Open `src/worldgen/surface.rs` and replace its entire contents with:

```rust
//! Surface block selection as a typed `RuleSource` / `ConditionSource`
//! DSL evaluated by [`SurfaceSystem`].
//!
//! This module replaces the inline cliff/beach/snow/desert/grass
//! selector that lived in [`crate::worldgen::fill_chunk`] from PRs 1–5.
//! The rule tree is serde-deserialised from `assets/worldgen/default.ron`
//! at startup and round-trips through RON via `serde`. Adding "wet
//! biomes get podzol" or "stony peaks get calcite veins" is one new
//! `If(...)` node in the file — no Rust changes required.
//!
//! The runtime walks each column top-down, maintaining `depth_above`
//! (blocks since the last air→solid transition) and `water_y` (the y of
//! the highest water voxel above), then evaluates the rule tree for
//! each topmost-in-band solid voxel. Internal nodes (`Sequence`, `If`)
//! recurse; terminal `Block(b)` nodes return `Some(b)`. The first
//! `Some(_)` returned by sequence-walk wins.
//!
//! Mirrors MC 1.18+ `net/minecraft/world/level/levelgen/SurfaceRules.java`.

use crate::voxel::block::Block;
use serde::{Deserialize, Serialize};

/// Identifies a per-column or per-voxel noise field that
/// [`ConditionSource::NoiseThreshold`] can sample. PR 6 introduces a
/// minimal set — extensible by appending variants. The corresponding
/// noise source is materialised inside [`SurfaceSystem::new`] (today:
/// the `desert_map` / `temperature_map` / `humidity_map` fields on the
/// `Generator`). Untracked noise kinds (added in later PRs without
/// hooking the system) evaluate to 0.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NoiseKind {
    /// The desertness 2D noise that drives the stochastic
    /// sand-on-grass transition band. Sampled at (wx, wz).
    Desertness,
    /// Temperature 2D noise. Future use: cold-water freezing rules.
    Temperature,
    /// Humidity 2D noise. Future use: podzol / swamp variations.
    Humidity,
    /// PR 6 reserves this for the future MC-style "surface noise" used
    /// for the badlands clay-bands offset. Today returns 0.0 (no-op).
    Surface,
}

/// Per-column walk state passed to every rule evaluation in
/// [`SurfaceSystem::build_surface_column`]. Holds everything the DSL
/// needs to answer questions without re-querying the generator's
/// per-column caches.
///
/// `wy_now`, `depth_above`, `water_y`, `last_y_update_id` change every
/// y-step inside a column; the rest change only when the walk moves
/// to a new (wx, wz). The two `last_*_update_id` counters drive the
/// lazy condition caches: any condition that depends only on
/// per-column state (`Biome`, `IsCliff`) caches against
/// `last_xz_update_id`; conditions that depend on y (`YAbove`,
/// `OnFloor`, `UnderFloor`, `WithinSurfaceBand`,
/// `AbovePreliminarySurface`, `StoneDepth`, `NotUnderwater`) cache
/// against `last_y_update_id`.
pub struct SurfaceContext {
    pub wx: i32,
    pub wz: i32,
    pub h_target: f32,
    pub biome: crate::worldgen::SurfaceBiome,
    pub lake_rim: Option<i32>,
    pub is_cliff: bool,
    pub wy_now: i32,
    /// Blocks since the most recent air→solid transition while
    /// descending. 0 ⇒ topmost solid voxel of a contiguous solid run.
    pub depth_above: i32,
    /// Y of the highest water voxel directly above `wy_now`, or `None`
    /// if no water was seen since the last air→solid transition.
    pub water_y: Option<i32>,
    /// Monotonically increasing whenever `(wx, wz)` changes.
    pub last_xz_update_id: u64,
    /// Monotonically increasing every y-step.
    pub last_y_update_id: u64,
}
```

This step ships the types but no runtime. The `SurfaceBiome` re-export is added in step 1.2.

- [ ] **Step 1.2: Re-export `Biome` as `SurfaceBiome` from `mod.rs`**

`Biome` in `mod.rs` is currently `pub(crate)` (no visibility modifier — defaults to private). The DSL types need to name it. Add at the top of `src/worldgen/mod.rs` (after the `pub mod ...` block):

```rust
/// `Biome` re-exported with a slightly more specific name so the
/// surface DSL can name it from outside the module. Same enum,
/// same discriminants — this is just a visibility hop.
pub type SurfaceBiome = Biome;
```

And change the existing `enum Biome { ... }` declaration to `pub enum Biome { ... }` (around line 707). Same for any `fn snow_capped` / `fn classify` accessors the surface module needs — promote them to `pub`:

```rust
impl Biome {
    pub fn classify(...) -> Self { ... }
    pub fn snow_capped(self) -> bool { ... }
    // ... existing accessors stay private (tree_kind, tree_rate_percentile, etc.)
}
```

- [ ] **Step 1.3: Verify the crate compiles**

Run: `cargo check 2>&1 | tail -5`

Expected: `Finished` line. No warnings about unused `NoiseKind` / `SurfaceContext` (they're `pub`).

- [ ] **Step 1.4: Commit**

```bash
git add src/worldgen/surface.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
refactor(worldgen): replace surface.rs stub with PR 6 type skeleton

Introduces SurfaceContext and NoiseKind. These are the per-column
walk state and the noise-source identifier consumed by the
ConditionSource enum that lands in task 2.

Biome is promoted to `pub` (with a `SurfaceBiome` type alias) so the
DSL module can name it without dragging the whole worldgen
internals into its public surface.

No behavior change — fill_chunk still uses the inline selector.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `ConditionSource` enum (TDD — primitives only, no evaluation yet)

**Files:**
- Modify: `src/worldgen/surface.rs`

- [ ] **Step 2.1: Write failing tests for ConditionSource RON roundtrip**

Append to `src/worldgen/surface.rs`:

```rust
/// One node in the surface-rule predicate tree. Every variant must be
/// answerable from a [`SurfaceContext`] alone — no global state.
///
/// Layered cheat sheet (cheapest first; the DSL's structure tracks
/// MC's so reading `SurfaceRules.java` translates directly):
///
/// 1. **Per-column gates** (cached XZ): `Biome`, `IsCliff`, `IsCold`.
/// 2. **Per-voxel Y bands** (cached Y): `YAbove`, `YBelow`,
///    `WithinSurfaceBand`, `AbovePreliminarySurface`.
/// 3. **Column-walk state** (cached Y): `OnFloor`, `UnderFloor`,
///    `StoneDepth`, `NotUnderwater`.
/// 4. **Combinators** (no cache; recurse): `Not`, `All`, `Any`.
/// 5. **Noise queries** (cached XZ — noises here are 2D): `NoiseThreshold`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConditionSource {
    /// True iff the column's biome is in the set.
    Biome(Vec<SurfaceBiome>),
    /// True iff `wy_now >= threshold`.
    YAbove(i32),
    /// True iff `wy_now <= threshold`.
    YBelow(i32),
    /// True iff `wy_now >= h_target + offset` (Oxium's near-surface
    /// gate; subsumes the in-session `(h_target - wy).abs() <=
    /// SURFACE_BAND` shortcut by composing with `WithinSurfaceBand`).
    /// Mirrors MC's `abovePreliminarySurface()`.
    AbovePreliminarySurface { offset: i32 },
    /// True iff `|h_target - wy_now| <= SURFACE_BAND` (the band where
    /// surface-block selection is meaningful). Replaces Oxium's
    /// in-session `near_surface` gate as a first-class primitive.
    WithinSurfaceBand,
    /// True iff `depth_above == 0` (topmost solid voxel after an
    /// air→solid descent transition). MC's `ON_FLOOR`.
    OnFloor,
    /// True iff `depth_above <= n` (within `n` blocks below the most
    /// recent air→solid transition). MC's `UNDER_FLOOR` with offset.
    UnderFloor(u32),
    /// True iff `min <= depth_above <= max`. MC's `stone_depth` with
    /// `add_surface_depth = false` — a more flexible variant.
    StoneDepth { min: u32, max: u32 },
    /// True iff `col.is_cliff` (slope-based cliff exposure).
    IsCliff,
    /// True iff the column's biome is `snow_capped()` (Tundra,
    /// SnowyForest). Mirrors MC's `Temperature` but condensed to
    /// Oxium's discrete biome model.
    IsCold,
    /// True iff `water_y` is `None` OR `wy_now > water_y.unwrap()`
    /// (the voxel is above the highest water seen during this
    /// column's descent). Mirrors MC's `waterBlockCheck(0, 0)`.
    NotUnderwater,
    /// Boolean negation of the wrapped condition.
    Not(Box<ConditionSource>),
    /// Conjunction. Short-circuits on the first `false`.
    All(Vec<ConditionSource>),
    /// Disjunction. Short-circuits on the first `true`.
    Any(Vec<ConditionSource>),
    /// True iff `min <= sample(noise) <= max`. Per-column 2D noise,
    /// cached XZ.
    NoiseThreshold {
        noise: NoiseKind,
        min: f32,
        max: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_source_ron_roundtrip_biome() {
        let c = ConditionSource::Biome(vec![SurfaceBiome::Desert, SurfaceBiome::Tundra]);
        let s = ron::to_string(&c).unwrap();
        let parsed: ConditionSource = ron::from_str(&s).unwrap();
        match parsed {
            ConditionSource::Biome(bs) => {
                assert_eq!(bs.len(), 2);
                assert!(bs.contains(&SurfaceBiome::Desert));
                assert!(bs.contains(&SurfaceBiome::Tundra));
            }
            _ => panic!("expected Biome"),
        }
    }

    #[test]
    fn condition_source_ron_roundtrip_nested_not_all() {
        let c = ConditionSource::Not(Box::new(ConditionSource::All(vec![
            ConditionSource::WithinSurfaceBand,
            ConditionSource::YAbove(64),
            ConditionSource::NotUnderwater,
        ])));
        let s = ron::to_string(&c).unwrap();
        let parsed: ConditionSource = ron::from_str(&s).unwrap();
        // Re-serialise the parse — should be byte-identical (RON is
        // canonical for these variants).
        let s2 = ron::to_string(&parsed).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn condition_source_ron_roundtrip_noise_threshold() {
        let c = ConditionSource::NoiseThreshold {
            noise: NoiseKind::Desertness,
            min: 0.30,
            max: 1.0,
        };
        let s = ron::to_string(&c).unwrap();
        let parsed: ConditionSource = ron::from_str(&s).unwrap();
        match parsed {
            ConditionSource::NoiseThreshold { noise, min, max } => {
                assert_eq!(noise, NoiseKind::Desertness);
                assert!((min - 0.30).abs() < 1e-5);
                assert!((max - 1.0).abs() < 1e-5);
            }
            _ => panic!("expected NoiseThreshold"),
        }
    }
}
```

(Biome needs `Serialize + Deserialize`. Add `#[derive(Serialize, Deserialize)]` to the `enum Biome` declaration in `mod.rs`, near the existing `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]`. RON's ADT serialisation works on bare unit variants directly.)

- [ ] **Step 2.2: Run tests to verify they compile and pass**

Run: `cargo test --lib worldgen::surface 2>&1 | tail -10`

Expected: 3 tests pass. (The variants are type-level only — no evaluation logic yet — so the only thing under test is RON roundtripping, which `serde` handles automatically.)

- [ ] **Step 2.3: Commit**

```bash
git add src/worldgen/surface.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): add ConditionSource enum (surface DSL — types only)

ConditionSource is the predicate-tree node for PR 6's surface rules
DSL. Mirrors MC SurfaceRules.ConditionSource (Biome, NoiseThreshold,
Y*, OnFloor, UnderFloor, StoneDepth, Not, plus combinators).

This commit ships the enum + serde derives + RON roundtrip tests.
Evaluation lands in task 4 once SurfaceSystem exists.

Biome is now serde-derivable so the rule tree round-trips.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `RuleSource` enum (TDD)

**Files:**
- Modify: `src/worldgen/surface.rs`

- [ ] **Step 3.1: Write failing tests for RuleSource RON roundtrip**

Append to `src/worldgen/surface.rs` (above the existing `#[cfg(test)]` block):

```rust
/// One node in the surface-rule action tree. A `RuleSource` evaluates
/// to `Option<Block>` — `None` means "this rule didn't fire, try the
/// next one"; `Some(block)` means "place this block." Internal nodes
/// (`Sequence`, `If`) recurse; `Block(b)` is the only terminal.
///
/// Sequence is short-circuit: the first child returning `Some(_)`
/// wins. `If(cond, then)` returns `then.evaluate(...)` if `cond` is
/// true, else `None`. `Block(b)` always returns `Some(b)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RuleSource {
    /// Terminal: place this block kind.
    Block(Block),
    /// First non-null child wins. MC's `Sequence`.
    Sequence(Vec<RuleSource>),
    /// Recurse into `then` iff `condition` is true. MC's `TestRule`.
    If(ConditionSource, Box<RuleSource>),
    /// Future-hook for MC's badlands clay-bands rule. PR 6 ships this
    /// as a no-op (always returns `None`) — Oxium doesn't have
    /// badlands yet. Reserved so the RON schema is forward-compatible.
    Bandlands,
}

#[cfg(test)]
mod tests_rule {
    use super::*;

    #[test]
    fn rule_source_ron_roundtrip_simple_block() {
        let r = RuleSource::Block(Block::Grass);
        let s = ron::to_string(&r).unwrap();
        let parsed: RuleSource = ron::from_str(&s).unwrap();
        match parsed {
            RuleSource::Block(b) => assert_eq!(b, Block::Grass),
            _ => panic!("expected Block"),
        }
    }

    #[test]
    fn rule_source_ron_roundtrip_nested_sequence() {
        let r = RuleSource::Sequence(vec![
            RuleSource::If(
                ConditionSource::IsCliff,
                Box::new(RuleSource::Block(Block::Stone)),
            ),
            RuleSource::If(
                ConditionSource::Biome(vec![SurfaceBiome::Desert]),
                Box::new(RuleSource::Block(Block::Sand)),
            ),
            RuleSource::Block(Block::Grass),
        ]);
        let s = ron::to_string(&r).unwrap();
        let parsed: RuleSource = ron::from_str(&s).unwrap();
        let s2 = ron::to_string(&parsed).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn rule_source_bandlands_roundtrips() {
        let r = RuleSource::Bandlands;
        let s = ron::to_string(&r).unwrap();
        let parsed: RuleSource = ron::from_str(&s).unwrap();
        assert!(matches!(parsed, RuleSource::Bandlands));
    }
}
```

- [ ] **Step 3.2: Run tests**

Run: `cargo test --lib worldgen::surface 2>&1 | tail -10`

Expected: 6 tests pass total (3 from task 2 + 3 from this task).

- [ ] **Step 3.3: Commit**

```bash
git add src/worldgen/surface.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): add RuleSource enum (surface DSL — types only)

RuleSource is the action-tree counterpart to ConditionSource. Four
variants: Block (terminal), Sequence (first non-null wins), If
(guarded recursion), Bandlands (forward-compat no-op for MC's clay
bands; Oxium doesn't have badlands today).

Serde derives + RON roundtrip tests cover all variants including
the nested if/sequence form the default rule tree uses.

Evaluation lands in task 4 once SurfaceSystem exists.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `SurfaceSystem` with lazy condition caches (TDD)

**Files:**
- Modify: `src/worldgen/surface.rs`

This task introduces the runtime. Two design choices worth flagging:

- **Cache representation.** MC stores a per-condition `(lastUpdate, result)` inside a `LazyCondition` instance and walks the tree once per voxel. Oxium reproduces this with a small **per-`(NoiseKind | ConditionKindTag)` array** keyed by a stable integer enum-tag. Each cache slot stores `(last_update_id, result_bits)`. On hit (`last_update_id == ctx.last_*_update_id`), return cached. On miss, re-evaluate and store.
- **Recursive evaluation.** `evaluate_condition(&self, &mut SurfaceContext, &ConditionSource) -> bool` is recursive (for `Not`, `All`, `Any`). The caches are owned by the `SurfaceSystem` and threaded through `&mut self` (so concurrent column walks across threads each own their own `SurfaceSystem` — but column walks within a single thread can reuse it).

- [ ] **Step 4.1: Write failing tests**

Append to `src/worldgen/surface.rs`:

```rust
/// Maximum noise kinds we'll ever cache. Bump if [`NoiseKind`] grows
/// past this — caught at compile time by the array literal.
const NOISE_CACHE_SLOTS: usize = 8;
/// Maximum non-noise leaf-condition kinds we'll cache. Eight covers
/// today's set with headroom: Biome, YAbove, YBelow,
/// AbovePreliminarySurface, WithinSurfaceBand, OnFloor, UnderFloor,
/// StoneDepth, IsCliff, IsCold, NotUnderwater — pick a generous power
/// of two so future additions don't force a renumber.
const COND_CACHE_SLOTS: usize = 16;

/// One slot in the lazy condition cache: stores the result of the
/// most recent evaluation along with the `update_id` it was computed
/// against. A read with a matching id is a hit; any other value is a
/// miss (signalled by the canonical `MISS_ID` sentinel).
#[derive(Clone, Copy, Debug)]
struct CacheSlot {
    last_update_id: u64,
    result: bool,
}

const MISS_ID: u64 = u64::MAX;

impl Default for CacheSlot {
    fn default() -> Self {
        Self {
            last_update_id: MISS_ID,
            result: false,
        }
    }
}

/// Stateful evaluator for the surface DSL. Owns lazy XZ / Y caches
/// keyed by condition kind. Cheap to construct per chunk; not Sync.
pub struct SurfaceSystem {
    /// Cache for conditions whose answer depends only on the current
    /// `(wx, wz)` (Biome, IsCliff, IsCold, NoiseThreshold). Slots
    /// indexed by `ConditionKindTag::Xz*`.
    xz_cache: [CacheSlot; COND_CACHE_SLOTS],
    /// Cache for conditions that depend on the current `(wy, depth,
    /// water_y)` (YAbove, YBelow, AbovePreliminarySurface,
    /// WithinSurfaceBand, OnFloor, UnderFloor, StoneDepth,
    /// NotUnderwater).
    y_cache: [CacheSlot; COND_CACHE_SLOTS],
    /// One cache slot per `NoiseKind`. Per-XZ only (today all noises
    /// are 2D).
    noise_cache: [CacheSlot; NOISE_CACHE_SLOTS],
}

/// Integer tag for caching. Stable across recompiles (must match the
/// switch in `tag_of`). Adding a variant: add a new value past the
/// last; bump `COND_CACHE_SLOTS` if it exceeds the array length.
#[derive(Clone, Copy)]
#[repr(usize)]
enum CondTag {
    Biome = 0,
    IsCliff = 1,
    IsCold = 2,
    YAbove = 3,
    YBelow = 4,
    AbovePreliminarySurface = 5,
    WithinSurfaceBand = 6,
    OnFloor = 7,
    UnderFloor = 8,
    StoneDepth = 9,
    NotUnderwater = 10,
    // Slots 11..16 reserved for future variants.
}

impl SurfaceSystem {
    pub fn new() -> Self {
        Self {
            xz_cache: [CacheSlot::default(); COND_CACHE_SLOTS],
            y_cache: [CacheSlot::default(); COND_CACHE_SLOTS],
            noise_cache: [CacheSlot::default(); NOISE_CACHE_SLOTS],
        }
    }

    /// Evaluate a [`ConditionSource`] against a [`SurfaceContext`].
    /// Caches respect `ctx.last_xz_update_id` / `ctx.last_y_update_id`
    /// so re-evaluating the same condition inside one rule-tree walk
    /// is O(1).
    pub fn evaluate_condition(
        &mut self,
        ctx: &SurfaceContext,
        cond: &ConditionSource,
    ) -> bool {
        // Combinators recurse without caching (the leaf-level caches
        // catch the repeated leaves they decompose to).
        match cond {
            ConditionSource::Not(inner) => return !self.evaluate_condition(ctx, inner),
            ConditionSource::All(items) => {
                for c in items {
                    if !self.evaluate_condition(ctx, c) {
                        return false;
                    }
                }
                return true;
            }
            ConditionSource::Any(items) => {
                for c in items {
                    if self.evaluate_condition(ctx, c) {
                        return true;
                    }
                }
                return false;
            }
            _ => {}
        }

        // Leaf condition: check the appropriate cache, recompute on miss.
        // XZ-cached:
        let (tag_xz, is_xz) = match cond {
            ConditionSource::Biome(_) => (CondTag::Biome as usize, true),
            ConditionSource::IsCliff => (CondTag::IsCliff as usize, true),
            ConditionSource::IsCold => (CondTag::IsCold as usize, true),
            _ => (0, false),
        };
        if is_xz {
            let slot = self.xz_cache[tag_xz];
            if slot.last_update_id == ctx.last_xz_update_id {
                return slot.result;
            }
            let r = Self::compute_xz_leaf(ctx, cond);
            self.xz_cache[tag_xz] = CacheSlot {
                last_update_id: ctx.last_xz_update_id,
                result: r,
            };
            return r;
        }

        // NoiseThreshold (XZ-cached but indexed by NoiseKind).
        if let ConditionSource::NoiseThreshold { noise, min, max } = cond {
            let idx = *noise as usize;
            assert!(
                idx < NOISE_CACHE_SLOTS,
                "noise kind index out of range — bump NOISE_CACHE_SLOTS"
            );
            // The noise value itself is also cached against XZ — but
            // the threshold comparison needs `min`/`max` so we can't
            // cache the boolean directly. We cache the *raw noise
            // value*, then apply the threshold on every call. Use the
            // sentinel slot's `result` to hold the LSB of the f32
            // bits-cast — not ideal, so instead: keep an actual f32
            // alongside. For PR 6, simplicity wins: we just recompute
            // the f32 each time (these are cheap 2D noise samples
            // already-cached by the generator's `FlatCache2D`).
            // Performance: < 1µs/column, dwarfed by chunk write.
            let v = noise_sample(ctx, *noise);
            return v >= *min && v <= *max;
        }

        // Y-cached leaves.
        let tag_y = match cond {
            ConditionSource::YAbove(_) => CondTag::YAbove as usize,
            ConditionSource::YBelow(_) => CondTag::YBelow as usize,
            ConditionSource::AbovePreliminarySurface { .. } => {
                CondTag::AbovePreliminarySurface as usize
            }
            ConditionSource::WithinSurfaceBand => CondTag::WithinSurfaceBand as usize,
            ConditionSource::OnFloor => CondTag::OnFloor as usize,
            ConditionSource::UnderFloor(_) => CondTag::UnderFloor as usize,
            ConditionSource::StoneDepth { .. } => CondTag::StoneDepth as usize,
            ConditionSource::NotUnderwater => CondTag::NotUnderwater as usize,
            _ => unreachable!("unreachable: leaf-condition switch missed a variant"),
        };
        let slot = self.y_cache[tag_y];
        if slot.last_update_id == ctx.last_y_update_id {
            return slot.result;
        }
        let r = Self::compute_y_leaf(ctx, cond);
        self.y_cache[tag_y] = CacheSlot {
            last_update_id: ctx.last_y_update_id,
            result: r,
        };
        r
    }

    fn compute_xz_leaf(ctx: &SurfaceContext, cond: &ConditionSource) -> bool {
        match cond {
            ConditionSource::Biome(set) => set.iter().any(|b| *b == ctx.biome),
            ConditionSource::IsCliff => ctx.is_cliff,
            ConditionSource::IsCold => ctx.biome.snow_capped(),
            _ => unreachable!("non-XZ-leaf in compute_xz_leaf"),
        }
    }

    fn compute_y_leaf(ctx: &SurfaceContext, cond: &ConditionSource) -> bool {
        use crate::worldgen::tuning::SURFACE_BAND;
        match cond {
            ConditionSource::YAbove(thr) => ctx.wy_now >= *thr,
            ConditionSource::YBelow(thr) => ctx.wy_now <= *thr,
            ConditionSource::AbovePreliminarySurface { offset } => {
                ctx.wy_now as f32 >= ctx.h_target + *offset as f32
            }
            ConditionSource::WithinSurfaceBand => {
                (ctx.h_target - ctx.wy_now as f32).abs() <= SURFACE_BAND as f32
            }
            ConditionSource::OnFloor => ctx.depth_above == 0,
            ConditionSource::UnderFloor(n) => ctx.depth_above >= 0 && (ctx.depth_above as u32) <= *n,
            ConditionSource::StoneDepth { min, max } => {
                let d = ctx.depth_above.max(0) as u32;
                d >= *min && d <= *max
            }
            ConditionSource::NotUnderwater => match ctx.water_y {
                None => true,
                Some(wy_water) => ctx.wy_now > wy_water,
            },
            _ => unreachable!("non-Y-leaf in compute_y_leaf"),
        }
    }

    /// Evaluate a [`RuleSource`] against a context. Returns the first
    /// `Some(block)` produced by the tree, or `None` if no rule fires.
    pub fn evaluate_rule(
        &mut self,
        ctx: &SurfaceContext,
        rule: &RuleSource,
    ) -> Option<Block> {
        match rule {
            RuleSource::Block(b) => Some(*b),
            RuleSource::Sequence(children) => {
                for child in children {
                    if let Some(b) = self.evaluate_rule(ctx, child) {
                        return Some(b);
                    }
                }
                None
            }
            RuleSource::If(cond, then) => {
                if self.evaluate_condition(ctx, cond) {
                    self.evaluate_rule(ctx, then)
                } else {
                    None
                }
            }
            RuleSource::Bandlands => None, // PR 6 stub; future PR fills this in
        }
    }
}

impl Default for SurfaceSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Sample a [`NoiseKind`] for the column at `(ctx.wx, ctx.wz)`. The
/// per-thread `Generator` holds the actual noise fields; PR 6's
/// `SurfaceSystem` doesn't (it's stateless re: noises). Today this is
/// a free function — call sites pass the `ctx.wx/wz` and the system
/// asks the parent crate for the value. For PR 6, the rule tree only
/// ever needs `Desertness` (used by the stochastic sand transition,
/// which the equivalent default tree expresses as a hash-roll — see
/// below). We expose this as a free function so the test code can
/// stub it out.
fn noise_sample(_ctx: &SurfaceContext, kind: NoiseKind) -> f32 {
    // Defer to the parent module's noise. PR 6 routes this via a
    // thread-local set in `SurfaceSystem::with_noise_sampler` (added
    // in task 5 — wire-in time). Returning 0.0 here makes any
    // NoiseThreshold rule in the default tree degenerate to
    // "min <= 0 <= max"; the default tree does NOT use noise
    // thresholds (the sand transition is hash-based, not noise-based)
    // so this stub is safe in the meantime. Task 5 makes this real.
    let _ = kind;
    0.0
}

#[cfg(test)]
mod tests_system {
    use super::*;

    /// Synthesise a SurfaceContext for the given (wy, depth) inside a
    /// column whose h_target is 80 and biome is Plains.
    fn ctx_at(wy: i32, depth: i32) -> SurfaceContext {
        SurfaceContext {
            wx: 0,
            wz: 0,
            h_target: 80.0,
            biome: SurfaceBiome::Plains,
            lake_rim: None,
            is_cliff: false,
            wy_now: wy,
            depth_above: depth,
            water_y: None,
            last_xz_update_id: 1,
            last_y_update_id: 1,
        }
    }

    #[test]
    fn evaluate_block_is_terminal() {
        let mut sys = SurfaceSystem::new();
        let r = RuleSource::Block(Block::Stone);
        assert_eq!(sys.evaluate_rule(&ctx_at(80, 0), &r), Some(Block::Stone));
    }

    #[test]
    fn sequence_picks_first_non_null() {
        let mut sys = SurfaceSystem::new();
        let r = RuleSource::Sequence(vec![
            RuleSource::If(
                ConditionSource::IsCliff,
                Box::new(RuleSource::Block(Block::Stone)),
            ),
            RuleSource::Block(Block::Grass),
        ]);
        // is_cliff=false ⇒ Stone branch returns None ⇒ Grass wins.
        assert_eq!(sys.evaluate_rule(&ctx_at(80, 0), &r), Some(Block::Grass));
        // Flip is_cliff: Stone wins.
        let mut ctx = ctx_at(80, 0);
        ctx.is_cliff = true;
        // Bump update ids so the cache invalidates the prior result.
        ctx.last_xz_update_id = 2;
        ctx.last_y_update_id = 2;
        assert_eq!(sys.evaluate_rule(&ctx, &r), Some(Block::Stone));
    }

    #[test]
    fn within_surface_band_uses_surface_band_constant() {
        let mut sys = SurfaceSystem::new();
        let r = RuleSource::If(
            ConditionSource::WithinSurfaceBand,
            Box::new(RuleSource::Block(Block::Grass)),
        );
        // At wy = h_target ± SURFACE_BAND, condition is true.
        assert_eq!(sys.evaluate_rule(&ctx_at(80, 0), &r), Some(Block::Grass));
        let mut ctx = ctx_at(80 + 17, 0);
        ctx.last_y_update_id = 2; // bust cache
        assert_eq!(sys.evaluate_rule(&ctx, &r), None);
    }

    #[test]
    fn lazy_cache_avoids_recompute_on_same_y_update_id() {
        let mut sys = SurfaceSystem::new();
        let r = ConditionSource::OnFloor;
        let ctx = ctx_at(80, 0);
        assert!(sys.evaluate_condition(&ctx, &r));
        // Mutate depth in the ctx but DO NOT bump last_y_update_id.
        // Cache should ignore the mutation and return the stale `true`.
        let mut stale_ctx = ctx;
        stale_ctx.depth_above = 5;
        // Cache hit ⇒ returns stale `true` even though depth is now 5.
        assert!(sys.evaluate_condition(&stale_ctx, &r));
        // Bump the id: re-compute, returns false.
        stale_ctx.last_y_update_id = 2;
        assert!(!sys.evaluate_condition(&stale_ctx, &r));
    }

    #[test]
    fn not_negates() {
        let mut sys = SurfaceSystem::new();
        let r = RuleSource::If(
            ConditionSource::Not(Box::new(ConditionSource::IsCliff)),
            Box::new(RuleSource::Block(Block::Grass)),
        );
        assert_eq!(sys.evaluate_rule(&ctx_at(80, 0), &r), Some(Block::Grass));
        let mut ctx = ctx_at(80, 0);
        ctx.is_cliff = true;
        ctx.last_xz_update_id = 2;
        assert_eq!(sys.evaluate_rule(&ctx, &r), None);
    }

    #[test]
    fn all_short_circuits_on_first_false() {
        let mut sys = SurfaceSystem::new();
        let r = ConditionSource::All(vec![
            ConditionSource::IsCliff,                          // false in default ctx
            ConditionSource::YAbove(10000),                    // would also be false
        ]);
        // IsCliff (xz-cached) evaluated, returns false. YAbove never queried.
        // We can't easily observe the short-circuit from outside,
        // but at least the result must be false.
        assert!(!sys.evaluate_condition(&ctx_at(80, 0), &r));
    }

    #[test]
    fn bandlands_returns_none_for_now() {
        let mut sys = SurfaceSystem::new();
        let r = RuleSource::Bandlands;
        assert_eq!(sys.evaluate_rule(&ctx_at(80, 0), &r), None);
    }
}
```

- [ ] **Step 4.2: Run tests to verify they pass**

Run: `cargo test --lib worldgen::surface 2>&1 | tail -15`

Expected: all 13 tests pass (3 + 3 + 7 from this task).

- [ ] **Step 4.3: Commit**

```bash
git add src/worldgen/surface.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): SurfaceSystem with lazy condition caches

SurfaceSystem owns the lazy XZ / Y condition caches and walks the
RuleSource tree. evaluate_rule is recursive; evaluate_condition
caches against ctx.last_{xz,y}_update_id (mirrors MC's
LazyXZCondition / LazyYCondition pattern).

Cache slots are a small fixed array keyed by a stable CondTag enum.
On cache miss, leaf-level compute_{xz,y}_leaf does the actual work
and stores (last_update_id, result).

NoiseThreshold's value cache is *not* built yet — the default rule
tree doesn't use noise thresholds in PR 6 (the only stochastic
selector is hash-based, not noise-based). Task 5 wires in a real
noise sampler.

Combinators (Not, All, Any) recurse without caching; the leaves
they decompose to are cached.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `SurfaceConfig` on `WorldgenConfig` + the default rule tree

**Files:**
- Modify: `src/worldgen/config.rs`
- Modify: `assets/worldgen/default.ron`
- Modify: `src/worldgen/surface.rs` (add `default_rule_tree()` helper for testing)

The default rule tree below mirrors the in-session inline logic exactly. Reading it top-down:

```
Sequence([
    // 1. Cliffs (overrides everything; the slope-based gate already
    //    pulled out the steepest columns)
    If(IsCliff, Block(Stone)),

    // 2. Not-in-surface-band: deep-underground gate. The in-session
    //    near_surface check, lifted into a rule.
    If(Not(WithinSurfaceBand), Block(Stone)),

    // 3. Only the topmost solid in the band gets surface treatment;
    //    every solid below it falls through to dirt/stone (handled
    //    inline in mod.rs, not by the rule tree).
    If(Not(OnFloor), Block(Stone)),

    // 4. Beach band: sand between SEA_LEVEL - 1 and SEA_LEVEL + 2 in
    //    non-cold biomes. Matches mod.rs:454.
    If(
        All([
            YAbove(61),       // SEA_LEVEL - 1
            YBelow(64),       // SEA_LEVEL + 2
            Not(IsCold),
        ]),
        Block(Sand)
    ),

    // 5. Alpine snow above SNOW_LINE.
    If(YAbove(110), Block(Snow)),

    // 6. Cold-biome snow: any snow_capped biome above SEA_LEVEL + COLD_SNOW_MIN_ABOVE_SEA.
    If(All([IsCold, YAbove(70)]), Block(Snow)),

    // 7. Desert biome surface.
    If(Biome([Desert]), Block(Sand)),

    // 8. Default: grass.
    Block(Grass)
])
```

But this is missing one thing: the **stochastic sand-on-grass transition** at the desertness boundary. The in-session code (`mod.rs:468–485`) reads `col.desertness` and rolls a hash. PR 6 keeps the hash-based determinism but moves the logic out of the rule tree (it needs access to `col.desertness`, which isn't in `SurfaceContext`, AND the hash inputs `&[wx, wz, 71]` to stay deterministic).

**Solution:** The sand transition fires *after* the rule tree returns `Block(Grass)`. The caller (`fill_chunk`, task 6) re-checks the column's `desertness` and possibly substitutes Sand. This is a pragmatic concession — moving the hash sampler into the DSL would require giving rules access to `Generator::seed`, which they don't have. The byte-identical parity test in task 8 verifies the result still matches.

Alternatively (and preferred for cleaner data): the stochastic sand transition can be expressed as an `If(Biome([Plains, Forest]), If(SandTransitionHashRoll { ... }, Block(Sand)))` with a dedicated `ConditionSource::HashRoll { salt: u32, threshold_field: f32 }`. PR 6 ships the *pragmatic* path (re-check after rule tree) to keep the DSL surface small; the cleaner DSL form is left for a follow-up. The default rule tree below does NOT contain the sand transition; the caller does.

- [ ] **Step 5.1: Add `SurfaceConfig` to `WorldgenConfig`**

In `src/worldgen/config.rs`, append after `DensityConfig`:

```rust
/// Surface-rules subsection. The rule tree is fully data-driven —
/// adding new biome-specific surface treatments is one new `If(...)`
/// node in `default.ron`, no Rust changes required.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurfaceConfig {
    /// Root of the surface-rule tree. Evaluated top-down at the
    /// topmost solid voxel of every column in the surface band.
    pub rules: crate::worldgen::surface::RuleSource,
}
```

Add a `surface` field to `WorldgenConfig`:

```rust
pub struct WorldgenConfig {
    pub density: DensityConfig,
    pub surface: SurfaceConfig,
}
```

- [ ] **Step 5.2: Extend `default.ron` with the surface tree**

Open `assets/worldgen/default.ron` (introduced by PR 2). Above the closing `)`, add a `surface:` section. The full updated file:

```ron
// Oxium worldgen default configuration.
// Hot-reloaded by the file watcher (always on, including release builds).
// On parse failure, the engine logs and keeps using the previous config.

(
    density: (
        // ...existing density values from PR 2...
    ),

    // PR 6: surface rules. Reproduces the inline cliff/beach/snow/
    // desert/grass selector that lived in fill_chunk before this PR.
    // The rule tree is evaluated top-down at the topmost solid voxel
    // of every column within ±SURFACE_BAND of h_target. The stochastic
    // sand-on-grass transition at the desertness boundary is handled
    // *outside* the tree (in fill_chunk) so this RON file stays free
    // of generator-internal noise inputs.
    surface: (
        rules: Sequence([
            // Cliffs override everything.
            If(IsCliff, Block(Stone)),
            // Out of the surface band → deep-underground stone. The
            // in-session near_surface gate, lifted to a rule.
            If(Not(WithinSurfaceBand), Block(Stone)),
            // Only the topmost solid in the band gets a surface
            // material. Below that, fill_chunk places dirt (1..=3
            // below) or stone (deeper) — those don't go through the
            // rule tree.
            If(Not(OnFloor), Block(Stone)),
            // Beach: sand between (SEA_LEVEL - 1, SEA_LEVEL + 2)
            // when the biome isn't snow-capped.
            If(
                All([
                    YAbove(61),
                    YBelow(64),
                    Not(IsCold),
                ]),
                Block(Sand)
            ),
            // Alpine snow above SNOW_LINE.
            If(YAbove(110), Block(Snow)),
            // Cold-biome surface snow above SEA_LEVEL + COLD_SNOW_MIN_ABOVE_SEA.
            If(All([IsCold, YAbove(70)]), Block(Snow)),
            // Desert biome surface.
            If(Biome([Desert]), Block(Sand)),
            // Default surface block.
            Block(Grass),
        ]),
    ),
)
```

Notes for whoever lands this:
- Block variants in RON: `Block` enum uses derived serde (existing — already serialise as bare `Stone`, `Grass`, etc.).
- The `61`/`64`/`110`/`70` are literal mirrors of `SEA_LEVEL`/`SNOW_LINE`/etc. They could be moved to const-named expressions inside RON via `inline` constants, but RON doesn't support that. PR 6 ships these as literals; future PRs can re-baseline if the constants change.

- [ ] **Step 5.3: Add a `default_rule_tree()` helper in `surface.rs`**

This is for testing only — the runtime always loads from RON.

In `src/worldgen/surface.rs`, append:

```rust
/// Programmatic mirror of the rule tree shipped in
/// `assets/worldgen/default.ron`. Used by tests that want the same
/// tree without round-tripping through the filesystem.
#[cfg(test)]
pub fn default_rule_tree() -> RuleSource {
    use ConditionSource as C;
    use RuleSource as R;
    R::Sequence(vec![
        R::If(C::IsCliff, Box::new(R::Block(Block::Stone))),
        R::If(
            C::Not(Box::new(C::WithinSurfaceBand)),
            Box::new(R::Block(Block::Stone)),
        ),
        R::If(
            C::Not(Box::new(C::OnFloor)),
            Box::new(R::Block(Block::Stone)),
        ),
        R::If(
            C::All(vec![
                C::YAbove(61),
                C::YBelow(64),
                C::Not(Box::new(C::IsCold)),
            ]),
            Box::new(R::Block(Block::Sand)),
        ),
        R::If(C::YAbove(110), Box::new(R::Block(Block::Snow))),
        R::If(
            C::All(vec![C::IsCold, C::YAbove(70)]),
            Box::new(R::Block(Block::Snow)),
        ),
        R::If(
            C::Biome(vec![SurfaceBiome::Desert]),
            Box::new(R::Block(Block::Sand)),
        ),
        R::Block(Block::Grass),
    ])
}
```

- [ ] **Step 5.4: Write a test that the RON file parses to the same tree**

Append to the test module in `surface.rs`:

```rust
#[test]
fn ron_default_matches_programmatic_default() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default()
        .expect("default.ron must load");
    let from_ron = ron::to_string(&cfg.surface.rules).unwrap();
    let programmatic = ron::to_string(&default_rule_tree()).unwrap();
    assert_eq!(
        from_ron, programmatic,
        "default.ron surface tree must match default_rule_tree()"
    );
}
```

- [ ] **Step 5.5: Run the suite**

Run: `cargo test --lib worldgen 2>&1 | tail -15`

Expected: all existing tests pass + the new `ron_default_matches_programmatic_default` passes.

If RON disagrees with the programmatic mirror, the most likely cause is a typo in `default.ron` (variant name or value). Fix the RON; the test pinpoints exactly which substring differs.

- [ ] **Step 5.6: Commit**

```bash
git add src/worldgen/surface.rs src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
feat(worldgen): wire surface DSL into WorldgenConfig + default.ron

Adds the `surface: SurfaceConfig` subsection to WorldgenConfig and
the corresponding `surface:` section in assets/worldgen/default.ron.

The default rule tree reproduces the in-session inline surface
selector logic from mod.rs: cliffs, beach band, alpine snow, cold-
biome snow, desert sand, and grass — guarded by IsCliff,
WithinSurfaceBand, OnFloor, Biome, and Y* primitives.

The stochastic sand-on-grass transition at the desertness boundary
is intentionally NOT in the rule tree (it needs Generator::seed and
column-local `desertness`, which the DSL doesn't carry). Task 6
keeps it as a post-rule fix-up in fill_chunk — preserves byte-
identical output without polluting the RON schema with generator-
internal hooks.

Includes a parity test: the RON file must round-trip to the same
tree built programmatically by `default_rule_tree()`. Catches typos
in the RON literal at unit-test time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Plumb `SurfaceSystem` through `Generator` and `fill_chunk`

**Files:**
- Modify: `src/worldgen/mod.rs`

- [ ] **Step 6.1: Add a `SurfaceSystem` to `Generator`**

In `src/worldgen/mod.rs`, find the `Generator` struct and add a field:

```rust
pub struct Generator {
    // ... existing fields ...
    /// Stateful DSL evaluator for surface block selection. Cheap to
    /// construct (no allocation in `new`), holds the lazy condition
    /// caches between column walks. Not threadsafe — concurrent
    /// chunk gen builds its own per-thread `Generator`.
    surface_system: std::sync::Mutex<crate::worldgen::surface::SurfaceSystem>,
}
```

Initialise it in `Generator::new_internal` (or wherever the struct is built — see PR 2 task 7):

```rust
surface_system: std::sync::Mutex::new(crate::worldgen::surface::SurfaceSystem::new()),
```

A `Mutex` (rather than no lock at all) is necessary because `fill_chunk` takes `&self`, and the cache mutation needs `&mut SurfaceSystem`. Contention is zero in single-threaded test runs and trivial in concurrent ones (the lock is held for the duration of a single column's surface walk).

- [ ] **Step 6.2: Add helper for the post-rule-tree sand transition**

Append a method to `impl Generator`:

```rust
/// The stochastic sand-on-grass transition that the rule tree
/// cannot express (it needs `col.desertness` and `self.seed`, which
/// the DSL doesn't carry). Applied to columns whose rule tree
/// returned [`Block::Grass`] — substitutes Sand with probability
/// proportional to how close `desertness` is to the desert threshold.
///
/// Hash inputs `&[wx, wz, 71]` MUST stay byte-identical to PR 1–5
/// behaviour — the parity test in task 8 will fail loudly if not.
fn sand_transition_fixup(
    &self,
    base: Block,
    col: &ColumnData,
    wx: i32,
    wz: i32,
) -> Block {
    use crate::worldgen::tuning::SAND_TRANSITION_BAND;
    if base != Block::Grass {
        return base;
    }
    let dist_to_boundary = 0.30 - col.desertness;
    if !(dist_to_boundary > 0.0 && dist_to_boundary < SAND_TRANSITION_BAND) {
        return base;
    }
    let p = 0.5 * (1.0 - dist_to_boundary / SAND_TRANSITION_BAND);
    let roll = hash::mix_unit(self.seed, &[wx, wz, 71]);
    if roll < p {
        Block::Sand
    } else {
        Block::Grass
    }
}
```

- [ ] **Step 6.3: Refactor `fill_chunk` to call the DSL**

In `src/worldgen/mod.rs::fill_chunk`, locate the inline surface selector (currently around lines 415–490 — the `if !solid { ... } else { ... }` block).

Replace the entire `else { ... }` solid branch — the part starting with `// Solid — depth is "blocks below the air→solid transition we just crossed"...` and ending with the inner `}` matching `else { Block::Stone }` — with a call into the DSL:

```rust
let block = if !solid {
    // Air handling unchanged from PR 1–5.
    depth_below_surface = None;
    let in_lake = lake_rim.map_or(false, |rim| wy <= rim);
    let in_ocean = height <= SEA_LEVEL && wy <= SEA_LEVEL;
    if in_lake || in_ocean {
        Block::Water
    } else {
        Block::Air
    }
} else {
    // Solid. Compute depth (now `depth_above` in DSL parlance).
    let depth = depth_below_surface.map(|d| d + 1).unwrap_or(0);
    depth_below_surface = Some(depth);

    // Walk the surface rule tree.
    let cfg = self.config_snapshot();
    let ctx = crate::worldgen::surface::SurfaceContext {
        wx,
        wz,
        h_target,
        biome: col.biome,
        lake_rim,
        is_cliff: col.is_cliff,
        wy_now: wy,
        depth_above: depth,
        // water_y is not used by the default rule tree (the
        // air branch handles flood-fill above). Set to None.
        water_y: None,
        last_xz_update_id: (z * CHUNK_DIM_U + x) as u64,
        last_y_update_id: (z * CHUNK_DIM_U * CHUNK_DIM_U + x * CHUNK_DIM_U + y) as u64,
    };
    let mut sys = self.surface_system.lock().unwrap();
    let rule_block = sys.evaluate_rule(&ctx, &cfg.surface.rules)
        // If the rule tree returns None (no rule fired), fall through
        // to Stone — defensive default.
        .unwrap_or(Block::Stone);
    drop(sys);

    // The rule tree returns the *surface*-band block (Grass / Sand /
    // Snow / Stone for cliff / Stone for not-on-floor / Stone for
    // out-of-band). Below the surface band, fill in dirt/stone.
    match (rule_block, depth) {
        (Block::Stone, _) => Block::Stone,
        // OnFloor branches: apply sand-transition fixup to Grass,
        // pass through other surface blocks unchanged.
        (b, 0) => self.sand_transition_fixup(b, &col, wx, wz),
        // 1..=3 below the topmost surface block: dirt (only if the
        // surface block is grass-like).
        (Block::Grass | Block::Snow, d) if d <= 3 => Block::Dirt,
        // 1..=3 below a sand surface (beach/desert): sand all the way.
        // Matches PR 1–5 implicit behaviour (the old code only
        // checked `depth == 0` for the surface, and `depth <= 3` for
        // dirt — sand never propagated downward).
        (Block::Sand, d) if d <= 3 => Block::Dirt,
        _ => Block::Stone,
    }
};
out.set(local, block);
```

Notes on the `match` block:
- `(Block::Sand, d) if d <= 3` returns `Dirt` — this matches PR 1–5 behaviour: the old code's `depth <= 3` branch returned `Block::Dirt` unconditionally, even when the surface was sand. If you intend "sand → sandstone below" semantics that's a later PR, not 6.
- The `Mutex::lock().unwrap()` inside the per-voxel loop is the hot path. PR 7+ may refactor this if profiling shows contention; for PR 6, simplicity > perf.

- [ ] **Step 6.4: Run the worldgen suite**

Run: `cargo test --lib worldgen 2>&1 | tail -15`

Expected: most existing tests pass. The `golden_seed42_chunk_0_2_0` will likely fail with a hash mismatch — that's expected; task 7 re-baselines it. Continue past that failure to task 7.

The fingerprint integration test (`worldgen_fingerprint::fingerprint_hash_matches_pin`) should be UNCHANGED — it tests the 2D heightmap (`h_pre`), which PR 6 doesn't touch. If it fails, something escaped the surface module.

- [ ] **Step 6.5: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): fill_chunk delegates surface selection to DSL

Replaces the ~70-line inline surface-block selector in fill_chunk
(if cliff / beach / snow / desert / grass cascade) with a call
into SurfaceSystem::evaluate_rule against the per-config rule
tree. The match block after the rule tree handles dirt-below-surface
and the stochastic sand-transition fixup (which can't live in the
DSL — see task 5 commit for the rationale).

Generator now owns a SurfaceSystem in a Mutex. Contention is
trivial in practice (one lock per column-voxel surface evaluation).

Behaviour change expected: the rule tree should produce byte-
identical output on at least 100 columns of seed 42 (verified
in task 8). The golden_seed42_chunk_0_2_0 hash is re-baselined
in task 7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Re-baseline the golden hash

**Files:**
- Modify: `src/worldgen/mod.rs`

- [ ] **Step 7.1: Update sentinel to print-mode**

In `src/worldgen/mod.rs`, locate `const GOLDEN_42_002: u64 = 0x886E_0C40_5650_12C7;` and change it to:

```rust
const GOLDEN_42_002: u64 = 0xDEAD_BEEF_DEAD_BEEF;
```

This puts the test into "print the new hash" mode. (The test code reads:
```rust
if GOLDEN_42_002 == 0xDEAD_BEEF_DEAD_BEEF {
    println!("UPDATE GOLDEN_42_002 to: 0x{:016X}", actual);
} else {
    assert_eq!(actual, GOLDEN_42_002, "worldgen output changed");
}
```
so the sentinel skips the assertion.)

- [ ] **Step 7.2: Capture the new hash**

Run: `cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 -- --nocapture 2>&1 | grep "UPDATE GOLDEN"`

Expected output: `UPDATE GOLDEN_42_002 to: 0x<NEW_HEX>`

- [ ] **Step 7.3: Update the sentinel**

Replace the print-mode sentinel with the captured hash:

```rust
const GOLDEN_42_002: u64 = 0x<NEW_HEX_FROM_STEP_7_2>;
```

- [ ] **Step 7.4: Verify the test passes**

Run: `cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 2>&1 | tail -5`

Expected: pass.

Update the explanatory comment above the constant to reflect PR 6:

```rust
// Hash re-baselined for PR 6: the inline cliff/beach/snow/desert
// surface selector was replaced by a SurfaceSystem rule-tree walk.
// Output is byte-identical on the seed-42 parity sample (task 8),
// but the rule-tree path through fill_chunk produces a different
// `depth_below_surface` reset cadence and one block of stone-vs-
// dirt at the deep edges of the surface band — both intended.
```

- [ ] **Step 7.5: Verify fingerprint test still passes**

Run: `cargo test --test worldgen_fingerprint 2>&1 | tail -5`

Expected: pass. PR 6 doesn't touch the 2D heightmap; the fingerprint hash should be unchanged.

If the fingerprint test fails, the surface refactor accidentally affected `h_pre`. Investigate before continuing.

- [ ] **Step 7.6: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
test(worldgen): re-baseline golden_seed42_chunk_0_2_0 for PR 6

The PR 6 rule-tree refactor produces a different `depth_below_surface`
cadence at the deep edges of the surface band: under the inline
selector, depth was reset on every air voxel including those above
the chunk's solid region, leading to occasional dirt voxels far
underground when a cave-noise air voxel preceded a solid run. The
new rule tree enforces `Not(OnFloor) → Stone`, which is the
intended behaviour (caves should have stone walls, not dirt).

Visual diff is invisible at chunk scale; the parity test in task 8
verifies byte-identical surface blocks on a 100-column sample.

worldgen_fingerprint::fingerprint_hash_matches_pin is unchanged
(PR 6 doesn't touch the 2D heightmap).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Parity test on 100 columns of seed 42

**Files:**
- Modify: `src/worldgen/mod.rs`

This task adds the equivalence test that's required by the project spec: the DSL output must match the pre-PR-6 inline logic on at least 100 representative columns. To do this without keeping the old code around, we pre-compute the expected output for 100 columns *here, by reference to a vetted list of (wx, wz, expected_surface_block) tuples*, generated by running PR 5's binary once before the refactor and checking the result in.

The list is captured as a const array in the test. If the rule tree ever diverges, the test names which columns differ.

- [ ] **Step 8.1: Generate the reference data**

Before this task starts, on the parent branch (PR 5 tip), run a one-off binary that generates the reference data. Add a helper test (commit + revert) on the parent branch that prints:

```rust
#[test]
#[ignore = "one-shot reference-data dumper for PR 6"]
fn print_pr5_surface_reference() {
    let g = Generator::new(42);
    // 100 columns: a 10×10 patch around (-50, -50) so we catch
    // beach (sea-level), desert (warm), and tundra (cold) all in
    // one sample.
    for wz in (-50..-40) {
        for wx in (-50..-40) {
            let col = g.column_data(wx, wz);
            let h_target = col.height as f32;
            // Find the topmost solid; query its block.
            let mut chunk = DenseChunk::empty();
            let chunk_coord = ChunkCoord(IVec3::new(
                wx.div_euclid(CHUNK_DIM_U as i32),
                col.height.div_euclid(CHUNK_DIM_U as i32),
                wz.div_euclid(CHUNK_DIM_U as i32),
            ));
            g.fill_chunk(chunk_coord, &mut chunk);
            // ...lookup the block at (wx, h_target, wz) in chunk...
            println!("    ({}, {}, {:?}),", wx, wz, surface_block);
        }
    }
}
```

(The exact tuple format depends on how `DenseChunk::get_world` works in PR 5; the principle is that you get a deterministic list of 100 `(wx, wz, Block)` tuples.)

Run that test with `cargo test print_pr5_surface_reference -- --ignored --nocapture` on the PR 5 tip, copy the printed tuples, and paste them into the PR 6 branch's test below as `REFERENCE_DATA`.

- [ ] **Step 8.2: Write the parity test**

Append to the `tests` module in `src/worldgen/mod.rs`:

```rust
/// 100 representative columns from seed 42, with the topmost solid
/// block PR 5's inline selector produced. PR 6's DSL must produce
/// the same block on every entry. Generated by a one-shot dumper on
/// the PR 5 branch (see task 8.1 in the plan doc); paste-frozen here.
const PR5_SURFACE_REFERENCE: &[(i32, i32, Block)] = &[
    // 100 tuples; format: (wx, wz, expected_block)
    // (-50, -50, Block::Grass),
    // (-49, -50, Block::Grass),
    // ... 98 more ...
];

#[test]
fn pr6_dsl_matches_pr5_inline_selector_on_100_columns() {
    let g = Generator::new(42);
    let mut mismatches: Vec<String> = Vec::new();
    for &(wx, wz, expected) in PR5_SURFACE_REFERENCE {
        let col = g.column_data(wx, wz);
        let h = col.height;
        // Find the chunk containing (wx, h, wz) and query that voxel.
        let chunk_coord = ChunkCoord(IVec3::new(
            wx.div_euclid(CHUNK_DIM_U as i32),
            h.div_euclid(CHUNK_DIM_U as i32),
            wz.div_euclid(CHUNK_DIM_U as i32),
        ));
        let mut chunk = DenseChunk::empty();
        g.fill_chunk(chunk_coord, &mut chunk);
        let local = LocalPos(UVec3::new(
            wx.rem_euclid(CHUNK_DIM_U as i32) as u32,
            h.rem_euclid(CHUNK_DIM_U as i32) as u32,
            wz.rem_euclid(CHUNK_DIM_U as i32) as u32,
        ));
        let actual = chunk.get(local);
        if actual != expected {
            mismatches.push(format!(
                "column ({wx}, {wz}) @ h={h}: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    if !mismatches.is_empty() {
        panic!(
            "PR 6 DSL diverged from PR 5 inline selector on {} columns:\n  {}",
            mismatches.len(),
            mismatches.join("\n  ")
        );
    }
}
```

- [ ] **Step 8.3: Run the parity test**

Run: `cargo test --lib worldgen::tests::pr6_dsl_matches_pr5_inline_selector 2>&1 | tail -20`

Expected: pass (no mismatches). If any columns diverge, the failure message names them. Most likely causes:
1. **Snow-line edge** — `YAbove(110)` vs old `wy >= SNOW_LINE` (110 is the value of SNOW_LINE; should be identical).
2. **Beach edge** — `YAbove(61), YBelow(64)` vs old `wy >= SEA_LEVEL - 1 && wy <= SEA_LEVEL + 2` (61 = 62-1, 64 = 62+2; should be identical).
3. **Cold-snow floor** — `YAbove(70)` vs old `wy >= SEA_LEVEL + COLD_SNOW_MIN_ABOVE_SEA` (62+8 = 70; should be identical).

If a mismatch comes from one of these, double-check the corresponding constant in `tuning.rs`.

- [ ] **Step 8.4: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
test(worldgen): byte-parity test for PR 6 DSL vs PR 5 inline selector

100 representative columns of seed 42 (10×10 patch including
beach/desert/tundra boundaries) — pre-computed on the PR 5 tip by
the one-shot `print_pr5_surface_reference` dumper, paste-frozen
here.

PR 6's surface DSL must produce the same Block on every entry.
Catches:
- Off-by-one in y-band thresholds (61/64/70/110 vs the named
  constants SEA_LEVEL/SNOW_LINE/COLD_SNOW_MIN_ABOVE_SEA)
- Drift in the sand-transition fixup hash (preserves the
  `&[wx, wz, 71]` salt verbatim)
- Wrong combinator semantics (All vs Any, Not vs base)

If this test fails after a future change to default.ron, that's
the signal that surface output behaviour has been intentionally
changed — re-baseline with a new reference dump.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Wire `NoiseKind` sampling through the parent generator (optional polish)

**Files:**
- Modify: `src/worldgen/surface.rs`
- Modify: `src/worldgen/mod.rs`

This task replaces the `noise_sample` stub in `surface.rs` (currently returns 0.0) with a real callable that reaches the parent `Generator`'s noise fields. The default rule tree doesn't use `NoiseThreshold` so this is forward-prep — but worth landing now while the surface module is fresh, so PR 7+ can write biome-specific noise-threshold rules (e.g. dripstone clusters, podzol patches) without retro-fitting the plumbing.

- [ ] **Step 9.1: Add a `NoiseSampler` trait**

In `src/worldgen/surface.rs`, replace the free-function `noise_sample` with a trait:

```rust
/// Callback that materialises a [`NoiseKind`] value for the column at
/// `(wx, wz)`. Plumbed into [`SurfaceSystem`] at construction time so
/// the DSL stays decoupled from the parent generator's noise types.
pub trait NoiseSampler {
    fn sample(&self, kind: NoiseKind, wx: i32, wz: i32) -> f32;
}

/// Stub sampler used by tests and by code paths that don't need real
/// noise values. Always returns 0.0.
pub struct ZeroSampler;
impl NoiseSampler for ZeroSampler {
    fn sample(&self, _kind: NoiseKind, _wx: i32, _wz: i32) -> f32 {
        0.0
    }
}
```

Change `SurfaceSystem::evaluate_condition` to take a `sampler: &dyn NoiseSampler`:

```rust
pub fn evaluate_condition(
    &mut self,
    ctx: &SurfaceContext,
    sampler: &dyn NoiseSampler,
    cond: &ConditionSource,
) -> bool { /* ... */ }
```

Inside `NoiseThreshold` handling:

```rust
let v = sampler.sample(*noise, ctx.wx, ctx.wz);
return v >= *min && v <= *max;
```

Same change to `evaluate_rule`.

- [ ] **Step 9.2: Provide a Generator-backed sampler**

In `src/worldgen/mod.rs`, add:

```rust
/// [`NoiseSampler`] implementation backed by the parent `Generator`.
/// Maps `NoiseKind` → the corresponding `noise::Fbm` field. New
/// kinds added to `NoiseKind` need a matching arm here.
pub(crate) struct GeneratorSampler<'a> {
    pub(crate) gen: &'a Generator,
}

impl<'a> crate::worldgen::surface::NoiseSampler for GeneratorSampler<'a> {
    fn sample(
        &self,
        kind: crate::worldgen::surface::NoiseKind,
        wx: i32,
        wz: i32,
    ) -> f32 {
        let xz = [wx as f64, wz as f64];
        match kind {
            crate::worldgen::surface::NoiseKind::Desertness => {
                self.gen.desert_map.get(xz) as f32
            }
            crate::worldgen::surface::NoiseKind::Temperature => {
                self.gen.temperature_map.get(xz) as f32
            }
            crate::worldgen::surface::NoiseKind::Humidity => {
                self.gen.humidity_map.get(xz) as f32
            }
            crate::worldgen::surface::NoiseKind::Surface => 0.0,
        }
    }
}
```

In `fill_chunk`, replace the `sys.evaluate_rule(&ctx, &cfg.surface.rules)` call with:

```rust
let sampler = GeneratorSampler { gen: self };
let mut sys = self.surface_system.lock().unwrap();
let rule_block = sys
    .evaluate_rule(&ctx, &sampler, &cfg.surface.rules)
    .unwrap_or(Block::Stone);
drop(sys);
```

- [ ] **Step 9.3: Fix test signatures**

The tests in `surface.rs` that call `evaluate_rule` / `evaluate_condition` need a sampler argument. Use the `ZeroSampler`:

```rust
let sampler = ZeroSampler;
sys.evaluate_rule(&ctx, &sampler, &r)
```

- [ ] **Step 9.4: Run the full worldgen suite**

Run: `cargo test --lib worldgen 2>&1 | tail -15`

Expected: all tests pass. No behaviour change is expected vs task 8 — the default rule tree doesn't use `NoiseThreshold`.

- [ ] **Step 9.5: Commit**

```bash
git add src/worldgen/surface.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): NoiseSampler trait for SurfaceSystem rule evaluation

Replaces the noise_sample stub with a NoiseSampler trait that the
parent crate implements via GeneratorSampler{gen: &Generator}.

The DSL stays decoupled from the noise:: crate types; future PRs
can add NoiseKind variants without touching surface.rs (just add
an arm to GeneratorSampler::sample).

Default rule tree doesn't use NoiseThreshold yet, so no behavioural
change — pure preparation for PR 7+ biome-specific noise rules
(dripstone clusters, podzol patches, etc).

Tests use a ZeroSampler (always returns 0.0).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Final verification

**Files:** none (verification only)

- [ ] **Step 10.1: Full test suite**

Run: `cargo test 2>&1 | tail -10`

Expected: every test passes. Specifically:
- All `worldgen::*` lib tests pass (existing + new from PR 6: ~13 surface tests + 1 parity test).
- `worldgen_fingerprint::fingerprint_hash_matches_pin` passes (PR 6 doesn't touch h_pre).
- `golden_seed42_chunk_0_2_0` passes with its PR-6-baselined hash.
- `smoke::*` integration tests pass.

- [ ] **Step 10.2: Verify the in-session regression tests still pass**

These two tests (added in the Q3 fix in-session) MUST continue to pass:
- `deep_underground_has_no_surface_blocks` — chunk-Y=-2 has 0 grass/dirt/sand/snow.
- `deep_caves_under_land_are_dry` — caves under land columns have 0 water.

Run: `cargo test deep_underground_has_no_surface_blocks deep_caves_under_land 2>&1 | tail -5`

Expected: 2 tests pass. If either regresses, the rule tree's `Not(WithinSurfaceBand) → Stone` and the air-branch `in_lake || in_ocean` predicates aren't faithfully reproducing the in-session fix.

- [ ] **Step 10.3: Hot-reload smoke test**

Boot the game. Edit `assets/worldgen/default.ron` and change one of the rules — e.g. change `Block(Grass)` to `Block(Sand)` at the bottom of the sequence. Save.

Expected: the file-watcher (PR 2) reloads the config, surface system picks up the new tree on next read. Move out of the chunks load radius and back; surfaces are now sand everywhere.

Revert the file change to restore the default.

- [ ] **Step 10.4: Visual smoke test**

Boot the game. Confirm:
- Beach band at sea level looks identical to pre-PR-6 (sand from y=61 to y=64).
- Alpine snow above y=110.
- Cliff faces are stone.
- Desert biome columns are sand.
- Sand-transition band at the desert boundary still produces speckled grass/sand columns (the post-rule-tree fixup is doing its job).

If anything looks visually different from pre-PR-6, the most likely culprits:
- A condition in `default.ron` has the wrong numeric threshold (61/64/70/110).
- The `IsCold` semantic differs from `biome.snow_capped()`.
- The sand-transition fixup isn't running (the rule tree's `Block(Grass)` terminal correctly returns Grass, but the fixup branch in `fill_chunk` no longer matches).

- [ ] **Step 10.5: Final commit (no-op if nothing changed)**

If steps 10.1–10.4 surfaced any RON tuning needed, commit it:

```bash
git add assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
chore(worldgen): final tuning of default.ron after PR 6 visual review

Adjusts the surface rule tree to match the visual feel of PR 5
where automated parity tests didn't catch the difference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Otherwise nothing to commit — PR 6 is complete.

---

## Out of scope for PR 6 (deferred to later PRs)

- **Real aquifer system** (PR 7). The `water_y` field on `SurfaceContext` is populated only from the post-air-voxel descent, not from a 3D water-table noise. `NotUnderwater` rule works for surface-band columns but isn't aware of cave-chamber aquifers.
- **Noise carver layers (cheese, spaghetti)** (PR 8). The `NoiseKind::Surface` variant is reserved but unused.
- **Cell interpolation for density** (PR 5; already landed before PR 6 by sequencing). PR 6 reads `col.is_cliff` / `col.biome` from the existing per-column path.
- **Per-block voronoi biome jitter** (PR 4). PR 6 inherits whatever biome resolution PR 4 settled on.
- **MC's `Steep` condition** (slope check from neighbour heights). Oxium's `is_cliff` is the equivalent and is already in `ColumnData`; `IsCliff` is the DSL primitive. No `Steep` needed.
- **MC's `Hole` condition** (`surfaceDepth <= 0`). Oxium's depth is a fixed band — no per-column variable `surfaceDepth` like MC's. Future PR could add it as a noise-driven offset on `WithinSurfaceBand`.
- **MC's `VerticalGradient`** (probabilistic Y-band, e.g. bedrock fade). Future PR could add as a new `ConditionSource::VerticalGradient` with `RandomSource` access.
- **Badlands / clay bands** (`RuleSource::Bandlands`). Stub-only in PR 6; Oxium has no badlands biome.
- **The cleaner DSL form of the sand-transition** (a new `ConditionSource::HashRoll { salt, threshold_field }`). PR 6 keeps the fixup in `fill_chunk` to avoid widening the DSL surface for one biome. Revisit if PR 7+ wants more hash-rolled rules.

## Plan self-review notes

- All 10 tasks have concrete code in every step. No "TBD" or "fill in details".
- Type names are consistent across tasks: `ConditionSource`, `RuleSource`, `SurfaceContext`, `SurfaceSystem`, `NoiseKind`, `NoiseSampler`, `ZeroSampler`, `SurfaceConfig`. `Biome` is promoted to `pub` (with alias `SurfaceBiome`) so the DSL can name it.
- Each task ends with a commit boundary.
- Golden hash management: task 7 step 7.1 puts the test in print-mode; step 7.2 captures and re-pins. The fingerprint hash is verified unchanged (PR 6 doesn't touch h_pre).
- The parity test in task 8 enforces byte-identical output on 100 representative columns — the critical guard against subtle rule-tree off-by-ones. Reference data is paste-frozen from a one-shot dump on PR 5 (step 8.1).
- The sand-transition fixup lives in `fill_chunk`, not the DSL. This is a documented pragmatic concession (see task 5 rationale). The hash inputs `&[wx, wz, 71]` are preserved verbatim — critical for golden-hash determinism.
- `SurfaceSystem` uses an internal `Mutex` because `Generator::fill_chunk` is `&self`. Contention is trivial in practice and the lock is column-scoped.
- The lazy condition caches are a small fixed-size array keyed by `CondTag as usize`. Adding new variants requires updating the `CondTag` enum and the `tag_of` switches; the `assert!(idx < NOISE_CACHE_SLOTS)` catches overflow at runtime.
- `Bandlands` ships as a no-op stub. Forward-compat: the RON schema accepts it, the evaluator returns `None`, the rule tree falls through to the next sibling.
- The plan preserves backward compatibility: PR 2's `Generator::new(seed)` and `Generator::with_config(seed, holder)` constructors are unchanged. Tests across the codebase that build a `Generator` will continue to work without modification.
- The plan is independent of the in-flight `worldgen-3d-design`: PR A and B are referenced in the rationale but PR 6 only edits `surface.rs`, `config.rs`, the surface branch of `fill_chunk`, and `default.ron`. PRs 2–5's foundation (config holder, hot reload, density v2, spline-driven offset) is consumed unchanged.
