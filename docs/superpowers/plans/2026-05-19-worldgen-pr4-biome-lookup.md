# Worldgen PR 4 — Multi-noise biome lookup (6D R-tree)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the nested-if `Biome::classify(temperature, humidity, is_desert)` in `mod.rs` with a 6D hyperbox lookup in `(temperature, humidity, continentalness, terrain_shape, depth, weirdness)` space. Biome claims are loaded from `assets/worldgen/default.ron` (PR 2's hot-reload pipeline), indexed by a fanout-6 R-tree with squared-L2 distance and a `ThreadLocal` last-leaf cache. Per-block Voronoi jitter (8-corner hash) gives organic biome borders without interpolating biome IDs. Oxium's existing 6 biomes (Tundra, SnowyForest, Plains, Forest, Desert, Tropical) are preserved — the change is *how* they're selected, not which ones exist.

**Architecture:** Five tightly-coupled sub-components landing together. (1) `Climate.Parameter` — a `(min, max)` `i64` interval, quantized from `f32` via `* 10000`. Matches MC's `net/minecraft/world/level/biome/Climate.java#Parameter`. (2) `ClimateParameterPoint` — a 6-Parameter hyperbox + a biome ID + a 7th `offset` slot (tie-breaker per MC). (3) `ClimateRTree` — fanout-6 spatial index built once at config load time; queries walk the tree with squared-L2 over per-axis hyperbox gaps. (4) `ClimateSampler` — wraps the existing temperature/humidity noise plus PR 3's continentalness + terrain_shape, plus a new `weirdness` noise channel (low frequency, like MC's "ridge"). Computes a `ClimateTargetPoint` at a given `(wx, wy, wz)`. (5) `Generator::biome_at(wx, wy, wz)` — applies per-block Voronoi jitter (8-corner hash, picks nearest jittered cell center), then queries the R-tree. `Biome::classify` is removed.

**Tech Stack:**
- Rust 2024 edition
- `noise` (already a dependency) — Simplex/Fbm for the `weirdness` channel
- PR 2's `WorldgenConfig` / `ConfigHolder` — biome table loaded from RON
- PR 2's `CubicSpline` — not used in PR 4 directly; reserved for PR 5+
- PR 3's `continentalness` + `terrain_shape` noise channels (PR 4 adds *placeholder* shims for these axes if PR 3 hasn't landed at integration time — see Task 4)

**Reference:** Architectural rationale is in `docs/superpowers/specs/2026-05-19-minecraft-worldgen-research.md` (Part 1 idea #3, Part 4 idea #3, Decisions Log Q5/Q6). MC source: `net/minecraft/world/level/biome/Climate.java` (the R-tree + Parameter + Sampler) and `net/minecraft/world/level/biome/OverworldBiomeBuilder.java` (the biome claim table structure). This plan does not re-argue those decisions.

---

### Task 1: `Climate.Parameter` type (TDD)

**Files:**
- Modify: `src/worldgen/climate.rs` (currently a 5-line stub)

- [ ] **Step 1.1: Write the failing tests**

Replace the contents of `src/worldgen/climate.rs` with the type stub and test module:

```rust
//! Climate parameters for the 6D biome lookup.
//!
//! A [`Parameter`] is an `(i64, i64)` interval quantized from `f32`
//! via `* QUANTIZATION_FACTOR`. A [`ParameterPoint`] bundles six such
//! intervals (one per climate axis) plus a 7th `offset` slot used as
//! a tie-breaker. A [`TargetPoint`] is the analogous 6-tuple of
//! quantized scalars (the "query point" in climate space).
//!
//! Algorithm mirrors Minecraft 1.18+
//! `net/minecraft/world/level/biome/Climate.java`. Quantization
//! factor is identical (10000) so claim values authored against MC's
//! literature translate 1:1.

use serde::{Deserialize, Serialize};

/// Quantization factor for converting `f32` climate values to `i64`
/// for hyperbox arithmetic. Matches MC's `QUANTIZATION_FACTOR`.
pub const QUANTIZATION_FACTOR: f32 = 10000.0;

/// Quantize a single `f32` climate value to the integer domain.
pub fn quantize(value: f32) -> i64 {
    (value * QUANTIZATION_FACTOR) as i64
}

/// Inverse of [`quantize`].
pub fn unquantize(value: i64) -> f32 {
    value as f32 / QUANTIZATION_FACTOR
}

/// A closed interval `[min, max]` in the quantized integer domain.
/// All climate intervals are inclusive on both ends (matches MC).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Parameter {
    pub min: i64,
    pub max: i64,
}

impl Parameter {
    /// Construct from quantized bounds (no validation; pre-quantized).
    pub const fn new(min: i64, max: i64) -> Self {
        Self { min, max }
    }

    /// Span from two unquantized `f32` endpoints. Quantizes inline.
    pub fn span(min: f32, max: f32) -> Self {
        assert!(min <= max, "Parameter::span requires min <= max");
        Self { min: quantize(min), max: quantize(max) }
    }

    /// Degenerate interval at a single quantized point.
    pub fn point(at: f32) -> Self {
        let q = quantize(at);
        Self { min: q, max: q }
    }

    /// Smallest non-negative gap from `target` to this interval (in
    /// quantized units). Zero when the target is inside `[min, max]`.
    /// Used as the squared-L2 distance summand for the R-tree.
    pub fn distance(&self, target: i64) -> i64 {
        let above = target - self.max;
        let below = self.min - target;
        if above > 0 {
            above
        } else if below > 0 {
            below
        } else {
            0
        }
    }

    /// Union with another `Parameter` (or treat `None` as identity).
    /// Used during R-tree subtree construction to derive the parent's
    /// bounding hyperbox per axis.
    pub fn union(&self, other: Option<Parameter>) -> Parameter {
        match other {
            None => *self,
            Some(o) => Parameter {
                min: self.min.min(o.min),
                max: self.max.max(o.max),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_and_unquantize_roundtrip() {
        assert_eq!(quantize(0.0), 0);
        assert_eq!(quantize(1.0), 10000);
        assert_eq!(quantize(-0.5), -5000);
        assert!((unquantize(quantize(0.337)) - 0.337).abs() < 1e-3);
    }

    #[test]
    fn distance_zero_when_target_inside() {
        let p = Parameter::span(-0.5, 0.5);
        assert_eq!(p.distance(quantize(0.0)), 0);
        assert_eq!(p.distance(quantize(-0.5)), 0);
        assert_eq!(p.distance(quantize(0.5)), 0);
    }

    #[test]
    fn distance_above_is_target_minus_max() {
        let p = Parameter::span(0.0, 0.2);
        // target at 0.5 → quantized 5000; max=2000; distance=3000.
        assert_eq!(p.distance(quantize(0.5)), 3000);
    }

    #[test]
    fn distance_below_is_min_minus_target() {
        let p = Parameter::span(0.3, 0.4);
        // target at 0.0; min=3000; distance=3000.
        assert_eq!(p.distance(quantize(0.0)), 3000);
    }

    #[test]
    fn union_with_none_is_identity() {
        let p = Parameter::span(0.0, 0.5);
        assert_eq!(p.union(None), p);
    }

    #[test]
    fn union_widens_to_outer_bounds() {
        let a = Parameter::span(0.0, 0.3);
        let b = Parameter::span(0.2, 0.7);
        let u = a.union(Some(b));
        assert_eq!(u.min, quantize(0.0));
        assert_eq!(u.max, quantize(0.7));
    }

    #[test]
    fn point_is_degenerate_interval() {
        let p = Parameter::point(0.42);
        assert_eq!(p.min, p.max);
        assert_eq!(p.min, quantize(0.42));
        assert_eq!(p.distance(p.min), 0);
        assert_eq!(p.distance(p.min + 1), 1);
    }

    #[test]
    fn ron_roundtrip_preserves_quantized_values() {
        let p = Parameter::span(-0.5, 0.7);
        let s = ron::to_string(&p).unwrap();
        let parsed: Parameter = ron::from_str(&s).unwrap();
        assert_eq!(parsed, p);
    }
}
```

Verify `src/worldgen/mod.rs` still has `pub mod climate;` declared (it should — PR 1 added it). If missing, add it.

- [ ] **Step 1.2: Run tests to verify they fail / compile**

Run: `cargo test --lib worldgen::climate 2>&1 | tail -15`

Expected: 8 tests pass on first run (the type and impl are inline). If anything fails it's a compile error — fix and re-run.

- [ ] **Step 1.3: Commit**

```bash
git add src/worldgen/climate.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): Climate::Parameter for 6D biome lookup

A quantized (min, max) interval in the integer climate domain
(QUANTIZATION_FACTOR = 10000, matches MC). Supports span/point
construction, gap distance from a target (the squared-L2 summand),
and union for parent-hyperbox derivation.

Foundation for the upcoming ParameterPoint / R-tree / Sampler
types (Tasks 2-7) that replace Biome::classify.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `ClimateParameterPoint` + `ClimateTargetPoint` (TDD)

**Files:**
- Modify: `src/worldgen/climate.rs`

- [ ] **Step 2.1: Write the failing tests**

Append to `src/worldgen/climate.rs` (above the `#[cfg(test)] mod tests` block):

```rust
/// A 6D climate hyperbox plus a biome ID and a tie-breaker offset.
///
/// Six axes match MC: temperature, humidity, continentalness,
/// terrain_shape (called "erosion" in MC), depth, weirdness. The
/// 7th `offset` slot acts as a tie-breaker: smaller offsets win
/// when two boxes are equidistant to the target. Default 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterPoint {
    pub temperature: Parameter,
    pub humidity: Parameter,
    pub continentalness: Parameter,
    pub terrain_shape: Parameter,
    pub depth: Parameter,
    pub weirdness: Parameter,
    #[serde(default)]
    pub offset: i64,
    /// Biome ID this hyperbox claims. Resolved against the biome
    /// enum at config-load time.
    pub biome: BiomeId,
}

/// String-keyed biome ID stored in RON. Resolved to the `Biome`
/// enum variant by [`ParameterList::new`] when the config loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiomeId {
    Tundra,
    SnowyForest,
    Plains,
    Forest,
    Desert,
    Tropical,
}

impl ParameterPoint {
    /// Build the "parameter space" array used by R-tree
    /// construction: 7 intervals (six axes + a degenerate
    /// `[offset, offset]` interval for the tie-breaker).
    pub fn parameter_space(&self) -> [Parameter; 7] {
        [
            self.temperature,
            self.humidity,
            self.continentalness,
            self.terrain_shape,
            self.depth,
            self.weirdness,
            Parameter::new(self.offset, self.offset),
        ]
    }

    /// Brute-force fitness: squared-L2 distance from `target` to
    /// this hyperbox, plus `offset²` (tie-breaker). Used by
    /// `ParameterList::find_value_brute_force` for testing and as
    /// the validation oracle for R-tree query correctness.
    pub fn fitness(&self, target: &TargetPoint) -> i64 {
        sq(self.temperature.distance(target.temperature))
            + sq(self.humidity.distance(target.humidity))
            + sq(self.continentalness.distance(target.continentalness))
            + sq(self.terrain_shape.distance(target.terrain_shape))
            + sq(self.depth.distance(target.depth))
            + sq(self.weirdness.distance(target.weirdness))
            + sq(self.offset)
    }
}

/// The 6D query point in quantized climate space. Built from
/// per-block noise samples via [`Sampler::sample`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetPoint {
    pub temperature: i64,
    pub humidity: i64,
    pub continentalness: i64,
    pub terrain_shape: i64,
    pub depth: i64,
    pub weirdness: i64,
}

impl TargetPoint {
    /// Quantize unquantized `f32` axis values into a TargetPoint.
    pub fn from_floats(
        temperature: f32,
        humidity: f32,
        continentalness: f32,
        terrain_shape: f32,
        depth: f32,
        weirdness: f32,
    ) -> Self {
        Self {
            temperature: quantize(temperature),
            humidity: quantize(humidity),
            continentalness: quantize(continentalness),
            terrain_shape: quantize(terrain_shape),
            depth: quantize(depth),
            weirdness: quantize(weirdness),
        }
    }

    /// Pack into a length-7 array matching `parameter_space()` so
    /// the R-tree distance loop can iterate uniformly. The 7th slot
    /// is 0 (the target has no offset of its own).
    pub fn to_array(&self) -> [i64; 7] {
        [
            self.temperature,
            self.humidity,
            self.continentalness,
            self.terrain_shape,
            self.depth,
            self.weirdness,
            0,
        ]
    }
}

/// Square an `i64` returning `i64`. Used by `fitness` and by the
/// R-tree distance metric. Overflows for very large arguments
/// (e.g. > 3 billion) — but quantized climate values stay within
/// ±20000, so squared values fit comfortably in `i64`.
#[inline]
fn sq(x: i64) -> i64 {
    x * x
}
```

Append to the existing `#[cfg(test)] mod tests` block:

```rust
    fn sample_pp() -> ParameterPoint {
        ParameterPoint {
            temperature: Parameter::span(-0.5, 0.5),
            humidity: Parameter::span(-1.0, 1.0),
            continentalness: Parameter::span(0.0, 1.0),
            terrain_shape: Parameter::span(-1.0, 1.0),
            depth: Parameter::span(0.0, 0.0),
            weirdness: Parameter::span(-1.0, 1.0),
            offset: 0,
            biome: BiomeId::Plains,
        }
    }

    #[test]
    fn parameter_space_is_seven_intervals() {
        let pp = sample_pp();
        let space = pp.parameter_space();
        assert_eq!(space.len(), 7);
        assert_eq!(space[0], pp.temperature);
        assert_eq!(space[6].min, 0);
        assert_eq!(space[6].max, 0);
    }

    #[test]
    fn fitness_zero_when_target_inside_box() {
        let pp = sample_pp();
        let target = TargetPoint::from_floats(0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        assert_eq!(pp.fitness(&target), 0);
    }

    #[test]
    fn fitness_grows_quadratically_with_gap() {
        let pp = sample_pp();
        // Target outside the temperature box by 0.1 (quantized 1000).
        let t1 = TargetPoint::from_floats(0.6, 0.0, 0.5, 0.0, 0.0, 0.0);
        // Target outside by 0.2 (quantized 2000).
        let t2 = TargetPoint::from_floats(0.7, 0.0, 0.5, 0.0, 0.0, 0.0);
        assert_eq!(pp.fitness(&t1), 1000 * 1000);
        assert_eq!(pp.fitness(&t2), 2000 * 2000);
        // 4× growth for 2× gap → quadratic.
    }

    #[test]
    fn fitness_includes_offset_squared() {
        let mut pp = sample_pp();
        pp.offset = 100;
        let target = TargetPoint::from_floats(0.0, 0.0, 0.5, 0.0, 0.0, 0.0);
        assert_eq!(pp.fitness(&target), 100 * 100);
    }

    #[test]
    fn target_to_array_has_seven_slots_with_zero_offset() {
        let t = TargetPoint::from_floats(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        let arr = t.to_array();
        assert_eq!(arr.len(), 7);
        assert_eq!(arr[6], 0);
        assert_eq!(arr[0], quantize(0.1));
    }
```

- [ ] **Step 2.2: Run tests to verify they pass**

Run: `cargo test --lib worldgen::climate 2>&1 | tail -10`

Expected: 13 tests pass total (8 from Task 1 + 5 new).

- [ ] **Step 2.3: Commit**

```bash
git add src/worldgen/climate.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): ClimateParameterPoint + TargetPoint

A ParameterPoint bundles six Parameter intervals (T, H, C, terrain
shape, depth, weirdness) plus a tie-breaker offset and the claimed
biome ID. parameter_space() returns the 7-slot array consumed by
the R-tree builder; fitness() computes the brute-force squared-L2
distance for testing.

TargetPoint is the analogous 6-tuple of quantized scalars (the
query point). Both pack/unpack to/from i64 arrays for tight inner
loops in the R-tree distance metric.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `ClimateRTree` build + squared-L2 search (TDD)

**Files:**
- Modify: `src/worldgen/climate.rs`

- [ ] **Step 3.1: Write the failing tests**

Append the R-tree implementation to `src/worldgen/climate.rs` (above the `#[cfg(test)] mod tests` block):

```rust
use std::cell::Cell;

/// Fanout-6 R-tree on `ParameterPoint`s indexed by their 6D hyperbox
/// (+ offset tie-breaker). Built once at config load. Queries walk
/// the tree from the root with squared-L2 over per-axis hyperbox
/// gaps; a `ThreadLocal` last-leaf cache short-circuits queries
/// whose target happens to lie inside the previous winner's
/// hyperbox.
///
/// Mirrors `net/minecraft/world/level/biome/Climate.java#RTree`.
pub struct RTree {
    root: Node,
    /// Per-thread cache: the leaf returned by the most recent query
    /// on this thread. Used as a seed candidate so a sequence of
    /// nearby queries can skip the full descent when the answer is
    /// stable (very common for chunk-fill: 1024 columns × similar
    /// climate ≈ same biome). Wrapped in `Cell` because `RTree` is
    /// shared `&self`; the leaf index is just a non-binding hint.
    last_leaf: thread_local::ThreadLocal<Cell<Option<usize>>>,
}

/// One node in the R-tree. Each node owns a 7-axis hyperbox
/// (`parameter_space`) that bounds every descendant. Leaves carry
/// a `(ParameterPoint, leaf_index)` pair; subtrees own a `Vec<Node>`
/// of children (up to `RTREE_FANOUT`).
enum Node {
    Leaf {
        parameter_space: [Parameter; 7],
        point: ParameterPoint,
        /// Index into `RTree::leaves` — used by the last-leaf cache.
        leaf_index: usize,
    },
    SubTree {
        parameter_space: [Parameter; 7],
        children: Vec<Node>,
    },
}

/// Branching factor. Matches MC's `CHILDREN_PER_NODE = 6`.
pub const RTREE_FANOUT: usize = 6;

impl Node {
    fn parameter_space(&self) -> &[Parameter; 7] {
        match self {
            Node::Leaf { parameter_space, .. } => parameter_space,
            Node::SubTree { parameter_space, .. } => parameter_space,
        }
    }

    /// Squared-L2 distance from `target` to this node's bounding
    /// hyperbox (zero if `target` is inside the box on every axis).
    fn distance(&self, target: &[i64; 7]) -> i64 {
        let space = self.parameter_space();
        let mut total = 0i64;
        for axis in 0..7 {
            total += sq(space[axis].distance(target[axis]));
        }
        total
    }
}

impl RTree {
    /// Build an R-tree from a non-empty slice of biome claims. The
    /// algorithm follows MC verbatim: when ≤6 claims remain, group
    /// them into a single subtree; otherwise pick the best axis (by
    /// minimum total bucketed-cost) to sort + bucket on, then
    /// recurse. Build is O(N log_6 N) — fine for ≤100 claims.
    pub fn new(points: &[ParameterPoint]) -> Self {
        assert!(!points.is_empty(), "RTree::new requires ≥1 claim");
        let leaves: Vec<Node> = points
            .iter()
            .enumerate()
            .map(|(i, p)| Node::Leaf {
                parameter_space: p.parameter_space(),
                point: *p,
                leaf_index: i,
            })
            .collect();
        let root = build(leaves);
        Self {
            root,
            last_leaf: thread_local::ThreadLocal::new(),
        }
    }

    /// Query the tree: returns the biome whose hyperbox is nearest
    /// (in squared-L2) to `target`, with ties broken by smaller
    /// `offset`.
    pub fn search(&self, target: &TargetPoint) -> BiomeId {
        let target_arr = target.to_array();
        // Seed the search with the last leaf's distance (if any) so
        // we can prune subtrees whose bounding-box distance already
        // exceeds the seed.
        let last_cell = self.last_leaf.get_or(|| Cell::new(None));
        let mut best_distance = i64::MAX;
        let mut best_point: Option<ParameterPoint> = None;
        let mut best_index: Option<usize> = None;
        if let Some(idx) = last_cell.get() {
            if let Some((p, d)) = leaf_at(&self.root, idx, &target_arr) {
                best_distance = d;
                best_point = Some(p);
                best_index = Some(idx);
            }
        }
        search_descend(
            &self.root,
            &target_arr,
            &mut best_distance,
            &mut best_point,
            &mut best_index,
        );
        last_cell.set(best_index);
        best_point.expect("RTree must have ≥1 claim by construction").biome
    }
}

/// Recursive descent. For each subtree child whose bounding-box
/// distance is strictly less than the current best leaf distance,
/// descend. Leaves update the best on direct comparison.
fn search_descend(
    node: &Node,
    target: &[i64; 7],
    best_distance: &mut i64,
    best_point: &mut Option<ParameterPoint>,
    best_index: &mut Option<usize>,
) {
    match node {
        Node::Leaf { point, leaf_index, .. } => {
            // For a Leaf, the bounding-box-to-target distance IS the
            // fitness (since the leaf's parameter_space == the
            // ParameterPoint's hyperbox), plus the offset². Use
            // fitness directly to include the offset tie-breaker.
            let d = point.fitness(&TargetPoint {
                temperature: target[0],
                humidity: target[1],
                continentalness: target[2],
                terrain_shape: target[3],
                depth: target[4],
                weirdness: target[5],
            });
            if d < *best_distance {
                *best_distance = d;
                *best_point = Some(*point);
                *best_index = Some(*leaf_index);
            }
        }
        Node::SubTree { children, .. } => {
            for child in children {
                let child_dist = child.distance(target);
                if child_dist < *best_distance {
                    search_descend(child, target, best_distance, best_point, best_index);
                }
            }
        }
    }
}

/// Find the leaf at `idx` by tree walk; return its (point, fitness)
/// for the given target. Used by the last-leaf cache seed. None if
/// the index is stale (shouldn't happen unless the tree is rebuilt
/// while a query is in flight — which the ConfigHolder semantics
/// prevent).
fn leaf_at(node: &Node, idx: usize, target: &[i64; 7]) -> Option<(ParameterPoint, i64)> {
    match node {
        Node::Leaf { point, leaf_index, .. } if *leaf_index == idx => {
            let d = point.fitness(&TargetPoint {
                temperature: target[0],
                humidity: target[1],
                continentalness: target[2],
                terrain_shape: target[3],
                depth: target[4],
                weirdness: target[5],
            });
            Some((*point, d))
        }
        Node::Leaf { .. } => None,
        Node::SubTree { children, .. } => {
            for child in children {
                if let Some(found) = leaf_at(child, idx, target) {
                    return Some(found);
                }
            }
            None
        }
    }
}

/// Bottom-up R-tree build. Mirrors MC's `build` recursion.
fn build(mut children: Vec<Node>) -> Node {
    assert!(!children.is_empty(), "build needs ≥1 child");
    if children.len() == 1 {
        return children.into_iter().next().unwrap();
    }
    if children.len() <= RTREE_FANOUT {
        // Small group: just sort by absolute axis-center sum (a
        // cheap proxy for locality) and wrap in a single subtree.
        children.sort_by_key(|n| {
            let space = n.parameter_space();
            (0..7)
                .map(|d| ((space[d].min + space[d].max) / 2).abs())
                .sum::<i64>()
        });
        let parameter_space = parent_hyperbox(&children);
        return Node::SubTree { parameter_space, children };
    }
    // ≥7 children: find the best axis (minimum total bucket cost)
    // to sort and partition on. Cost = sum of |max - min| across
    // all axes, for each bucket. MC's heuristic.
    let mut best_axis = 0usize;
    let mut best_cost = i64::MAX;
    let mut best_buckets: Vec<Vec<Node>> = Vec::new();
    for axis in 0..7 {
        let mut probe = children.iter().enumerate().collect::<Vec<_>>();
        probe.sort_by_key(|(_, n)| {
            let p = n.parameter_space()[axis];
            (p.min + p.max) / 2
        });
        let sorted_indices: Vec<usize> = probe.into_iter().map(|(i, _)| i).collect();
        let sorted: Vec<Node> = sorted_indices
            .iter()
            .map(|&i| clone_node(&children[i]))
            .collect();
        let buckets = bucketize(sorted);
        let cost: i64 = buckets
            .iter()
            .map(|b| {
                let hb = parent_hyperbox(b);
                (0..7).map(|d| (hb[d].max - hb[d].min).abs()).sum::<i64>()
            })
            .sum();
        if cost < best_cost {
            best_cost = cost;
            best_axis = axis;
            best_buckets = buckets;
        }
    }
    // Re-sort children list by best axis (absolute center this
    // time, like MC's second pass) and rebucket.
    children.sort_by_key(|n| {
        let p = n.parameter_space()[best_axis];
        ((p.min + p.max) / 2).abs()
    });
    let _ = best_buckets;
    let buckets = bucketize(children);
    let built: Vec<Node> = buckets.into_iter().map(build).collect();
    let parameter_space = parent_hyperbox(&built);
    Node::SubTree { parameter_space, children: built }
}

/// Clone a Node tree (Nodes hold `Vec<Node>`, not `Arc`, so this
/// is a deep clone used by the multi-axis probe).
fn clone_node(node: &Node) -> Node {
    match node {
        Node::Leaf { parameter_space, point, leaf_index } => Node::Leaf {
            parameter_space: *parameter_space,
            point: *point,
            leaf_index: *leaf_index,
        },
        Node::SubTree { parameter_space, children } => Node::SubTree {
            parameter_space: *parameter_space,
            children: children.iter().map(clone_node).collect(),
        },
    }
}

/// Group a sorted list of nodes into ≤`RTREE_FANOUT` buckets,
/// each holding `6^floor(log_6(n-ε))` consecutive nodes. Mirrors
/// MC's `bucketize`: floor-log-base-6 of (n - 0.01) gives the
/// expected bucket count, then nodes are packed in order.
fn bucketize(nodes: Vec<Node>) -> Vec<Vec<Node>> {
    let n = nodes.len();
    if n <= 1 {
        return vec![nodes];
    }
    // expected = 6^floor(log_6(n - 0.01))
    let expected = 6f64.powf(((n as f64) - 0.01).log(6.0).floor()) as usize;
    let expected = expected.max(1);
    let mut buckets: Vec<Vec<Node>> = Vec::new();
    let mut current: Vec<Node> = Vec::new();
    for node in nodes {
        current.push(node);
        if current.len() >= expected {
            buckets.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        buckets.push(current);
    }
    buckets
}

/// Compute the parent hyperbox covering every child node.
fn parent_hyperbox(children: &[Node]) -> [Parameter; 7] {
    let mut bounds: [Option<Parameter>; 7] = [None; 7];
    for child in children {
        let space = child.parameter_space();
        for axis in 0..7 {
            bounds[axis] = Some(space[axis].union(bounds[axis]));
        }
    }
    let mut result = [Parameter::new(0, 0); 7];
    for axis in 0..7 {
        result[axis] = bounds[axis].expect("parent_hyperbox needs ≥1 child");
    }
    result
}
```

Add the `thread_local` crate to `Cargo.toml`'s `[dependencies]` block (if not already present):

```toml
thread_local = "1.1"
```

Append to the `#[cfg(test)] mod tests` block:

```rust
    fn make_two_box_tree() -> RTree {
        let cold = ParameterPoint {
            temperature: Parameter::span(-1.0, 0.0),
            humidity: Parameter::span(-1.0, 1.0),
            continentalness: Parameter::span(-1.0, 1.0),
            terrain_shape: Parameter::span(-1.0, 1.0),
            depth: Parameter::span(0.0, 0.0),
            weirdness: Parameter::span(-1.0, 1.0),
            offset: 0,
            biome: BiomeId::Tundra,
        };
        let hot = ParameterPoint {
            temperature: Parameter::span(0.0, 1.0),
            humidity: Parameter::span(-1.0, 1.0),
            continentalness: Parameter::span(-1.0, 1.0),
            terrain_shape: Parameter::span(-1.0, 1.0),
            depth: Parameter::span(0.0, 0.0),
            weirdness: Parameter::span(-1.0, 1.0),
            offset: 0,
            biome: BiomeId::Desert,
        };
        RTree::new(&[cold, hot])
    }

    #[test]
    fn rtree_two_boxes_lookup_cold_target() {
        let tree = make_two_box_tree();
        let target = TargetPoint::from_floats(-0.5, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(tree.search(&target), BiomeId::Tundra);
    }

    #[test]
    fn rtree_two_boxes_lookup_hot_target() {
        let tree = make_two_box_tree();
        let target = TargetPoint::from_floats(0.7, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(tree.search(&target), BiomeId::Desert);
    }

    #[test]
    fn rtree_returns_nearest_when_no_box_contains_target() {
        let tree = make_two_box_tree();
        // Both boxes span depth=0; query with depth=0.5 → both
        // equally far on depth axis; tie broken by temperature.
        let target = TargetPoint::from_floats(-0.2, 0.0, 0.0, 0.0, 0.5, 0.0);
        // -0.2 is inside cold's temperature box → cold wins.
        assert_eq!(tree.search(&target), BiomeId::Tundra);
    }

    #[test]
    fn rtree_brute_force_oracle_agrees_for_random_targets() {
        // Build a 6-box tree (one per Oxium biome) and validate
        // R-tree against brute-force fitness on 200 random targets.
        let points = vec![
            ParameterPoint {
                temperature: Parameter::span(-1.0, -0.3),
                humidity: Parameter::span(-1.0, 0.0),
                continentalness: Parameter::span(-1.0, 1.0),
                terrain_shape: Parameter::span(-1.0, 1.0),
                depth: Parameter::span(0.0, 0.0),
                weirdness: Parameter::span(-1.0, 1.0),
                offset: 0,
                biome: BiomeId::Tundra,
            },
            ParameterPoint {
                temperature: Parameter::span(-1.0, -0.3),
                humidity: Parameter::span(0.0, 1.0),
                continentalness: Parameter::span(-1.0, 1.0),
                terrain_shape: Parameter::span(-1.0, 1.0),
                depth: Parameter::span(0.0, 0.0),
                weirdness: Parameter::span(-1.0, 1.0),
                offset: 0,
                biome: BiomeId::SnowyForest,
            },
            ParameterPoint {
                temperature: Parameter::span(-0.3, 0.2),
                humidity: Parameter::span(-1.0, 0.05),
                continentalness: Parameter::span(-1.0, 1.0),
                terrain_shape: Parameter::span(-1.0, 1.0),
                depth: Parameter::span(0.0, 0.0),
                weirdness: Parameter::span(-1.0, 1.0),
                offset: 0,
                biome: BiomeId::Plains,
            },
            ParameterPoint {
                temperature: Parameter::span(-0.3, 0.2),
                humidity: Parameter::span(0.05, 1.0),
                continentalness: Parameter::span(-1.0, 1.0),
                terrain_shape: Parameter::span(-1.0, 1.0),
                depth: Parameter::span(0.0, 0.0),
                weirdness: Parameter::span(-1.0, 1.0),
                offset: 0,
                biome: BiomeId::Forest,
            },
            ParameterPoint {
                temperature: Parameter::span(0.55, 1.0),
                humidity: Parameter::span(-1.0, 1.0),
                continentalness: Parameter::span(-1.0, 1.0),
                terrain_shape: Parameter::span(-1.0, 1.0),
                depth: Parameter::span(0.0, 0.0),
                weirdness: Parameter::span(-1.0, 1.0),
                offset: 0,
                biome: BiomeId::Desert,
            },
            ParameterPoint {
                temperature: Parameter::span(0.2, 0.55),
                humidity: Parameter::span(0.05, 1.0),
                continentalness: Parameter::span(-1.0, 1.0),
                terrain_shape: Parameter::span(-1.0, 1.0),
                depth: Parameter::span(0.0, 0.0),
                weirdness: Parameter::span(-1.0, 1.0),
                offset: 0,
                biome: BiomeId::Tropical,
            },
        ];
        let tree = RTree::new(&points);
        // Deterministic test: walk an 11×11 grid in (T, H) at fixed
        // C=E=D=W=0; assert tree result matches brute force.
        for i in 0..11 {
            for j in 0..11 {
                let t = -1.0 + 0.2 * i as f32;
                let h = -1.0 + 0.2 * j as f32;
                let target = TargetPoint::from_floats(t, h, 0.0, 0.0, 0.0, 0.0);
                // Brute force: min fitness across all points.
                let bf = points
                    .iter()
                    .min_by_key(|p| p.fitness(&target))
                    .unwrap()
                    .biome;
                let rt = tree.search(&target);
                assert_eq!(rt, bf, "mismatch at T={}, H={}", t, h);
            }
        }
    }

    #[test]
    fn rtree_last_leaf_cache_is_consistent() {
        // Two successive queries at the same target must return
        // the same biome (the cache shouldn't corrupt results).
        let tree = make_two_box_tree();
        let target = TargetPoint::from_floats(0.5, 0.0, 0.0, 0.0, 0.0, 0.0);
        let a = tree.search(&target);
        let b = tree.search(&target);
        assert_eq!(a, b);
    }
```

- [ ] **Step 3.2: Verify tests fail / compile errors first**

Run: `cargo test --lib worldgen::climate 2>&1 | tail -20`

Expected: either compile errors (if `thread_local` isn't yet a dep) or some new tests pass / some fail. If compile errors, add the dep and re-run.

- [ ] **Step 3.3: Iterate until tests pass**

Most likely points of failure:
- The `bucketize` "expected" formula edge case at `n=2` (log produces a non-integer; floor should give 1).
- The `search_descend` early-return logic when the cached best is already the answer.

Fix bugs until: `cargo test --lib worldgen::climate 2>&1 | tail -10` shows 18 tests pass total (13 prior + 5 R-tree).

- [ ] **Step 3.4: Commit**

```bash
git add src/worldgen/climate.rs Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
feat(worldgen): ClimateRTree — fanout-6 spatial index over biome claims

Build phase: O(N log_6 N) recursive bucket-by-best-axis matching
MC's net/minecraft/world/level/biome/Climate.java#build. Query phase:
descent with subtree pruning by parent-hyperbox distance, leaves
compared on full fitness (includes the offset tie-breaker).

ThreadLocal last-leaf cache: each chunk-fill thread keeps the
previously-returned leaf as a seed candidate. Chunk fills have
strong climate locality (1024 columns × similar T/H/C/E/W/D), so
the cache hit-rate dominates and avoids the full descent.

Validated against brute-force fitness over an 11×11 (T, H) grid
on six representative claims — R-tree must agree with brute force
on every cell.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `weirdness` noise channel + `Sampler` (TDD)

**Files:**
- Modify: `src/worldgen/heightmap.rs` (add `WeirdnessNoise` next to `DensityNoise`)
- Modify: `src/worldgen/climate.rs` (add `Sampler`)
- Modify: `src/worldgen/mod.rs` (add `weirdness` field on `Generator`)

- [ ] **Step 4.1: Write the failing tests**

Append to the heightmap test module in `src/worldgen/heightmap.rs`:

```rust
    #[test]
    fn weirdness_noise_in_range_and_deterministic() {
        let w = WeirdnessNoise::new(42);
        let a = w.evaluate(100, 200);
        let b = w.evaluate(100, 200);
        assert_eq!(a, b, "deterministic for same coord");
        // Simplex returns in approx [-1, 1].
        assert!(a >= -1.5 && a <= 1.5, "weirdness out of expected range: {a}");
    }

    #[test]
    fn weirdness_varies_over_long_distance() {
        let w = WeirdnessNoise::new(42);
        let a = w.evaluate(0, 0);
        let b = w.evaluate(2000, 2000);
        assert!((a - b).abs() > 0.05, "weirdness should vary over 2000 blocks");
    }
```

Append to `src/worldgen/heightmap.rs` (alongside `DensityNoise`):

```rust
/// "Weirdness" noise: a low-frequency 2D field that drives biome
/// variants (sunflower plains, ice spikes, etc. in MC). Independent
/// of climate (T/H) and macro shape (C, terrain_shape). PR 4 uses it
/// as the 6th axis of the biome lookup; values feed in raw — PR 5+
/// may apply a `peaksAndValleys` fold for the heightmap pipeline.
pub struct WeirdnessNoise {
    field: Fbm<Simplex>,
}

impl WeirdnessNoise {
    pub fn new(seed: u64) -> Self {
        let field = Fbm::<Simplex>::new(seed.wrapping_add(307) as u32)
            .set_octaves(3)
            .set_frequency(1.0 / 500.0)
            .set_persistence(0.5);
        Self { field }
    }

    /// Sample at world `(wx, wz)` — depth-invariant (matches MC).
    pub fn evaluate(&self, wx: i32, wz: i32) -> f32 {
        self.field.get([wx as f64, wz as f64]) as f32
    }
}
```

Append to `src/worldgen/climate.rs` (above the `#[cfg(test)] mod tests` block):

```rust
use crate::worldgen::heightmap::WeirdnessNoise;
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

/// Wraps every climate noise channel + the y-clamped-gradient
/// `depth` field. Produces a [`TargetPoint`] at any world position.
///
/// Channels:
/// - **temperature**, **humidity** — existing Oxium FBMs (Generator
///   owns them; the Sampler holds references).
/// - **continentalness**, **terrain_shape** — PR 3 channels. Until
///   PR 3 lands, both default to constant-zero (placeholder); the
///   axes are still present in queries so claim hyperboxes don't
///   need to be rewritten when PR 3 hooks up real channels.
/// - **weirdness** — PR 4's new low-freq channel ([`WeirdnessNoise`]).
/// - **depth** — `yClampedGradient(wy)`: +1 at `y_min`, -1 at
///   `y_max`. Lifted from PR 2's `DensityConfig`.
pub struct Sampler<'a> {
    pub temperature: &'a Fbm<Simplex>,
    pub humidity: &'a Fbm<Simplex>,
    pub continentalness: &'a Fbm<Simplex>,
    pub terrain_shape: &'a Fbm<Simplex>,
    pub weirdness: &'a WeirdnessNoise,
    pub y_min: i32,
    pub y_max: i32,
}

impl<'a> Sampler<'a> {
    /// Sample the 6D target point at world position `(wx, wy, wz)`.
    /// Continentalness, terrain_shape, weirdness, T, H are
    /// y-independent — only `depth` varies with `wy`.
    pub fn sample(&self, wx: i32, wy: i32, wz: i32) -> TargetPoint {
        let xz = [wx as f64, wz as f64];
        let t = self.temperature.get(xz) as f32;
        let h = self.humidity.get(xz) as f32;
        let c = self.continentalness.get(xz) as f32;
        let e = self.terrain_shape.get(xz) as f32;
        let w = self.weirdness.evaluate(wx, wz);
        let d = self.depth_at(wy);
        TargetPoint::from_floats(t, h, c, e, d, w)
    }

    /// `yClampedGradient(wy)`: +1 at `y_min`, -1 at `y_max`,
    /// linear in between, clamped outside [y_min, y_max]. Matches
    /// MC's `Mth.clampedMap(wy, y_min, y_max, +1, -1)`.
    fn depth_at(&self, wy: i32) -> f32 {
        let t = (wy - self.y_min) as f32 / (self.y_max - self.y_min) as f32;
        let depth = 1.0 - 2.0 * t;
        depth.clamp(-1.0, 1.0)
    }
}
```

In `src/worldgen/mod.rs`, locate the `Generator` struct definition. Add the new fields (next to `temperature_map`, `humidity_map`):

```rust
    /// PR 3 placeholder: continentalness noise. Real PR 3 channel
    /// replaces this with a plate-distance-driven field; PR 4
    /// inserts a low-freq FBM stub so the axis exists. Tuned so
    /// values stay near 0 — biome claims that span the full ±1
    /// range will fire for now (continentalness-gating is PR 3's
    /// job).
    continentalness_map: Fbm<Simplex>,
    /// PR 3 placeholder: terrain_shape noise (MC calls this
    /// "erosion"). Low-freq 2D FBM, same scale as continentalness.
    terrain_shape_map: Fbm<Simplex>,
    /// PR 4: weirdness — the 6th axis of the biome lookup.
    weirdness: heightmap::WeirdnessNoise,
```

In `Generator::new`, after the `humidity_map` construction, add:

```rust
        let continentalness_map = Fbm::<Simplex>::new(seed.wrapping_add(11) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 2048.0)
            .set_persistence(0.5);
        let terrain_shape_map = Fbm::<Simplex>::new(seed.wrapping_add(12) as u32)
            .set_octaves(2)
            .set_frequency(1.0 / 2048.0)
            .set_persistence(0.5);
        let weirdness = heightmap::WeirdnessNoise::new(seed);
```

And add the fields to the `Self { ... }` literal at the end of `Generator::new`.

Add a method to `Generator`:

```rust
    /// Build a climate [`Sampler`] that borrows this generator's
    /// noise fields. Use for `Sampler::sample(wx, wy, wz)` inside
    /// `column_data_with` and `fill_chunk`.
    pub fn climate_sampler(&self) -> crate::worldgen::climate::Sampler<'_> {
        let cfg = self.config_snapshot();
        crate::worldgen::climate::Sampler {
            temperature: &self.temperature_map,
            humidity: &self.humidity_map,
            continentalness: &self.continentalness_map,
            terrain_shape: &self.terrain_shape_map,
            weirdness: &self.weirdness,
            y_min: cfg.density.y_min,
            y_max: cfg.density.y_max,
        }
    }
```

- [ ] **Step 4.2: Run tests to verify they pass**

Run: `cargo test --lib worldgen::heightmap::tests::weirdness_noise 2>&1 | tail -10`

Expected: 2 tests pass.

Then verify `Sampler::sample` is reachable:

Run: `cargo build --lib 2>&1 | tail -5`

Expected: builds cleanly.

- [ ] **Step 4.3: Add a smoke test for `Sampler::sample`**

Append to the climate test module:

```rust
    #[test]
    fn sampler_produces_deterministic_target_point() {
        // Construct stub noise fields. Skip via Generator since we
        // need real seeded FBMs; sample_at via the helper.
        use crate::worldgen::Generator;
        let g = Generator::new(42);
        let s = g.climate_sampler();
        let t1 = s.sample(100, 0, 200);
        let t2 = s.sample(100, 0, 200);
        assert_eq!(t1, t2, "sampler must be deterministic");
    }

    #[test]
    fn sampler_depth_at_y_min_is_plus_one_quantized() {
        use crate::worldgen::Generator;
        let g = Generator::new(42);
        let s = g.climate_sampler();
        let t = s.sample(0, s.y_min, 0);
        assert_eq!(t.depth, quantize(1.0));
    }

    #[test]
    fn sampler_depth_at_y_max_is_minus_one_quantized() {
        use crate::worldgen::Generator;
        let g = Generator::new(42);
        let s = g.climate_sampler();
        let t = s.sample(0, s.y_max, 0);
        assert_eq!(t.depth, quantize(-1.0));
    }
```

Run: `cargo test --lib worldgen::climate 2>&1 | tail -10`

Expected: 21 tests pass total (18 prior + 3 sampler).

- [ ] **Step 4.4: Commit**

```bash
git add src/worldgen/climate.rs src/worldgen/heightmap.rs src/worldgen/mod.rs Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
feat(worldgen): weirdness noise + ClimateSampler

WeirdnessNoise is a new low-frequency 2D Simplex FBM (3 octaves,
500-block period, seed.wrapping_add(307)). Sits alongside
DensityNoise / HeightmapNoise in heightmap.rs.

ClimateSampler bundles the six noise channels (T, H, C,
terrain_shape, weirdness) plus a y-clamped-gradient `depth`
derived from the WorldgenConfig's y_min/y_max. Sampler::sample
returns a quantized TargetPoint at any world (wx, wy, wz).

Continentalness and terrain_shape are placeholders until PR 3
lands real channels; the axes exist so claim hyperboxes don't
need to be rewritten when PR 3 hooks up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `ParameterList` wrapper + RON-loadable biome table

**Files:**
- Modify: `src/worldgen/climate.rs`
- Modify: `src/worldgen/config.rs` (add `biomes` section to `WorldgenConfig`)
- Modify: `assets/worldgen/default.ron`

- [ ] **Step 5.1: Add `ParameterList` and biome resolution**

Append to `src/worldgen/climate.rs` (above the `#[cfg(test)] mod tests` block):

```rust
/// Owned collection of biome claims + a built R-tree. Construct
/// once per config load; queries take `&self`.
pub struct ParameterList {
    pub claims: Vec<ParameterPoint>,
    tree: RTree,
}

impl ParameterList {
    /// Build an R-tree over the given claims.
    pub fn new(claims: Vec<ParameterPoint>) -> Self {
        let tree = RTree::new(&claims);
        Self { claims, tree }
    }

    /// Query: nearest claim's biome ID.
    pub fn find(&self, target: &TargetPoint) -> BiomeId {
        self.tree.search(target)
    }

    /// Brute-force oracle for tests.
    #[cfg(test)]
    pub fn find_brute_force(&self, target: &TargetPoint) -> BiomeId {
        self.claims
            .iter()
            .min_by_key(|p| p.fitness(target))
            .expect("non-empty")
            .biome
    }
}
```

- [ ] **Step 5.2: Add a `biomes` section to `WorldgenConfig`**

In `src/worldgen/config.rs`, add the import:

```rust
use crate::worldgen::climate::ParameterPoint;
```

Add to the `WorldgenConfig` struct:

```rust
pub struct WorldgenConfig {
    pub density: DensityConfig,
    /// PR 4: biome claims for the 6D R-tree lookup. Replaces the
    /// nested-if `Biome::classify`. Hot-reloadable: edit the RON
    /// file to retune claim boxes without recompiling.
    pub biomes: BiomeConfig,
}
```

Add the new struct (after `DensityConfig`):

```rust
/// Biome lookup configuration. PR 4 introduces this section; later
/// PRs extend it with surface-rule conditions and tree-density
/// per-claim overrides.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BiomeConfig {
    /// Per-block Voronoi jitter cell size (blocks). Borders wave
    /// with this period. MC uses ~4 (quart resolution); Oxium
    /// preserves 24-block period to match existing visual
    /// frequency.
    pub jitter_period: u32,
    /// Jitter amplitude in noise-value units (added to the
    /// hashed cell center). 0.5 means the jittered center can
    /// shift up to half a cell.
    pub jitter_amplitude: f32,
    /// Biome claims. Order is meaningful only for tie-breaking
    /// (R-tree query is order-independent for distinct claims).
    pub claims: Vec<ParameterPoint>,
}

impl WorldgenConfig {
    /// Build the `ParameterList` from the RON-loaded claims. Called
    /// by `Generator::with_config` after a hot-reload swap.
    pub fn build_biome_table(&self) -> crate::worldgen::climate::ParameterList {
        crate::worldgen::climate::ParameterList::new(self.biomes.claims.clone())
    }
}
```

- [ ] **Step 5.3: Extend `assets/worldgen/default.ron`**

Append to `assets/worldgen/default.ron` (inside the top-level tuple, after the `density: (...)` block):

```ron
    biomes: (
        // Per-block Voronoi jitter for organic borders.
        jitter_period: 24,
        jitter_amplitude: 0.5,
        // Six claim hyperboxes. The terrain_shape, continentalness,
        // and weirdness axes span the full ±1 range for now
        // (PR 3 will narrow continentalness for ocean/mountain
        // gating; further PRs can add terrain_shape and weirdness
        // variants). The lookup currently behaves like a 2D (T, H)
        // partition with depth=0 — equivalent to the old classify()
        // but data-driven and ready for additional axes.
        claims: [
            // Tundra: cold, dry, any geography.
            (
                temperature:     (min: -10000, max: -3000),
                humidity:        (min: -10000, max: 0),
                continentalness: (min: -10000, max: 10000),
                terrain_shape:   (min: -10000, max: 10000),
                depth:           (min: 0, max: 0),
                weirdness:       (min: -10000, max: 10000),
                offset: 0,
                biome: Tundra,
            ),
            // SnowyForest: cold, humid.
            (
                temperature:     (min: -10000, max: -3000),
                humidity:        (min: 0, max: 10000),
                continentalness: (min: -10000, max: 10000),
                terrain_shape:   (min: -10000, max: 10000),
                depth:           (min: 0, max: 0),
                weirdness:       (min: -10000, max: 10000),
                offset: 0,
                biome: SnowyForest,
            ),
            // Plains: temperate, dry.
            (
                temperature:     (min: -3000, max: 2000),
                humidity:        (min: -10000, max: 500),
                continentalness: (min: -10000, max: 10000),
                terrain_shape:   (min: -10000, max: 10000),
                depth:           (min: 0, max: 0),
                weirdness:       (min: -10000, max: 10000),
                offset: 0,
                biome: Plains,
            ),
            // Forest: temperate, humid.
            (
                temperature:     (min: -3000, max: 2000),
                humidity:        (min: 500, max: 10000),
                continentalness: (min: -10000, max: 10000),
                terrain_shape:   (min: -10000, max: 10000),
                depth:           (min: 0, max: 0),
                weirdness:       (min: -10000, max: 10000),
                offset: 0,
                biome: Forest,
            ),
            // Desert: hot, any humidity, inland-ish, low erosion.
            (
                temperature:     (min: 5500, max: 10000),
                humidity:        (min: -10000, max: 10000),
                continentalness: (min: -10000, max: 10000),
                terrain_shape:   (min: -10000, max: 10000),
                depth:           (min: 0, max: 0),
                weirdness:       (min: -10000, max: 10000),
                offset: 0,
                biome: Desert,
            ),
            // Tropical: warm-hot, humid.
            (
                temperature:     (min: 2000, max: 5500),
                humidity:        (min: 500, max: 10000),
                continentalness: (min: -10000, max: 10000),
                terrain_shape:   (min: -10000, max: 10000),
                depth:           (min: 0, max: 0),
                weirdness:       (min: -10000, max: 10000),
                offset: 0,
                biome: Tropical,
            ),
        ],
    ),
```

- [ ] **Step 5.4: Add tests for ParameterList + config wiring**

Append to the climate test module:

```rust
    #[test]
    fn parameter_list_resolves_six_biome_table() {
        use crate::worldgen::config::WorldgenConfig;
        let cfg = WorldgenConfig::bundled_default().expect("default.ron");
        let table = cfg.build_biome_table();
        assert!(!table.claims.is_empty());
        // Smoke check: cold target → Tundra or SnowyForest.
        let cold_dry = TargetPoint::from_floats(-0.8, -0.5, 0.0, 0.0, 0.0, 0.0);
        let b = table.find(&cold_dry);
        assert!(matches!(b, BiomeId::Tundra | BiomeId::SnowyForest));
        // Hot dry → Desert.
        let hot = TargetPoint::from_floats(0.8, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(table.find(&hot), BiomeId::Desert);
        // Hot wet → Tropical.
        let tropical = TargetPoint::from_floats(0.4, 0.8, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(table.find(&tropical), BiomeId::Tropical);
        // Temperate dry → Plains.
        let plains = TargetPoint::from_floats(0.0, -0.5, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(table.find(&plains), BiomeId::Plains);
        // Temperate wet → Forest.
        let forest = TargetPoint::from_floats(0.0, 0.5, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(table.find(&forest), BiomeId::Forest);
    }

    #[test]
    fn parameter_list_rtree_matches_brute_force_on_default() {
        use crate::worldgen::config::WorldgenConfig;
        let cfg = WorldgenConfig::bundled_default().expect("default.ron");
        let table = cfg.build_biome_table();
        // 13×13 grid in (T, H), depth=0.
        for i in 0..13 {
            for j in 0..13 {
                let t = -1.0 + i as f32 * (2.0 / 12.0);
                let h = -1.0 + j as f32 * (2.0 / 12.0);
                let target = TargetPoint::from_floats(t, h, 0.0, 0.0, 0.0, 0.0);
                assert_eq!(
                    table.find(&target),
                    table.find_brute_force(&target),
                    "R-tree disagrees with brute force at T={t}, H={h}"
                );
            }
        }
    }
```

- [ ] **Step 5.5: Run the suite**

Run: `cargo test --lib worldgen::climate 2>&1 | tail -15`

Expected: 23 tests pass total (21 prior + 2 new).

Run: `cargo test --lib worldgen::config 2>&1 | tail -10`

Expected: existing tests still pass (the new `biomes` section is loaded; `bundled_default_loads_and_parses` should still succeed).

- [ ] **Step 5.6: Commit**

```bash
git add src/worldgen/climate.rs src/worldgen/config.rs assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
feat(worldgen): ParameterList + biome claim table in default.ron

ParameterList owns the Vec<ParameterPoint> claims and the
built RTree. WorldgenConfig::build_biome_table() constructs one
from the RON-loaded biomes section; the Generator calls this
once per config load.

default.ron now ships six claims (one per Oxium biome) covering
the (T, H) plane with depth=0 and other axes wide-open. The
claim hyperboxes match the existing Biome::classify thresholds:
- COLD_THRESHOLD=-0.30, FOREST_HUMIDITY=0.05, desert at T>0.55
This is the migration-equivalence config — task 7 swaps the
runtime call site, golden hashes need re-baselining.

R-tree query is validated against brute-force fitness over a
13×13 (T, H) grid: must agree on every cell.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Per-block Voronoi jitter (TDD)

**Files:**
- Modify: `src/worldgen/climate.rs`

- [ ] **Step 6.1: Write the failing tests**

Append to the climate test module:

```rust
    #[test]
    fn voronoi_jitter_is_deterministic() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let (a_x, a_z) = voronoi_jitter(100, 200, 42, &cfg.biomes);
        let (b_x, b_z) = voronoi_jitter(100, 200, 42, &cfg.biomes);
        assert_eq!((a_x, a_z), (b_x, b_z));
    }

    #[test]
    fn voronoi_jitter_returns_one_of_eight_neighbors() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        let period = cfg.biomes.jitter_period as i32;
        for wx in [-50, 0, 13, 47] {
            for wz in [-50, 0, 13, 47] {
                let (jx, jz) = voronoi_jitter(wx, wz, 42, &cfg.biomes);
                // The returned cell center must be in the 2×2 grid
                // of cells surrounding (wx, wz). Adjacent diagonally
                // (the 8 corners of the surrounding 2x2 cell block,
                // collapsed to 4 unique cell origins).
                let cx = wx.div_euclid(period);
                let cz = wz.div_euclid(period);
                let jitter_cx = jx.div_euclid(period);
                let jitter_cz = jz.div_euclid(period);
                let dx = jitter_cx - cx;
                let dz = jitter_cz - cz;
                assert!(
                    dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1,
                    "jitter at ({wx}, {wz}) escaped 2×2 neighborhood: \
                     ({jx}, {jz}) → cell ({jitter_cx}, {jitter_cz}) vs ({cx}, {cz})"
                );
            }
        }
    }

    #[test]
    fn voronoi_jitter_changes_across_long_distance() {
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
        // Two coords in different jitter cells must (almost always)
        // resolve to different cell centers.
        let (a_x, a_z) = voronoi_jitter(0, 0, 42, &cfg.biomes);
        let (b_x, b_z) = voronoi_jitter(1000, 1000, 42, &cfg.biomes);
        assert_ne!((a_x, a_z), (b_x, b_z));
    }
```

- [ ] **Step 6.2: Implement `voronoi_jitter`**

Append to `src/worldgen/climate.rs` (above the `#[cfg(test)] mod tests` block):

```rust
use crate::worldgen::config::BiomeConfig;
use crate::worldgen::hash::hash_2d;

/// Per-block Voronoi jitter for organic biome borders.
///
/// Returns the world-space `(jx, jz)` of the *nearest jittered*
/// cell center to `(wx, wz)`. Algorithm: hash each of the 8
/// surrounding cell origins, offset them by a `jitter_amplitude`-
/// scaled fraction of `jitter_period`, then return the offset
/// that's closest to `(wx, wz)` in squared L2.
///
/// Use the returned `(jx, jz)` as the world coord for the
/// `Sampler::sample` call — every block within ~1 jitter_period
/// snaps to the same jittered center, so biome borders wave at
/// that frequency without interpolating biome IDs (which would
/// produce checkerboards near 3-way junctions).
///
/// Mirrors MC's approach in `Climate.Sampler` + per-quart hashing
/// (Oxium operates at block resolution, not quart).
pub fn voronoi_jitter(wx: i32, wz: i32, seed: u64, cfg: &BiomeConfig) -> (i32, i32) {
    let period = cfg.jitter_period as i32;
    let amp = cfg.jitter_amplitude;
    // The cell containing (wx, wz).
    let cx = wx.div_euclid(period);
    let cz = wz.div_euclid(period);
    let mut best_dx = 0i64;
    let mut best_dz = 0i64;
    let mut best_d_sq = i64::MAX;
    // Walk the 2×2 neighborhood centered on (cx, cz). For each cell
    // origin, hash to compute the jittered center; pick the nearest.
    // (MC walks a 2×2 grid because the cell-origin → center vector
    // is at most ±jitter_amplitude·period long, so a 2×2 covers
    // every possible nearest center.)
    for dx in 0..=1 {
        for dz in 0..=1 {
            let ox = cx + dx;
            let oz = cz + dz;
            // Cell origin in world coords.
            let origin_wx = ox * period;
            let origin_wz = oz * period;
            // Hash to derive a sub-cell offset in [-amp, amp]·period.
            let h = hash_2d(seed.wrapping_add(0x510B_B0A8), ox, oz);
            let h2 = hash_2d(seed.wrapping_add(0xA105_55EE), ox, oz);
            let frac_x = ((h & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
            let frac_z = ((h2 & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
            let jitter_x = (frac_x * amp * period as f32) as i32;
            let jitter_z = (frac_z * amp * period as f32) as i32;
            let jx = origin_wx + jitter_x;
            let jz = origin_wz + jitter_z;
            let dxw = (jx - wx) as i64;
            let dzw = (jz - wz) as i64;
            let d_sq = dxw * dxw + dzw * dzw;
            if d_sq < best_d_sq {
                best_d_sq = d_sq;
                best_dx = jx as i64;
                best_dz = jz as i64;
            }
        }
    }
    (best_dx as i32, best_dz as i32)
}
```

Verify `crate::worldgen::hash::hash_2d` exists (it's used widely by Oxium's plates/caves modules). If the signature differs, adapt — the function name is the only stable identifier; arg order may need re-derivation.

- [ ] **Step 6.3: Run tests**

Run: `cargo test --lib worldgen::climate 2>&1 | tail -10`

Expected: 26 tests pass total (23 prior + 3 jitter).

If `voronoi_jitter_returns_one_of_eight_neighbors` fails, the most likely cause is the 2×2 walk only covers cells `(cx, cz) .. (cx+1, cz+1)` — but the test asserts `dx ∈ [-1, 1]`. The expected behavior is that the jitter can pull the nearest center from any of the 4 surrounding cells; if your impl walks `(cx-1..=cx, cz-1..=cz)` instead, you'll still pass the test but the visual coverage differs subtly. Match MC's `(cx, cx+1)` × `(cz, cz+1)` walk.

- [ ] **Step 6.4: Commit**

```bash
git add src/worldgen/climate.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): per-block Voronoi jitter for biome borders

voronoi_jitter(wx, wz, seed, cfg) walks the 2×2 cell neighborhood
around (wx, wz), hashes each cell origin to derive a sub-cell
offset in [-amplitude, amplitude]·period, and returns the world
coord of the *nearest* jittered cell center to (wx, wz).

Use the returned (jx, jz) as the world coord for Sampler::sample
— every block within roughly one jitter_period snaps to the same
jittered center, so biome borders wave at that frequency without
interpolating biome IDs (which would produce checkerboards near
3-way climate junctions).

jitter_period=24 and jitter_amplitude=0.5 in default.ron preserve
Oxium's existing visual border frequency (24 blocks per wave).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `Generator::biome_at` + remove `Biome::classify`

**Files:**
- Modify: `src/worldgen/mod.rs`

- [ ] **Step 7.1: Write the failing test**

Append to the worldgen test module in `src/worldgen/mod.rs`:

```rust
    #[test]
    fn biome_at_returns_one_of_six_biomes() {
        let g = Generator::new(42);
        // Sample biome at 100 random columns.
        for i in 0..100 {
            let wx = i * 37 - 1850;
            let wz = i * 71 + 1200;
            let b = g.biome_at(wx, 64, wz);
            assert!(
                matches!(
                    b,
                    Biome::Tundra
                        | Biome::SnowyForest
                        | Biome::Plains
                        | Biome::Forest
                        | Biome::Desert
                        | Biome::Tropical
                ),
                "unexpected biome at ({wx}, {wz}): {:?}",
                b
            );
        }
    }

    #[test]
    fn biome_at_is_voronoi_jittered() {
        // The point of jitter is that biome doesn't change at exact
        // cell boundaries — small steps within a jitter cell stay
        // on the same biome. Take one column with a known biome,
        // assert the column 3 blocks away is most likely the same.
        let g = Generator::new(42);
        let mut same = 0;
        let mut total = 0;
        for wx_base in (0..100).step_by(10) {
            for wz_base in (0..100).step_by(10) {
                let base = g.biome_at(wx_base, 64, wz_base);
                let near = g.biome_at(wx_base + 3, 64, wz_base + 3);
                if base == near {
                    same += 1;
                }
                total += 1;
            }
        }
        // At least 80% of small-step pairs should share a biome
        // (the jitter snaps both to the same cell most of the time).
        let ratio = same as f32 / total as f32;
        assert!(ratio > 0.75, "small-step biome stability too low: {ratio}");
    }

    #[test]
    fn biome_at_changes_across_long_distance() {
        // A column at (0, 0) and (4000, 4000) should *almost always*
        // be different biomes (climate noises decorrelate at
        // ~512-block scale).
        let g = Generator::new(42);
        let a = g.biome_at(0, 64, 0);
        let b = g.biome_at(4000, 64, 4000);
        // Not a hard assertion — just sanity that the function
        // isn't returning a constant. If both happen to be the
        // same biome, try another long-distance pair.
        let c = g.biome_at(-3000, 64, 5000);
        let any_diff = a != b || a != c || b != c;
        assert!(any_diff, "biome_at appears to return constant: {a:?}");
    }
```

- [ ] **Step 7.2: Implement `biome_at` and store the `ParameterList`**

In `src/worldgen/mod.rs`, add the field to `Generator`:

```rust
    /// PR 4: built R-tree over the biome claims. Rebuilt on
    /// config hot-reload. Wrapped in `Arc` so reads are
    /// cheap; the holder swaps the whole `Arc` atomically.
    biome_table: std::sync::Arc<crate::worldgen::climate::ParameterList>,
```

In the `Generator::with_config` body, after the config is loaded, build the table:

```rust
        let cfg_snapshot = config.load();
        let biome_table = std::sync::Arc::new(cfg_snapshot.build_biome_table());
```

And include `biome_table` in the `Self { ... }` literal.

Add the method:

```rust
impl Generator {
    /// PR 4: look up the biome at world `(wx, wy, wz)` via the 6D
    /// R-tree. Applies per-block Voronoi jitter (8-corner hash,
    /// nearest cell center) before sampling, so biome borders wave
    /// at `cfg.biomes.jitter_period` without interpolation.
    pub fn biome_at(&self, wx: i32, wy: i32, wz: i32) -> Biome {
        let cfg = self.config_snapshot();
        let (jx, jz) = crate::worldgen::climate::voronoi_jitter(
            wx, wz, self.seed, &cfg.biomes,
        );
        let sampler = self.climate_sampler();
        let target = sampler.sample(jx, wy, jz);
        let id = self.biome_table.find(&target);
        Biome::from_id(id)
    }
}
```

Add the `Biome::from_id` helper next to the `Biome` enum:

```rust
impl Biome {
    /// Map a `BiomeId` (RON-loaded enum from the climate table)
    /// to the existing internal `Biome` enum. Trivial 1:1 for now;
    /// later PRs can add underground variants without breaking
    /// the public `Biome` shape.
    fn from_id(id: crate::worldgen::climate::BiomeId) -> Self {
        use crate::worldgen::climate::BiomeId;
        match id {
            BiomeId::Tundra => Biome::Tundra,
            BiomeId::SnowyForest => Biome::SnowyForest,
            BiomeId::Plains => Biome::Plains,
            BiomeId::Forest => Biome::Forest,
            BiomeId::Desert => Biome::Desert,
            BiomeId::Tropical => Biome::Tropical,
        }
    }
}
```

- [ ] **Step 7.3: Replace `Biome::classify` call in `column_data_with`**

In `src/worldgen/mod.rs::column_data_with`, locate the existing climate sampling block:

```rust
        let desertness_raw = self.desert_map.get(xz) as f32;
        let desertness = desertness_raw + jitter;
        let is_desert = desertness > 0.30;

        let temperature_raw = self.temperature_map.get(xz) as f32;
        let temperature = temperature_raw - jitter;
        let humidity_raw = self.humidity_map.get(xz) as f32;
        let humidity = humidity_raw + self.biome_jitter_rot(wx, wz);
        let biome = Biome::classify(temperature, humidity, is_desert);
```

Replace with:

```rust
        // PR 4: biome lookup via 6D R-tree (Voronoi-jittered).
        // The previous jitter / desert / temperature / humidity
        // FBM reads are no longer used for biome selection —
        // they're encapsulated in `biome_at` via the Sampler.
        // The `desertness` field on ColumnData is preserved for
        // the sand-transition band logic in `fill_chunk`; we
        // sample it directly here (still using the same FBM).
        let desertness_raw = self.desert_map.get(xz) as f32;
        let desertness = desertness_raw + jitter;
        // Sample the biome at this column. wy=SEA_LEVEL is the
        // canonical surface column — biomes are 2D for now
        // (depth=0 in every claim), so wy choice is moot.
        let biome = self.biome_at(wx, SEA_LEVEL, wz);
```

- [ ] **Step 7.4: Remove `Biome::classify`**

Delete the `impl Biome { fn classify(...) -> Self { ... } }` method from `src/worldgen/mod.rs`. The other `impl Biome` methods (`snow_capped`, `tree_rate_percentile`, `tree_kind`, the new `from_id`) stay.

Verify no other code references `Biome::classify`:

```bash
grep -n "Biome::classify" /Users/fdatoo/Developer/oxium/src/ -r 2>&1
```

Expected: no matches (other than the deleted line — verify it's gone).

- [ ] **Step 7.5: Run the worldgen tests**

Run: `cargo test --lib worldgen 2>&1 | tail -20`

Expected: most tests pass. Several will need updates:
- `cold_biome_caps_with_snow` — should still pass (Tundra/SnowyForest still cap with snow).
- `biome_diversity_in_8k_scan` — should still find ≥3 biome variants in the scan; semantics unchanged.
- `golden_seed42_chunk_0_2_0` — will fail (biome distribution differs subtly because Voronoi jitter ≠ FBM jitter). Update the golden hash in step 8.

- [ ] **Step 7.6: Commit (golden hash update deferred to Task 8)**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): biome_at — 6D R-tree lookup with Voronoi jitter

Replaces Biome::classify (nested if-elif on T/H/desertness) with
Generator::biome_at(wx, wy, wz). Internally:
  1. Voronoi-jitter the (wx, wz) to the nearest jittered cell
     center for organic borders.
  2. Sample the 6D climate target point at (jx, wy, jz).
  3. R-tree lookup to find the nearest claim's biome ID.
  4. Map BiomeId → existing Biome enum via Biome::from_id.

The desertness FBM is preserved for the sand-transition band
logic in fill_chunk (it's not a biome axis, just a per-column
threshold for surface-block stochastic sand placement).

column_data_with's old jitter / desert / temperature / humidity
samples for biome are dead code now — they live inside the
Sampler (which the biome_at call uses). The remaining direct
sample in column_data_with is desertness only.

Golden hash re-baselining: deferred to Task 8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Golden hash re-baselining + fingerprint check

**Files:**
- Modify: `src/worldgen/mod.rs` (the `GOLDEN_42_002` constant)
- Possibly modify: `tests/worldgen_fingerprint.rs` (`EXPECTED_HASH`)

- [ ] **Step 8.1: Put `golden_seed42_chunk_0_2_0` into print-mode**

In `src/worldgen/mod.rs`, change:

```rust
const GOLDEN_42_002: u64 = 0x<current_value>;
```

to:

```rust
const GOLDEN_42_002: u64 = 0xDEAD_BEEF_DEAD_BEEF;
```

- [ ] **Step 8.2: Capture the new chunk-golden hash**

Run: `cargo test --lib worldgen::tests::golden_seed42_chunk_0_2_0 -- --nocapture 2>&1 | grep "UPDATE GOLDEN"`

Expected output: `UPDATE GOLDEN_42_002 to: 0x<NEW_HEX>`

- [ ] **Step 8.3: Re-pin the chunk-golden hash**

Update the sentinel:

```rust
const GOLDEN_42_002: u64 = 0x<NEW_HEX_FROM_STEP_8_2>;
```

- [ ] **Step 8.4: Decide whether the fingerprint test needs updating**

Run: `cargo test --test worldgen_fingerprint 2>&1 | tail -15`

The fingerprint test samples `h_pre` (the 2D pre-river heightmap from `plates.rs` + `HeightmapNoise`). PR 4 doesn't touch `h_pre`, `plates`, `hydrology`, or `heightmap.rs::h_pre` — so the fingerprint hash should NOT change.

Expected: pass with the previous pin.

If it fails, that means PR 4 inadvertently touched the 2D heightmap pipeline — investigate before proceeding (the noise channel additions don't change `h_pre`; the most likely culprit is an accidental edit to `Generator::new` that reorders the existing FBM seeds, perturbing `temperature_map` etc. but those don't feed `h_pre` either — so any failure here is a real bug to chase).

- [ ] **Step 8.5: Run the full suite**

Run: `cargo test 2>&1 | tail -15`

Expected: all tests pass. Common failures to fix:
- `biome_diversity_in_8k_scan` — if the assertion is "find at least N biomes in 8k columns" and N is close to all six, the new lookup might land 5 biomes instead of 6 in that particular scan window. Adjust N if so; the biome SET hasn't shrunk, just the distribution.

- [ ] **Step 8.6: Commit**

```bash
git add src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
chore(worldgen): re-baseline golden_seed42_chunk_0_2_0 after PR 4

Voronoi jitter ≠ FBM-perturbation jitter, so biome borders in the
test chunk shift slightly. Surface blocks (Snow/Sand/Grass) follow
the new borders. The chunk hash changes; the chunk is still valid.

worldgen_fingerprint::fingerprint_hash_matches_pin unchanged — the
2D heightmap (plates + h_pre + warped FBM) is untouched by PR 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Hot-reload integration (rebuild biome table on config change)

**Files:**
- Modify: `src/worldgen/config.rs` (extend `spawn_watcher` callback)
- Modify: `src/worldgen/mod.rs` (expose a `rebuild_biome_table` hook)

- [ ] **Step 9.1: Write the failing test**

Append to the worldgen test module:

```rust
    #[test]
    fn hot_reload_swaps_biome_table() {
        use crate::worldgen::config::{ConfigHolder, WorldgenConfig};
        use crate::worldgen::climate::{ParameterPoint, Parameter, BiomeId};
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let holder = ConfigHolder::new(cfg);
        let g = Generator::with_config(42, holder.clone());

        // Sample biome at a known-Forest column (temperate, wet).
        // The default claim table puts Forest at T ∈ [-0.3, 0.2],
        // H ∈ [0.05, 1.0].
        let _ = g.biome_at(0, 64, 0);

        // Hot-swap a config where Forest is replaced with Tropical
        // claim covering the full plane. Newly-sampled biomes must
        // pick up the change.
        let mut new_cfg = (*holder.load()).clone();
        new_cfg.biomes.claims = vec![ParameterPoint {
            temperature: Parameter::span(-1.0, 1.0),
            humidity: Parameter::span(-1.0, 1.0),
            continentalness: Parameter::span(-1.0, 1.0),
            terrain_shape: Parameter::span(-1.0, 1.0),
            depth: Parameter::span(0.0, 0.0),
            weirdness: Parameter::span(-1.0, 1.0),
            offset: 0,
            biome: BiomeId::Tropical,
        }];
        holder.swap(new_cfg);

        // Now we need the Generator to rebuild its biome_table.
        // The hook is `Generator::reload_biome_table()`.
        g.reload_biome_table();

        // Every column should now be Tropical.
        for i in 0..10 {
            let wx = i * 31;
            let wz = i * 71;
            assert_eq!(g.biome_at(wx, 64, wz), Biome::Tropical);
        }
    }
```

- [ ] **Step 9.2: Add a `reload_biome_table` hook**

In `src/worldgen/mod.rs`, modify the `biome_table` field's type to enable mutation:

```rust
    biome_table: arc_swap::ArcSwap<crate::worldgen::climate::ParameterList>,
```

In `Generator::with_config`:

```rust
        let cfg_snapshot = config.load();
        let biome_table = arc_swap::ArcSwap::new(
            std::sync::Arc::new(cfg_snapshot.build_biome_table()),
        );
```

Replace `self.biome_table.find(...)` reads with `self.biome_table.load().find(...)`. Add the reload method:

```rust
impl Generator {
    /// Rebuild the biome R-tree from the current config snapshot.
    /// Called by the file-watcher callback after a hot-reload swap.
    pub fn reload_biome_table(&self) {
        let cfg = self.config_snapshot();
        self.biome_table.store(std::sync::Arc::new(cfg.build_biome_table()));
    }
}
```

- [ ] **Step 9.3: Wire the watcher to call reload**

Look at PR 2's `spawn_watcher` signature in `src/worldgen/config.rs`. It takes a `ConfigHolder` but knows nothing about the `Generator`. PR 2 deliberately kept these decoupled.

Option A (lightweight): expose a side-channel. Add a `Vec<Box<dyn Fn() + Send + Sync>>` to `ConfigHolder` (or pass a callback into `spawn_watcher`):

```rust
pub fn spawn_watcher_with_callback<F>(
    path: PathBuf,
    holder: ConfigHolder,
    on_reload: F,
) -> anyhow::Result<Debouncer<...>>
where
    F: Fn() + Send + Sync + 'static,
{
    // ... existing logic ...
    // After `holder.swap(cfg)`, also call `on_reload()`.
    on_reload();
    // ...
}
```

In the app startup site (PR 2 Task 11), pass `move || generator.reload_biome_table()` as the callback. The `Generator` needs to be `Arc<Generator>` (or wrapped in some shared state) for the callback's lifetime — match the app's existing arc-wrapping pattern.

(Plan-level note: this requires touching the app entry point — if the integration risks being noisy, defer the file-watcher wiring to a follow-up and ship PR 4 with explicit `reload_biome_table()` calls from tests + a TODO in the app entry point. The tests in step 9.1 pass with explicit reload regardless.)

- [ ] **Step 9.4: Run the test**

Run: `cargo test --lib worldgen::tests::hot_reload_swaps_biome_table 2>&1 | tail -10`

Expected: PASS.

- [ ] **Step 9.5: Run the full suite**

Run: `cargo test 2>&1 | tail -15`

Expected: all tests pass.

- [ ] **Step 9.6: Commit**

```bash
git add src/worldgen/mod.rs src/worldgen/config.rs
git commit -m "$(cat <<'EOF'
feat(worldgen): rebuild biome R-tree on hot-reload

Generator::reload_biome_table() rebuilds the biome R-tree from
the current config snapshot. The biome_table field is now an
arc_swap::ArcSwap<ParameterList> so reload is lock-free for
readers.

spawn_watcher_with_callback extends PR 2's hot-reload mechanism
with a post-swap callback. The app entry point passes
generator.reload_biome_table as the callback; tests call it
directly (no watcher needed in unit tests).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Smoke + perf sanity

**Files:** none (verification only)

- [ ] **Step 10.1: Run the full test suite once more**

Run: `cargo test 2>&1 | tail -15`

Expected: every test passes. Pay attention to:
- `worldgen::climate::tests::*` — 26+ unit tests covering Parameter, R-tree, ParameterList, Sampler, voronoi_jitter.
- `worldgen::tests::*` — including `biome_at_returns_one_of_six_biomes`, `biome_at_is_voronoi_jittered`, `biome_at_changes_across_long_distance`, `hot_reload_swaps_biome_table`, `golden_seed42_chunk_0_2_0` (re-baselined), `biome_diversity_in_8k_scan`, `cold_biome_caps_with_snow`.
- `worldgen_fingerprint::*` — 2D heightmap fingerprint, unchanged.

- [ ] **Step 10.2: Perf sanity — R-tree query is faster than brute force**

A simple bench-style test (skip if Oxium doesn't have a bench harness; this is a one-shot sanity check):

```bash
cargo test --release --lib worldgen::climate -- --nocapture --test-threads=1 2>&1 | grep -i "time\|elapsed"
```

Manually time a 1M-query loop in the test if needed:

```rust
    #[test]
    #[ignore = "perf — run with `cargo test --release -- --ignored`"]
    fn rtree_query_is_at_least_5x_faster_than_brute_force() {
        use crate::worldgen::config::WorldgenConfig;
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let table = cfg.build_biome_table();
        let targets: Vec<_> = (0..10000)
            .map(|i| {
                TargetPoint::from_floats(
                    (i as f32 / 10000.0) * 2.0 - 1.0,
                    ((i * 7) as f32 / 10000.0) % 2.0 - 1.0,
                    0.0, 0.0, 0.0, 0.0,
                )
            })
            .collect();
        let t0 = std::time::Instant::now();
        for t in &targets {
            let _ = table.find(t);
        }
        let rtree_ns = t0.elapsed().as_nanos();
        let t0 = std::time::Instant::now();
        for t in &targets {
            let _ = table.find_brute_force(t);
        }
        let bf_ns = t0.elapsed().as_nanos();
        let ratio = bf_ns as f64 / rtree_ns as f64;
        eprintln!("rtree={rtree_ns}ns, brute_force={bf_ns}ns, ratio={ratio:.2}x");
        // For 6 claims the R-tree win is small — should be ≥5×.
        assert!(ratio > 5.0, "R-tree should be ≥5× faster; got {ratio:.2}x");
    }
```

Run: `cargo test --release --lib worldgen::climate -- --ignored 2>&1 | tail -5`

Expected: passes, prints a ratio (typically ≥10× because the ThreadLocal last-leaf cache short-circuits most queries when biomes are clustered).

If the ratio is <5×, investigate — the last-leaf cache might not be wired correctly.

- [ ] **Step 10.3: Visual smoke test (manual)**

Boot the game with the release-built binary. Observe:
- Six distinct biomes are visible across a 1km walk (Tundra cold caps, SnowyForest with trees, Plains grass, Forest dense trees, Desert sand, Tropical palms).
- Biome borders **wave** at the ~24-block period (the jitter signature) instead of cutting along straight contour lines or the previous FBM-noise jitter.
- No checkerboard artifacts at 3-way biome junctions (Voronoi snapping prevents per-block alternation).
- Edit `assets/worldgen/default.ron`: change the Desert temperature box's `min` from `5500` to `3000` and save. Walk out of and back into chunks. The Desert footprint should expand visibly into formerly-Plains/Tropical territory.

Failures to investigate:
- Sharp axis-aligned cell boundaries (no wave): jitter not applied — check `voronoi_jitter` is called in `biome_at`.
- Always one biome: the R-tree is returning a stuck leaf — check the `last_leaf` cache reset logic, or the claim hyperboxes have malformed overlapping `min > max` ranges.

- [ ] **Step 10.4: Commit (no-op if nothing changed)**

If steps 10.1–10.3 prompted any RON tuning, commit it:

```bash
git add assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
chore(worldgen): tune biome jitter/claims after PR 4 visual review

Final tuning of the per-block jitter amplitude and biome claim
hyperboxes after visual smoke test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Otherwise nothing to commit — PR 4 is complete.

---

### Task 11: Removal of dead jitter helpers + tuning constants

**Files:**
- Modify: `src/worldgen/mod.rs`
- Modify: `src/worldgen/tuning.rs`

- [ ] **Step 11.1: Mark `Generator::biome_jitter` / `biome_jitter_rot` deprecated**

The PR 4 lookup uses Voronoi jitter inside `voronoi_jitter`, not the FBM-perturbation jitter. The old `biome_jitter` / `biome_jitter_rot` methods on `Generator` are only used by `column_data_with` for the *legacy* desertness/T/H perturbation — but PR 4 still uses them indirectly (the sand-transition band keeps the `desertness` jitter perturbation).

Audit: `grep -n "biome_jitter\b\|biome_jitter_rot\b" src/worldgen/mod.rs`

If the only remaining callers are inside `column_data_with` for the `desertness` perturbation, keep them. If they're now unused, mark `#[allow(dead_code)]` and add a TODO to remove in a follow-up.

- [ ] **Step 11.2: Mark `COLD_THRESHOLD` and `FOREST_HUMIDITY` deprecated**

In `src/worldgen/tuning.rs`, the constants `COLD_THRESHOLD = -0.30` and `FOREST_HUMIDITY = 0.05` were only used by `Biome::classify` (which is gone). Mark them:

```rust
#[deprecated(note = "PR 4 moved biome thresholds into assets/worldgen/default.ron's biomes.claims")]
pub const COLD_THRESHOLD: f32 = -0.30;
#[deprecated(note = "PR 4 moved biome thresholds into assets/worldgen/default.ron's biomes.claims")]
pub const FOREST_HUMIDITY: f32 = 0.05;
```

Leave them in place to avoid breaking any stragglers; future PRs remove them entirely.

Audit `mod.rs`'s top-level `use crate::worldgen::tuning::{... COLD_THRESHOLD, FOREST_HUMIDITY ...}` block and remove these two imports if no other code uses them.

- [ ] **Step 11.3: Run the suite**

Run: `cargo test 2>&1 | tail -10`

Expected: all tests pass; deprecation warnings on the two constants are acceptable.

- [ ] **Step 11.4: Commit**

```bash
git add src/worldgen/tuning.rs src/worldgen/mod.rs
git commit -m "$(cat <<'EOF'
chore(worldgen): mark legacy biome-threshold constants deprecated

COLD_THRESHOLD and FOREST_HUMIDITY are no longer the source of
truth — the biome partitioning lives in default.ron's
biomes.claims, edited per-claim hyperbox. The constants are
preserved with #[deprecated] so any test/util straggler keeps
compiling; future PRs drop them entirely.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Final verification

**Files:** none (verification only)

- [ ] **Step 12.1: Full test suite + lints**

Run: `cargo test 2>&1 | tail -15`

Expected: every test passes.

Run: `cargo clippy --all-targets 2>&1 | tail -15`

Expected: no new warnings. Deprecation warnings on `COLD_THRESHOLD` / `FOREST_HUMIDITY` are accepted.

- [ ] **Step 12.2: Verify the public API**

Run: `cargo doc --no-deps --lib 2>&1 | tail -5`

Expected: builds with no broken intradoc links. The new public symbols are: `worldgen::climate::{Parameter, ParameterPoint, TargetPoint, BiomeId, Sampler, ParameterList, RTree, QUANTIZATION_FACTOR, RTREE_FANOUT, voronoi_jitter}` and `worldgen::Generator::biome_at`, `Generator::reload_biome_table`, `Generator::climate_sampler`.

- [ ] **Step 12.3: Check the watcher round-trip in release**

Run: `cargo run --release` (full game). Edit `assets/worldgen/default.ron`'s Desert temperature box (move `min` from `5500` to `2000`). Save. Walk out of and back into chunks. Observe: log line indicates reload, Desert expands into the Plains/Tropical region.

- [ ] **Step 12.4: Final commit (no-op if nothing changed)**

```bash
# If anything was tuned during 12.1–12.3:
git add assets/worldgen/default.ron
git commit -m "$(cat <<'EOF'
chore(worldgen): final default.ron tuning post-PR-4

Adjustments after the visual smoke + hot-reload test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Otherwise PR 4 is complete.

---

## Out of scope for PR 4 (deferred to later PRs)

- **Real continentalness + terrain_shape noise channels.** PR 4 uses placeholder low-freq FBM stubs for the C and terrain_shape axes. PR 3 replaces these with the plate-distance signed continentalness + per-plate-biased terrain_shape noise; biome claim hyperboxes that span the full ±1 range for C and terrain_shape today will be narrowed in PR 3+ (oceanic vs inland, low-erosion mountain bands).
- **Underground biomes (depth ≠ 0 claims).** All six current claims sit at `depth: (min: 0, max: 0)`. The depth axis exists in the lookup so PR 6+ can add a `DripstoneCaves`-style underground biome by adding claims with `depth: (min: 2000, max: 9000)` — no code change required.
- **Weirdness-driven biome variants** (ice spikes, sunflower plains). The weirdness axis exists and is sampled; claim hyperboxes can be cloned and narrowed on weirdness to create variants without new noises or new axes.
- **Cell-level (quart) jitter resolution.** MC operates at 4-block quart granularity to amortize per-quart sampler costs across 64 voxels. Oxium PR 4 operates at block granularity (jitter cell ≈ 24 blocks). PR 5's cell interpolation will introduce quart-level caching; biome lookup can opt in to quart-resolution then if perf measurements show benefit.
- **Per-claim surface-rule overrides.** The current Biome enum carries `snow_capped`, `tree_rate_percentile`, `tree_kind` as methods. PR 6 (surface rules DSL) can fold these into the claim record (`biome: ResolvedSurface { snow_capped: true, tree_rate: 12, ... }`) so the RON authoring surface is uniform. PR 4 keeps the existing enum dispatch.
- **`peaksAndValleys` fold on weirdness for height splines.** MC folds weirdness via `pv(w) = -(||w| - 2/3| - 1/3) * 3` before feeding it into height splines. PR 4 uses raw weirdness for biome lookup (matches MC's biome usage). The folded form is PR 5+'s concern.
- **Spawn-point search.** MC's `Climate.findSpawnPosition` walks the climate field looking for the best-fitness target. Not needed for Oxium today — players spawn at world origin. Add if/when meta-game progression needs it.

## Plan self-review notes

- All 12 tasks have concrete code in every step. Real Rust — no `// implementation here` placeholders.
- Type names are consistent across tasks: `Parameter`, `ParameterPoint`, `TargetPoint`, `BiomeId`, `RTree`, `ParameterList`, `Sampler`, `WeirdnessNoise`, `BiomeConfig`. The "Climate." prefix in the spec is dropped — Rust uses module qualification (`climate::Parameter`), so the names don't need to repeat the module.
- Each task ends with a commit boundary. Total: 12 commits.
- Golden hash management: Task 8 step 8.1 puts the test in print-mode; step 8.3 captures and re-pins. Fingerprint hash should not change.
- The plan preserves `Generator::new(seed)` semantics (PR 2's invariant) — existing 50+ tests don't break wholesale. The biome enum and its trait methods (`snow_capped`, etc.) are preserved verbatim; only the *selector* changes.
- Quantization constant matches MC exactly (`10000`) — claim values authored against MC's literature translate 1:1.
- R-tree fanout is 6 (matches MC). RTree::search behavior is validated against brute-force fitness on a deterministic 13×13 (T, H) grid (`parameter_list_rtree_matches_brute_force_on_default`).
- The plan accepts (acknowledged in Task 4 and 5) that continentalness/terrain_shape are placeholders until PR 3. Claim hyperboxes span the full ±1 range on those axes so the lookup behaves as a 2D (T, H) partition during PR 4's lifetime — equivalent to the old `classify()` but data-driven.
- Per-block Voronoi jitter algorithm (Task 6) is spelled out explicitly: 2×2 cell walk, hash both cells for sub-cell offset, take nearest. Matches MC's `Climate.Sampler.findSpawnPosition`-adjacent jitter and the in-tree quart-level jitter.
- Hot-reload integration (Task 9) honors PR 2's decoupled `ConfigHolder` + `spawn_watcher` design by adding a post-swap callback (`spawn_watcher_with_callback`) — no shared mutable state added to `ConfigHolder` itself.
- The `last_leaf` ThreadLocal cache uses `thread_local::ThreadLocal<Cell<Option<usize>>>` (the `thread_local` crate, not `std::thread_local!`). The crate is a one-line `Cargo.toml` addition. Justification: queries from chunk-fill threads must share the holder across threads (`RTree` is `&self`), and the `std::thread_local!` macro can only declare static-lifetime locals — incompatible with a runtime-built tree. `thread_local::ThreadLocal` provides a `Send + Sync` thread-keyed holder that fits the pattern.
- Estimated LOC: ~620 (climate.rs ≈ 480, mod.rs ≈ 60, config.rs ≈ 40, heightmap.rs ≈ 20, default.ron ≈ 75). Matches the spec's ~600 budget.
