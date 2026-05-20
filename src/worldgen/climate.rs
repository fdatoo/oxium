//! Multi-noise biome lookup (PR 4).
//!
//! Every column / voxel has six climate channels:
//!
//! | Channel           | Source                                             |
//! |-------------------|----------------------------------------------------|
//! | `temperature`     | Generator::temperature_map (low-freq Fbm)          |
//! | `humidity`        | Generator::humidity_map (low-freq Fbm)             |
//! | `continentalness` | plates::signed_continentalness (Voronoi)           |
//! | `terrain_shape`   | HeightmapNoise::terrain_shape (low-freq Fbm + bias)|
//! | `depth`           | `y_clamped_gradient(wy)` ∈ [-1, 1.5]                |
//! | `weirdness`       | new Fbm noise (mid-freq, mirrors MC's ridge axis)  |
//!
//! Each biome owns one or more axis-aligned hyperboxes in this 6D
//! space (a [`ParameterPoint`]). A query at `(t, h, c, s, d, w)`
//! finds the *nearest* biome by squared-L2 distance over per-axis
//! gaps (zero inside the box on that axis, gap-squared outside).
//!
//! All distances are computed in integer space after quantization by
//! [`QUANTIZATION_FACTOR`] (10000), matching MC's
//! `net/minecraft/world/level/biome/Climate.java`. Integers give
//! byte-deterministic equality comparison across platforms.
//!
//! The lookup is wrapped in an [`RTree`] with fanout 6 and a per-
//! thread last-leaf cache so adjacent voxels (which almost always
//! hit the same leaf) prune the tree on the first child check.

use crate::worldgen::Biome;
use serde::{Deserialize, Serialize};
use std::cell::Cell;

/// Quantization factor for climate values. f32 → i64 via
/// `(v * QUANTIZATION_FACTOR) as i64`. Matches MC's `Climate`.
pub const QUANTIZATION_FACTOR: f32 = 10000.0;

/// R-tree fanout. Matches MC.
pub const RTREE_FANOUT: usize = 6;

/// Number of climate axes (excluding the synthetic `offset`
/// tie-breaker slot, which is always 0 for biome entries).
pub const PARAMETER_COUNT: usize = 7;

/// Quantize a climate value `v ∈ [-2, 2]` to a signed integer.
pub fn quantize(v: f32) -> i64 {
    (v * QUANTIZATION_FACTOR) as i64
}

/// One axis of a biome's claim — a closed interval `[min, max]` in
/// quantized integer space.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Parameter {
    pub min: i64,
    pub max: i64,
}

impl Parameter {
    /// A single-value point on this axis (degenerate interval).
    pub fn point(v: f32) -> Self {
        let q = quantize(v);
        Self { min: q, max: q }
    }

    /// A range claim on this axis.
    pub fn span(min: f32, max: f32) -> Self {
        Self {
            min: quantize(min),
            max: quantize(max),
        }
    }

    /// A claim spanning the entire range — biome doesn't care about
    /// this axis.
    pub fn any() -> Self {
        Self {
            min: quantize(-2.0),
            max: quantize(2.0),
        }
    }

    /// Gap from `target` to this interval. 0 if inside; positive
    /// distance to the nearest edge if outside.
    pub fn distance(&self, target: i64) -> i64 {
        if target < self.min {
            self.min - target
        } else if target > self.max {
            target - self.max
        } else {
            0
        }
    }
}

/// One biome's claim on the 6D climate space. Each `Parameter` is
/// an interval on its axis; `offset` is a synthetic tie-breaker
/// (always 0 for biome entries — kept for parity with MC's
/// `ParameterPoint` and to let future code add weighted preferences
/// without a schema change).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterPoint {
    pub temperature: Parameter,
    pub humidity: Parameter,
    pub continentalness: Parameter,
    pub terrain_shape: Parameter,
    pub depth: Parameter,
    pub weirdness: Parameter,
    pub offset: i64,
    pub biome: Biome,
}

impl ParameterPoint {
    /// Squared-L2 distance from a quantized target to this hyperbox.
    /// Each axis contributes `gap^2`; `0` inside, `gap^2` outside.
    pub fn fitness(&self, target: &TargetPoint) -> i64 {
        let dt = self.temperature.distance(target.temperature);
        let dh = self.humidity.distance(target.humidity);
        let dc = self.continentalness.distance(target.continentalness);
        let ds = self.terrain_shape.distance(target.terrain_shape);
        let dd = self.depth.distance(target.depth);
        let dw = self.weirdness.distance(target.weirdness);
        let do_ = target.offset - self.offset;
        dt * dt + dh * dh + dc * dc + ds * ds + dd * dd + dw * dw + do_ * do_
    }

    /// Axis-i parameter for the R-tree bucket-by-best-axis build.
    fn param(&self, axis: usize) -> Parameter {
        match axis {
            0 => self.temperature,
            1 => self.humidity,
            2 => self.continentalness,
            3 => self.terrain_shape,
            4 => self.depth,
            5 => self.weirdness,
            6 => Parameter {
                min: self.offset,
                max: self.offset,
            },
            _ => panic!("axis index out of range"),
        }
    }
}

/// A quantized query point in the 6D climate space.
#[derive(Clone, Copy, Debug)]
pub struct TargetPoint {
    pub temperature: i64,
    pub humidity: i64,
    pub continentalness: i64,
    pub terrain_shape: i64,
    pub depth: i64,
    pub weirdness: i64,
    pub offset: i64,
}

impl TargetPoint {
    /// Build a query point from raw f32 climate values.
    pub fn new(
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
            offset: 0,
        }
    }
}

/// One node of the R-tree. Inner nodes hold up to [`RTREE_FANOUT`]
/// children and a hyperbox that bounds all descendants. Leaves hold
/// a single `ParameterPoint`.
#[derive(Clone, Debug)]
enum RTreeNode {
    Leaf {
        entry_idx: u32,
    },
    Inner {
        /// Hyperbox bounding all children (per-axis min/max).
        bbox: [Parameter; PARAMETER_COUNT],
        children: Vec<RTreeNode>,
    },
}

impl RTreeNode {
    /// Squared-L2 distance from `target` to this node's hyperbox.
    /// For a leaf, this is the underlying biome's fitness.
    fn distance(&self, entries: &[ParameterPoint], target: &TargetPoint) -> i64 {
        match self {
            RTreeNode::Leaf { entry_idx } => entries[*entry_idx as usize].fitness(target),
            RTreeNode::Inner { bbox, .. } => {
                let dt = bbox[0].distance(target.temperature);
                let dh = bbox[1].distance(target.humidity);
                let dc = bbox[2].distance(target.continentalness);
                let ds = bbox[3].distance(target.terrain_shape);
                let dd = bbox[4].distance(target.depth);
                let dw = bbox[5].distance(target.weirdness);
                let do_ = bbox[6].distance(target.offset);
                dt * dt + dh * dh + dc * dc + ds * ds + dd * dd + dw * dw + do_ * do_
            }
        }
    }

    /// Find the leaf with minimum fitness against `target`, pruning
    /// subtrees whose hyperbox distance ≥ best-so-far.
    fn search(
        &self,
        entries: &[ParameterPoint],
        target: &TargetPoint,
        best: &mut SearchState,
    ) {
        match self {
            RTreeNode::Leaf { entry_idx } => {
                let d = entries[*entry_idx as usize].fitness(target);
                if d < best.distance {
                    best.distance = d;
                    best.leaf_idx = *entry_idx;
                }
            }
            RTreeNode::Inner { children, .. } => {
                // Order children by node distance ascending — the
                // first ones are most likely to improve `best`, so
                // we set a tight pruning bound quickly.
                let mut child_dists: Vec<(usize, i64)> = children
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, c.distance(entries, target)))
                    .collect();
                child_dists.sort_by_key(|x| x.1);
                for (i, d) in child_dists {
                    if d >= best.distance {
                        break;
                    }
                    children[i].search(entries, target, best);
                }
            }
        }
    }
}

struct SearchState {
    distance: i64,
    leaf_idx: u32,
}

/// Bounded R-tree of biome claims. Built once from a
/// [`ParameterList`] at world creation; queried per-voxel during
/// chunk fill.
#[derive(Clone, Debug)]
pub struct RTree {
    root: RTreeNode,
}

impl RTree {
    /// Build the tree by recursively bucketing entries along the
    /// axis with the lowest total bucket-span cost (MC's heuristic).
    pub fn build(entries: &[ParameterPoint]) -> Self {
        assert!(!entries.is_empty(), "RTree needs ≥ 1 entry");
        let indices: Vec<u32> = (0..entries.len() as u32).collect();
        let root = Self::build_node(entries, indices);
        Self { root }
    }

    fn build_node(entries: &[ParameterPoint], indices: Vec<u32>) -> RTreeNode {
        if indices.len() == 1 {
            return RTreeNode::Leaf {
                entry_idx: indices[0],
            };
        }
        if indices.len() <= RTREE_FANOUT {
            // Leaf-level inner node: each child is a leaf.
            let children: Vec<RTreeNode> = indices
                .iter()
                .map(|&i| RTreeNode::Leaf { entry_idx: i })
                .collect();
            return RTreeNode::Inner {
                bbox: union_bbox(entries, &indices),
                children,
            };
        }
        // Pick the axis with lowest total bucket-span cost.
        let bucket_size = bucket_size_for(indices.len());
        let mut best_axis = 0;
        let mut best_cost = i64::MAX;
        for axis in 0..PARAMETER_COUNT {
            let mut sorted = indices.clone();
            sorted.sort_by_key(|&i| midpoint(entries[i as usize].param(axis)));
            let cost = total_span_cost(entries, &sorted, bucket_size);
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
            }
        }
        // Bucket along that axis.
        let mut sorted = indices.clone();
        sorted.sort_by_key(|&i| midpoint(entries[i as usize].param(best_axis)));
        let mut children = Vec::new();
        for chunk in sorted.chunks(bucket_size) {
            children.push(Self::build_node(entries, chunk.to_vec()));
        }
        RTreeNode::Inner {
            bbox: union_bbox(entries, &indices),
            children,
        }
    }

    /// Find the biome whose ParameterPoint is closest to `target`.
    /// Returns `(entry_idx, distance)` so callers can detect
    /// arbitrarily-bad mismatches if needed.
    pub fn search(&self, entries: &[ParameterPoint], target: &TargetPoint) -> u32 {
        let mut state = SearchState {
            distance: i64::MAX,
            leaf_idx: 0,
        };
        // Seed the search from the per-thread last leaf so adjacent
        // queries prune quickly on the first child distance check.
        LAST_LEAF.with(|last| {
            let idx = last.get();
            if (idx as usize) < entries.len() {
                state.distance = entries[idx as usize].fitness(target);
                state.leaf_idx = idx;
            }
        });
        self.root.search(entries, target, &mut state);
        LAST_LEAF.with(|last| last.set(state.leaf_idx));
        state.leaf_idx
    }
}

thread_local! {
    /// Per-thread cache of the last winning leaf. Adjacent voxel
    /// queries almost always hit the same biome, so seeding the
    /// search with this leaf's fitness sets a tight pruning bound.
    static LAST_LEAF: Cell<u32> = const { Cell::new(0) };
}

fn midpoint(p: Parameter) -> i64 {
    (p.min + p.max) / 2
}

fn bucket_size_for(n: usize) -> usize {
    // Largest power of FANOUT that's ≤ n. Matches MC's
    // `Math.pow(6, Math.floor(log6(n)))`. Then `n / bucket_size`
    // ≤ FANOUT children per node.
    let mut k = 1;
    while k * RTREE_FANOUT <= n {
        k *= RTREE_FANOUT;
    }
    k
}

fn total_span_cost(entries: &[ParameterPoint], sorted: &[u32], bucket_size: usize) -> i64 {
    let mut cost = 0i64;
    for chunk in sorted.chunks(bucket_size) {
        for axis in 0..PARAMETER_COUNT {
            let bbox = chunk
                .iter()
                .map(|&i| entries[i as usize].param(axis))
                .fold(Parameter { min: i64::MAX, max: i64::MIN }, |acc, p| {
                    Parameter {
                        min: acc.min.min(p.min),
                        max: acc.max.max(p.max),
                    }
                });
            cost = cost.saturating_add(bbox.max - bbox.min);
        }
    }
    cost
}

fn union_bbox(entries: &[ParameterPoint], indices: &[u32]) -> [Parameter; PARAMETER_COUNT] {
    let mut bbox = [Parameter {
        min: i64::MAX,
        max: i64::MIN,
    }; PARAMETER_COUNT];
    for &i in indices {
        let e = &entries[i as usize];
        bbox[0].min = bbox[0].min.min(e.temperature.min);
        bbox[0].max = bbox[0].max.max(e.temperature.max);
        bbox[1].min = bbox[1].min.min(e.humidity.min);
        bbox[1].max = bbox[1].max.max(e.humidity.max);
        bbox[2].min = bbox[2].min.min(e.continentalness.min);
        bbox[2].max = bbox[2].max.max(e.continentalness.max);
        bbox[3].min = bbox[3].min.min(e.terrain_shape.min);
        bbox[3].max = bbox[3].max.max(e.terrain_shape.max);
        bbox[4].min = bbox[4].min.min(e.depth.min);
        bbox[4].max = bbox[4].max.max(e.depth.max);
        bbox[5].min = bbox[5].min.min(e.weirdness.min);
        bbox[5].max = bbox[5].max.max(e.weirdness.max);
        bbox[6].min = bbox[6].min.min(e.offset);
        bbox[6].max = bbox[6].max.max(e.offset);
    }
    bbox
}

/// A parsed [`ParameterPoint`] list plus its built [`RTree`]. The
/// public API biome lookups go through here.
pub struct ParameterList {
    pub entries: Vec<ParameterPoint>,
    pub tree: RTree,
}

impl ParameterList {
    pub fn new(entries: Vec<ParameterPoint>) -> Self {
        let tree = RTree::build(&entries);
        Self { entries, tree }
    }

    /// Look up the biome at the given climate point.
    pub fn lookup(&self, target: &TargetPoint) -> Biome {
        let idx = self.tree.search(&self.entries, target);
        self.entries[idx as usize].biome
    }
}

/// Per-block Voronoi-style biome jitter. At a given block position,
/// hash-jitter 8 nearby quart corners and pick the nearest jittered
/// corner's biome. Produces organic borders without interpolating
/// biome IDs.
///
/// Returns a `(qx, qz)` quart offset to apply to the lookup. The
/// caller queries `biome_at(quart_jittered(wx, wz, ...))`.
pub fn voronoi_jitter_offset(seed: u64, wx: i32, wy: i32, wz: i32) -> (i32, i32) {
    let qx = wx.div_euclid(4);
    let qz = wz.div_euclid(4);
    let qy = wy.div_euclid(4);
    let mut best_dist = i64::MAX;
    let mut best_qx = qx;
    let mut best_qz = qz;
    // 2×2×2 neighbourhood of quart corners.
    for dx in 0..2 {
        for dy in 0..2 {
            for dz in 0..2 {
                let cqx = qx + dx;
                let cqy = qy + dy;
                let cqz = qz + dz;
                // Two-axis jitter on the corner so adjacent quart
                // cells produce squiggly biome borders.
                let jx = crate::worldgen::hash::mix_unit(
                    seed,
                    &[cqx, cqy, cqz, 0],
                ) as f64;
                let jz = crate::worldgen::hash::mix_unit(
                    seed,
                    &[cqx, cqy, cqz, 1],
                ) as f64;
                let jitter_x = jx * 0.9 - 0.45;
                let jitter_z = jz * 0.9 - 0.45;
                let cx_world = (cqx as f64 + jitter_x) * 4.0;
                let cz_world = (cqz as f64 + jitter_z) * 4.0;
                let dx_w = cx_world - wx as f64;
                let dz_w = cz_world - wz as f64;
                let d = (dx_w * dx_w + dz_w * dz_w) as i64;
                if d < best_dist {
                    best_dist = d;
                    best_qx = cqx;
                    best_qz = cqz;
                }
            }
        }
    }
    (best_qx * 4 + 2 - wx, best_qz * 4 + 2 - wz)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_entries() -> Vec<ParameterPoint> {
        vec![
            ParameterPoint {
                temperature: Parameter::span(-1.0, -0.3),
                humidity: Parameter::any(),
                continentalness: Parameter::any(),
                terrain_shape: Parameter::any(),
                depth: Parameter::any(),
                weirdness: Parameter::any(),
                offset: 0,
                biome: Biome::Tundra,
            },
            ParameterPoint {
                temperature: Parameter::span(0.3, 1.0),
                humidity: Parameter::any(),
                continentalness: Parameter::any(),
                terrain_shape: Parameter::any(),
                depth: Parameter::any(),
                weirdness: Parameter::any(),
                offset: 0,
                biome: Biome::Desert,
            },
            ParameterPoint {
                temperature: Parameter::span(-0.3, 0.3),
                humidity: Parameter::span(0.0, 1.0),
                continentalness: Parameter::any(),
                terrain_shape: Parameter::any(),
                depth: Parameter::any(),
                weirdness: Parameter::any(),
                offset: 0,
                biome: Biome::Forest,
            },
            ParameterPoint {
                temperature: Parameter::span(-0.3, 0.3),
                humidity: Parameter::span(-1.0, 0.0),
                continentalness: Parameter::any(),
                terrain_shape: Parameter::any(),
                depth: Parameter::any(),
                weirdness: Parameter::any(),
                offset: 0,
                biome: Biome::Plains,
            },
        ]
    }

    #[test]
    fn parameter_distance_is_zero_inside() {
        let p = Parameter::span(-0.5, 0.5);
        assert_eq!(p.distance(0), 0);
        assert_eq!(p.distance(p.min), 0);
        assert_eq!(p.distance(p.max), 0);
    }

    #[test]
    fn parameter_distance_is_gap_outside() {
        let p = Parameter::span(0.0, 1.0);
        assert_eq!(p.distance(quantize(-0.5)), quantize(0.5));
        assert_eq!(p.distance(quantize(1.5)), quantize(0.5));
    }

    #[test]
    fn rtree_finds_cold_biome_at_cold_point() {
        let entries = test_entries();
        let list = ParameterList::new(entries);
        let t = TargetPoint::new(-0.8, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(list.lookup(&t), Biome::Tundra);
    }

    #[test]
    fn rtree_finds_hot_biome_at_hot_point() {
        let entries = test_entries();
        let list = ParameterList::new(entries);
        let t = TargetPoint::new(0.8, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(list.lookup(&t), Biome::Desert);
    }

    #[test]
    fn rtree_picks_forest_when_temperate_and_wet() {
        let entries = test_entries();
        let list = ParameterList::new(entries);
        let t = TargetPoint::new(0.0, 0.5, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(list.lookup(&t), Biome::Forest);
    }

    #[test]
    fn rtree_picks_plains_when_temperate_and_dry() {
        let entries = test_entries();
        let list = ParameterList::new(entries);
        let t = TargetPoint::new(0.0, -0.5, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(list.lookup(&t), Biome::Plains);
    }

    #[test]
    fn rtree_finds_nearest_when_outside_all_boxes() {
        // No biome claims temperature 5.0; it should still pick the
        // hottest available (Desert).
        let entries = test_entries();
        let list = ParameterList::new(entries);
        let t = TargetPoint::new(5.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(list.lookup(&t), Biome::Desert);
    }

    #[test]
    fn ron_roundtrip_preserves_entries() {
        let entries = test_entries();
        let s = ron::to_string(&entries).unwrap();
        let parsed: Vec<ParameterPoint> = ron::from_str(&s).unwrap();
        assert_eq!(parsed.len(), entries.len());
        let list = ParameterList::new(parsed);
        let t = TargetPoint::new(-0.8, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(list.lookup(&t), Biome::Tundra);
    }

    #[test]
    fn voronoi_jitter_offset_is_deterministic() {
        let (dx1, dz1) = voronoi_jitter_offset(42, 100, 64, 200);
        let (dx2, dz2) = voronoi_jitter_offset(42, 100, 64, 200);
        assert_eq!((dx1, dz1), (dx2, dz2));
    }

    #[test]
    fn voronoi_jitter_offset_is_small() {
        // Jitter shouldn't move us more than a few blocks.
        for wx in (-100..100).step_by(7) {
            for wz in (-100..100).step_by(7) {
                let (dx, dz) = voronoi_jitter_offset(42, wx, 64, wz);
                assert!(dx.abs() <= 8 && dz.abs() <= 8, "jitter too large: {dx},{dz}");
            }
        }
    }
}
