//! Surface rules DSL (PR 6).
//!
//! Replaces the inline `if/else` surface-block selector in
//! `mod.rs::fill_chunk` with a data-driven tree of
//! [`ConditionSource`] and [`RuleSource`] nodes, modeled after
//! Minecraft 1.18+ `SurfaceRules.java`.
//!
//! The rule tree is held in `WorldgenConfig::surface` and walked once per
//! solid voxel during chunk fill. A [`SurfaceContext`] carries per-voxel
//! column state (`h_target`, `is_cliff`, `depth_below_surface`, `biome`,
//! etc.) and is passed to the root [`RuleSource`], which short-circuits
//! on the first matching rule.
//!
//! ### Key concepts
//!
//! - **Conditions** ([`ConditionSource`]): predicates over the column state
//!   — e.g. `IsCliff`, `WithinSurfaceBand(N)`, `Biome([Desert])`.
//! - **Rules** ([`RuleSource`]): a block to place when the condition is
//!   met, or a sequence / conditional chain.
//! - **`SurfaceContext`**: carries `depth_below_surface` (blocks below the
//!   most recent air→solid transition), which is updated by the chunk fill
//!   loop as it scans top-down through each column.
//!
//! ### Hot reload
//!
//! The rule tree lives in `assets/worldgen/default.ron`'s `surface` field.
//! It can be edited while the engine runs; the file watcher swaps it
//! atomically and newly generated chunks pick it up immediately.
//!
//! See `docs/book/content/part-4-chunk-fill/4.6-surface-rules.mdx` and
//! `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`.

use crate::voxel::block::Block;
use crate::worldgen::Biome;
use crate::worldgen::config::WorldgenConfig;
use crate::worldgen::hash;
use serde::{Deserialize, Serialize};

/// All state available to a [`ConditionSource`] / [`RuleSource`]
/// during a single voxel evaluation.
pub struct SurfaceContext<'a> {
    pub wx: i32,
    pub wy: i32,
    pub wz: i32,
    pub h_target: i32,
    pub biome: Biome,
    pub is_cliff: bool,
    pub desertness: f32,
    pub depth_below_surface: i32,
    /// Unified water surface Y for this column (see `ColumnData::water_surface_y`).
    pub water_surface_y: Option<i32>,
    pub seed: u64,
    pub cfg: &'a WorldgenConfig,
    pub sea_level: i32,
}

/// Predicate over a [`SurfaceContext`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConditionSource {
    Always(bool),
    IsCliff,
    IsCold,
    OnFloor,
    UnderFloor(i32),
    YAbove(i32),
    YBelow(i32),
    WithinSurfaceBand(i32),
    Biome(Vec<Biome>),
    AbovePreliminarySurface(i32),
    Not(Box<ConditionSource>),
    All(Vec<ConditionSource>),
    Any(Vec<ConditionSource>),
    BeachBand {
        below_sea: i32,
        above_sea: i32,
    },
    SandTransitionRoll {
        temp_min: f32,
        probability: f32,
    },
    /// True when `ctx.water_surface_y` is set and `ctx.wy ≤ water_surface_y
    /// - offset`. Use `offset: 0` to match any submerged voxel, or a
    /// positive offset to match voxels that are at least `offset` blocks
    /// below the water surface (useful for transition layers).
    BelowWaterSurface {
        offset: i32,
    },
}

impl ConditionSource {
    pub fn eval(&self, ctx: &SurfaceContext) -> bool {
        match self {
            ConditionSource::Always(v) => *v,
            ConditionSource::IsCliff => ctx.is_cliff,
            ConditionSource::IsCold => snow_capped(ctx.biome),
            ConditionSource::OnFloor => ctx.depth_below_surface == 0,
            ConditionSource::UnderFloor(n) => ctx.depth_below_surface <= *n,
            ConditionSource::YAbove(y) => ctx.wy >= *y,
            ConditionSource::YBelow(y) => ctx.wy <= *y,
            ConditionSource::WithinSurfaceBand(w) => (ctx.h_target - ctx.wy).abs() <= *w,
            ConditionSource::Biome(set) => set.contains(&ctx.biome),
            ConditionSource::AbovePreliminarySurface(off) => ctx.wy >= ctx.h_target + *off,
            ConditionSource::Not(c) => !c.eval(ctx),
            ConditionSource::All(cs) => cs.iter().all(|c| c.eval(ctx)),
            ConditionSource::Any(cs) => cs.iter().any(|c| c.eval(ctx)),
            ConditionSource::BeachBand {
                below_sea,
                above_sea,
            } => ctx.wy >= ctx.sea_level - below_sea && ctx.wy <= ctx.sea_level + above_sea,
            ConditionSource::SandTransitionRoll {
                temp_min,
                probability,
            } => {
                if ctx.desertness < *temp_min || matches!(ctx.biome, Biome::Desert) {
                    return false;
                }
                let roll = hash::mix_unit(ctx.seed, &[ctx.wx, ctx.wz, 71]);
                roll < *probability
            }
            ConditionSource::BelowWaterSurface { offset } => {
                ctx.water_surface_y.map_or(false, |w| ctx.wy <= w - offset)
            }
        }
    }
}

/// A rule that, when matched, produces a [`Block`] for the voxel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RuleSource {
    Block(Block),
    Sequence(Vec<RuleSource>),
    If {
        condition: ConditionSource,
        then: Box<RuleSource>,
    },
}

impl RuleSource {
    /// Walk the rule tree; return the matching block or `None` if
    /// every guarded branch falls through.
    pub fn apply(&self, ctx: &SurfaceContext) -> Option<Block> {
        match self {
            RuleSource::Block(b) => Some(*b),
            RuleSource::Sequence(rules) => {
                for r in rules {
                    if let Some(b) = r.apply(ctx) {
                        return Some(b);
                    }
                }
                None
            }
            RuleSource::If { condition, then } => {
                if condition.eval(ctx) {
                    then.apply(ctx)
                } else {
                    None
                }
            }
        }
    }
}

/// Holds the parsed surface rule tree and exposes the
/// `surface_block` query used by the chunk-fill hot path.
pub struct SurfaceSystem {
    pub rules: RuleSource,
}

impl SurfaceSystem {
    pub fn new(rules: RuleSource) -> Self {
        Self { rules }
    }

    /// Evaluate the surface rule tree. Falls back to `Block::Stone`
    /// if every branch falls through (defensive — a well-formed
    /// rule tree always has a terminal `Block`).
    pub fn surface_block(&self, ctx: &SurfaceContext) -> Block {
        self.rules.apply(ctx).unwrap_or(Block::Stone)
    }
}

fn snow_capped(b: Biome) -> bool {
    matches!(b, Biome::Tundra | Biome::SnowyForest)
}

#[cfg(test)]
#[path = "surface_tests.rs"]
mod tests;
