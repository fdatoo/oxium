//! Surface rules DSL (PR 6).
//!
//! Replaces the inline `if/else` surface-block selector in
//! `mod.rs::fill_chunk` with a data-driven tree of
//! [`ConditionSource`] and [`RuleSource`] nodes, modeled after
//! Minecraft 1.18+ `SurfaceRules.java`.
//!
//! The rule tree is held in `WorldgenConfig::surface` and walked
//! once per voxel by [`SurfaceSystem::surface_block`]. A
//! [`SurfaceContext`] carries the per-voxel column state
//! (`h_target`, `is_cliff`, `depth_below_surface`, `biome`, etc.).

use crate::voxel::block::Block;
use crate::worldgen::config::WorldgenConfig;
use crate::worldgen::hash;
use crate::worldgen::Biome;
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
    pub lake_rim: Option<i32>,
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
mod tests {
    use super::*;

    fn cfg() -> WorldgenConfig {
        WorldgenConfig::bundled_default().unwrap()
    }

    fn ctx<'a>(
        cfg: &'a WorldgenConfig,
        wy: i32,
        depth: i32,
        biome: Biome,
        is_cliff: bool,
    ) -> SurfaceContext<'a> {
        SurfaceContext {
            wx: 0,
            wy,
            wz: 0,
            h_target: 80,
            biome,
            is_cliff,
            desertness: 0.0,
            depth_below_surface: depth,
            lake_rim: None,
            seed: 42,
            cfg,
            sea_level: 62,
        }
    }

    #[test]
    fn on_floor_fires_only_at_depth_zero() {
        let cfg = cfg();
        let on_floor = ConditionSource::OnFloor;
        assert!(on_floor.eval(&ctx(&cfg, 80, 0, Biome::Plains, false)));
        assert!(!on_floor.eval(&ctx(&cfg, 79, 1, Biome::Plains, false)));
    }

    #[test]
    fn under_floor_n_includes_through_n() {
        let cfg = cfg();
        let u3 = ConditionSource::UnderFloor(3);
        for d in 0..=3 {
            assert!(u3.eval(&ctx(&cfg, 80 - d, d, Biome::Plains, false)));
        }
        assert!(!u3.eval(&ctx(&cfg, 76, 4, Biome::Plains, false)));
    }

    #[test]
    fn beach_band_in_range() {
        let cfg = cfg();
        let bb = ConditionSource::BeachBand {
            below_sea: 1,
            above_sea: 2,
        };
        assert!(bb.eval(&ctx(&cfg, 61, 0, Biome::Plains, false)));
        assert!(bb.eval(&ctx(&cfg, 64, 0, Biome::Plains, false)));
        assert!(!bb.eval(&ctx(&cfg, 60, 0, Biome::Plains, false)));
        assert!(!bb.eval(&ctx(&cfg, 65, 0, Biome::Plains, false)));
    }

    #[test]
    fn within_surface_band_window() {
        let cfg = cfg();
        let band = ConditionSource::WithinSurfaceBand(16);
        assert!(band.eval(&ctx(&cfg, 64, 0, Biome::Plains, false)));
        assert!(band.eval(&ctx(&cfg, 96, 0, Biome::Plains, false)));
        assert!(!band.eval(&ctx(&cfg, 63, 0, Biome::Plains, false)));
        assert!(!band.eval(&ctx(&cfg, 97, 0, Biome::Plains, false)));
    }

    #[test]
    fn not_inverts() {
        let cfg = cfg();
        let n = ConditionSource::Not(Box::new(ConditionSource::Always(false)));
        assert!(n.eval(&ctx(&cfg, 0, 0, Biome::Plains, false)));
    }

    #[test]
    fn all_short_circuits_on_false() {
        let cfg = cfg();
        let all = ConditionSource::All(vec![
            ConditionSource::Always(true),
            ConditionSource::Always(false),
            ConditionSource::Always(true),
        ]);
        assert!(!all.eval(&ctx(&cfg, 0, 0, Biome::Plains, false)));
    }

    #[test]
    fn any_short_circuits_on_true() {
        let cfg = cfg();
        let any = ConditionSource::Any(vec![
            ConditionSource::Always(false),
            ConditionSource::Always(true),
        ]);
        assert!(any.eval(&ctx(&cfg, 0, 0, Biome::Plains, false)));
    }

    #[test]
    fn sequence_returns_first_match() {
        let cfg = cfg();
        let r = RuleSource::Sequence(vec![
            RuleSource::If {
                condition: ConditionSource::Always(false),
                then: Box::new(RuleSource::Block(Block::Sand)),
            },
            RuleSource::If {
                condition: ConditionSource::Always(true),
                then: Box::new(RuleSource::Block(Block::Grass)),
            },
            RuleSource::Block(Block::Stone),
        ]);
        let c = ctx(&cfg, 80, 0, Biome::Plains, false);
        assert_eq!(r.apply(&c), Some(Block::Grass));
    }

    #[test]
    fn cliff_to_stone() {
        let cfg = cfg();
        let r = RuleSource::Sequence(vec![
            RuleSource::If {
                condition: ConditionSource::IsCliff,
                then: Box::new(RuleSource::Block(Block::Stone)),
            },
            RuleSource::Block(Block::Grass),
        ]);
        let c_cliff = ctx(&cfg, 80, 0, Biome::Plains, true);
        let c_flat = ctx(&cfg, 80, 0, Biome::Plains, false);
        assert_eq!(r.apply(&c_cliff), Some(Block::Stone));
        assert_eq!(r.apply(&c_flat), Some(Block::Grass));
    }

    #[test]
    fn snow_capped_biomes() {
        assert!(snow_capped(Biome::Tundra));
        assert!(snow_capped(Biome::SnowyForest));
        assert!(!snow_capped(Biome::Plains));
        assert!(!snow_capped(Biome::Forest));
        assert!(!snow_capped(Biome::Desert));
        assert!(!snow_capped(Biome::Tropical));
    }

    #[test]
    fn ron_roundtrip_preserves_rules() {
        let r = RuleSource::Sequence(vec![
            RuleSource::If {
                condition: ConditionSource::IsCliff,
                then: Box::new(RuleSource::Block(Block::Stone)),
            },
            RuleSource::If {
                condition: ConditionSource::All(vec![
                    ConditionSource::OnFloor,
                    ConditionSource::YAbove(110),
                ]),
                then: Box::new(RuleSource::Block(Block::Snow)),
            },
            RuleSource::Block(Block::Grass),
        ]);
        let s = ron::to_string(&r).unwrap();
        let parsed: RuleSource = ron::from_str(&s).unwrap();
        let cfg = cfg();
        let c = ctx(&cfg, 80, 0, Biome::Plains, true);
        assert_eq!(parsed.apply(&c), Some(Block::Stone));
    }
}
