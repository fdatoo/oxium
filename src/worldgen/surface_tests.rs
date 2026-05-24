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
        water_surface_y: None,
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
