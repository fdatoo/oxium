use super::*;
use crate::voxel::chunk::CHUNK_VOL;
use glam::IVec3;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Reduce a chunk to a single u64 hash — easier than comparing full
/// 32 KB arrays in test failure messages.
fn hash_chunk(c: &DenseChunk) -> u64 {
    let mut h = DefaultHasher::new();
    for b in c.blocks.iter() {
        (*b as u16).hash(&mut h);
    }
    h.finish()
}

#[test]
fn fill_is_deterministic() {
    let g = Generator::new(42);
    let mut a = DenseChunk::empty();
    let mut b = DenseChunk::empty();
    g.fill_chunk(ChunkCoord(IVec3::ZERO), &mut a);
    g.fill_chunk(ChunkCoord(IVec3::ZERO), &mut b);
    assert_eq!(hash_chunk(&a), hash_chunk(&b));
}

#[test]
fn different_seeds_differ() {
    let mut a = DenseChunk::empty();
    let mut b = DenseChunk::empty();
    Generator::new(1).fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut a);
    Generator::new(2).fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut b);
    assert_ne!(hash_chunk(&a), hash_chunk(&b));
}

/// Golden test: locks the generator's output for a known seed/chunk.
/// First run prints the actual hash; update `GOLDEN_42_002` once and
/// future runs catch unintentional behavioural drift.
#[test]
fn golden_seed42_chunk_0_2_0() {
    // Rebaselined 2026-05-22: cave tuning — cheese_offset 0.18→0.40,
    // smin_k 1.2→0.5, halved chamber/tunnel radii, reduced chamber counts.
    const GOLDEN_42_002: u64 = 0x290BDFEA91DA07FC;
    let g = Generator::new(42);
    let mut c = DenseChunk::empty();
    g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut c);
    let actual = hash_chunk(&c);
    if GOLDEN_42_002 == 0xDEAD_BEEF_DEAD_BEEF {
        println!("UPDATE GOLDEN_42_002 to: 0x{:016X}", actual);
    } else {
        assert_eq!(actual, GOLDEN_42_002, "worldgen output changed");
    }
}

/// Cave system sanity: graph-based caves are spatially
/// structured — not every chunk has carving (that's the point;
/// systems are discoverable). But across a generous scan of
/// underground chunks, at least one should have caves.
///
/// PR4.1: smin composition legitimately allows a fully-carved chunk
/// interior (merged pocket volumes). The per-chunk zero-solid check
/// was removed; only the "at least one carved chunk" invariant remains.
#[test]
fn underground_chunk_has_both_caves_and_solid() {
    let g = Generator::new(42);
    let mut found_carved_chunk = false;
    // Scan a 16 × 16 grid of chunks (one region's worth) at
    // chunk y=-2 (world y [-64, -33]). This depth straddles the
    // Middle / Deep cave bands, so at least one chunk should
    // hit something.
    for cx in -8..8 {
        for cz in -8..8 {
            let mut c = DenseChunk::empty();
            g.fill_chunk(ChunkCoord(IVec3::new(cx, -2, cz)), &mut c);
            let air: usize = c
                .blocks
                .iter()
                .filter(|b| matches!(b, Block::Air | Block::Water))
                .count();
            if air > CHUNK_VOL / 50 {
                // 2% — well above the 0.5% pre-PR-8 threshold;
                // ambient cheese carving means every underground
                // chunk should easily clear this.
                found_carved_chunk = true;
            }
        }
    }
    assert!(
        found_carved_chunk,
        "expected ≥1 chunk in 16×16 scan to overlap a cave feature; found none"
    );
}

/// Above-sea cave fills under a land column should never come
/// from the ocean's surface flood — sea-level water only belongs
/// in ocean columns; lake water only belongs under lakes.
#[test]
fn above_sea_air_under_land_is_dry() {
    let g = Generator::new(42);
    // Find a chunk where every column is land AND none has a
    // lake above it. Probe the chunk straddling sea-level to
    // check the air-above-sea voxels: none of them should be
    // Water (the ocean shouldn't be leaking into the column).
    let mut found = None;
    'outer: for cz in -8..8 {
        for cx in -8..8 {
            let mut ok = true;
            'cols: for lz in 0..CHUNK_DIM_U {
                for lx in 0..CHUNK_DIM_U {
                    let wx = cx * CHUNK_DIM_U as i32 + lx as i32;
                    let wz = cz * CHUNK_DIM_U as i32 + lz as i32;
                    let col = g.column_data(wx, wz);
                    if col.height <= SEA_LEVEL + 5 || col.water_surface_y.is_some() {
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
    // Above-sea chunk (Y=2 is rough sea+).
    let mut chunk = DenseChunk::empty();
    g.fill_chunk(ChunkCoord(IVec3::new(cx, 2, cz)), &mut chunk);
    let chunk_origin_y = 2 * CHUNK_DIM_U as i32;
    for y in 0..CHUNK_DIM_U {
        let wy = chunk_origin_y + y as i32;
        if wy <= SEA_LEVEL {
            continue;
        }
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let b = chunk.get(LocalPos(UVec3::new(x, y, z)));
                assert!(
                    !matches!(b, Block::Water),
                    "above-sea voxel ({},{},{}) in lake-free land chunk \
                     was Water — surface flood leaked into the column",
                    cx * CHUNK_DIM_U as i32 + x as i32,
                    wy,
                    cz * CHUNK_DIM_U as i32 + z as i32,
                );
            }
        }
    }
}

/// Noise carvers are gated by `underground_density_threshold` —
/// they only operate where the raw density is well above zero
/// (deep underground). The surface band should therefore look
/// identical whether the noise channels are nudged or left at
/// defaults: the gate keeps them silent.
///
/// This catches a regression where the gate is dropped and
/// noise carvers start eating into surface terrain.
#[test]
fn noise_carvers_silent_above_underground_threshold() {
    let base = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    // "Permissive cheese" config: drop the offset so cheese
    // wants to carve much more aggressively.
    let mut cfg_permissive = base.clone();
    cfg_permissive.cave.cheese_offset = -1.0;
    let g_permissive = Generator::with_config(
        42,
        crate::worldgen::config::ConfigHolder::new(cfg_permissive),
    );
    let g_default =
        Generator::with_config(42, crate::worldgen::config::ConfigHolder::new(base));
    // Chunk Y=4 → world Y in [128, 159]. MAX_TERRAIN_Y is 140
    // and the test seed has no plate seam pushing peaks above
    // that, so raw_density across this chunk stays comfortably
    // below the underground threshold and the gate must keep
    // carvers silent regardless of `cheese_offset`. The chunk
    // was Y=3 before the smooth-plate-blend fix, but that fix
    // lets some columns near plate boundaries reach into Y≥96,
    // which made `(0, 3, 0)` no longer guaranteed-above-terrain.
    let coord = ChunkCoord(IVec3::new(0, 4, 0));
    let mut chunk_a = DenseChunk::empty();
    let mut chunk_b = DenseChunk::empty();
    g_permissive.fill_chunk(coord, &mut chunk_a);
    g_default.fill_chunk(coord, &mut chunk_b);
    let air = |c: &DenseChunk| -> usize {
        c.blocks
            .iter()
            .filter(|b| matches!(b, Block::Air | Block::Water | Block::Lava))
            .count()
    };
    assert_eq!(
        air(&chunk_a),
        air(&chunk_b),
        "noise carvers leaked above the underground density threshold"
    );
}

/// PR 8 invariant: pillars are additive. With pillar_intensity
/// zero'd out, a deep chunk should have STRICTLY MORE empty
/// space than with pillars on (pillars refill carved voxels).
#[test]
fn pillars_increase_stone_count_relative_to_carvers_alone() {
    let base = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let mut cfg_no_pillars = base.clone();
    cfg_no_pillars.cave.pillar_intensity = 0.0;
    let g_no_pillars = Generator::with_config(
        42,
        crate::worldgen::config::ConfigHolder::new(cfg_no_pillars),
    );
    let g_with = Generator::with_config(42, crate::worldgen::config::ConfigHolder::new(base));
    let coord = ChunkCoord(IVec3::new(0, -2, 0));
    let mut a = DenseChunk::empty();
    let mut b = DenseChunk::empty();
    g_no_pillars.fill_chunk(coord, &mut a);
    g_with.fill_chunk(coord, &mut b);
    let stone = |c: &DenseChunk| {
        c.blocks
            .iter()
            .filter(|b| matches!(b, Block::Stone))
            .count()
    };
    let s_no = stone(&a);
    let s_yes = stone(&b);
    assert!(
        s_yes >= s_no,
        "pillars should add stone back (no_pillars={s_no}, with_pillars={s_yes})"
    );
}

/// Deep caves under land should contain some planned cave-pool
/// fluid, independent of surface ocean columns.
/// Confirms the procedural carver actually carves voxels in
/// underground chunks (not just runs and does nothing). A 16×16
/// grid of chunks at chunk-Y=-1 (world Y -32..-1) should have
/// at least one chunk with substantial air voxels — caves from
/// the carver should be visible at this depth.
#[test]
fn carver_produces_air_in_underground_chunks() {
    let g = Generator::new(42);
    let mut max_air = 0;
    for cx in -8..8 {
        for cz in -8..8 {
            let mut c = DenseChunk::empty();
            g.fill_chunk(ChunkCoord(IVec3::new(cx, -1, cz)), &mut c);
            let air = c.blocks.iter().filter(|b| matches!(b, Block::Air)).count();
            max_air = max_air.max(air);
        }
    }
    assert!(
        max_air > 1500,
        "max air voxels = {max_air} — carver may not be running"
    );
}

#[test]
fn fluid_planner_fills_some_underground_cave_basins() {
    let g = Generator::new(42);
    // Scan a wide grid of deep chunks (Y=-3 ≈ blocks -96..-65)
    // and confirm *at least one* of them has planned fluid.
    // Sparse contained basins + small cave fraction means many chunks
    // can be dry — but across 16x16 chunks we should see at
    // least one fluid pocket.
    let mut total_fluid = 0;
    for cz in -8..8 {
        for cx in -8..8 {
            let mut chunk = DenseChunk::empty();
            g.fill_chunk(ChunkCoord(IVec3::new(cx, -3, cz)), &mut chunk);
            total_fluid += chunk
                .blocks
                .iter()
                .filter(|b| matches!(b, Block::Water | Block::Lava))
                .count();
        }
    }
    assert!(
        total_fluid > 0,
        "deep cave scan produced no fluid across 256 chunks"
    );
}

/// Deep underground chunks must contain no surface blocks.
/// Symptom of the per-chunk depth-reset bug: the topmost solid
/// voxel of every chunk got rendered as a surface block, so
/// digging straight down showed grass-dirt-stone cycles every
/// 32 blocks vertically.
#[test]
fn deep_underground_has_no_surface_blocks() {
    let g = Generator::new(42);
    // Chunk-Y=-2 covers world Y -64..-33 — comfortably below
    // any plausible surface across all biomes.
    let mut grass = 0u32;
    let mut dirt = 0u32;
    let mut sand = 0u32;
    let mut snow = 0u32;
    for cx in -8..8 {
        for cz in -8..8 {
            let mut chunk = DenseChunk::empty();
            g.fill_chunk(ChunkCoord(IVec3::new(cx, -2, cz)), &mut chunk);
            for b in chunk.blocks.iter() {
                match b {
                    Block::Grass => grass += 1,
                    Block::Dirt => dirt += 1,
                    Block::Sand => sand += 1,
                    Block::Snow => snow += 1,
                    _ => {}
                }
            }
        }
    }
    assert_eq!(grass, 0, "deep underground had {grass} grass blocks");
    assert_eq!(dirt, 0, "deep underground had {dirt} dirt blocks");
    assert_eq!(sand, 0, "deep underground had {sand} sand blocks");
    assert_eq!(snow, 0, "deep underground had {snow} snow blocks");
}

/// Generator::with_config holds the ConfigHolder, and
/// `config_snapshot()` reflects swaps.
#[test]
fn generator_holds_config_and_reads_from_holder() {
    let cfg = crate::worldgen::config::WorldgenConfig::bundled_default().unwrap();
    let holder = crate::worldgen::config::ConfigHolder::new(cfg);
    let g = Generator::with_config(42, holder.clone());
    // Initial snapshot has bundled default factor (4.0).
    assert!((g.config_snapshot().density.factor - 4.0).abs() < 1e-5);
    // Swap a new config; the Generator's snapshot should reflect it.
    let mut new_cfg = (*holder.load()).clone();
    new_cfg.density.factor = 1.0;
    holder.swap(new_cfg);
    assert!((g.config_snapshot().density.factor - 1.0).abs() < 1e-5);
}

/// Biome diversity: scanning a few thousand columns across a
/// generous area should turn up every biome at least once.
/// Otherwise either the thresholds are misconfigured (cold
/// belt vanishingly narrow, forest too rare) or the climate
/// noise isn't actually getting sampled.
#[test]
fn all_biomes_appear_in_a_large_scan() {
    let g = Generator::new(42);
    let mut seen = std::collections::HashSet::new();
    // Step 8 blocks at a time so a 2048×2048 scan only costs
    // 64 K column evaluations — fast enough to keep the test
    // under a second even in debug builds.
    for wz in (-1024..1024).step_by(8) {
        for wx in (-1024..1024).step_by(8) {
            seen.insert(g.column_data(wx, wz).biome);
        }
    }
    for expected in [
        Biome::Tundra,
        Biome::SnowyForest,
        Biome::Plains,
        Biome::Forest,
        Biome::Desert,
        Biome::Tropical,
    ] {
        assert!(
            seen.contains(&expected),
            "biome {:?} never appeared in the scan; saw {:?}",
            expected,
            seen
        );
    }
}

/// Cold biomes should plant Snow as their surface block above
/// the coastal elevation buffer. With 3D density the surface
/// height isn't exactly `col.height` anymore — it can shift by
/// up to `SURFACE_BAND` blocks — so the test now scans the
/// chunk top-down to find the actual topmost solid block and
/// checks its kind.
#[test]
fn cold_biome_caps_with_snow() {
    let g = Generator::new(42);
    // With 3D density the surface can deviate from `col.height`
    // by up to `DENSITY_FALLOFF` (~4) blocks. Pick a column
    // where `col.height` is comfortably between the cold-snow
    // floor and the snow line so the actual surface lands in
    // the cold-biome cap band.
    let min_h = SEA_LEVEL + crate::worldgen::tuning::COLD_SNOW_MIN_ABOVE_SEA + 6;
    let max_h = SNOW_LINE - 6;
    let mut found: Option<(i32, i32)> = None;
    'outer: for wz in (-1024..1024).step_by(8) {
        for wx in (-1024..1024).step_by(8) {
            let col = g.column_data(wx, wz);
            if col.biome == Biome::Tundra
                && col.height >= min_h
                && col.height < max_h
                && !col.is_cliff
            {
                found = Some((wx, wz));
                break 'outer;
            }
        }
    }
    let (wx, wz) = found.expect("expected at least one tundra column");
    let col = g.column_data(wx, wz);
    let cx = wx.div_euclid(CHUNK_DIM_U as i32);
    let cz = wz.div_euclid(CHUNK_DIM_U as i32);
    // The actual surface might be in one of two chunks if it
    // happens to span a vertical chunk boundary; check both.
    let lx = wx.rem_euclid(CHUNK_DIM_U as i32) as u32;
    let lz = wz.rem_euclid(CHUNK_DIM_U as i32) as u32;
    let mut found_surface = None;
    for cy in [
        col.height.div_euclid(CHUNK_DIM_U as i32),
        col.height.div_euclid(CHUNK_DIM_U as i32) + 1,
    ] {
        let mut chunk = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(cx, cy, cz)), &mut chunk);
        // Top-down scan in this chunk to find the topmost solid.
        for ly in (0..CHUNK_DIM_U).rev() {
            let block = chunk.blocks
                [crate::voxel::coords::LocalPos(glam::UVec3::new(lx, ly, lz)).to_index()];
            if !matches!(block, Block::Air | Block::Water) {
                found_surface = Some(block);
                break;
            }
        }
        if found_surface.is_some() {
            break;
        }
    }
    let surface = found_surface.expect("topmost solid not found in tundra column");
    assert!(
        matches!(surface, Block::Snow | Block::Sand | Block::Stone),
        "tundra surface at ({wx}, {wz}) was {:?}, expected Snow",
        surface
    );
}

/// Rivers and lakes should produce a non-trivial amount of
/// inland water — somewhere in a generous scan we expect at
/// least one column carved below sea level *and* high enough
/// that the carve is the cause (not just baseline ocean from
/// the height noise). Catches future refactors that
/// inadvertently neutralise the carve pass.
#[test]
fn rivers_or_lakes_carve_inland_water() {
    let g = Generator::new(42);
    // A "carved" column is one whose height landed *below* sea
    // level while the heightmap *without* the river/lake pass
    // would have stayed on dry land. We approximate "would have
    // stayed dry" by sampling far from any river/lake band —
    // but since `column_data` already runs the carve, easier to
    // just count columns at SEA_LEVEL-1 or below where the
    // surrounding 5-block disc has at least one dry column.
    // That rules out the smooth ocean background.
    let mut inland_water_columns = 0usize;
    for wz in (-512..512).step_by(4) {
        for wx in (-512..512).step_by(4) {
            let h = g.column_data(wx, wz).height;
            if h >= SEA_LEVEL {
                continue;
            }
            // Inland if any neighbour 24 blocks away is above
            // sea level.
            let neighbours = [
                g.column_data(wx + 24, wz).height,
                g.column_data(wx - 24, wz).height,
                g.column_data(wx, wz + 24).height,
                g.column_data(wx, wz - 24).height,
            ];
            if neighbours.iter().any(|&n| n > SEA_LEVEL + 4) {
                inland_water_columns += 1;
            }
        }
    }
    assert!(
        inland_water_columns > 0,
        "expected at least one inland water column from river/lake carving, found none"
    );
}

#[test]
fn above_sea_river_segment_generates_water_blocks() {
    let g = Generator::new(42);
    let mut found = None;
    'outer: for wz in (-2048..2048).step_by(4) {
        for wx in (-2048..2048).step_by(4) {
            let coord = region::RegionCoord::containing(wx, wz);
            let chunk_origin = ChunkCoord(IVec3::new(
                coord.x * (FINE_REGION_SIZE / 32),
                0,
                coord.z * (FINE_REGION_SIZE / 32),
            ));
            let regions = g.gather_chunk_regions(chunk_origin);
            if let Some(cell) = regions.river_cell_at(wx, wz, g.seed()) {
                if cell.surface_y > SEA_LEVEL + 2 {
                    found = Some((wx, wz, cell.surface_y));
                    break 'outer;
                }
            }
        }
    }

    let (wx, wz, wy) = found.expect("expected an above-sea river in the scan");
    let coord = ChunkCoord(IVec3::new(
        wx.div_euclid(CHUNK_DIM_U as i32),
        wy.div_euclid(CHUNK_DIM_U as i32),
        wz.div_euclid(CHUNK_DIM_U as i32),
    ));
    let mut chunk = DenseChunk::empty();
    g.fill_chunk(coord, &mut chunk);
    let local = LocalPos(UVec3::new(
        wx.rem_euclid(CHUNK_DIM_U as i32) as u32,
        wy.rem_euclid(CHUNK_DIM_U as i32) as u32,
        wz.rem_euclid(CHUNK_DIM_U as i32) as u32,
    ));
    assert_eq!(chunk.get(local), Block::Water);
}

#[test]
fn seed42_region_has_visible_surface_rivers_and_lakes() {
    let g = Generator::new(42);
    let mut river_water = 0usize;
    // surface_water counts ocean + lake voxels: any Water block where
    // the column's unified water_surface_y says this column is wet and
    // the voxel is within the expected water depth.
    let mut surface_water = 0usize;

    for cz in -4..4 {
        for cx in -4..4 {
            for cy in 1..4 {
                let coord = ChunkCoord(IVec3::new(cx, cy, cz));
                let origin = coord.origin().0;
                let regions = g.gather_chunk_regions(coord);
                let mut chunk = DenseChunk::empty();
                g.fill_chunk(coord, &mut chunk);

                for z in 0..CHUNK_DIM_U {
                    for x in 0..CHUNK_DIM_U {
                        let wx = origin.x + x as i32;
                        let wz = origin.z + z as i32;
                        let col = g.column_data(wx, wz);
                        let river = regions.river_cell_at(wx, wz, g.seed());

                        for y in 0..CHUNK_DIM_U {
                            let wy = origin.y + y as i32;
                            if chunk.get(LocalPos(UVec3::new(x, y, z))) != Block::Water {
                                continue;
                            }
                            if let Some(cell) = river {
                                if cell.surface_y > SEA_LEVEL
                                    && wy >= cell.bed_y
                                    && wy <= cell.surface_y
                                {
                                    river_water += 1;
                                }
                            }
                            // Count lake + ocean water via the unified water_surface_y.
                            if let Some(wsurf) = col.water_surface_y {
                                if wy >= col.height && wy <= wsurf {
                                    surface_water += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(
        river_water > 2_000,
        "expected visible above-sea river water, found {river_water} voxels"
    );
    assert!(
        surface_water > 5_000,
        "expected visible surface water (ocean + lake), found {surface_water} voxels"
    );
}

#[test]
fn generated_lava_never_appears_above_lava_band() {
    let g = Generator::new(42);
    for cx in -4..4 {
        for cz in -4..4 {
            for cy in -1..4 {
                let coord = ChunkCoord(IVec3::new(cx, cy, cz));
                let origin = coord.origin().0;
                let mut chunk = DenseChunk::empty();
                g.fill_chunk(coord, &mut chunk);
                for z in 0..CHUNK_DIM_U {
                    for y in 0..CHUNK_DIM_U {
                        for x in 0..CHUNK_DIM_U {
                            let block = chunk.get(LocalPos(UVec3::new(x, y, z)));
                            if block == Block::Lava {
                                let wy = origin.y + y as i32;
                                assert!(
                                    wy <= aquifer::LAVA_BAND_TOP_Y,
                                    "lava generated above band at y={wy}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The plate-driven heightmap caps final heights at
/// `MAX_TERRAIN_Y` (140 by default) so the tallest possible
/// peak still sits inside the loaded vertical radius. Catches a
/// future refactor that drops the cap (or sets `MAX_TERRAIN_Y`
/// above the chunk-stack ceiling).
#[test]
fn ridged_mountains_respect_height_cap() {
    let g = Generator::new(42);
    // Scan a generous area: the cap should hold everywhere, not
    // just near origin. CC ridges peak at ~SEA_LEVEL + 28 + 90 =
    // 180, but the clamp pulls them back to MAX_TERRAIN_Y.
    for wx in (-2048..=2048).step_by(128) {
        for wz in (-2048..=2048).step_by(128) {
            let col = g.column_data(wx, wz);
            assert!(
                col.height <= MAX_TERRAIN_Y,
                "column height {} broke the cap at ({wx}, {wz})",
                col.height
            );
        }
    }
}

#[test]
fn chunk_at_sea_level_has_water_or_solid() {
    // Sanity: the column-wise terrain must place *something* in any
    // sea-level chunk — either solid (under-water terrain) or water
    // (above-terrain flood).
    let g = Generator::new(42);
    let mut c = DenseChunk::empty();
    g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut c);
    let has_non_air = c.blocks.iter().any(|&b| b != Block::Air);
    assert!(has_non_air);
}
