//! Post-pass surface corrections applied after terrain fill and fluid
//! placement.
//!
//! Two problems are fixed here that cannot be handled in the main voxel
//! loop because they require a full-chunk view after all carving is done:
//!
//! - **Vertical-run clamp** — caps any continuous air column at
//!   [`MAX_VERTICAL_AIR_RUN`] voxels so players cannot fall into an
//!   effectively-infinite pit from the chunk below.
//! - **Surface block fixer** (Terasology-borrowed technique) — two sub-passes:
//!   - *Pass A (ceiling grass removal)*: a Grass/Snow/Sand/Dirt block with
//!     Air both above and below is a cave-ceiling artefact; replace with Stone.
//!   - *Pass B (cave-floor surface block)*: when a cave breaches the
//!     heightmap, the first solid voxel below the breach should receive the
//!     climate-correct surface block rather than bare Stone. Spreads
//!     laterally by [`SURFACE_SPREAD`] voxels.
//!
//! Both passes are O(chunk voxels) and allocation-free.
//!
//! See `docs/book/content/part-4-chunk-fill/4.7-surface-fixer.mdx`.

use crate::voxel::block::Block;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{CHUNK_DIM_U, LocalPos};
use crate::worldgen::columns::ColumnData;
use crate::worldgen::config::WorldgenConfig;
use crate::worldgen::surface;
use crate::worldgen::tuning::{MAX_VERTICAL_AIR_RUN, SEA_LEVEL, SURFACE_SPREAD};
use glam::{IVec3, UVec3};

/// Apply all three post-pass surface corrections to `out` in place.
///
/// # Parameters
/// - `out` — the chunk being built (modified in place).
/// - `columns` — 32×32 column data in row-major `z * 32 + x` order.
/// - `origin` — world-space origin of the chunk's (0,0,0) corner.
/// - `cfg` — the active worldgen config (surface rule tree, etc.).
/// - `seed` — world seed forwarded to the surface rule for stochastic
///   surface-block decisions.
///
/// # Panics
/// Panics in debug if `columns.len() != 1024`.
pub(crate) fn apply_surface_post_passes(
    out: &mut DenseChunk,
    columns: &[ColumnData],
    origin: IVec3,
    cfg: &WorldgenConfig,
    seed: u64,
) {
    debug_assert_eq!(columns.len(), (CHUNK_DIM_U as usize).pow(2));

    apply_vertical_run_clamp(out);
    apply_surface_block_fixer(out, columns, origin, cfg, seed);
}

// ── Vertical run clamp ────────────────────────────────────────────────────

/// Cap any continuous vertical air column at `MAX_VERTICAL_AIR_RUN` voxels.
///
/// Prevents fall hazards caused by cave shafts or natural overhangs that
/// extend through most of a chunk's height. The cap is a per-chunk
/// approximation: a run that straddles the chunk boundary can reach up to
/// `2 × MAX_VERTICAL_AIR_RUN` across it, which is accepted as harmless.
fn apply_vertical_run_clamp(out: &mut DenseChunk) {
    for x in 0..CHUNK_DIM_U {
        for z in 0..CHUNK_DIM_U {
            let mut run = 0i32;
            for y in 0..CHUNK_DIM_U {
                let pos = LocalPos(UVec3::new(x, y, z));
                if out.get(pos) == Block::Air {
                    run += 1;
                    if run > MAX_VERTICAL_AIR_RUN {
                        out.set(pos, Block::Stone);
                        run = 0;
                    }
                } else {
                    run = 0;
                }
            }
        }
    }
}

// ── Surface block fixer ───────────────────────────────────────────────────

/// A block that the terrain / surface-rule passes treat as a top surface.
///
/// Grass, Dirt, Sand, and Snow are the only blocks the surface rule tree
/// stamps on top of terrain. Any of these with open faces above *and* below
/// is a cave-ceiling artefact.
fn is_surface_block(b: Block) -> bool {
    matches!(b, Block::Grass | Block::Sand | Block::Snow | Block::Dirt)
}

/// Open face: a voxel that is passable (does not count as solid terrain).
///
/// Used to detect cave breaches (the surface-height voxel is open) and to
/// identify the top face of a cave floor (the block above it is open).
fn is_surface_open(b: Block) -> bool {
    matches!(b, Block::Air | Block::Water | Block::Lava)
}

/// How many voxels to search downward from the surface height before
/// giving up and treating the open region as a buried chamber rather than
/// a surface breach.
///
/// A breach produces a cave mouth visible from outside; a buried chamber
/// is fully enclosed underground. The distinction matters because only
/// breaches should receive a surface-block stamp on their floor — a buried
/// chamber's floor is stone by design.
const MAX_BREACH_SEARCH_DEPTH: i32 = 4;

fn apply_surface_block_fixer(
    out: &mut DenseChunk,
    columns: &[ColumnData],
    origin: IVec3,
    cfg: &WorldgenConfig,
    seed: u64,
) {
    let dim = CHUNK_DIM_U;

    // ── Pass A: remove ceiling grass ─────────────────────────────────────
    //
    // Ceiling grass arises when a cave entrance or sinkhole shaft is carved
    // *above* `h_target`. The topmost-solid block above the cave interior
    // then sits inside `WithinSurfaceBand(16)` AND above `h_target - 1`, so
    // the surface rule legitimately stamps Grass/Snow/Sand — but the block
    // has Air on both sides. Replace with Stone.
    for x in 0..dim {
        for z in 0..dim {
            for y in 1..(dim - 1) {
                let pos = LocalPos(UVec3::new(x, y, z));
                let above = LocalPos(UVec3::new(x, y + 1, z));
                let below = LocalPos(UVec3::new(x, y - 1, z));
                if is_surface_block(out.get(pos))
                    && out.get(above) == Block::Air
                    && out.get(below) == Block::Air
                {
                    out.set(pos, Block::Stone);
                }
            }
        }
    }

    // ── Pass B: cave-floor surface block ─────────────────────────────────
    //
    // For each XZ column: if `h_target` (col.height) falls inside this
    // chunk's Y range AND the voxel at that height is Air, a cave has
    // breached the surface. Find the first solid voxel below and stamp the
    // climate-correct surface block. Then spread laterally by SURFACE_SPREAD.
    //
    // Skipped for wet columns (water_surface_y.is_some()): the force-flood
    // pass filled the water column with Air above col.height;
    // apply_surface_fluids will stamp Water there. Stamping a surface block
    // into a flooded column would produce grass underwater.
    for x in 0..dim {
        for z in 0..dim {
            let wx = origin.x + x as i32;
            let wz = origin.z + z as i32;
            let col = columns[z as usize * dim as usize + x as usize];
            if col.water_surface_y.is_some() {
                continue;
            }
            let h_target = col.height;

            // Is h_target inside this chunk's Y range?
            let ly_at_h = h_target - origin.y;
            if ly_at_h < 0 || ly_at_h >= dim as i32 {
                continue;
            }
            // Is the voxel at h_target open? (cave breached the surface)
            let at_surface = LocalPos(UVec3::new(x, ly_at_h as u32, z));
            if !is_surface_open(out.get(at_surface)) {
                continue;
            }

            // Scan downward for the first solid voxel in this chunk.
            // Limit search to MAX_BREACH_SEARCH_DEPTH: a voxel that's
            // Air with solid stone 30 blocks below is a buried chamber,
            // not a surface breach, and shouldn't get a surface stamp.
            let mut floor_ly = ly_at_h - 1;
            let mut steps = 0;
            while floor_ly >= 0
                && is_surface_open(out.get(LocalPos(UVec3::new(x, floor_ly as u32, z))))
                && steps < MAX_BREACH_SEARCH_DEPTH
            {
                floor_ly -= 1;
                steps += 1;
            }
            if floor_ly < 0 || steps >= MAX_BREACH_SEARCH_DEPTH {
                continue; // Floor is too deep; this is a buried chamber, not a surface breach.
            }

            // Determine the appropriate surface block for this column using
            // the same climate-driven surface rule tree as the main fill,
            // but with h_target set to the cave floor position so the
            // surface-band and above-preliminary-surface checks pass.
            let floor_wy = origin.y + floor_ly;
            let surf_ctx = surface::SurfaceContext {
                wx,
                wy: floor_wy,
                wz,
                h_target: floor_wy, // floor IS the new local surface
                biome: col.biome,
                is_cliff: col.is_cliff,
                desertness: col.desertness,
                depth_below_surface: 0,
                water_surface_y: col.water_surface_y,
                seed,
                cfg,
                sea_level: SEA_LEVEL,
            };
            let surface_block = cfg.surface.apply(&surf_ctx).unwrap_or(Block::Grass);

            // Only replace Stone (bare cave floor) — don't overwrite
            // water / lava / already-surface blocks. Additionally verify the
            // block below the floor is also solid (not Air) to avoid
            // misidentifying a floating block (e.g. a ceiling converted from
            // grass by Pass A) as a cave floor.
            let floor_pos = LocalPos(UVec3::new(x, floor_ly as u32, z));
            let floor_is_true_floor = floor_ly == 0
                || out.get(LocalPos(UVec3::new(x, (floor_ly - 1) as u32, z))) != Block::Air;
            if out.get(floor_pos) == Block::Stone && floor_is_true_floor {
                out.set(floor_pos, surface_block);
            }

            // Lateral spread: for neighbours within SURFACE_SPREAD in XZ
            // that also have air at the floor_wy level and stone below,
            // apply the same surface block.
            for dx in -SURFACE_SPREAD..=SURFACE_SPREAD {
                for dz in -SURFACE_SPREAD..=SURFACE_SPREAD {
                    if dx == 0 && dz == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;
                    if nx < 0 || nx >= dim as i32 {
                        continue;
                    }
                    if nz < 0 || nz >= dim as i32 {
                        continue;
                    }
                    // The neighbour's floor: scan from the same ly_at_h
                    // level downward. Same MAX_BREACH_SEARCH_DEPTH cap.
                    let mut nly = ly_at_h - 1;
                    let mut nsteps = 0;
                    while nly >= 0
                        && is_surface_open(
                            out.get(LocalPos(UVec3::new(nx as u32, nly as u32, nz as u32))),
                        )
                        && nsteps < MAX_BREACH_SEARCH_DEPTH
                    {
                        nly -= 1;
                        nsteps += 1;
                    }
                    if nly < 0 || nsteps >= MAX_BREACH_SEARCH_DEPTH {
                        continue;
                    }
                    let n_floor_pos = LocalPos(UVec3::new(nx as u32, nly as u32, nz as u32));
                    let n_above_pos = LocalPos(UVec3::new(nx as u32, (nly + 1) as u32, nz as u32));
                    // Apply only if the top face is open (floor, not buried)
                    // AND the block below is also solid (not a floating block).
                    let n_below_is_solid = nly == 0
                        || out.get(LocalPos(UVec3::new(nx as u32, (nly - 1) as u32, nz as u32)))
                            != Block::Air;
                    if is_surface_open(out.get(n_above_pos))
                        && out.get(n_floor_pos) == Block::Stone
                        && n_below_is_solid
                    {
                        out.set(n_floor_pos, surface_block);
                    }
                }
            }
        }
    }
}
