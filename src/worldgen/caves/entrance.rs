//! Cave entrance rolling — Sinkhole, CliffMouth, and Skylight.
//!
//! [`roll_entrances`] walks every chamber in a system and decides, with a
//! band-weighted probability, whether that chamber should have a surface
//! entrance. Three entrance types are tried in priority order:
//!
//! 1. **Sinkhole** — the chamber top is within [`SINKHOLE_DEPTH_MAX`] of
//!    `h_pre`. A narrow vertical shaft is carved from the surface down to
//!    the chamber roof; the shaft top is extended by [`SURFACE_BAND`] to
//!    clear any 3D-density bumps above the nominal surface height that
//!    would otherwise produce floating terrain islands around the opening.
//!
//! 2. **Cliff mouth** — a cliff column (slope above [`CLIFF_SLOPE_THRESH`])
//!    is reachable within [`CLIFF_ENTRANCE_DIST`] in any of 16 radial
//!    directions from the chamber centre. The entrance is pinned at the
//!    cliff column's surface height so it blends naturally into the rock
//!    face rather than hanging in mid-air.
//!
//! 3. **Skylight** — the chamber top is between [`SKYLIGHT_DEPTH_MIN`] and
//!    [`SKYLIGHT_DEPTH_MAX`] below the surface. The shaft is the same shape
//!    as a sinkhole but narrower in appearance because the depth gap makes
//!    it read as a beam of light rather than a collapse feature.
//!
//! Exactly one entrance per chamber is generated; a chamber may have no
//! entrance if the band entrance probability gate rejects it or if none of
//! the three conditions hold.
//!
//! See `docs/book/content/part-3-region-build/3.5-caves.mdx`.

use super::ctx::CaveCtx;
use super::style::{DepthBand, SALT_ENTRANCE_PROB};
use crate::worldgen::hash::mix_unit;
use crate::worldgen::region::{Chamber, Entrance, EntranceKind};
use crate::worldgen::terrain_ref::TerrainRef;
use crate::worldgen::tuning::*;
use glam::IVec3;

/// Roll Sinkhole / CliffMouth / Skylight entrances for each chamber.
///
/// Each chamber is independently gated by a uniform roll against
/// `band.entrance_prob()`. Chambers that pass are checked for Sinkhole →
/// CliffMouth → Skylight in priority order; the first matching condition
/// yields one entrance.
///
/// Returns a `Vec<Entrance>` — may be empty if no chambers qualify or all
/// rolls fail.
pub(super) fn roll_entrances(
    ctx: CaveCtx,
    chambers: &[Chamber],
    band: DepthBand,
    terrain: TerrainRef<'_>,
) -> Vec<Entrance> {
    let CaveCtx {
        seed,
        coord,
        system_idx,
    } = ctx;
    let mut entrances: Vec<Entrance> = Vec::new();
    for (ci, chamber) in chambers.iter().enumerate() {
        let try_roll = mix_unit(
            seed,
            &[coord.x, coord.z, system_idx, SALT_ENTRANCE_PROB, ci as i32],
        );
        if try_roll >= band.entrance_prob() {
            continue;
        }
        let cwx = chamber.center.x as i32;
        let cwy_top = (chamber.center.y + chamber.radii.0.y) as i32;
        let cwz = chamber.center.z as i32;
        let surface_h = terrain.heightmap.h_pre(
            seed,
            chamber.center.x,
            chamber.center.z,
            terrain.climate,
            terrain.density,
        ) as i32;

        // 1. Sinkhole: chamber top close enough to the surface to collapse.
        // The shaft extends `SURFACE_BAND` above `h_pre` to clear any
        // 3D-density bumps that would leave floating terrain fragments above
        // the entrance opening.
        if surface_h - cwy_top <= SINKHOLE_DEPTH_MAX && surface_h - cwy_top >= -2 {
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::Sinkhole,
                surface: IVec3::new(cwx, surface_h + SURFACE_BAND, cwz),
            });
            continue;
        }

        // 2. Cliff mouth: scan 16 evenly-spaced radial directions for a cliff
        // column within CLIFF_ENTRANCE_DIST. The first hit wins — scanning
        // stops early so nearby cliffs always win over distant ones.
        let mut found_cliff: Option<IVec3> = None;
        for step in 0..16 {
            let theta = step as f32 * std::f32::consts::TAU / 16.0;
            let cwx_f = chamber.center.x + theta.cos() * CLIFF_ENTRANCE_DIST as f32;
            let cwz_f = chamber.center.z + theta.sin() * CLIFF_ENTRANCE_DIST as f32;
            if terrain
                .heightmap
                .is_cliff(seed, cwx_f, cwz_f, terrain.climate, terrain.density)
            {
                found_cliff = Some(IVec3::new(
                    cwx_f as i32,
                    terrain
                        .heightmap
                        .h_pre(seed, cwx_f, cwz_f, terrain.climate, terrain.density)
                        as i32,
                    cwz_f as i32,
                ));
                break;
            }
        }
        if let Some(surface) = found_cliff {
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::CliffMouth,
                surface,
            });
            continue;
        }

        // 3. Skylight: chamber top deep enough for a dramatic light shaft but
        // not so deep that the shaft would be invisible from above. The shaft
        // shape is identical to a sinkhole; the depth range is what produces
        // the distinct "beam of light" aesthetic.
        let dy = surface_h - cwy_top;
        if (SKYLIGHT_DEPTH_MIN..=SKYLIGHT_DEPTH_MAX).contains(&dy) {
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::Skylight,
                surface: IVec3::new(cwx, surface_h + SURFACE_BAND, cwz),
            });
        }
    }
    entrances
}
