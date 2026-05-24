//! Cave system builder — orchestrates chamber placement, tunnel graph,
//! entrance rolling, AABB finalization, and the public
//! [`build_systems_for_region`] entry point.
//!
//! The heavy lifting is delegated to sibling modules:
//!
//! - [`super::chamber`] — Poisson-disk chamber placement
//! - [`super::tunnel`] — Kruskal MST tunnel graph
//! - [`super::entrance`] — Sinkhole / CliffMouth / Skylight rollers
//!
//! This file owns: system-count rolling, bounding-box placement,
//! AABB finalization, and the [`style_params`] helper that maps a
//! [`CaveStyle`] variant to the corresponding numeric ranges from the
//! hot-reloadable config.
use super::chamber::sample_chambers;
use super::connectors::build_vertical_connectors;
use super::ctx::{CaveCtx, StyleParams};
use super::entrance::roll_entrances;
use super::pools::derive_cave_pools;
use super::style::{
    CaveStyle, DepthBand, SALT_BB_ORIGIN_X, SALT_BB_ORIGIN_Z, SALT_SYSTEM_COUNT, pick_style,
};
use super::tunnel::connect_chambers_mst;
use crate::worldgen::hash::mix_u32;
use crate::worldgen::region::{
    CaveSystem, Chamber, Entrance, FineRegion, RegionCoord, SystemBoundingBox, Tunnel,
};
use crate::worldgen::terrain_ref::TerrainRef;
use crate::worldgen::tuning::*;
use glam::{IVec3, Vec3};

/// Map a [`CaveStyle`] variant to its per-style numeric ranges.
///
/// Extracted once per system and threaded as `&StyleParams` into
/// `sample_chambers` and `connect_chambers_mst`. This avoids carrying
/// the entire `CaveConfig` deeper than necessary.
pub(super) fn style_params(
    style: CaveStyle,
    table: &crate::worldgen::config::CaveStyleTable,
) -> StyleParams {
    match style {
        CaveStyle::Cathedral => StyleParams {
            chamber_count: table.cathedral_chamber_count,
            r_xz: table.cathedral_r_xz,
            r_y: table.cathedral_r_y,
            tunnel_r: table.cathedral_tunnel_r,
        },
        CaveStyle::Warren => StyleParams {
            chamber_count: table.warren_chamber_count,
            r_xz: table.warren_r_xz,
            r_y: table.warren_r_y,
            tunnel_r: table.warren_tunnel_r,
        },
        CaveStyle::Slot => StyleParams {
            chamber_count: table.slot_chamber_count,
            r_xz: table.slot_r_xz,
            r_y: table.slot_r_y,
            tunnel_r: table.slot_tunnel_r,
        },
        CaveStyle::Sump => StyleParams {
            chamber_count: table.sump_chamber_count,
            r_xz: table.sump_r_xz,
            r_y: table.sump_r_y,
            tunnel_r: table.sump_tunnel_r,
        },
        CaveStyle::Karst => StyleParams {
            chamber_count: table.karst_chamber_count,
            r_xz: table.karst_r_xz,
            r_y: table.karst_r_y,
            tunnel_r: table.karst_tunnel_r,
        },
    }
}

/// Build all cave systems for the given fine region. Each system is
/// deterministically derived from `(seed, coord, system_idx)`.
///
/// `CAVE_SYSTEMS_PER_REGION` acts as a compile-time safety cap;
/// `cave_cfg.systems_per_region_max` is the hot-reloadable override.
/// The effective maximum is the minimum of the two.
pub(crate) fn build_systems_for_region(
    seed: u64,
    coord: RegionCoord,
    terrain: TerrainRef<'_>,
    region: &mut FineRegion,
    cave_cfg: &crate::worldgen::config::CaveConfig,
) {
    let n_min = CAVE_SYSTEMS_PER_REGION.0;
    let n_max = cave_cfg
        .systems_per_region_max
        .min(CAVE_SYSTEMS_PER_REGION.1);
    let n = n_min + (mix_u32(seed, &[coord.x, coord.z, SALT_SYSTEM_COUNT]) % (n_max - n_min + 1));
    region.cave_systems.clear();
    for system_idx in 0..n as i32 {
        let ctx = CaveCtx {
            seed,
            coord,
            system_idx,
        };
        let sys = build_system(ctx, terrain, cave_cfg);
        region.cave_systems.push(sys);
    }
    build_vertical_connectors(seed, coord, cave_cfg, region);
    derive_cave_pools(seed, region);
}

/// Roll the system bounding box within the region at the given Y extents.
///
/// The box origin is randomised within the region footprint. The box is
/// allowed to straddle the region boundary — neighbouring regions are
/// consulted via the 3×3 neighbourhood at chunk fill time so nothing is
/// dropped.
fn roll_bounding_box(ctx: CaveCtx, y_min: i32, y_max: i32) -> SystemBoundingBox {
    let CaveCtx {
        seed,
        coord,
        system_idx,
    } = ctx;
    let bb_size = IVec3::new(
        CAVE_SYSTEM_BB_HALF_EXTENT,
        (y_max - y_min).min(64),
        CAVE_SYSTEM_BB_HALF_EXTENT,
    );
    let region_origin = coord.origin();
    let bb_origin_x = region_origin.0
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, SALT_BB_ORIGIN_X])
            % (FINE_REGION_SIZE - bb_size.x).max(1) as u32) as i32
        - (bb_size.x / 2);
    let bb_origin_z = region_origin.1
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, SALT_BB_ORIGIN_Z])
            % (FINE_REGION_SIZE - bb_size.z).max(1) as u32) as i32
        - (bb_size.z / 2);
    let bb_origin_y = y_min;
    let min = IVec3::new(bb_origin_x, bb_origin_y, bb_origin_z);
    SystemBoundingBox {
        min,
        max: min + bb_size,
    }
}

/// Expand the rolled bounding box to conservatively cover everything the
/// SDF functions might carve: chamber ellipsoids, tunnel capsule control
/// points, and entrance shaft footprints.
///
/// The chunk-side AABB intersection test must never miss a carve, so the
/// expansion is deliberate and slightly generous (1-block halo on chambers
/// and tunnel endpoints).
fn finalize_aabb(
    initial: SystemBoundingBox,
    chambers: &[Chamber],
    tunnels: &[Tunnel],
    entrances: &[Entrance],
) -> SystemBoundingBox {
    let mut min = initial.min;
    let mut max = initial.max;
    for c in chambers {
        let cmin = (c.center - c.radii.0 - Vec3::splat(1.0)).floor();
        let cmax = (c.center + c.radii.0 + Vec3::splat(1.0)).ceil();
        min.x = min.x.min(cmin.x as i32);
        min.y = min.y.min(cmin.y as i32);
        min.z = min.z.min(cmin.z as i32);
        max.x = max.x.max(cmax.x as i32);
        max.y = max.y.max(cmax.y as i32);
        max.z = max.z.max(cmax.z as i32);
    }
    for t in tunnels {
        for p in &t.control_points {
            min.x = min.x.min((p.x - t.radius.0 - 1.0) as i32);
            min.y = min.y.min((p.y - t.radius.0 - 1.0) as i32);
            min.z = min.z.min((p.z - t.radius.0 - 1.0) as i32);
            max.x = max.x.max((p.x + t.radius.0 + 1.0) as i32);
            max.y = max.y.max((p.y + t.radius.0 + 1.0) as i32);
            max.z = max.z.max((p.z + t.radius.0 + 1.0) as i32);
        }
    }
    for e in entrances {
        // Vertical shaft extent — clip downward to chamber, upward to surface.
        min.y = min.y.min(e.surface.y);
        max.y = max.y.max(e.surface.y + 1);
        min.x = min.x.min(e.surface.x - ENTRANCE_BB_EXPAND);
        max.x = max.x.max(e.surface.x + ENTRANCE_BB_EXPAND);
        min.z = min.z.min(e.surface.z - ENTRANCE_BB_EXPAND);
        max.z = max.z.max(e.surface.z + ENTRANCE_BB_EXPAND);
    }
    SystemBoundingBox { min, max }
}

/// Build one cave system deterministically from `ctx`.
///
/// Order of operations:
/// 1. Pick depth band → Y extents → roll bounding box.
/// 2. Pick style → extract `StyleParams`.
/// 3. Place chambers (Poisson-disk, [`sample_chambers`]).
/// 4. Connect chambers (Kruskal MST, [`connect_chambers_mst`]).
/// 5. Roll entrances ([`roll_entrances`]).
/// 6. Expand AABB to include all carved geometry ([`finalize_aabb`]).
fn build_system(
    ctx: CaveCtx,
    terrain: TerrainRef<'_>,
    cave_cfg: &crate::worldgen::config::CaveConfig,
) -> CaveSystem {
    let CaveCtx {
        seed,
        coord,
        system_idx,
    } = ctx;
    let band = DepthBand::pick(seed, system_idx, coord);
    let (y_min, y_max) = band.range();

    let style = pick_style(seed, coord, system_idx, band, cave_cfg);
    let sp = style_params(style, &cave_cfg.style_table);

    let bb = roll_bounding_box(ctx, y_min, y_max);
    let chambers = sample_chambers(ctx, bb, style, &sp, cave_cfg);
    let tunnels = connect_chambers_mst(ctx, &chambers, &sp);
    let entrances = roll_entrances(ctx, &chambers, band, terrain);
    let bbox = finalize_aabb(bb, &chambers, &tunnels, &entrances);

    CaveSystem {
        bbox,
        chambers,
        tunnels,
        entrances,
        style,
        trunk: None,
        vertical_connectors: vec![],
    }
}
