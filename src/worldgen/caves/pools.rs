//! Cave pool derivation — qualifying large, deep-enough chambers get fluid
//! bodies (water or lava) placed at their floor.
use super::style::SALT_POOL_LAVA;
use crate::worldgen::aquifer::LAVA_BAND_TOP_Y;
use crate::worldgen::fluid::FluidBodyKind;
use crate::worldgen::hash::mix_unit;
use crate::worldgen::region::FineRegion;
use crate::worldgen::tuning::*;

/// Derive `CavePool` entries for every qualifying chamber in the region.
///
/// A chamber qualifies when:
/// - Its minimum XZ semi-axis ≥ `POOL_MIN_RADIUS_XZ` (large enough to look like a real pool).
/// - Its ceiling `(center.y + radii.y)` sits at least `POOL_TOP_CLEARANCE` blocks below `SEA_LEVEL`
///   (pool must be underground, not breaching the surface water).
///
/// The pool surface is placed at `floor + height * POOL_SURFACE_FRACTION`, clamped so there
/// is always at least 1 block of fluid and `POOL_TOP_CLEARANCE` blocks of air above.
/// Deep chambers (top below `LAVA_BAND_TOP_Y`) roll for lava with probability `POOL_LAVA_PROB`.
pub(super) fn derive_cave_pools(seed: u64, region: &mut FineRegion) {
    region.cave_pools.clear();
    // Work on indices to avoid borrow conflicts.
    let system_count = region.cave_systems.len();
    for si in 0..system_count {
        let chamber_count = region.cave_systems[si].chambers.len();
        for ci in 0..chamber_count {
            let ch = region.cave_systems[si].chambers[ci];
            let min_xz = ch.radii.0.x.min(ch.radii.0.z);
            if min_xz < POOL_MIN_RADIUS_XZ {
                continue;
            }
            let ceiling = (ch.center.y + ch.radii.0.y).ceil() as i32;
            if ceiling > SEA_LEVEL - POOL_TOP_CLEARANCE {
                continue;
            }
            let floor = (ch.center.y - ch.radii.0.y).floor() as i32;
            let height = ((ch.radii.0.y * 2.0) as i32).max(1);
            let surface_y_raw = floor + (height as f32 * POOL_SURFACE_FRACTION) as i32;
            // Clamp: must have at least 1 block of fluid and POOL_TOP_CLEARANCE air above.
            let surface_y = surface_y_raw
                .max(floor + 1)
                .min(ceiling - POOL_TOP_CLEARANCE);
            if surface_y <= floor {
                continue;
            }
            // Lava if the entire chamber sits below the lava band ceiling.
            let is_lava = ceiling <= LAVA_BAND_TOP_Y && {
                let roll = mix_unit(
                    seed,
                    &[
                        ch.center.x as i32,
                        ch.center.y as i32,
                        ch.center.z as i32,
                        SALT_POOL_LAVA,
                    ],
                );
                roll < POOL_LAVA_PROB
            };
            use crate::worldgen::region::CavePool;
            region.cave_pools.push(CavePool {
                center: ch.center,
                radii: ch.radii,
                surface_y,
                bed_y: floor,
                kind: if is_lava {
                    FluidBodyKind::LavaPool
                } else {
                    FluidBodyKind::CavePool
                },
            });
        }
    }
}
