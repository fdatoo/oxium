//! Live terrain feature search over the worldgen LRU cache.
//!
//! Exposes [`find`] for locating named terrain features — cave entrances, lava
//! pools, water bodies, biome patches, river channels — by scanning concentric
//! Chebyshev rings of fine regions outward from an origin point. The worldgen
//! LRU cache acts as the search index: materialising a region once caches it
//! for every subsequent call, so repeated searches in the same area are cheap.
//!
//! ### Usage (from `oxium-probe inspect --find`)
//!
//! ```no_run
//! use oxium::worldgen::{Generator, features::{self, FeatureKind}};
//! let g = Generator::new(42);
//! let hits = features::find(&g, 0, 0, FeatureKind::Lava, 4096, 3).unwrap();
//! ```
//!
//! See `docs/superpowers/plans/let-s-plan-a-proper-stateful-sloth.md` for the
//! full harness design.

use anyhow::bail;

use crate::worldgen::biome::Biome;
use crate::worldgen::fluid::FluidBodyKind;
use crate::worldgen::region::{self, FineRegion, RegionCoord};
use crate::worldgen::tuning::{FINE_CELL, FINE_CELLS_PER_REGION, FINE_REGION_SIZE};
use crate::worldgen::Generator;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A terrain feature category to search for. Maps one-to-one to the
/// `--find <kind>` argument accepted by `oxium-probe inspect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    /// Surface water column (ocean, lake, or river surface).
    Water,
    /// Lava pool inside a cave chamber.
    Lava,
    /// Cave entrance to the surface: sinkhole, cliff mouth, or skylight.
    CaveEntrance,
    /// River channel segment (from hydrology `RiverSegment` list).
    RiverChannel,
    /// Column whose biome is [`Biome::Forest`].
    Forest,
    /// Column whose biome is [`Biome::Tropical`].
    Tropical,
    /// Column whose biome is [`Biome::Desert`].
    Desert,
    /// Column whose biome is [`Biome::Tundra`].
    Tundra,
    /// Column whose biome is [`Biome::Plains`].
    Plains,
    /// Column whose biome is [`Biome::SnowyForest`].
    SnowyForest,
}

impl std::str::FromStr for FeatureKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "water" => Ok(Self::Water),
            "lava" => Ok(Self::Lava),
            "cave" | "cave_entrance" => Ok(Self::CaveEntrance),
            "river" | "river_channel" => Ok(Self::RiverChannel),
            "forest" => Ok(Self::Forest),
            "tropical" => Ok(Self::Tropical),
            "desert" => Ok(Self::Desert),
            "tundra" => Ok(Self::Tundra),
            "plains" => Ok(Self::Plains),
            "snowy_forest" => Ok(Self::SnowyForest),
            other => bail!(
                "unknown feature kind {:?}; valid: water, lava, cave, river, \
                 forest, tropical, desert, tundra, plains, snowy_forest",
                other
            ),
        }
    }
}

/// A located terrain feature, one entry in the result list from [`find`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct FeatureHit {
    /// What type of feature was found.
    pub kind: FeatureKind,
    /// World-space position [x, y, z]. For point features (cave entrance,
    /// lava pool) this is the top of the feature; for column-based features
    /// (biome, water) this is the surface height of the representative column.
    pub pos: [i32; 3],
    /// Fine region [x, z] that contained this feature.
    pub region: [i32; 2],
    /// Horizontal distance in blocks from the search origin to this hit.
    pub distance_blocks: f32,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Locate up to `count` occurrences of `kind` within `max_radius_blocks` of
/// `(origin_wx, origin_wz)`.
///
/// Scans concentric Chebyshev rings of fine regions (each 512 × 512 blocks)
/// outward from the origin's region until `count` hits are found or the
/// radius is exhausted. The worldgen LRU cache is the search index — regions
/// are materialised on first access and reused across calls.
///
/// Returns `Ok(hits)` sorted by ascending distance; `hits.len() ≤ count`.
/// Returns `Err` if no hits were found within the radius.
pub fn find(
    g: &Generator,
    origin_wx: i32,
    origin_wz: i32,
    kind: FeatureKind,
    max_radius_blocks: i32,
    count: usize,
) -> anyhow::Result<Vec<FeatureHit>> {
    let center = RegionCoord::containing(origin_wx, origin_wz);
    let max_ring = (max_radius_blocks + FINE_REGION_SIZE - 1) / FINE_REGION_SIZE;

    let mut hits: Vec<FeatureHit> = Vec::new();

    'search: for ring in 0..=max_ring {
        for (rdx, rdz) in chebyshev_ring(ring) {
            let coord = RegionCoord {
                x: center.x + rdx,
                z: center.z + rdz,
            };
            let fine = region::get_fine(&g.fine_cache, coord, || g.build_fine_region(coord));
            search_region(&fine, g, kind, origin_wx, origin_wz, max_radius_blocks, &mut hits);
            if hits.len() >= count {
                break 'search;
            }
        }
    }

    if hits.is_empty() {
        bail!(
            "no {:?} found within {} blocks of ({}, {})",
            kind,
            max_radius_blocks,
            origin_wx,
            origin_wz,
        );
    }

    hits.sort_by(|a, b| {
        a.distance_blocks
            .partial_cmp(&b.distance_blocks)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(count);
    Ok(hits)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Returns the (rdx, rdz) region-offset pairs on the Chebyshev shell at
/// `ring` distance. Ring 0 yields only (0, 0). Ring r > 0 yields the 8r
/// cells on the perimeter of the square at Chebyshev distance r.
fn chebyshev_ring(ring: i32) -> Vec<(i32, i32)> {
    if ring == 0 {
        return vec![(0, 0)];
    }
    let r = ring;
    // 4 sides of the square, corners shared between top/bottom rows
    let cap = (8 * r) as usize;
    let mut cells = Vec::with_capacity(cap);
    for dx in -r..=r {
        cells.push((dx, -r));
        cells.push((dx, r));
    }
    for dz in (-r + 1)..r {
        cells.push((-r, dz));
        cells.push((r, dz));
    }
    cells
}

/// Append any hits of `kind` in `region` that fall within `max_radius_blocks`
/// of the origin to `out`. Returns immediately once one biome/water hit is
/// found (to avoid flooding results with adjacent same-biome columns).
fn search_region(
    fine: &FineRegion,
    g: &Generator,
    kind: FeatureKind,
    origin_wx: i32,
    origin_wz: i32,
    max_radius_blocks: i32,
    out: &mut Vec<FeatureHit>,
) {
    let max_sq = (max_radius_blocks as f64) * (max_radius_blocks as f64);

    match kind {
        FeatureKind::Lava => {
            for pool in &fine.cave_pools {
                if pool.kind != FluidBodyKind::LavaPool {
                    continue;
                }
                let cx = pool.center.x as i32;
                let cz = pool.center.z as i32;
                let dist_sq = horiz_dist_sq(cx, cz, origin_wx, origin_wz);
                if dist_sq > max_sq {
                    continue;
                }
                out.push(FeatureHit {
                    kind,
                    pos: [cx, pool.surface_y, cz],
                    region: [fine.coord.x, fine.coord.z],
                    distance_blocks: dist_sq.sqrt() as f32,
                });
            }
        }

        FeatureKind::CaveEntrance => {
            for sys in &fine.cave_systems {
                for entrance in &sys.entrances {
                    let p = entrance.surface;
                    let dist_sq = horiz_dist_sq(p.x, p.z, origin_wx, origin_wz);
                    if dist_sq > max_sq {
                        continue;
                    }
                    out.push(FeatureHit {
                        kind,
                        pos: [p.x, p.y, p.z],
                        region: [fine.coord.x, fine.coord.z],
                        distance_blocks: dist_sq.sqrt() as f32,
                    });
                }
            }
        }

        FeatureKind::RiverChannel => {
            for seg in &fine.segments {
                let (fx, fz) = seg.from;
                let dist_sq = horiz_dist_sq(fx, fz, origin_wx, origin_wz);
                if dist_sq > max_sq {
                    continue;
                }
                out.push(FeatureHit {
                    kind,
                    pos: [fx, seg.water_y, fz],
                    region: [fine.coord.x, fine.coord.z],
                    distance_blocks: dist_sq.sqrt() as f32,
                });
            }
        }

        // Column-based searches: sample on a coarser stride (4 × FINE_CELL = 32 blocks)
        // to avoid spending time on 4096 column_data calls per region. This still provides
        // one sample point per 32 × 32 block area, enough to locate any biome patch.
        _ => {
            let stride = 4; // in fine cells; gives a 32-block sample spacing
            let (rx, rz) = fine.coord.origin();
            'col: for iz in (0..FINE_CELLS_PER_REGION).step_by(stride) {
                for ix in (0..FINE_CELLS_PER_REGION).step_by(stride) {
                    let wx = rx + ix * FINE_CELL;
                    let wz = rz + iz * FINE_CELL;
                    let dist_sq = horiz_dist_sq(wx, wz, origin_wx, origin_wz);
                    if dist_sq > max_sq {
                        continue;
                    }
                    let col = g.column_data(wx, wz);
                    let matches = match kind {
                        FeatureKind::Water => col.water_surface_y.is_some(),
                        FeatureKind::Forest => col.biome == Biome::Forest,
                        FeatureKind::Tropical => col.biome == Biome::Tropical,
                        FeatureKind::Desert => col.biome == Biome::Desert,
                        FeatureKind::Tundra => col.biome == Biome::Tundra,
                        FeatureKind::Plains => col.biome == Biome::Plains,
                        FeatureKind::SnowyForest => col.biome == Biome::SnowyForest,
                        // Handled by the outer match arms above
                        FeatureKind::Lava | FeatureKind::CaveEntrance | FeatureKind::RiverChannel => {
                            unreachable!()
                        }
                    };
                    if matches {
                        let y = col.water_surface_y.unwrap_or(col.height);
                        out.push(FeatureHit {
                            kind,
                            pos: [wx, y, wz],
                            region: [fine.coord.x, fine.coord.z],
                            distance_blocks: dist_sq.sqrt() as f32,
                        });
                        // One representative point per region is enough; the caller
                        // accumulates across regions to build the full result list.
                        break 'col;
                    }
                }
            }
        }
    }
}

#[inline]
fn horiz_dist_sq(ax: i32, az: i32, bx: i32, bz: i32) -> f64 {
    let dx = (ax - bx) as f64;
    let dz = (az - bz) as f64;
    dx * dx + dz * dz
}
