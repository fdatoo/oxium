//! Graph-based cave systems plus ambient noise carvers.
//!
//! Each 512×512-block fine region deterministically rolls 0–3 cave systems.
//! A system is a small graph of ellipsoidal chambers connected by spline
//! tunnels (minimum spanning tree + 1–2 loop edges), placed inside a
//! per-system bounding box at one of three depth bands:
//!
//! - **Shallow** (`CAVE_BAND_SHALLOW`): near surface, high entrance rate.
//! - **Middle** (`CAVE_BAND_MIDDLE`): mid-depth, moderate entrance rate.
//! - **Deep** (`CAVE_BAND_DEEP`): very deep, rare entrances.
//!
//! Each system is assigned a style ([`CaveStyle`]) that parametrises
//! chamber count, radii, and tunnel widths. Adjacent systems in different
//! bands may be linked by vertical connectors.
//!
//! ### Carving at chunk fill time
//!
//! Carving is deferred to chunk fill. For each system whose bounding box
//! intersects the chunk, signed SDF functions (`cave_sdf`, `trunks_sdf`,
//! `entrance_sdf`) return a positive intensity wherever a voxel lies inside
//! a chamber or tunnel. The fill loop subtracts these from the base density
//! via `smin` (smooth-min), so cave walls have soft, chamfered edges.
//!
//! The `CAVE_SURFACE_BUFFER` guard preserves the grass cap by refusing to
//! apply the graph SDFs within `CAVE_SURFACE_BUFFER` blocks of `h_pre`.
//! Entrance features (sinkholes, cliff mouths, skylights) bypass this guard
//! via a separate `entrance_sdf` gate so intentional cave openings can still
//! breach the surface.
//!
//! ### Ambient noise carvers
//!
//! Alongside the graph systems, two MC-derived ambient carvers fire on every
//! underground voxel:
//!
//! - **Cheese** ([`cheese_contribution`]): threshold-sampled 3D FBM produces
//!   Swiss-cheese-like isolated pockets. A `cave_layer²` stratification term
//!   concentrates carving at specific depth bands.
//! - **Terasology ambient** ([`terasology_ambient`]): two independent FBM
//!   channels are intersected; carving occurs where both are near zero —
//!   geometrically a disk in 2D noise space. Disk radius grows with depth.
//!
//! Both layers are sampled via a `CarverEvaluator` corner-lattice trilerp
//! (same 9³ pattern as `density_graph::CellEvaluator`) to avoid per-voxel
//! FBM cost.
//!
//! See `docs/superpowers/specs/2026-05-21-cave-system-overhaul-design.md`,
//! `docs/book/content/part-3-region-build/3.6-cave-systems.mdx`, and
//! `docs/book/content/part-4-chunk-fill/4.3-composing-caves.mdx`.
use crate::worldgen::aquifer::LAVA_BAND_TOP_Y;
use crate::worldgen::fluid::FluidBodyKind;
use crate::worldgen::hash::{mix_range, mix_u32, mix_unit};
use crate::worldgen::heightmap::HeightmapNoise;
use crate::worldgen::region::{
    CavePool, CaveSystem, Chamber, Entrance, EntranceKind, FineRegion, RegionCoord, Tunnel,
};
use crate::worldgen::tuning::*;
use glam::{IVec3, Vec3};
use noise::{Fbm, NoiseFn, Simplex};

/// Distinct cave-system personalities, rolled per system from the
/// region cell id and the system's depth band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaveStyle {
    /// Few large chambers, wide tunnels. Deep-band-biased.
    Cathedral,
    /// Many small chambers, narrow tunnels. Shallow-band-biased.
    Warren,
    /// XZ-stretched chambers, narrow vertical sheets. Mid-band-biased.
    Slot,
    /// Low-clustered chambers (flooded look). Deep-band-biased.
    Sump,
    /// Default — medium chambers, medium tunnels.
    Karst,
}

/// Depth band a system belongs to. Drives bounding-box Y placement
/// and entrance-roll probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthBand {
    Shallow,
    Middle,
    Deep,
}

impl DepthBand {
    fn pick(seed: u64, system_id: i32, region: RegionCoord) -> Self {
        // Three-way roll: 35% Shallow, 35% Middle, 30% Deep.
        let r = mix_unit(seed, &[region.x, region.z, system_id, 100]);
        if r < 0.35 {
            DepthBand::Shallow
        } else if r < 0.70 {
            DepthBand::Middle
        } else {
            DepthBand::Deep
        }
    }

    /// `[y_min, y_max]` for chamber placement (inclusive).
    fn range(self) -> (i32, i32) {
        match self {
            DepthBand::Shallow => CAVE_BAND_SHALLOW,
            DepthBand::Middle => CAVE_BAND_MIDDLE,
            DepthBand::Deep => CAVE_BAND_DEEP,
        }
    }

    fn entrance_prob(self) -> f32 {
        match self {
            DepthBand::Shallow => ENTRANCE_PROB_SHALLOW,
            DepthBand::Middle => ENTRANCE_PROB_MIDDLE,
            DepthBand::Deep => ENTRANCE_PROB_DEEP,
        }
    }
}

/// Roll a `CaveStyle` deterministically from `(seed, region_coord,
/// system_idx, band)`. Band-weighted via `CaveStyleTable`.
pub fn pick_style(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    band: DepthBand,
    cfg: &crate::worldgen::config::CaveConfig,
) -> CaveStyle {
    let u = mix_unit(seed, &[coord.x, coord.z, system_idx, 7000]);
    let weights = match band {
        DepthBand::Shallow => &cfg.style_table.style_weights_shallow,
        DepthBand::Middle => &cfg.style_table.style_weights_middle,
        DepthBand::Deep => &cfg.style_table.style_weights_deep,
    };
    let mut acc = 0.0;
    let styles = [
        CaveStyle::Cathedral,
        CaveStyle::Warren,
        CaveStyle::Slot,
        CaveStyle::Sump,
        CaveStyle::Karst,
    ];
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if u <= acc {
            return styles[i];
        }
    }
    CaveStyle::Karst
}

/// Build all cave systems for the given fine region. Each system is
/// deterministically derived from `(seed, coord, system_idx)`.
pub fn build_systems_for_region(
    seed: u64,
    coord: RegionCoord,
    heightmap: &HeightmapNoise,
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
    region: &mut FineRegion,
    cave_cfg: &crate::worldgen::config::CaveConfig,
) {
    // Decide how many systems this region hosts.
    // CAVE_SYSTEMS_PER_REGION acts as a compile-time safety cap;
    // cave_cfg.systems_per_region_max is the hot-reloadable config value.
    let n_min = CAVE_SYSTEMS_PER_REGION.0;
    let n_max = cave_cfg
        .systems_per_region_max
        .min(CAVE_SYSTEMS_PER_REGION.1);
    let n = n_min + (mix_u32(seed, &[coord.x, coord.z, 1]) % (n_max - n_min + 1));
    region.cave_systems.clear();
    for system_idx in 0..n as i32 {
        let sys = build_system(
            seed, coord, system_idx, heightmap, climate, density, cave_cfg,
        );
        region.cave_systems.push(sys);
    }
    build_vertical_connectors(seed, coord, cave_cfg, region);
    derive_cave_pools(seed, region);
}

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
fn derive_cave_pools(seed: u64, region: &mut FineRegion) {
    region.cave_pools.clear();
    // Work on indices to avoid borrow conflicts.
    let system_count = region.cave_systems.len();
    for si in 0..system_count {
        let chamber_count = region.cave_systems[si].chambers.len();
        for ci in 0..chamber_count {
            let ch = region.cave_systems[si].chambers[ci];
            let min_xz = ch.radii.x.min(ch.radii.z);
            if min_xz < POOL_MIN_RADIUS_XZ {
                continue;
            }
            let ceiling = (ch.center.y + ch.radii.y).ceil() as i32;
            if ceiling > SEA_LEVEL - POOL_TOP_CLEARANCE {
                continue;
            }
            let floor = (ch.center.y - ch.radii.y).floor() as i32;
            let height = ((ch.radii.y * 2.0) as i32).max(1);
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
                        99_001,
                    ],
                );
                roll < POOL_LAVA_PROB
            };
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

/// For each pair of systems in adjacent bands within `region`, roll
/// `vertical_connector_prob`. If passing, push a `Tunnel` from the upper
/// system's lowest chamber to the lower system's highest chamber.
pub fn build_vertical_connectors(
    seed: u64,
    coord: RegionCoord,
    cfg: &crate::worldgen::config::CaveConfig,
    region: &mut FineRegion,
) {
    if cfg.vertical_connector_prob <= 0.0 {
        return;
    }
    // Snapshot which band each system belongs to (avoids borrow conflict).
    let bands: Vec<DepthBand> = region
        .cave_systems
        .iter()
        .map(|s| {
            let cy = (s.bb_min.y + s.bb_max.y) / 2;
            if cy >= CAVE_BAND_SHALLOW.0 {
                DepthBand::Shallow
            } else if cy >= CAVE_BAND_MIDDLE.0 {
                DepthBand::Middle
            } else {
                DepthBand::Deep
            }
        })
        .collect();
    let band_idx = |b: DepthBand| -> u8 {
        match b {
            DepthBand::Shallow => 0,
            DepthBand::Middle => 1,
            DepthBand::Deep => 2,
        }
    };
    for i in 0..region.cave_systems.len() {
        for j in 0..region.cave_systems.len() {
            if i == j {
                continue;
            }
            // Only Shallow→Middle or Middle→Deep (i is upper, j is lower).
            if band_idx(bands[j]) != band_idx(bands[i]) + 1 {
                continue;
            }
            let u = mix_unit(seed, &[coord.x, coord.z, i as i32, j as i32, 9500]);
            if u > cfg.vertical_connector_prob {
                continue;
            }
            // Upper system's lowest chamber (minimum center.y).
            let upper_lowest = region.cave_systems[i]
                .chambers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.center.y.partial_cmp(&b.center.y).unwrap())
                .map(|(idx, _)| idx);
            // Lower system's highest chamber (maximum center.y).
            let lower_highest = region.cave_systems[j]
                .chambers
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.center.y.partial_cmp(&b.center.y).unwrap())
                .map(|(idx, _)| idx);
            let (Some(a_idx), Some(b_idx)) = (upper_lowest, lower_highest) else {
                continue;
            };
            let pa = region.cave_systems[i].chambers[a_idx].center;
            let pb = region.cave_systems[j].chambers[b_idx].center;
            let connector = Tunnel {
                control_points: vec![pa, pb],
                radius: cfg.vertical_connector_r,
            };
            region.cave_systems[i].vertical_connectors.push(connector);
        }
    }
}

/// For each cave system in the centre region of `regions`, roll
/// `trunk_prob`. If passing, link to the nearest chamber-0 in any 8-
/// neighbour region. The trunk runs from the system's chamber 0 through
/// a 40-block perpendicular-offset midpoint to the other system's
/// chamber 0.
///
/// Architecture note: trunks are cross-region features — a single
/// `FineRegion` build cannot see neighbour regions (they may not exist
/// yet in the cache). `build_trunks` takes an owned `[(RegionCoord,
/// FineRegion)]` slice that covers all regions of interest and is
/// called:
///   * from tests (on owned data — `trunk` field verified directly), and
///   * from `cave_sdf` / `cave_air` on-the-fly (trunk geometry
///     recomputed from the flat `&[&CaveSystem]` slice already
///     aggregated across the 3×3 region halo).
///
/// Salt 9000 = trunk probability roll; 9001 = midpoint offset roll.
pub fn build_trunks(
    seed: u64,
    cfg: &crate::worldgen::config::CaveConfig,
    regions: &mut [(RegionCoord, FineRegion)],
) {
    if cfg.trunk_prob <= 0.0 {
        return;
    }
    // Snapshot (region_index, system_index, coord, chamber-0 center)
    // for lookup — avoids borrow conflict when writing .trunk later.
    let snap: Vec<(usize, usize, RegionCoord, glam::Vec3)> = regions
        .iter()
        .enumerate()
        .flat_map(|(ri, (coord, region))| {
            region
                .cave_systems
                .iter()
                .enumerate()
                .filter_map(|(si, sys)| sys.chambers.first().map(|c| (ri, si, *coord, c.center)))
                .collect::<Vec<_>>()
        })
        .collect();

    // For each system, find the nearest chamber-0 in an 8-neighbour region.
    for &(ri, si, coord, my_center) in &snap {
        // Stable salt: chamber-0 center coords (integer-rounded). Stable
        // across chunks because chamber centers are immutable; unique per
        // system within a region because Poisson sampling enforces
        // min_spacing > 0.
        let salt_x = my_center.x as i32;
        let salt_y = my_center.y as i32;
        let salt_z = my_center.z as i32;
        let u = mix_unit(seed, &[coord.x, coord.z, salt_x, salt_y, salt_z, 9000]);
        if u > cfg.trunk_prob {
            continue;
        }
        // Find nearest neighbour-region chamber-0 (different region index,
        // within 1 step in both x and z).
        let mut best: Option<(glam::Vec3, f32)> = None;
        for &(ori, _osi, other_coord, other_center) in &snap {
            if ori == ri {
                continue;
            }
            let dx = (other_coord.x - coord.x).abs();
            let dz = (other_coord.z - coord.z).abs();
            if dx > 1 || dz > 1 {
                continue;
            }
            let d = (other_center - my_center).length();
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((other_center, d));
            }
        }
        let Some((other_center, _)) = best else {
            continue;
        };
        // Mid-arc: midpoint with a 40-block perpendicular offset in XZ.
        let axis = other_center - my_center;
        let len = (axis.x * axis.x + axis.z * axis.z).sqrt().max(1.0);
        let perp = glam::Vec3::new(-axis.z / len, 0.0, axis.x / len);
        let o =
            (mix_unit(seed, &[coord.x, coord.z, salt_x, salt_y, salt_z, 9001]) * 2.0 - 1.0) * 40.0;
        let mid = glam::Vec3::new(
            (my_center.x + other_center.x) * 0.5 + perp.x * o,
            (my_center.y + other_center.y) * 0.5,
            (my_center.z + other_center.z) * 0.5 + perp.z * o,
        );
        regions[ri].1.cave_systems[si].trunk = Some(Tunnel {
            control_points: vec![my_center, mid, other_center],
            radius: cfg.trunk_r,
        });
    }
}

/// Per-style parameter set extracted from the style table.
struct StyleParams {
    chamber_count: (u32, u32),
    r_xz: (f32, f32),
    r_y: (f32, f32),
    tunnel_r: (f32, f32),
}

fn style_params(style: CaveStyle, table: &crate::worldgen::config::CaveStyleTable) -> StyleParams {
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

/// Build one cave system inside region `coord`.
///
/// ### Chamber placement (Poisson-disk rejection sampling)
///
/// Candidate chamber centers are drawn uniformly at random inside the
/// system bounding box. A candidate is rejected if it lies within
/// `mean_radius × POISSON_MIN_SPACING_MULT` of any already-placed
/// chamber. This minimum-distance constraint prevents rooms from
/// overlapping or crowding into impenetrable clusters. Each attempt
/// increments an independent counter; after `max_tries` attempts the
/// sampling stops — if a target chamber count can't be reached, the
/// system uses however many fit, down to a minimum of 1.
///
/// ### Tunnel graph (Kruskal's MST)
///
/// After all chambers are placed, every pair of chambers becomes a
/// candidate edge weighted by their 3D Euclidean distance. Kruskal's
/// minimum spanning tree algorithm selects the subset of edges that
/// connects all chambers with the smallest total tunnel length. Path-
/// compression union-find gives near-O(α(n)) per edge, amortised over
/// the whole system. Additionally `MST_EXTRA_LOOPS.0..=1` short non-MST
/// edges are added back to introduce cycles — without loops the cave
/// graph is a tree and every room has exactly one entrance/exit, which
/// feels unnatural.
fn build_system(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    heightmap: &HeightmapNoise,
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
    cave_cfg: &crate::worldgen::config::CaveConfig,
) -> CaveSystem {
    let band = DepthBand::pick(seed, system_idx, coord);
    let (y_min, y_max) = band.range();

    // Roll the style for this system.
    let style = pick_style(seed, coord, system_idx, band, cave_cfg);
    let sp = style_params(style, &cave_cfg.style_table);

    // Bounding-box footprint inside the region. The box is allowed to
    // straddle the region boundary — neighbouring regions consult our
    // systems via the 3 × 3 region neighbourhood at chunk fill time.
    let bb_size = IVec3::new(220, (y_max - y_min).min(64), 220);
    let region_origin = coord.origin();
    let bb_origin_x = region_origin.0
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, 10])
            % (FINE_REGION_SIZE - bb_size.x).max(1) as u32) as i32
        - (bb_size.x / 2);
    let bb_origin_z = region_origin.1
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, 11])
            % (FINE_REGION_SIZE - bb_size.z).max(1) as u32) as i32
        - (bb_size.z / 2);
    let bb_origin_y = y_min;
    let bb_min = IVec3::new(bb_origin_x, bb_origin_y, bb_origin_z);
    let bb_max = bb_min + bb_size;

    // Chamber count — from style table.
    let (cn_min, cn_max) = sp.chamber_count;
    let chamber_count =
        cn_min + (mix_u32(seed, &[coord.x, coord.z, system_idx, 20]) % (cn_max - cn_min + 1));

    // Pre-compute Sump bb_center_y / bb_half_y for the bias formula.
    let bb_center_y = (bb_min.y + bb_max.y) as f32 * 0.5;
    let bb_half_y = (bb_max.y - bb_min.y) as f32 * 0.5;

    // Poisson-disk-like rejection sampling for chamber centers.
    let mut chambers: Vec<Chamber> = Vec::with_capacity(chamber_count as usize);
    let mut tries = 0u32;
    let max_tries = 200u32;
    let mut attempt = 0i32;
    while chambers.len() < chamber_count as usize && tries < max_tries {
        let sx = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 30, attempt],
            0.0,
            (bb_max.x - bb_min.x) as f32,
        );
        let sy = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 31, attempt],
            0.0,
            (bb_max.y - bb_min.y) as f32,
        );
        let sz = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 32, attempt],
            0.0,
            (bb_max.z - bb_min.z) as f32,
        );

        // Depth multiplier: deeper = larger chambers.
        let cy_raw = bb_min.y as f32 + sy;
        let depth_mult = 1.0 + cave_cfg.depth_scale * ((40.0 - cy_raw).max(0.0) / 80.0);

        let rx = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 40, attempt],
            sp.r_xz.0,
            sp.r_xz.1,
        ) * depth_mult;
        let ry = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 41, attempt],
            sp.r_y.0,
            sp.r_y.1,
        ) * depth_mult;
        let rz = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 42, attempt],
            sp.r_xz.0,
            sp.r_xz.1,
        ) * depth_mult;

        // Slot deliberately drops rz and derives both XZ radii from rx for
        // the elongated-in-X look. Other styles use independent rx, rz.
        let (rx_final, rz_final) = if style == CaveStyle::Slot {
            (rx * 1.4, rx * 0.6)
        } else {
            (rx, rz)
        };

        // Sump style: bias chambers low in the bounding box (floor cluster).
        let cy_final = if style == CaveStyle::Sump {
            bb_center_y - bb_half_y * 0.4 + (cy_raw - bb_center_y).abs() * 0.5
        } else {
            cy_raw
        };

        let center = Vec3::new(bb_min.x as f32 + sx, cy_final, bb_min.z as f32 + sz);
        let radii = Vec3::new(rx_final, ry, rz_final);
        let mean_r = (rx_final + ry + rz_final) / 3.0;
        let min_spacing = mean_r * POISSON_MIN_SPACING_MULT;
        // Reject if too close to any existing chamber.
        let too_close = chambers.iter().any(|c| {
            let d = c.center - center;
            d.length() < min_spacing
        });
        attempt += 1;
        tries += 1;
        if too_close {
            continue;
        }
        chambers.push(Chamber { center, radii });
    }

    // MST via Kruskal.
    let mut tunnels: Vec<Tunnel> = Vec::new();
    if chambers.len() >= 2 {
        let mut edges: Vec<(f32, usize, usize)> = Vec::new();
        for i in 0..chambers.len() {
            for j in (i + 1)..chambers.len() {
                let d = (chambers[i].center - chambers[j].center).length();
                edges.push((d, i, j));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut parent: Vec<usize> = (0..chambers.len()).collect();
        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        let mut mst_edges: Vec<(usize, usize)> = Vec::new();
        for &(_, a, b) in &edges {
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            if ra != rb {
                parent[ra] = rb;
                mst_edges.push((a, b));
            }
        }
        // Optional loop edges: pick the shortest edges not already in
        // the MST.
        let extra_loop_count = MST_EXTRA_LOOPS.0
            + (mix_u32(seed, &[coord.x, coord.z, system_idx, 50])
                % (MST_EXTRA_LOOPS.1 - MST_EXTRA_LOOPS.0 + 1));
        let mut added_extras = 0u32;
        for &(_, a, b) in &edges {
            if added_extras >= extra_loop_count {
                break;
            }
            if mst_edges
                .iter()
                .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
            {
                continue;
            }
            mst_edges.push((a, b));
            added_extras += 1;
        }
        // Build a tunnel for each edge — radius from per-style range.
        for (idx, &(a, b)) in mst_edges.iter().enumerate() {
            let radius = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 60, idx as i32],
                sp.tunnel_r.0,
                sp.tunnel_r.1,
            );
            let pa = chambers[a].center;
            let pb = chambers[b].center;
            // Catmull-Rom-friendly control polyline: pa, mid +
            // domain-warped perpendicular offset 1, mid +
            // perpendicular offset 2, pb. Offsets break the otherwise
            // straight line into a meander; the 4-point polyline is
            // sampled later by the SDF as N straight capsules.
            let axis = (pb - pa).normalize_or_zero();
            // Two arbitrary orthogonal axes for perpendicular warp.
            let world_up = Vec3::Y;
            let perp1 = axis.cross(world_up).try_normalize().unwrap_or(Vec3::X);
            let perp2 = axis.cross(perp1).try_normalize().unwrap_or(Vec3::Z);
            let amp = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 61, idx as i32],
                radius * 1.0,
                radius * 1.5,
            ) * 4.0;
            let off1_u = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 62, idx as i32],
                -amp,
                amp,
            );
            let off1_v = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 63, idx as i32],
                -amp,
                amp,
            );
            let off2_u = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 64, idx as i32],
                -amp,
                amp,
            );
            let off2_v = mix_range(
                seed,
                &[coord.x, coord.z, system_idx, 65, idx as i32],
                -amp,
                amp,
            );
            let p1 = pa.lerp(pb, 0.33) + perp1 * off1_u + perp2 * off1_v;
            let p2 = pa.lerp(pb, 0.66) + perp1 * off2_u + perp2 * off2_v;
            tunnels.push(Tunnel {
                control_points: vec![pa, p1, p2, pb],
                radius,
            });
        }
    }

    // Entrance rolls.
    let mut entrances: Vec<Entrance> = Vec::new();
    for (ci, chamber) in chambers.iter().enumerate() {
        let try_roll = mix_unit(seed, &[coord.x, coord.z, system_idx, 70, ci as i32]);
        if try_roll >= band.entrance_prob() {
            continue;
        }
        let cwx = chamber.center.x as i32;
        let cwy_top = (chamber.center.y + chamber.radii.y) as i32;
        let cwz = chamber.center.z as i32;
        let surface_h =
            heightmap.h_pre(seed, chamber.center.x, chamber.center.z, climate, density) as i32;
        // 1. Sinkhole.
        if surface_h - cwy_top <= SINKHOLE_DEPTH_MAX && surface_h - cwy_top >= -2 {
            // Extend the shaft top by SURFACE_BAND so it carves through any
            // 3D-density bumps above h_pre — otherwise those bumps become
            // floating terrain islands above the entrance opening.
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::Sinkhole,
                surface: IVec3::new(cwx, surface_h + SURFACE_BAND as i32, cwz),
            });
            continue;
        }
        // 2. Cliff mouth: look for a steep-slope column within
        //    CLIFF_ENTRANCE_DIST. Sample a few directions.
        let mut found_cliff: Option<IVec3> = None;
        for step in 0..16 {
            let theta = step as f32 * std::f32::consts::TAU / 16.0;
            let cwx_f = chamber.center.x + theta.cos() * CLIFF_ENTRANCE_DIST as f32;
            let cwz_f = chamber.center.z + theta.sin() * CLIFF_ENTRANCE_DIST as f32;
            if heightmap.is_cliff(seed, cwx_f, cwz_f, climate, density) {
                found_cliff = Some(IVec3::new(
                    cwx_f as i32,
                    heightmap.h_pre(seed, cwx_f, cwz_f, climate, density) as i32,
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
        // 3. Skylight: chamber 30–60 below surface.
        let dy = surface_h - cwy_top;
        if (30..=60).contains(&dy) {
            entrances.push(Entrance {
                chamber_idx: ci as u32,
                kind: EntranceKind::Skylight,
                surface: IVec3::new(cwx, surface_h + SURFACE_BAND as i32, cwz),
            });
        }
    }

    // Compute the system's actual bounding box including chambers,
    // tunnels, and any entrance shafts so the chunk-side AABB
    // intersection test catches everything we might carve.
    let mut bb_min = bb_min;
    let mut bb_max = bb_max;
    for c in &chambers {
        let cmin = (c.center - c.radii - Vec3::splat(1.0)).floor();
        let cmax = (c.center + c.radii + Vec3::splat(1.0)).ceil();
        bb_min.x = bb_min.x.min(cmin.x as i32);
        bb_min.y = bb_min.y.min(cmin.y as i32);
        bb_min.z = bb_min.z.min(cmin.z as i32);
        bb_max.x = bb_max.x.max(cmax.x as i32);
        bb_max.y = bb_max.y.max(cmax.y as i32);
        bb_max.z = bb_max.z.max(cmax.z as i32);
    }
    for t in &tunnels {
        for p in &t.control_points {
            bb_min.x = bb_min.x.min((p.x - t.radius - 1.0) as i32);
            bb_min.y = bb_min.y.min((p.y - t.radius - 1.0) as i32);
            bb_min.z = bb_min.z.min((p.z - t.radius - 1.0) as i32);
            bb_max.x = bb_max.x.max((p.x + t.radius + 1.0) as i32);
            bb_max.y = bb_max.y.max((p.y + t.radius + 1.0) as i32);
            bb_max.z = bb_max.z.max((p.z + t.radius + 1.0) as i32);
        }
    }
    for e in &entrances {
        // Vertical shaft extent — clip downward to chamber, upward to
        // surface.
        bb_min.y = bb_min.y.min(e.surface.y);
        bb_max.y = bb_max.y.max(e.surface.y + 1);
        bb_min.x = bb_min.x.min(e.surface.x - 4);
        bb_max.x = bb_max.x.max(e.surface.x + 4);
        bb_min.z = bb_min.z.min(e.surface.z - 4);
        bb_max.z = bb_max.z.max(e.surface.z + 4);
    }

    CaveSystem {
        bb_min,
        bb_max,
        chambers,
        tunnels,
        entrances,
        style,
        trunk: None,
        vertical_connectors: vec![],
    }
}

// ── Per-chunk carve ──────────────────────────────────────────────────

/// Soft SDF: positive inside chambers / tunnels, 0 outside.
/// Peak `CAVE_SDF_INTENSITY` deep inside; tapers smoothly to 0 at
/// the strict geometric boundary. The density-based fill chunk
/// subtracts this from the per-voxel density so cave walls have
/// soft, chamfered edges instead of the pixel-sharp ellipsoid /
/// capsule boundaries.
pub fn cave_sdf(wx: i32, wy: i32, wz: i32, systems: &[&CaveSystem]) -> f32 {
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    let mut max_sdf: f32 = 0.0;
    for sys in systems {
        if !system_bb_contains(sys, wx, wy, wz) {
            continue;
        }
        // Chamber ellipsoid: normalised squared distance. <= 1 inside,
        // > 1 outside. Soft falloff over the [0, 1] range.
        for c in &sys.chambers {
            let d = p - c.center;
            let ratio =
                (d.x / c.radii.x).powi(2) + (d.y / c.radii.y).powi(2) + (d.z / c.radii.z).powi(2);
            if ratio <= 1.0 {
                // 1.0 at center → 0.0 at boundary.
                let sdf = (1.0 - ratio) * CAVE_SDF_INTENSITY;
                if sdf > max_sdf {
                    max_sdf = sdf;
                }
            }
        }
        // Tunnel capsule along control polyline.
        for t in &sys.tunnels {
            if t.control_points.len() < 2 {
                continue;
            }
            for i in 0..t.control_points.len() - 1 {
                let a = t.control_points[i];
                let b = t.control_points[i + 1];
                let ab = b - a;
                let len_sq = ab.length_squared();
                if len_sq < 1e-6 {
                    continue;
                }
                let t_param = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                let dist = (p - closest).length();
                if dist <= t.radius {
                    let sdf = (1.0 - dist / t.radius) * CAVE_SDF_INTENSITY;
                    if sdf > max_sdf {
                        max_sdf = sdf;
                    }
                }
            }
        }
        // Vertical connectors between adjacent-band systems in the same
        // region. Same capsule SDF math as tunnels.
        for t in &sys.vertical_connectors {
            if t.control_points.len() < 2 {
                continue;
            }
            for i in 0..t.control_points.len() - 1 {
                let a = t.control_points[i];
                let b = t.control_points[i + 1];
                let ab = b - a;
                let len_sq = ab.length_squared();
                if len_sq < 1e-6 {
                    continue;
                }
                let t_param = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                let dist = (p - closest).length();
                if dist <= t.radius {
                    let sdf = (1.0 - dist / t.radius) * CAVE_SDF_INTENSITY;
                    if sdf > max_sdf {
                        max_sdf = sdf;
                    }
                }
            }
        }
        // Cross-region trunk (populated by build_trunks; None in the
        // production cache path, Some in test-built regions).
        if let Some(trunk) = &sys.trunk {
            for i in 0..trunk.control_points.len().saturating_sub(1) {
                let a = trunk.control_points[i];
                let b = trunk.control_points[i + 1];
                let ab = b - a;
                let len_sq = ab.length_squared();
                if len_sq < 1e-6 {
                    continue;
                }
                let t_param = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                let dist = (p - closest).length();
                if dist <= trunk.radius {
                    let sdf = (1.0 - dist / trunk.radius) * CAVE_SDF_INTENSITY;
                    if sdf > max_sdf {
                        max_sdf = sdf;
                    }
                }
            }
        }
    }
    max_sdf
}

/// Compute the cross-region trunk SDF on the fly from a flat `systems`
/// slice spanning multiple regions. Called at chunk-fill time where
/// `sys.trunk` is always `None` (regions are immutable in the Arc
/// cache). Returns the maximum trunk SDF value (positive inside,
/// 0 outside).
///
/// For each system, derives its region coord from its chamber-0
/// world position, finds the nearest chamber-0 in a different region,
/// and evaluates the deterministic 3-point capsule trunk.
pub fn trunks_sdf(
    wx: i32,
    wy: i32,
    wz: i32,
    systems: &[&CaveSystem],
    seed: u64,
    trunk_r: f32,
    trunk_prob: f32,
) -> f32 {
    if trunk_prob <= 0.0 || systems.len() < 2 {
        return 0.0;
    }
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    let mut max_sdf = 0.0_f32;

    // Snapshot chamber-0 center + derived region coord for each system.
    let snap: Vec<(RegionCoord, Vec3)> = systems
        .iter()
        .filter_map(|s| {
            s.chambers.first().map(|c| {
                (
                    RegionCoord::containing(c.center.x as i32, c.center.z as i32),
                    c.center,
                )
            })
        })
        .collect();

    for (si, sys) in systems.iter().enumerate() {
        let Some(c0) = sys.chambers.first() else {
            continue;
        };
        let my_coord = RegionCoord::containing(c0.center.x as i32, c0.center.z as i32);
        let my_center = c0.center;

        // Stable salt: chamber-0 center coords (integer-rounded). Stable
        // across chunks because chamber centers are immutable; unique per
        // system within a region because Poisson sampling enforces
        // min_spacing > 0.
        let salt_x = c0.center.x as i32;
        let salt_y = c0.center.y as i32;
        let salt_z = c0.center.z as i32;

        let u = mix_unit(
            seed,
            &[my_coord.x, my_coord.z, salt_x, salt_y, salt_z, 9000],
        );
        if u > trunk_prob {
            continue;
        }

        // Find nearest chamber-0 in a neighbouring region (different
        // region, within 1 step in both x and z).
        let mut best_center: Option<Vec3> = None;
        let mut best_dist = f32::MAX;
        for (j, (other_coord, other_center)) in snap.iter().enumerate() {
            if j == si {
                continue;
            }
            let dx = (other_coord.x - my_coord.x).abs();
            let dz = (other_coord.z - my_coord.z).abs();
            if dx > 1 || dz > 1 {
                continue;
            }
            // Require different region (not same region coord).
            if *other_coord == my_coord {
                continue;
            }
            let d = (*other_center - my_center).length();
            if d < best_dist {
                best_dist = d;
                best_center = Some(*other_center);
            }
        }
        let Some(other_center) = best_center else {
            continue;
        };

        // Build trunk geometry (mirrors build_trunks).
        let axis = other_center - my_center;
        let len = (axis.x * axis.x + axis.z * axis.z).sqrt().max(1.0);
        let perp = Vec3::new(-axis.z / len, 0.0, axis.x / len);
        let o = (mix_unit(
            seed,
            &[my_coord.x, my_coord.z, salt_x, salt_y, salt_z, 9001],
        ) * 2.0
            - 1.0)
            * 40.0;
        let mid = Vec3::new(
            (my_center.x + other_center.x) * 0.5 + perp.x * o,
            (my_center.y + other_center.y) * 0.5,
            (my_center.z + other_center.z) * 0.5 + perp.z * o,
        );
        let control = [my_center, mid, other_center];

        // Capsule SDF along the 3-point polyline.
        for i in 0..2 {
            let a = control[i];
            let b = control[i + 1];
            let ab = b - a;
            let len_sq = ab.length_squared();
            if len_sq < 1e-6 {
                continue;
            }
            let t_param = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
            let closest = a + ab * t_param;
            let dist = (p - closest).length();
            if dist <= trunk_r {
                let sdf = (1.0 - dist / trunk_r) * CAVE_SDF_INTENSITY;
                if sdf > max_sdf {
                    max_sdf = sdf;
                }
            }
        }
    }
    max_sdf
}

/// Soft SDF for entrance features (sinkholes, cliff mouths,
/// skylights). Always punches through regardless of surface buffer;
/// peak intensity twice the regular cave SDF so entrances reliably
/// breach the density even right at the surface.
pub fn entrance_sdf(wx: i32, wy: i32, wz: i32, systems: &[&CaveSystem]) -> f32 {
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    let mut max_sdf: f32 = 0.0;
    for sys in systems {
        if !system_bb_contains(sys, wx, wy, wz) {
            continue;
        }
        for e in &sys.entrances {
            let ch = match sys.chambers.get(e.chamber_idx as usize) {
                Some(c) => c,
                None => continue,
            };
            let intensity = CAVE_SDF_INTENSITY * 2.0;
            match e.kind {
                EntranceKind::Sinkhole => {
                    let chamber_top_y = ch.center.y + ch.radii.y;
                    if wy as f32 >= chamber_top_y - 1.0 && wy as f32 <= e.surface.y as f32 {
                        let dx = p.x - e.surface.x as f32;
                        let dz = p.z - e.surface.z as f32;
                        let depth = (e.surface.y as f32 - wy as f32).max(0.0);
                        let max_depth = (e.surface.y as f32 - chamber_top_y).max(1.0);
                        let r = 2.0 + (depth / max_depth) * 1.0;
                        let d = (dx * dx + dz * dz).sqrt();
                        if d <= r {
                            let sdf = (1.0 - d / r) * intensity;
                            if sdf > max_sdf {
                                max_sdf = sdf;
                            }
                        }
                    }
                }
                EntranceKind::Skylight => {
                    let chamber_top_y = ch.center.y + ch.radii.y;
                    if wy as f32 >= chamber_top_y - 1.0 && wy as f32 <= e.surface.y as f32 {
                        let dx = p.x - e.surface.x as f32;
                        let dz = p.z - e.surface.z as f32;
                        let d = (dx * dx + dz * dz).sqrt();
                        if d <= 1.5 {
                            let sdf = (1.0 - d / 1.5) * intensity;
                            if sdf > max_sdf {
                                max_sdf = sdf;
                            }
                        }
                    }
                }
                EntranceKind::CliffMouth => {
                    let chamber_p = ch.center;
                    let cliff_p = Vec3::new(e.surface.x as f32, chamber_p.y, e.surface.z as f32);
                    let ab = cliff_p - chamber_p;
                    let len_sq = ab.length_squared();
                    if len_sq < 1e-6 {
                        continue;
                    }
                    let t_param = ((p - chamber_p).dot(ab) / len_sq).clamp(0.0, 1.0);
                    let closest = chamber_p + ab * t_param;
                    let d = (p - closest).length();
                    if d <= 2.5 {
                        let sdf = (1.0 - d / 2.5) * intensity;
                        if sdf > max_sdf {
                            max_sdf = sdf;
                        }
                    }
                }
            }
        }
    }
    max_sdf
}

/// Does the cell at `(wx, wy, wz)` lie inside any chamber or tunnel
/// of any cave system whose bounding box covers it? Surface buffer is
/// enforced *outside* this function; this only answers the geometric
/// question.
pub fn cave_air(wx: i32, wy: i32, wz: i32, systems: &[&CaveSystem]) -> bool {
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    for sys in systems {
        if !system_bb_contains(sys, wx, wy, wz) {
            continue;
        }
        // Ellipsoid SDF: (dx/rx)² + (dy/ry)² + (dz/rz)² <= 1.
        for c in &sys.chambers {
            let d = p - c.center;
            let ratio =
                (d.x / c.radii.x).powi(2) + (d.y / c.radii.y).powi(2) + (d.z / c.radii.z).powi(2);
            if ratio <= 1.0 {
                return true;
            }
        }
        // Tunnel: distance to the polyline (sampling the 4-point
        // control polyline as 3 straight capsules — a piecewise
        // approximation of the Catmull-Rom).
        for t in &sys.tunnels {
            if t.control_points.len() < 2 {
                continue;
            }
            for i in 0..t.control_points.len() - 1 {
                let a = t.control_points[i];
                let b = t.control_points[i + 1];
                let ab = b - a;
                let t_param = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                if (p - closest).length() <= t.radius {
                    return true;
                }
            }
        }
        // Vertical connectors between adjacent-band systems in the same
        // region. Same capsule SDF math as tunnels.
        for t in &sys.vertical_connectors {
            if t.control_points.len() < 2 {
                continue;
            }
            for i in 0..t.control_points.len() - 1 {
                let a = t.control_points[i];
                let b = t.control_points[i + 1];
                let ab = b - a;
                let t_param = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                if (p - closest).length() <= t.radius {
                    return true;
                }
            }
        }
        // Cross-region trunk (populated by build_trunks; None in the
        // production cache path, Some in test-built regions).
        if let Some(trunk) = &sys.trunk {
            for i in 0..trunk.control_points.len().saturating_sub(1) {
                let a = trunk.control_points[i];
                let b = trunk.control_points[i + 1];
                let ab = b - a;
                let t_param = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                let closest = a + ab * t_param;
                if (p - closest).length() <= trunk.radius {
                    return true;
                }
            }
        }
    }
    false
}

/// Does `(wx, wy, wz)` lie inside one of the system's entrance
/// features (sinkhole shaft, skylight, cliff mouth)? Entrances
/// override the surface buffer.
pub fn entrance_air(wx: i32, wy: i32, wz: i32, systems: &[&CaveSystem]) -> bool {
    let p = Vec3::new(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5);
    for sys in systems {
        if !system_bb_contains(sys, wx, wy, wz) {
            continue;
        }
        for e in &sys.entrances {
            let ch = match sys.chambers.get(e.chamber_idx as usize) {
                Some(c) => c,
                None => continue,
            };
            match e.kind {
                EntranceKind::Sinkhole => {
                    // Vertical shaft from chamber top up to surface,
                    // radius 2.5, slightly funnel-shaped at top.
                    let chamber_top_y = ch.center.y + ch.radii.y;
                    if wy as f32 >= chamber_top_y - 1.0 && wy as f32 <= e.surface.y as f32 {
                        let dx = p.x - e.surface.x as f32;
                        let dz = p.z - e.surface.z as f32;
                        // Funnel: radius grows from 2 at chamber top to
                        // 3 at surface.
                        let depth = (e.surface.y as f32 - wy as f32).max(0.0);
                        let max_depth = (e.surface.y as f32 - chamber_top_y).max(1.0);
                        let r = 2.0 + (depth / max_depth) * 1.0;
                        if (dx * dx + dz * dz).sqrt() <= r {
                            return true;
                        }
                    }
                }
                EntranceKind::Skylight => {
                    // Narrow 1.5-block radius shaft from chamber top
                    // to surface.
                    let chamber_top_y = ch.center.y + ch.radii.y;
                    if wy as f32 >= chamber_top_y - 1.0 && wy as f32 <= e.surface.y as f32 {
                        let dx = p.x - e.surface.x as f32;
                        let dz = p.z - e.surface.z as f32;
                        if (dx * dx + dz * dz).sqrt() <= 1.5 {
                            return true;
                        }
                    }
                }
                EntranceKind::CliffMouth => {
                    // Horizontal tunnel from chamber to the cliff
                    // anchor. Capsule with radius 2.5.
                    let chamber_p = ch.center;
                    let cliff_p = Vec3::new(e.surface.x as f32, chamber_p.y, e.surface.z as f32);
                    let ab = cliff_p - chamber_p;
                    let len_sq = ab.length_squared();
                    if len_sq < 1e-6 {
                        continue;
                    }
                    let t_param = ((p - chamber_p).dot(ab) / len_sq).clamp(0.0, 1.0);
                    let closest = chamber_p + ab * t_param;
                    if (p - closest).length() <= 2.5 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Is `wy` inside any cave system's "carved range"? Used as a quick
/// rejection — only check the expensive ellipsoid / capsule SDFs if
/// at least one system's Y extent covers the column.
pub fn any_system_y_in_range(wy: i32, systems: &[&CaveSystem]) -> bool {
    systems.iter().any(|s| wy >= s.bb_min.y && wy <= s.bb_max.y)
}

#[inline]
fn system_bb_contains(sys: &CaveSystem, wx: i32, wy: i32, wz: i32) -> bool {
    wx >= sys.bb_min.x
        && wx <= sys.bb_max.x
        && wy >= sys.bb_min.y
        && wy <= sys.bb_max.y
        && wz >= sys.bb_min.z
        && wz <= sys.bb_max.z
}

// ── Noise carver layers ──────────────────────────────────────────────
//
// Cheese / pillars: MC-style ambient noise-based cave density.
// Live alongside the graph cave systems.
// The contributions wire into fill_chunk's `cave_contribution`
// composition (cheese subtracts from density; pillars add back).

use crate::worldgen::config::CaveConfig;
use crate::worldgen::noise_channel::build_channel;

/// All MC-derived noise channels needed for the cheese and pillar
/// carvers. Built once per Generator.
pub struct NoiseCarvers {
    // Cheese.
    pub cheese: Fbm<Simplex>,
    /// `cave_layer` — the regional gating noise added to cheese as
    /// `intensity * layer²`. See [`CaveConfig::cave_layer`].
    pub cave_layer: Fbm<Simplex>,
    // Pillars.
    pub pillar: Fbm<Simplex>,
    pub pillar_rareness: Fbm<Simplex>,
    pub pillar_thickness: Fbm<Simplex>,
    // Terasology ambient.
    pub tera_a: Fbm<Simplex>,
    pub tera_b: Fbm<Simplex>,
}

impl NoiseCarvers {
    /// Build all channels from their config descriptors. Each
    /// channel uses a different seed-salt so they're uncorrelated.
    pub fn new(seed: u64, cfg: &CaveConfig) -> Self {
        Self {
            cheese: build_channel(&cfg.cheese, seed, 1001),
            cave_layer: build_channel(&cfg.cave_layer, seed, 1011),
            pillar: build_channel(&cfg.pillar, seed, 1006),
            pillar_rareness: build_channel(&cfg.pillar_rareness, seed, 1007),
            pillar_thickness: build_channel(&cfg.pillar_thickness, seed, 1008),
            tera_a: build_channel(&cfg.tera_a, seed, 2000),
            tera_b: build_channel(&cfg.tera_b, seed, 2001),
        }
    }
}

/// MC-style cheese cave contribution.
///
/// Where the signed FBM noise is negative enough (below `cheese_offset`
/// as a threshold), the cheese term goes negative, carving a hole. The
/// "cheese" metaphor: if you sample a random 3D FBM and threshold it at
/// zero, you get a Swiss-cheese-like collection of blobs where the field
/// dips below the threshold, each blob being an isolated pocket of air.
///
/// The `cave_layer² × intensity` term adds horizontal stratification:
/// `cave_layer` is a low-frequency noise that controls which horizontal
/// strata are rich in caves. Where `cave_layer ≈ 0`, the `layer²` term
/// is near-zero so the raw cheese signal dominates and carves freely.
/// Where `|cave_layer|` is large, the `layer²` term is strongly positive,
/// pushing the total toward solid and suppressing caves in that stratum.
/// This produces the characteristic Minecraft "cave layer" banding —
/// caves that cluster at specific depths rather than distributing evenly.
///
/// `term1` (raw cheese signal) + `term2` (surface suppression, using
/// `raw_density` as a proxy for depth near the surface) + `layerized`.
/// The caller composes via `smin(density, cheese, k)` so the whole
/// signed value participates in the soft-blend.
///
/// ```text
///   term1      = clamp(cheese_offset + cheese_noise, -1, 1)
///   term2      = clamp(supp_offset + supp_slope × raw_density,
///                      supp_min, supp_max)
///   layerized  = cave_layer_intensity × layer²
///   result     = term1 + term2 + layerized
/// ```
pub fn cheese_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    raw_density: f32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    let xz_scale = cfg.cheese_xz_scale as f64;
    let y_scale = cfg.cheese_y_scale as f64;
    let cheese = carvers.cheese.get([
        wx as f64 * xz_scale,
        wy as f64 * y_scale,
        wz as f64 * xz_scale,
    ]) as f32;
    let term1 = (cfg.cheese_offset + cheese).clamp(-1.0, 1.0);
    let supp = (cfg.cheese_suppression_offset + cfg.cheese_suppression_slope * raw_density)
        .clamp(cfg.cheese_suppression_min, cfg.cheese_suppression_max);

    // MC-parity `layerizedCaverns`: add `intensity * layer²` so
    // cheese carving is regionally gated by horizontal strata.
    // Without this the cheese clamp at -1 makes ~50% of deep
    // voxels carve, producing uniform swiss cheese.
    let layer = carvers.cave_layer.get([
        wx as f64 * cfg.cave_layer_xz_scale as f64,
        wy as f64 * cfg.cave_layer_y_scale as f64,
        wz as f64 * cfg.cave_layer_xz_scale as f64,
    ]) as f32;
    let layerized = cfg.cave_layer_intensity * layer * layer;

    term1 + supp + layerized
}

/// Per-voxel pillar contribution. Returns a non-negative value in
/// `[0, pillar_intensity]` that is ADDED to density (not subtracted)
/// in `fill_chunk`'s composition. Composition order matters:
/// pillars apply AFTER all cave subtractions so they can refill
/// previously-carved voxels — the MC "columns inside open caves"
/// look.
///
/// MC formula (from `data/.../caves/pillars.json`):
///
/// ```text
///   pillar_raw  = 2 * noise(pillar, xz_scale, y_scale)
///   pillar_rare = -1 - noise(pillar_rareness)
///   thickness   = (0.55 + 0.55 * noise(pillar_thickness))^3
///   pillars     = (pillar_raw + pillar_rare) * thickness
///   range_choice: if pillars >= cutoff → pillars, else 0
/// ```
pub fn pillar_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    let p_x = wx as f64 * cfg.pillar_xz_scale as f64;
    let p_y = wy as f64 * cfg.pillar_y_scale as f64;
    let p_z = wz as f64 * cfg.pillar_xz_scale as f64;
    let pillar_raw = 2.0 * carvers.pillar.get([p_x, p_y, p_z]) as f32;
    let pillar_rare = -1.0
        - carvers
            .pillar_rareness
            .get([wx as f64, wy as f64, wz as f64]) as f32;
    let thickness_noise = carvers
        .pillar_thickness
        .get([wx as f64, wy as f64, wz as f64]) as f32;
    let thickness = (0.55 + 0.55 * thickness_noise).powi(3);
    let raw = (pillar_raw + pillar_rare) * thickness;
    if raw < cfg.pillar_cutoff {
        return 0.0;
    }
    let depth = (raw - cfg.pillar_cutoff).clamp(0.0, 1.0);
    cfg.pillar_intensity * depth
}

/// Polynomial smooth-min (Inigo Quilez's C1 smooth-min).
///
/// Produces a soft blend between two SDF surfaces within a blending radius
/// `k`. When `k = 0`, this is ordinary `min(a, b)`. Increasing `k` merges
/// nearby cave chambers and tunnels into one organic-looking connected
/// volume rather than leaving hard intersections where SDFs meet.
///
/// The formula: `min(a, b) - h²·k/4` where `h = max(0, k - |a-b|) / k`.
/// This is continuous and has a continuous first derivative at the blend
/// boundary; the maximum "pull below min" is exactly `k/4`.
///
/// Used to merge cave SDFs near layer boundaries so close-but-not-touching
/// pockets connect into one volume instead of staying as isolated bubbles.
#[inline]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = ((k - (a - b).abs()).max(0.0)) / k;
    a.min(b) - h * h * k * 0.25
}

/// Terasology-style depth-driven 2-noise disk carver.
///
/// Two independently seeded 3D FBM noise channels (`tera_a`, `tera_b`) are
/// each evaluated at the same scaled position. Geometrically, the pair
/// `(n0, n1)` defines a point in 2D noise space; carving occurs where that
/// point falls inside a disk of radius `freq_depth` centred near the
/// origin. Because noise values cluster near zero, the disk selects a
/// thin connected manifold — visually a set of nearly-horizontal tubes
/// threading through the rock, mimicking the stratigraphy-following
/// caves found in real karst.
///
/// Two depth-driven offsets modulate the disk:
/// - `freq_reduction` shifts the disk center off-axis near the surface,
///   suppressing carving there (`tera_supp` controls the suppression
///   depth). This replaces the blunt `CAVE_SURFACE_BUFFER` for tera-caves.
/// - `freq_depth` grows with depth so caves become more frequent
///   underground. The growth rate is `tera_thresh_depth`.
///
/// The Y axis is sampled at `tera_y_factor × freq` to squash the noise
/// vertically, keeping the tubes lean-horizontal.
///
/// Returns signed density: negative = carve, positive = solid. Output is
/// scaled by `* 5.0` to align its magnitude with the cheese carver for
/// downstream `smin` composition. Typical range: `[-5.4, +6.7]`.
///
/// Reference: `org.terasology.caves.CaveFacetProvider`.
pub fn terasology_ambient(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
    surface_y: f32,
) -> f32 {
    let depth = (surface_y - wy as f32).max(0.0);
    let freq_reduction = (cfg.tera_supp - depth / cfg.tera_supp_depth).max(0.0);
    let freq_depth = cfg.tera_thresh_base + depth / cfg.tera_thresh_depth;
    let freq = 1.0 / cfg.tera_wave;
    let wy_scaled = wy as f32 * cfg.tera_y_factor;
    let n0 = carvers.tera_a.get([
        (wx as f32 * freq) as f64,
        (wy_scaled * freq) as f64,
        (wz as f32 * freq) as f64,
    ]) as f32;
    let n1 = carvers.tera_b.get([
        (wx as f32 * freq) as f64,
        (wy_scaled * freq) as f64,
        (wz as f32 * freq) as f64,
    ]) as f32
        + freq_reduction;
    ((n0 * n0 + n1 * n1).sqrt() - freq_depth) * 5.0
    // scale: align magnitude with cheese carver for downstream smin composition
}

// ── Carver evaluator (corner-lattice trilerp) ────────────────────────
//
// The per-voxel `cheese_contribution` and `pillar_contribution` each
// issue several FBM samples per voxel. With 32³ voxels per chunk and
// 2-octave FBM, that dominates the chunk-fill cost.
//
// `CarverEvaluator` mirrors `density_graph::CellEvaluator`: sample
// each underlying noise on a 9³ corner lattice (4-block spacing,
// 729 corners per chunk), then trilerp per voxel. The per-voxel
// formulas (clamps, `raw_density`-dependent suppression, gates) run
// unchanged on the lerped values, so cave shapes track the original
// formulas to within FBM's local smoothness — visually equivalent at
// 4-block resolution, the same precedent as the base density.

const CARVER_CELL_SIZE: i32 = 4;
const CARVER_CELL_COUNT: usize =
    (crate::voxel::coords::CHUNK_DIM_U as usize) / CARVER_CELL_SIZE as usize; // 8
const CARVER_CORNER_COUNT: usize = CARVER_CELL_COUNT + 1; // 9
const CARVER_CORNER_CUBE: usize = CARVER_CORNER_COUNT * CARVER_CORNER_COUNT * CARVER_CORNER_COUNT; // 729

/// One corner's worth of pre-sampled noise. Keeping the channels
/// AoS means each voxel touches 8 contiguous corner structs instead
/// of striding separate `Vec<f32>` arenas — better cache behavior
/// in the per-voxel inner loop.
#[derive(Default, Clone, Copy)]
struct CarverCorner {
    cheese: f32,
    cave_layer: f32,
    pillar: f32,
    pillar_rare: f32,
    pillar_thick: f32,
    tera_a: f32,
    tera_b: f32,
}

/// Pre-sampled noise lattice for the carver layers. Built once per
/// chunk fill; `*_at` accessors trilerp from the 8 surrounding
/// corners and apply the original per-voxel formula.
pub struct CarverEvaluator {
    corners: Box<[CarverCorner]>,
    origin: IVec3,
}

impl CarverEvaluator {
    pub fn new(carvers: &NoiseCarvers, cfg: &CaveConfig, chunk_origin: IVec3) -> Self {
        let mut corners = vec![CarverCorner::default(); CARVER_CORNER_CUBE].into_boxed_slice();
        for cz in 0..CARVER_CORNER_COUNT {
            for cy in 0..CARVER_CORNER_COUNT {
                for cx in 0..CARVER_CORNER_COUNT {
                    let wx = chunk_origin.x + (cx as i32) * CARVER_CELL_SIZE;
                    let wy = chunk_origin.y + (cy as i32) * CARVER_CELL_SIZE;
                    let wz = chunk_origin.z + (cz as i32) * CARVER_CELL_SIZE;
                    let idx = corner_index(cx, cy, cz);

                    let cheese = carvers.cheese.get([
                        wx as f64 * cfg.cheese_xz_scale as f64,
                        wy as f64 * cfg.cheese_y_scale as f64,
                        wz as f64 * cfg.cheese_xz_scale as f64,
                    ]) as f32;
                    let cave_layer = carvers.cave_layer.get([
                        wx as f64 * cfg.cave_layer_xz_scale as f64,
                        wy as f64 * cfg.cave_layer_y_scale as f64,
                        wz as f64 * cfg.cave_layer_xz_scale as f64,
                    ]) as f32;

                    let pillar = carvers.pillar.get([
                        wx as f64 * cfg.pillar_xz_scale as f64,
                        wy as f64 * cfg.pillar_y_scale as f64,
                        wz as f64 * cfg.pillar_xz_scale as f64,
                    ]) as f32;
                    let pillar_rare = carvers
                        .pillar_rareness
                        .get([wx as f64, wy as f64, wz as f64])
                        as f32;
                    let pillar_thick = carvers
                        .pillar_thickness
                        .get([wx as f64, wy as f64, wz as f64])
                        as f32;

                    let tera_freq = 1.0 / cfg.tera_wave;
                    let tera_wy = wy as f32 * cfg.tera_y_factor;
                    let tera_a = carvers.tera_a.get([
                        (wx as f32 * tera_freq) as f64,
                        (tera_wy * tera_freq) as f64,
                        (wz as f32 * tera_freq) as f64,
                    ]) as f32;
                    let tera_b = carvers.tera_b.get([
                        (wx as f32 * tera_freq) as f64,
                        (tera_wy * tera_freq) as f64,
                        (wz as f32 * tera_freq) as f64,
                    ]) as f32;

                    corners[idx] = CarverCorner {
                        cheese,
                        cave_layer,
                        pillar,
                        pillar_rare,
                        pillar_thick,
                        tera_a,
                        tera_b,
                    };
                }
            }
        }
        Self {
            corners,
            origin: chunk_origin,
        }
    }

    #[inline]
    fn lerp_coords(&self, wx: i32, wy: i32, wz: i32) -> LerpCoords {
        let lx = wx - self.origin.x;
        let ly = wy - self.origin.y;
        let lz = wz - self.origin.z;
        let cx = ((lx / CARVER_CELL_SIZE) as usize).min(CARVER_CELL_COUNT - 1);
        let cy = ((ly / CARVER_CELL_SIZE) as usize).min(CARVER_CELL_COUNT - 1);
        let cz = ((lz / CARVER_CELL_SIZE) as usize).min(CARVER_CELL_COUNT - 1);
        let tx = (lx - cx as i32 * CARVER_CELL_SIZE) as f32 / CARVER_CELL_SIZE as f32;
        let ty = (ly - cy as i32 * CARVER_CELL_SIZE) as f32 / CARVER_CELL_SIZE as f32;
        let tz = (lz - cz as i32 * CARVER_CELL_SIZE) as f32 / CARVER_CELL_SIZE as f32;
        LerpCoords {
            cx,
            cy,
            cz,
            tx,
            ty,
            tz,
        }
    }

    /// Trilinear interp of one channel — `get` picks the channel from a
    /// `CarverCorner`. The Y→X→Z lerp order matches
    /// `density_graph::CellEvaluator::evaluate`.
    #[inline]
    fn trilerp<F: Fn(&CarverCorner) -> f32>(&self, c: &LerpCoords, get: F) -> f32 {
        let i = |x: usize, y: usize, z: usize| &self.corners[corner_index(x, y, z)];
        let c000 = get(i(c.cx, c.cy, c.cz));
        let c100 = get(i(c.cx + 1, c.cy, c.cz));
        let c010 = get(i(c.cx, c.cy + 1, c.cz));
        let c110 = get(i(c.cx + 1, c.cy + 1, c.cz));
        let c001 = get(i(c.cx, c.cy, c.cz + 1));
        let c101 = get(i(c.cx + 1, c.cy, c.cz + 1));
        let c011 = get(i(c.cx, c.cy + 1, c.cz + 1));
        let c111 = get(i(c.cx + 1, c.cy + 1, c.cz + 1));
        let xz00 = c000 + (c010 - c000) * c.ty;
        let xz10 = c100 + (c110 - c100) * c.ty;
        let xz01 = c001 + (c011 - c001) * c.ty;
        let xz11 = c101 + (c111 - c101) * c.ty;
        let z0 = xz00 + (xz10 - xz00) * c.tx;
        let z1 = xz01 + (xz11 - xz01) * c.tx;
        z0 + (z1 - z0) * c.tz
    }

    /// Same shape as `cheese_contribution`: `term1 + supp + layerized`.
    /// Trilerps the noise channels; the clamps and `raw_density`-driven
    /// suppression term run unchanged per voxel.
    pub fn cheese_at(&self, wx: i32, wy: i32, wz: i32, raw_density: f32, cfg: &CaveConfig) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let cheese = self.trilerp(&lc, |c| c.cheese);
        let cave_layer = self.trilerp(&lc, |c| c.cave_layer);
        let term1 = (cfg.cheese_offset + cheese).clamp(-1.0, 1.0);
        let supp = (cfg.cheese_suppression_offset + cfg.cheese_suppression_slope * raw_density)
            .clamp(cfg.cheese_suppression_min, cfg.cheese_suppression_max);
        let layerized = cfg.cave_layer_intensity * cave_layer * cave_layer;
        term1 + supp + layerized
    }

    /// Same shape as `pillar_contribution`. The cutoff gate runs per voxel.
    pub fn pillar_at(&self, wx: i32, wy: i32, wz: i32, cfg: &CaveConfig) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let pillar = self.trilerp(&lc, |c| c.pillar);
        let pillar_rare_n = self.trilerp(&lc, |c| c.pillar_rare);
        let thickness_noise = self.trilerp(&lc, |c| c.pillar_thick);
        let pillar_raw = 2.0 * pillar;
        let pillar_rare = -1.0 - pillar_rare_n;
        let thickness = (0.55 + 0.55 * thickness_noise).powi(3);
        let raw = (pillar_raw + pillar_rare) * thickness;
        if raw < cfg.pillar_cutoff {
            return 0.0;
        }
        let depth = (raw - cfg.pillar_cutoff).clamp(0.0, 1.0);
        cfg.pillar_intensity * depth
    }

    /// Trilerp the pre-sampled tera_a and tera_b noise values, then run the
    /// same `terasology_ambient` arithmetic on the lerped result. Exact at
    /// corners by construction.
    pub fn terasology_ambient_at(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        cfg: &CaveConfig,
        surface_y: f32,
    ) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let n0_raw = self.trilerp(&lc, |c| c.tera_a);
        let n1_raw = self.trilerp(&lc, |c| c.tera_b);
        let depth = (surface_y - wy as f32).max(0.0);
        let freq_reduction = (cfg.tera_supp - depth / cfg.tera_supp_depth).max(0.0);
        let freq_depth = cfg.tera_thresh_base + depth / cfg.tera_thresh_depth;
        let n1 = n1_raw + freq_reduction;
        ((n0_raw * n0_raw + n1 * n1).sqrt() - freq_depth) * 5.0
    }
}

#[derive(Clone, Copy)]
struct LerpCoords {
    cx: usize,
    cy: usize,
    cz: usize,
    tx: f32,
    ty: f32,
    tz: f32,
}

#[inline]
fn corner_index(cx: usize, cy: usize, cz: usize) -> usize {
    cx + cy * CARVER_CORNER_COUNT + cz * CARVER_CORNER_COUNT * CARVER_CORNER_COUNT
}

#[cfg(test)]
#[path = "caves_tests.rs"]
mod tests;
