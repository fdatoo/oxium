//! Hydrology: D8 flow accumulation at fine (8 m) and macro (64 m)
//! resolution, sink-fill into lakes, trunk-river injection, river
//! segment extraction, valley carving.
//!
//! ### Algorithm summary
//!
//! A region's hydrology is built on a 5 × 5-region window of fine
//! cells (2-region halo on each side, total 320 × 320 cells at 8 m
//! per cell). Operating on the full window means flow that crosses
//! into the region from outside is computed correctly — and two
//! adjacent regions that share a halo see the same flow direction
//! at the shared border (with the caveat that closed basins bigger
//! than the window terrace at region boundaries; the macro pass
//! catches the big ones).
//!
//! 1. **Sample h_pre** at every fine cell in the window.
//! 2. **Sink fill** (priority-queue Planchon-Darboux) — every cell
//!    ends up with a non-strictly-decreasing path to the window
//!    boundary. Cells whose filled height exceeds their natural
//!    height are flagged as lake water; the filled value is the
//!    lake's rim elevation.
//! 3. **Macro trunk injection** — for each macro cell flagged as a
//!    trunk river by the macro pass, inject `macro_acc * 64` units
//!    of starting accumulation at the fine cell at the macro cell's
//!    center. Trunk drainage from outside the visible fine window
//!    appears as already-fat rivers entering the region.
//! 4. **D8 flow direction** on the filled heightmap → each cell
//!    points to its single steepest-downhill neighbour. Lake-rim
//!    cells get an outflow direction (over the rim toward the
//!    lowest neighbour outside the lake).
//! 5. **Flow accumulation** — topological sort cells by filled
//!    height descending; each cell donates its own area plus any
//!    injected trunk units to its downstream neighbour. O(N).
//! 6. **River cells** = cells where accumulation ≥ `RIVER_THRESH`.
//!    Width = `clamp(sqrt(acc) * SCALE, MIN, MAX)`.
//! 7. **Extract region-interior data** into the cache (boxed
//!    slices, halo discarded).
//! 8. **Build river segment list** by walking each river cell to
//!    its downstream neighbour — the cell-center to next-cell-center
//!    polyline plus a domain-warped meander offset.
//!
//! ### Macro pass
//!
//! Same algorithm at coarser resolution (64 m cells, 1-macro-region
//! halo). Operates on 128 × 128 cells per macro region (window 3 ×
//! 128 = 384 cells); sees a 24 km drainage horizon — enough for
//! continental-scale trunk rivers and inland-basin lakes.

use crate::worldgen::heightmap::HeightmapNoise;
use crate::worldgen::region::{
    bitset_get, bitset_set, FineRegion, MacroCache, MacroRegion, MacroRegionCoord,
    RegionCoord, RiverSegment,
};
use crate::worldgen::tuning::*;
use std::collections::BinaryHeap;

// ── D8 direction encoding ─────────────────────────────────────────────

/// 8 compass directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW.
/// 8 = sink / no downhill (lake centre after fill, before outflow
/// direction is computed).
pub const DIR_NONE: u8 = 8;

/// (dx, dz) offsets for each direction code.
pub const DIR_OFFSETS: [(i32, i32); 8] = [
    (0, -1),   // N
    (1, -1),   // NE
    (1, 0),    // E
    (1, 1),    // SE
    (0, 1),    // S
    (-1, 1),   // SW
    (-1, 0),   // W
    (-1, -1),  // NW
];

/// Diagonal moves cost √2 longer than cardinal moves; the D8 slope
/// test divides height-drop by distance so a 1-block drop diagonally
/// loses to a 1-block drop cardinally (correctly — the diagonal slope
/// is shallower).
const DIR_DIST: [f32; 8] = [
    1.0,                    // N
    std::f32::consts::SQRT_2, // NE
    1.0,                    // E
    std::f32::consts::SQRT_2, // SE
    1.0,                    // S
    std::f32::consts::SQRT_2, // SW
    1.0,                    // W
    std::f32::consts::SQRT_2, // NW
];

// ── Grid view ─────────────────────────────────────────────────────────

/// A square grid view used by the D8 / fill / accumulation routines.
/// Both fine and macro passes operate on an instance of this struct.
struct Grid {
    /// Number of cells along one edge.
    n: usize,
    /// Heightmap samples at cell centers. Length = `n * n`.
    h: Vec<i16>,
    /// Filled heightmap (Planchon-Darboux output). Length = `n * n`.
    h_fill: Vec<i16>,
    /// D8 flow direction per cell. `DIR_NONE` until `compute_flow` runs.
    flow_dir: Vec<u8>,
    /// Flow accumulation. `0` until `compute_acc` runs (cells start
    /// with `1` + any injected trunk units).
    flow_acc: Vec<u32>,
    /// Trunk injection (in fine-cell units) added to starting
    /// accumulation. Only the fine pass uses this.
    trunk_injection: Vec<u32>,
}

impl Grid {
    #[inline]
    fn idx(&self, ix: usize, iz: usize) -> usize {
        iz * self.n + ix
    }

    #[inline]
    fn in_bounds(&self, ix: i32, iz: i32) -> bool {
        ix >= 0 && iz >= 0 && (ix as usize) < self.n && (iz as usize) < self.n
    }

    /// Priority-queue Planchon-Darboux sink fill. Every cell ends up
    /// with `h_fill[c] >= h[c]` and a non-strictly-decreasing path to
    /// the window boundary. `epsilon` is added between adjacent cells
    /// to break flat-plateau ties; we use 0 for integer heights —
    /// flat plateaus aren't a problem at our chunk scale.
    fn sink_fill(&mut self) {
        let n = self.n;
        // Min-heap of (filled height, packed (ix, iz)) for cells whose
        // h_fill is already determined.
        #[derive(PartialEq, Eq)]
        struct Item {
            h: i16,
            ix: u16,
            iz: u16,
        }
        impl Ord for Item {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Reverse for min-heap.
                other.h.cmp(&self.h)
            }
        }
        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        // Initialise: h_fill = h on the boundary, +∞ inside.
        self.h_fill.fill(i16::MAX);
        let mut heap = BinaryHeap::new();
        for ix in 0..n {
            for iz in 0..n {
                let on_boundary = ix == 0 || iz == 0 || ix == n - 1 || iz == n - 1;
                if on_boundary {
                    let h = self.h[iz * n + ix];
                    self.h_fill[iz * n + ix] = h;
                    heap.push(Item {
                        h,
                        ix: ix as u16,
                        iz: iz as u16,
                    });
                }
            }
        }
        // Pop lowest cell; propagate raising-or-keeping to unfilled
        // neighbours.
        while let Some(Item { h, ix, iz }) = heap.pop() {
            let ix = ix as i32;
            let iz = iz as i32;
            for d in 0..8 {
                let (dx, dz) = DIR_OFFSETS[d];
                let nx = ix + dx;
                let nz = iz + dz;
                if !self.in_bounds(nx, nz) {
                    continue;
                }
                let ni = (nz as usize) * n + (nx as usize);
                if self.h_fill[ni] != i16::MAX {
                    continue;
                }
                // Lift to current frontier height if the natural
                // height is lower; otherwise keep natural height.
                let raised = self.h[ni].max(h);
                self.h_fill[ni] = raised;
                heap.push(Item {
                    h: raised,
                    ix: nx as u16,
                    iz: nz as u16,
                });
            }
        }
    }

    /// Compute D8 flow direction on the *filled* heightmap. Every
    /// cell picks the steepest-downhill neighbour by `slope = drop /
    /// distance`. Cells at the boundary point inward toward their
    /// best neighbour (the boundary itself can't flow off the grid).
    fn compute_flow(&mut self) {
        let n = self.n;
        for iz in 0..n {
            for ix in 0..n {
                let idx = self.idx(ix, iz);
                let h_here = self.h_fill[idx];
                let mut best_slope = 0.0_f32;
                let mut best_dir = DIR_NONE;
                for d in 0..8 {
                    let (dx, dz) = DIR_OFFSETS[d];
                    let nx = ix as i32 + dx;
                    let nz = iz as i32 + dz;
                    if !self.in_bounds(nx, nz) {
                        continue;
                    }
                    let ni = (nz as usize) * n + (nx as usize);
                    let drop = (h_here - self.h_fill[ni]) as f32;
                    if drop <= 0.0 {
                        continue;
                    }
                    let slope = drop / DIR_DIST[d];
                    if slope > best_slope {
                        best_slope = slope;
                        best_dir = d as u8;
                    }
                }
                self.flow_dir[idx] = best_dir;
            }
        }
    }

    /// Compute upstream flow accumulation in topological order.
    /// Each cell donates `1 + trunk_injection[c]` to its downstream
    /// neighbour. After this, `flow_acc[c]` is the total upstream
    /// drainage that reaches `c` (in fine cells).
    fn compute_acc(&mut self) {
        let n = self.n;
        // Initialise: each cell contributes 1 + injection.
        for i in 0..(n * n) {
            self.flow_acc[i] = 1 + self.trunk_injection[i];
        }
        // Process cells highest-first so a downstream donation
        // doesn't get stomped by a later upstream donation.
        let mut order: Vec<u32> = (0..(n * n) as u32).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.h_fill[i as usize]));
        for &i in &order {
            let i = i as usize;
            let dir = self.flow_dir[i];
            if dir == DIR_NONE {
                continue;
            }
            let ix = i % n;
            let iz = i / n;
            let (dx, dz) = DIR_OFFSETS[dir as usize];
            let nx = ix as i32 + dx;
            let nz = iz as i32 + dz;
            if !self.in_bounds(nx, nz) {
                continue;
            }
            let ni = (nz as usize) * n + (nx as usize);
            self.flow_acc[ni] = self.flow_acc[ni].saturating_add(self.flow_acc[i]);
        }
    }
}

// ── Macro pass ────────────────────────────────────────────────────────

/// Build the macro region for `coord` from noise alone. Pure in
/// `(seed, coord)`.
pub fn build_macro_region(
    seed: u64,
    coord: MacroRegionCoord,
    heightmap: &HeightmapNoise,
) -> MacroRegion {
    let halo = MACRO_HALO_REGIONS;
    let inner = MACRO_CELLS_PER_REGION;
    let n = ((1 + 2 * halo) * inner) as usize;
    let origin_x = (coord.x - halo) * MACRO_REGION_SIZE;
    let origin_z = (coord.z - halo) * MACRO_REGION_SIZE;

    let mut grid = Grid {
        n,
        h: vec![0i16; n * n],
        h_fill: vec![0i16; n * n],
        flow_dir: vec![DIR_NONE; n * n],
        flow_acc: vec![0u32; n * n],
        trunk_injection: vec![0u32; n * n],
    };

    // Sample h_pre at the cell centers of the macro window.
    for iz in 0..n {
        for ix in 0..n {
            let wx = origin_x + (ix as i32) * MACRO_CELL + MACRO_CELL / 2;
            let wz = origin_z + (iz as i32) * MACRO_CELL + MACRO_CELL / 2;
            grid.h[iz * n + ix] = heightmap.h_pre(seed, wx as f32, wz as f32) as i16;
        }
    }

    grid.sink_fill();
    grid.compute_flow();
    grid.compute_acc();

    // Extract the macro region's interior (drop halo).
    let mut flow_dir = vec![DIR_NONE; (inner * inner) as usize];
    let mut flow_acc = vec![0u32; (inner * inner) as usize];
    let bs_bytes = (inner as usize * inner as usize).div_ceil(8);
    let mut is_trunk = vec![0u8; bs_bytes];
    let mut is_lake = vec![0u8; bs_bytes];
    let mut lake_rim = vec![0i16; (inner * inner) as usize];
    let halo_cells = (halo * inner) as usize;
    for iz in 0..(inner as usize) {
        for ix in 0..(inner as usize) {
            let src = (iz + halo_cells) * n + (ix + halo_cells);
            let dst = iz * (inner as usize) + ix;
            flow_dir[dst] = grid.flow_dir[src];
            flow_acc[dst] = grid.flow_acc[src];
            if grid.flow_acc[src] >= MACRO_RIVER_THRESH {
                bitset_set(&mut is_trunk, dst, true);
            }
            if grid.h_fill[src] > grid.h[src] {
                bitset_set(&mut is_lake, dst, true);
                lake_rim[dst] = grid.h_fill[src];
            }
        }
    }

    MacroRegion {
        coord,
        flow_dir: flow_dir.into_boxed_slice(),
        flow_acc: flow_acc.into_boxed_slice(),
        is_trunk: is_trunk.into_boxed_slice(),
        is_lake: is_lake.into_boxed_slice(),
        lake_rim: lake_rim.into_boxed_slice(),
    }
}

// ── Fine pass ─────────────────────────────────────────────────────────

/// Build the hydrology layer of a fine region. Requires access to the
/// macro cache so trunk drainage from outside the fine window can be
/// injected.
/// Read-only snapshots of the four cardinal-neighbour fine regions of
/// a region currently being built. Each entry is `Some` iff the
/// neighbour is already in the fine cache; `gather_neighbour_edges`
/// never triggers a build. Names denote the direction *to* the
/// neighbour. Corner neighbours are omitted — diagonal contact is one
/// cell and not worth the bookkeeping.
pub struct NeighbourEdges {
    pub west: Option<std::sync::Arc<crate::worldgen::region::FineRegion>>,
    pub east: Option<std::sync::Arc<crate::worldgen::region::FineRegion>>,
    pub north: Option<std::sync::Arc<crate::worldgen::region::FineRegion>>,
    pub south: Option<std::sync::Arc<crate::worldgen::region::FineRegion>>,
}

impl NeighbourEdges {
    /// All-`None` view. Equivalent to the pre-PR-1 free-edge behaviour.
    pub fn empty() -> Self {
        Self {
            west: None,
            east: None,
            north: None,
            south: None,
        }
    }
}

/// Build a [`NeighbourEdges`] for `coord` by peeking the four cardinal
/// neighbours in the fine cache. Cold neighbours stay `None`.
pub fn gather_neighbour_edges(
    coord: RegionCoord,
    fine_cache: &crate::worldgen::region::FineCache,
) -> NeighbourEdges {
    NeighbourEdges {
        west: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord { x: coord.x - 1, z: coord.z },
        ),
        east: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord { x: coord.x + 1, z: coord.z },
        ),
        north: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord { x: coord.x, z: coord.z - 1 },
        ),
        south: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord { x: coord.x, z: coord.z + 1 },
        ),
    }
}

pub fn build_fine_hydro(
    seed: u64,
    coord: RegionCoord,
    heightmap: &HeightmapNoise,
    macro_cache: &MacroCache,
    fine_cache: &crate::worldgen::region::FineCache,
    region: &mut FineRegion,
) {
    // PR 1: gather neighbour edges. Task 3 consumes them.
    let _neighbours = gather_neighbour_edges(coord, fine_cache);
    let halo = FINE_HALO_REGIONS;
    let inner = FINE_CELLS_PER_REGION;
    let n = ((1 + 2 * halo) * inner) as usize;
    let origin_x = (coord.x - halo) * FINE_REGION_SIZE;
    let origin_z = (coord.z - halo) * FINE_REGION_SIZE;

    let mut grid = Grid {
        n,
        h: vec![0i16; n * n],
        h_fill: vec![0i16; n * n],
        flow_dir: vec![DIR_NONE; n * n],
        flow_acc: vec![0u32; n * n],
        trunk_injection: vec![0u32; n * n],
    };

    // Sample h_pre at fine cell centers.
    for iz in 0..n {
        for ix in 0..n {
            let wx = origin_x + (ix as i32) * FINE_CELL + FINE_CELL / 2;
            let wz = origin_z + (iz as i32) * FINE_CELL + FINE_CELL / 2;
            grid.h[iz * n + ix] = heightmap.h_pre(seed, wx as f32, wz as f32) as i16;
        }
    }

    grid.sink_fill();

    // Trunk injection: for each macro cell inside the window flagged
    // as trunk, find the fine cell at its center and add
    // `macro_acc * (FINE_PER_MACRO * FINE_PER_MACRO)` to the
    // injection (rescaling macro-cell-area units to fine-cell-area).
    //
    // The macro window only needs to cover what our fine window can
    // see. Compute which macro cells our window touches.
    let macro_unit = MACRO_CELL;
    let fine_per_macro_axis = FINE_PER_MACRO; // 8
    // Determine which macro regions overlap our fine window.
    let win_min_x = origin_x;
    let win_min_z = origin_z;
    let win_max_x = origin_x + (n as i32) * FINE_CELL;
    let win_max_z = origin_z + (n as i32) * FINE_CELL;
    let mr_min = MacroRegionCoord::containing(win_min_x, win_min_z);
    let mr_max = MacroRegionCoord::containing(win_max_x - 1, win_max_z - 1);
    for mrz in mr_min.z..=mr_max.z {
        for mrx in mr_min.x..=mr_max.x {
            let mr_coord = MacroRegionCoord { x: mrx, z: mrz };
            let mr = crate::worldgen::region::get_macro(macro_cache, mr_coord, || {
                build_macro_region(seed, mr_coord, heightmap)
            });
            let mr_origin = mr_coord.origin();
            for miz in 0..(MACRO_CELLS_PER_REGION as usize) {
                for mix in 0..(MACRO_CELLS_PER_REGION as usize) {
                    let mi = miz * (MACRO_CELLS_PER_REGION as usize) + mix;
                    let mwx = mr_origin.0 + (mix as i32) * macro_unit + macro_unit / 2;
                    let mwz = mr_origin.1 + (miz as i32) * macro_unit + macro_unit / 2;
                    // Convert to fine-cell index inside the window.
                    let fix = (mwx - origin_x).div_euclid(FINE_CELL);
                    let fiz = (mwz - origin_z).div_euclid(FINE_CELL);
                    if fix < 0 || fiz < 0 || fix as usize >= n || fiz as usize >= n {
                        continue;
                    }
                    let fi = (fiz as usize) * n + (fix as usize);
                    if bitset_get(&mr.is_trunk, mi) {
                        let macro_acc = mr.flow_acc[mi];
                        let area_factor = (fine_per_macro_axis * fine_per_macro_axis) as u32;
                        grid.trunk_injection[fi] = grid
                            .trunk_injection[fi]
                            .saturating_add(macro_acc.saturating_mul(area_factor));
                    }
                    // Also propagate macro lake rims into the fine
                    // grid: any fine cell that falls inside a macro
                    // lake gets its h_fill raised to the macro rim
                    // (if higher than the fine fill).
                    if bitset_get(&mr.is_lake, mi) {
                        let rim = mr.lake_rim[mi];
                        // Stamp the macro lake over the 8×8 block of
                        // fine cells covered by this macro cell.
                        for dz in 0..fine_per_macro_axis {
                            for dx in 0..fine_per_macro_axis {
                                let cx = fix as i32 + dx - fine_per_macro_axis / 2;
                                let cz = fiz as i32 + dz - fine_per_macro_axis / 2;
                                if cx < 0 || cz < 0 || cx as usize >= n || cz as usize >= n {
                                    continue;
                                }
                                let ci = (cz as usize) * n + (cx as usize);
                                if grid.h[ci] < rim {
                                    grid.h_fill[ci] = grid.h_fill[ci].max(rim);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    grid.compute_flow();
    grid.compute_acc();

    // Extract the region's interior.
    let halo_cells = (halo * inner) as usize;
    let inner_u = inner as usize;
    region.flow_dir.fill(DIR_NONE);
    region.flow_acc.fill(0);
    for b in region.is_river.iter_mut() {
        *b = 0;
    }
    for b in region.is_lake.iter_mut() {
        *b = 0;
    }
    region.width.fill(0.0);
    region.lake_rim.fill(0);

    for iz in 0..inner_u {
        for ix in 0..inner_u {
            let src = (iz + halo_cells) * n + (ix + halo_cells);
            let dst = iz * inner_u + ix;
            region.flow_dir[dst] = grid.flow_dir[src];
            region.flow_acc[dst] = grid.flow_acc[src];
            region.h_pre[dst] = grid.h[src];
            if grid.flow_acc[src] >= RIVER_THRESH {
                bitset_set(&mut region.is_river, dst, true);
                let w = ((grid.flow_acc[src] as f32).sqrt() * RIVER_WIDTH_SCALE)
                    .clamp(MIN_RIVER_WIDTH, MAX_RIVER_WIDTH);
                region.width[dst] = w;
            }
            if grid.h_fill[src] > grid.h[src] {
                bitset_set(&mut region.is_lake, dst, true);
                region.lake_rim[dst] = grid.h_fill[src];
            }
        }
    }

    // Build river segments inside the region.
    region.segments.clear();
    let region_origin_x = coord.x * FINE_REGION_SIZE;
    let region_origin_z = coord.z * FINE_REGION_SIZE;
    for iz in 0..inner_u {
        for ix in 0..inner_u {
            let dst = iz * inner_u + ix;
            if !bitset_get(&region.is_river, dst) {
                continue;
            }
            let dir = region.flow_dir[dst];
            if dir == DIR_NONE {
                continue;
            }
            let (dx, dz) = DIR_OFFSETS[dir as usize];
            let from = (
                region_origin_x + (ix as i32) * FINE_CELL + FINE_CELL / 2,
                region_origin_z + (iz as i32) * FINE_CELL + FINE_CELL / 2,
            );
            let to = (from.0 + dx * FINE_CELL, from.1 + dz * FINE_CELL);
            let width = region.width[dst];
            // Mouth: segment is at the downstream side of land — if
            // the downstream cell's h_pre is ≤ SEA_LEVEL.
            let nx = ix as i32 + dx;
            let nz = iz as i32 + dz;
            let mouth = nx >= 0
                && nz >= 0
                && nx < inner as i32
                && nz < inner as i32
                && region.h_pre[(nz as usize) * inner_u + (nx as usize)] <= SEA_LEVEL as i16;
            region.segments.push(RiverSegment {
                from,
                to,
                width,
                mouth,
            });
        }
    }
}

// ── Valley carving (queried per column at chunk-fill time) ────────────

/// Return the depth (blocks) the river/valley pass should subtract
/// from `h_pre` at world `(wx, wz)`. Looks at all river segments in
/// the chunk's local fine region plus its 8 immediate neighbours
/// (so a river that exits one region carves the valley continuously
/// into the next).
pub fn valley_carve(
    wx: i32,
    wz: i32,
    region: &FineRegion,
    neighbours: &[Option<&FineRegion>; 8],
    seed: u64,
) -> f32 {
    let mut max_depth: f32 = 0.0;
    for_each_segment(region, neighbours, |seg| {
        let d = perpendicular_distance(wx, wz, seg, seed);
        let width = if seg.mouth {
            seg.width * MOUTH_FLARE_MULT
        } else {
            seg.width
        };
        let half_w = width * 0.5;
        let half_valley = width * VALLEY_HALF_WIDTH_MULT;
        let depth = if d <= half_w {
            RIVER_BED_DEPTH as f32
        } else if d < half_valley {
            let t = (d - half_w) / (half_valley - half_w);
            // Smoothstep falloff.
            let s = 1.0 - t * t * (3.0 - 2.0 * t);
            s * RIVER_BED_DEPTH as f32
        } else {
            0.0
        };
        if depth > max_depth {
            max_depth = depth;
        }
    });
    max_depth
}

/// Iterate over every river segment in `region` and `neighbours`.
fn for_each_segment<F: FnMut(&RiverSegment)>(
    region: &FineRegion,
    neighbours: &[Option<&FineRegion>; 8],
    mut f: F,
) {
    for s in &region.segments {
        f(s);
    }
    for opt in neighbours {
        if let Some(r) = opt {
            for s in &r.segments {
                f(s);
            }
        }
    }
}

/// Perpendicular distance (blocks) from world `(wx, wz)` to the
/// segment's perturbed centerline. Adds a domain-warped offset
/// scaled by width so trunk rivers meander hard while small streams
/// stay nearly straight.
fn perpendicular_distance(wx: i32, wz: i32, seg: &RiverSegment, seed: u64) -> f32 {
    let p = (wx as f32, wz as f32);
    let a = (seg.from.0 as f32, seg.from.1 as f32);
    let b = (seg.to.0 as f32, seg.to.1 as f32);
    let ab = (b.0 - a.0, b.1 - a.1);
    let len_sq = ab.0 * ab.0 + ab.1 * ab.1;
    if len_sq < 1e-6 {
        // Degenerate segment — fall back to point distance.
        let dx = p.0 - a.0;
        let dz = p.1 - a.1;
        return (dx * dx + dz * dz).sqrt();
    }
    let t = (((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / len_sq).clamp(0.0, 1.0);
    let proj = (a.0 + ab.0 * t, a.1 + ab.1 * t);
    // Perpendicular offset based on a quick deterministic hash of the
    // projected point — substitute for a real Simplex meander to
    // avoid pulling another noise field per query. Same shape /
    // smoothness as a hash-noise interpolation.
    let amp = (seg.width * MEANDER_AMP_PER_WIDTH).min(MAX_MEANDER_AMP);
    // Sample a smooth 1D noise along the segment's parametric `t`:
    // hash adjacent buckets and linearly interpolate.
    let along_world = (proj.0 - seg.from.0 as f32).hypot(proj.1 - seg.from.1 as f32);
    let bucket = (along_world / 8.0).floor() as i32;
    let frac = along_world / 8.0 - bucket as f32;
    let h0 =
        crate::worldgen::hash::mix_range(seed, &[seg.from.0, seg.from.1, bucket], -1.0, 1.0);
    let h1 =
        crate::worldgen::hash::mix_range(seed, &[seg.from.0, seg.from.1, bucket + 1], -1.0, 1.0);
    let offset = (h0 * (1.0 - frac) + h1 * frac) * amp;
    // Perpendicular unit vector.
    let len = len_sq.sqrt();
    let perp = (-ab.1 / len, ab.0 / len);
    let perturbed = (proj.0 + perp.0 * offset, proj.1 + perp.1 * offset);
    let dx = p.0 - perturbed.0;
    let dz = p.1 - perturbed.1;
    (dx * dx + dz * dz).sqrt()
}

/// Look up the lake rim at world `(wx, wz)`. Returns `Some(rim_y)` if
/// the column is inside a lake; `None` otherwise. Used by chunk fill
/// to flood lake water above the original heightmap.
pub fn lake_rim_at(wx: i32, wz: i32, region: &FineRegion) -> Option<i32> {
    let region_origin = (
        region.coord.x * FINE_REGION_SIZE,
        region.coord.z * FINE_REGION_SIZE,
    );
    let lx = wx - region_origin.0;
    let lz = wz - region_origin.1;
    let ix = lx.div_euclid(FINE_CELL);
    let iz = lz.div_euclid(FINE_CELL);
    if ix < 0 || iz < 0 || ix >= FINE_CELLS_PER_REGION || iz >= FINE_CELLS_PER_REGION {
        return None;
    }
    let i = (iz * FINE_CELLS_PER_REGION + ix) as usize;
    if bitset_get(&region.is_lake, i) {
        Some(region.lake_rim[i] as i32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(n: usize, h: Vec<i16>) -> Grid {
        Grid {
            n,
            h_fill: vec![0i16; n * n],
            flow_dir: vec![DIR_NONE; n * n],
            flow_acc: vec![0u32; n * n],
            trunk_injection: vec![0u32; n * n],
            h,
        }
    }

    #[test]
    fn d8_picks_steepest_downhill() {
        // 3x3 with center higher than all neighbours. D8 picks the
        // *steepest slope* = drop / distance, not just the lowest
        // neighbour. With NW at value 1 (drop 8, slope 8/√2 ≈ 5.66)
        // and N at value 2 (drop 7, slope 7/1 = 7.0), N wins on
        // slope even though NW has the larger absolute drop.
        //   1 2 3
        //   4 9 5
        //   6 7 8
        let h = vec![1i16, 2, 3, 4, 9, 5, 6, 7, 8];
        let mut g = make_grid(3, h);
        g.h_fill = g.h.clone();
        g.compute_flow();
        assert_eq!(g.flow_dir[1 * 3 + 1], 0); // N
    }

    #[test]
    fn sink_fill_raises_local_minimum() {
        // 3x3 with a pit in the middle:
        //   9 9 9
        //   9 0 9
        //   9 9 9
        let h = vec![9i16, 9, 9, 9, 0, 9, 9, 9, 9];
        let mut g = make_grid(3, h.clone());
        g.sink_fill();
        // Center cell should be raised to at least 9 (the rim).
        assert!(
            g.h_fill[1 * 3 + 1] >= 9,
            "center pit should fill to ≥ 9, got {}",
            g.h_fill[1 * 3 + 1]
        );
    }

    #[test]
    fn flow_acc_concentrates_on_lowest_path() {
        // A 5x1 ramp dropping linearly: 5, 4, 3, 2, 1. (Make it 5x5
        // with all rows identical so D8 has a clear east-direction
        // flow.) Verify acc strictly increases from the high end to
        // the low end along the spine.
        let n = 5;
        let mut h = Vec::with_capacity(n * n);
        for iz in 0..n {
            for ix in 0..n {
                let _ = iz;
                h.push((n as i16 - 1 - ix as i16) * 10);
            }
        }
        let mut g = make_grid(n, h);
        g.sink_fill();
        g.compute_flow();
        g.compute_acc();
        // Cell at (0,2), (1,2), (2,2), (3,2): accumulation should
        // increase along the row toward x=n-1 (= the low end).
        let prev = g.flow_acc[2 * n + 0];
        for ix in 1..n {
            let here = g.flow_acc[2 * n + ix];
            assert!(
                here >= prev,
                "acc should be non-decreasing down the ramp: at ix={ix} got {here}, prev {prev}"
            );
        }
    }

    #[test]
    fn build_macro_region_is_deterministic() {
        let hm = HeightmapNoise::new(42);
        let coord = MacroRegionCoord { x: 0, z: 0 };
        let m1 = build_macro_region(42, coord, &hm);
        let m2 = build_macro_region(42, coord, &hm);
        // Compare a small sample of cells.
        assert_eq!(m1.flow_dir, m2.flow_dir);
        assert_eq!(m1.flow_acc, m2.flow_acc);
        assert_eq!(m1.is_trunk, m2.is_trunk);
    }

    #[test]
    fn fine_hydro_produces_some_river_cells() {
        // Scan a 5 × 5 grid of regions around the origin. At least one
        // should produce river cells. Some regions are pure ocean and
        // won't have any; we just need one with land + drainage.
        let hm = HeightmapNoise::new(42);
        let macro_cache = crate::worldgen::region::fresh_macro_cache();
        let fine_cache = crate::worldgen::region::fresh_fine_cache();
        let mut total_river_cells = 0usize;
        for z in -2..=2 {
            for x in -2..=2 {
                let coord = RegionCoord { x, z };
                let mut region =
                    crate::worldgen::region::build_fine_region_placeholder(coord);
                region.coord = coord;
                build_fine_hydro(42, coord, &hm, &macro_cache, &fine_cache, &mut region);
                let n = (FINE_CELLS_PER_REGION * FINE_CELLS_PER_REGION) as usize;
                total_river_cells +=
                    (0..n).filter(|&i| bitset_get(&region.is_river, i)).count();
            }
        }
        assert!(
            total_river_cells > 0,
            "no river cells across the 5×5 region scan at seed=42"
        );
    }

    #[test]
    fn river_width_monotonic_downstream() {
        // Flow accumulation only ever increases downstream, so width
        // (which is monotone in acc) should too. Pick any river cell
        // in a built region and chase its flow_dir; widths must be
        // non-decreasing.
        let hm = HeightmapNoise::new(42);
        let macro_cache = crate::worldgen::region::fresh_macro_cache();
        let fine_cache = crate::worldgen::region::fresh_fine_cache();
        let coord = RegionCoord { x: 0, z: 0 };
        let mut region = crate::worldgen::region::build_fine_region_placeholder(coord);
        region.coord = coord;
        build_fine_hydro(42, coord, &hm, &macro_cache, &fine_cache, &mut region);
        let n = FINE_CELLS_PER_REGION as usize;
        for iz in 1..n - 1 {
            for ix in 1..n - 1 {
                let idx = iz * n + ix;
                if !bitset_get(&region.is_river, idx) {
                    continue;
                }
                let dir = region.flow_dir[idx];
                if dir == DIR_NONE {
                    continue;
                }
                let (dx, dz) = DIR_OFFSETS[dir as usize];
                let nx = ix as i32 + dx;
                let nz = iz as i32 + dz;
                if nx < 0 || nz < 0 || nx as usize >= n || nz as usize >= n {
                    continue;
                }
                let ni = (nz as usize) * n + (nx as usize);
                if !bitset_get(&region.is_river, ni) {
                    continue;
                }
                assert!(
                    region.width[ni] >= region.width[idx] - 1e-3,
                    "width decreased downstream at ({ix},{iz}) → ({nx},{nz}): {} → {}",
                    region.width[idx],
                    region.width[ni]
                );
            }
        }
    }
}
