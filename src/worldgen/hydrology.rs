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
    FineRegion, MacroCache, MacroRegion, MacroRegionCoord, RegionCoord, RiverSegment,
    RiverSegmentKind, bitset_get, bitset_set,
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
    (0, -1),  // N
    (1, -1),  // NE
    (1, 0),   // E
    (1, 1),   // SE
    (0, 1),   // S
    (-1, 1),  // SW
    (-1, 0),  // W
    (-1, -1), // NW
];

/// Diagonal moves cost √2 longer than cardinal moves; the D8 slope
/// test divides height-drop by distance so a 1-block drop diagonally
/// loses to a 1-block drop cardinally (correctly — the diagonal slope
/// is shallower).
const DIR_DIST: [f32; 8] = [
    1.0,                      // N
    std::f32::consts::SQRT_2, // NE
    1.0,                      // E
    std::f32::consts::SQRT_2, // SE
    1.0,                      // S
    std::f32::consts::SQRT_2, // SW
    1.0,                      // W
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
    /// PR 1: per-cell inbound direction from a cached neighbour at
    /// the corresponding window-edge cell. `DIR_NONE` (8) = no hint.
    /// When set, `compute_flow` forbids picking the OPPOSITE direction
    /// (which would form a 2-cycle across the seam).
    inbound_dir: Vec<u8>,
    /// PR 1: per-cell additional starting accumulation supplied by a
    /// cached neighbour. Added to `compute_acc`'s `1 + trunk_injection`
    /// initial value so a fat river keeps its magnitude across seams.
    inbound_acc: Vec<u32>,
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
                // PR 1: if a neighbour stitch marked this cell with an
                // inbound direction, forbid picking the opposite (would
                // form a 2-cycle across the seam). Inbound dir `d` →
                // forbidden self-dir is `(d + 4) & 7`.
                let forbidden_dir = if self.inbound_dir[idx] != DIR_NONE {
                    (self.inbound_dir[idx] + 4) & 7
                } else {
                    DIR_NONE
                };
                let mut best_slope = 0.0_f32;
                let mut best_dir = DIR_NONE;
                for d in 0..8 {
                    if d as u8 == forbidden_dir {
                        continue;
                    }
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
        // Initialise: each cell contributes 1 + injection + (PR 1)
        // any inbound accumulation donated by a cached neighbour at
        // a stitched boundary cell.
        for i in 0..(n * n) {
            self.flow_acc[i] = (1u32)
                .saturating_add(self.trunk_injection[i])
                .saturating_add(self.inbound_acc[i]);
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
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
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
        inbound_dir: vec![DIR_NONE; n * n],
        inbound_acc: vec![0u32; n * n],
    };

    // Sample h_pre at the cell centers of the macro window.
    for iz in 0..n {
        for ix in 0..n {
            let wx = origin_x + (ix as i32) * MACRO_CELL + MACRO_CELL / 2;
            let wz = origin_z + (iz as i32) * MACRO_CELL + MACRO_CELL / 2;
            grid.h[iz * n + ix] =
                heightmap.h_pre(seed, wx as f32, wz as f32, climate, density) as i16;
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
            RegionCoord {
                x: coord.x - 1,
                z: coord.z,
            },
        ),
        east: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord {
                x: coord.x + 1,
                z: coord.z,
            },
        ),
        north: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord {
                x: coord.x,
                z: coord.z - 1,
            },
        ),
        south: crate::worldgen::region::peek_fine(
            fine_cache,
            RegionCoord {
                x: coord.x,
                z: coord.z + 1,
            },
        ),
    }
}

pub fn build_fine_hydro(
    seed: u64,
    coord: RegionCoord,
    heightmap: &HeightmapNoise,
    climate: &crate::worldgen::config::ClimateConfig,
    density: &crate::worldgen::config::DensityConfig,
    macro_cache: &MacroCache,
    fine_cache: &crate::worldgen::region::FineCache,
    region: &mut FineRegion,
) {
    // PR 1: gather neighbour edges (peek-only, no build).
    let neighbours = gather_neighbour_edges(coord, fine_cache);
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
        inbound_dir: vec![DIR_NONE; n * n],
        inbound_acc: vec![0u32; n * n],
    };

    // Sample h_pre at fine cell centers.
    for iz in 0..n {
        for ix in 0..n {
            let wx = origin_x + (ix as i32) * FINE_CELL + FINE_CELL / 2;
            let wz = origin_z + (iz as i32) * FINE_CELL + FINE_CELL / 2;
            grid.h[iz * n + ix] =
                heightmap.h_pre(seed, wx as f32, wz as f32, climate, density) as i16;
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
                build_macro_region(seed, mr_coord, heightmap, climate, density)
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
                        grid.trunk_injection[fi] = grid.trunk_injection[fi]
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

    // PR 1: stitch neighbour-edge inbound hints into this region's
    // interior-boundary cells. Window-grid layout: this region's
    // interior occupies indices in `[halo_cells, halo_cells + inner)`
    // on both axes, where `halo_cells = halo * inner`. The neighbour
    // regions' INTERIOR cells (size = inner * inner) are what we read.
    {
        let halo_cells = (halo * inner) as usize;
        let inner_u = inner as usize;

        // West neighbour: its east-most interior column flows into our
        // west-most interior column when its flow_dir == 2 (east).
        if let Some(west) = neighbours.west.as_ref() {
            for iz in 0..inner_u {
                let neigh_idx = iz * inner_u + (inner_u - 1);
                if west.flow_dir[neigh_idx] != 2 {
                    continue;
                }
                let grid_idx = (halo_cells + iz) * n + halo_cells;
                grid.inbound_dir[grid_idx] = 2;
                grid.inbound_acc[grid_idx] =
                    grid.inbound_acc[grid_idx].saturating_add(west.flow_acc[neigh_idx]);
            }
        }
        // East neighbour: its west-most interior column flows into our
        // east-most interior column when its flow_dir == 6 (west).
        if let Some(east) = neighbours.east.as_ref() {
            for iz in 0..inner_u {
                let neigh_idx = iz * inner_u;
                if east.flow_dir[neigh_idx] != 6 {
                    continue;
                }
                let grid_idx = (halo_cells + iz) * n + (halo_cells + inner_u - 1);
                grid.inbound_dir[grid_idx] = 6;
                grid.inbound_acc[grid_idx] =
                    grid.inbound_acc[grid_idx].saturating_add(east.flow_acc[neigh_idx]);
            }
        }
        // North neighbour: its south-most interior row flows into our
        // north-most interior row when its flow_dir == 4 (south).
        if let Some(north) = neighbours.north.as_ref() {
            for ix in 0..inner_u {
                let neigh_idx = (inner_u - 1) * inner_u + ix;
                if north.flow_dir[neigh_idx] != 4 {
                    continue;
                }
                let grid_idx = halo_cells * n + (halo_cells + ix);
                grid.inbound_dir[grid_idx] = 4;
                grid.inbound_acc[grid_idx] =
                    grid.inbound_acc[grid_idx].saturating_add(north.flow_acc[neigh_idx]);
            }
        }
        // South neighbour: its north-most interior row flows into our
        // south-most interior row when its flow_dir == 0 (north).
        if let Some(south) = neighbours.south.as_ref() {
            for ix in 0..inner_u {
                let neigh_idx = ix;
                if south.flow_dir[neigh_idx] != 0 {
                    continue;
                }
                let grid_idx = (halo_cells + inner_u - 1) * n + (halo_cells + ix);
                grid.inbound_dir[grid_idx] = 0;
                grid.inbound_acc[grid_idx] =
                    grid.inbound_acc[grid_idx].saturating_add(south.flow_acc[neigh_idx]);
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
            // Tag as lake only when the sink-fill raised the cell by at
            // least LAKE_MIN_NATURAL_DEPTH blocks AND the cell's natural
            // height is not deep ocean.
            //
            // The ≥ LAKE_MIN_NATURAL_DEPTH guard suppresses 1-block-deep
            // "scratch" basins that sink-fill creates at every slight
            // terrain depression. Without it, Step-5 carving
            // (MIN_LAKE_BED_DROP = 3) deepens those scratches into visible
            // ponds even though the basin is topographically insignificant.
            //
            // The ≥ SEA_LEVEL-6 guard prevents ocean cells from being
            // tagged as lakes; ocean is handled by the plate-driven
            // predicate in column_data_with, and tagging it here would
            // produce a non-flat "tilted ocean" rim.
            let natural_depth = grid.h_fill[src] - grid.h[src];
            if natural_depth >= LAKE_MIN_NATURAL_DEPTH as i16 && grid.h[src] >= SEA_LEVEL as i16 - 6
            {
                bitset_set(&mut region.is_lake, dst, true);
                region.lake_rim[dst] = grid.h_fill[src];
                // Guarantee at least MIN_LAKE_BED_DROP blocks of open water
                // above the terrain floor so lakes have visible depth.
                region.lake_bed_depth[dst] = natural_depth.max(MIN_LAKE_BED_DROP as i16);
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
                && nx < inner
                && nz < inner
                && region.h_pre[(nz as usize) * inner_u + (nx as usize)] <= SEA_LEVEL as i16;
            let src = (iz + halo_cells) * n + (ix + halo_cells);
            let here_h = grid.h[src] as i32;
            let (downstream_h, downstream_h_fill) =
                if nx >= 0 && nz >= 0 && nx < inner && nz < inner {
                    let ds = (nz as usize + halo_cells) * n + (nx as usize + halo_cells);
                    (grid.h[ds] as i32, grid.h_fill[ds] as i32)
                } else {
                    (here_h, grid.h_fill[src] as i32)
                };
            let drop = here_h - downstream_h;
            let kind = if drop >= 12 {
                RiverSegmentKind::Waterfall
            } else if drop >= 5 {
                RiverSegmentKind::Rapid
            } else {
                RiverSegmentKind::Channel
            };
            let water_y = if mouth {
                SEA_LEVEL
            } else {
                // Use sink-fill heights rather than raw terrain heights so
                // adjacent segments share a coherent, monotone water surface.
                // h_fill is non-decreasing along the upstream direction by
                // the Planchon-Darboux guarantee, eliminating the per-segment
                // stepping that produced visible water walls.
                (grid.h_fill[src] as i32)
                    .min(downstream_h_fill)
                    .max(SEA_LEVEL + 1)
            };
            let bed_y = water_y - RIVER_BED_DEPTH;
            region.segments.push(RiverSegment {
                from,
                to,
                width,
                water_y,
                bed_y,
                kind,
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

/// Precompute valley-carve depths for an entire 32×32 chunk column grid
/// in one pass over the segment list. Much faster than calling `valley_carve`
/// per-column because it AABB-culls each segment to only the columns it
/// can reach, then only computes `perpendicular_distance` for those columns.
///
/// `origin_wx` and `origin_wz` are the world-space X/Z of the chunk's
/// (0,0) column (i.e. `chunk_coord.x * 32` and `chunk_coord.z * 32`).
pub fn valley_grid(
    origin_wx: i32,
    origin_wz: i32,
    region: &FineRegion,
    neighbours: &[Option<&FineRegion>; 8],
    seed: u64,
) -> [[f32; 32]; 32] {
    const DIM: i32 = 32;
    let mut grid = [[0.0f32; 32]; 32];

    for_each_segment(region, neighbours, |seg| {
        let width = if seg.mouth {
            seg.width * MOUTH_FLARE_MULT
        } else {
            seg.width
        };
        let half_w = width * 0.5;
        let half_valley = width * VALLEY_HALF_WIDTH_MULT;

        // Compute segment AABB expanded by half_valley, then clip to chunk.
        let seg_min_wx = ((seg.from.0.min(seg.to.0) as f32) - half_valley).floor() as i32;
        let seg_max_wx = ((seg.from.0.max(seg.to.0) as f32) + half_valley).ceil() as i32;
        let seg_min_wz = ((seg.from.1.min(seg.to.1) as f32) - half_valley).floor() as i32;
        let seg_max_wz = ((seg.from.1.max(seg.to.1) as f32) + half_valley).ceil() as i32;

        // Clip to chunk world-space bounds; convert to local indices.
        let lx_min = ((seg_min_wx - origin_wx).max(0) as usize).min((DIM - 1) as usize);
        let lx_max = ((seg_max_wx - origin_wx).max(0) as usize).min((DIM - 1) as usize);
        let lz_min = ((seg_min_wz - origin_wz).max(0) as usize).min((DIM - 1) as usize);
        let lz_max = ((seg_max_wz - origin_wz).max(0) as usize).min((DIM - 1) as usize);

        // Guard: if the segment AABB doesn't intersect this chunk at all,
        // both ranges will be clamped to the same side and lx_min > lx_max
        // or lz_min > lz_max. Skip before entering the inner loop.
        if seg_min_wx > origin_wx + DIM - 1
            || seg_max_wx < origin_wx
            || seg_min_wz > origin_wz + DIM - 1
            || seg_max_wz < origin_wz
        {
            return;
        }

        for lz in lz_min..=lz_max {
            for lx in lx_min..=lx_max {
                let wx = origin_wx + lx as i32;
                let wz = origin_wz + lz as i32;
                let d = perpendicular_distance(wx, wz, seg, seed);
                let depth = if d <= half_w {
                    RIVER_BED_DEPTH as f32
                } else if d < half_valley {
                    let t = (d - half_w) / (half_valley - half_w);
                    // Smoothstep falloff — same formula as `valley_carve`.
                    let s = 1.0 - t * t * (3.0 - 2.0 * t);
                    s * RIVER_BED_DEPTH as f32
                } else {
                    0.0
                };
                if depth > grid[lz][lx] {
                    grid[lz][lx] = depth;
                }
            }
        }
    });

    // Second pass: lake bed carve. For every column already inside a lake
    // fine cell, apply its lake_bed_depth (guaranteed ≥ MIN_LAKE_BED_DROP).
    // This runs after the river pass so rivers inside lake basins keep their
    // bed depth where it exceeds the lake bed depth.
    let all_regions =
        std::iter::once(Some(region)).chain(neighbours.iter().map(|o| o.map(|r| r as &FineRegion)));
    for reg in all_regions.flatten() {
        let (rx, rz) = reg.coord.origin();
        for lz in 0..DIM as usize {
            for lx in 0..DIM as usize {
                let wx = origin_wx + lx as i32;
                let wz = origin_wz + lz as i32;
                let ix = (wx - rx).div_euclid(FINE_CELL);
                let iz = (wz - rz).div_euclid(FINE_CELL);
                if ix < 0 || iz < 0 || ix >= FINE_CELLS_PER_REGION || iz >= FINE_CELLS_PER_REGION {
                    continue;
                }
                let i = (iz * FINE_CELLS_PER_REGION + ix) as usize;
                if bitset_get(&reg.is_lake, i) {
                    let depth = reg.lake_bed_depth[i] as f32;
                    if depth > grid[lz][lx] {
                        grid[lz][lx] = depth;
                    }
                }
            }
        }
    }

    grid
}

/// Iterate over every river segment in `region` and `neighbours`.
pub(crate) fn for_each_segment<F: FnMut(&RiverSegment)>(
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
pub(crate) fn perpendicular_distance(wx: i32, wz: i32, seg: &RiverSegment, seed: u64) -> f32 {
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
    // Hoist the sqrt here so we can reuse `len` for both `along_world` and
    // the perpendicular unit vector, eliminating the redundant `hypot` call
    // that would otherwise perform a second sqrt on the same quantity.
    let len = len_sq.sqrt();
    let t = (((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / len_sq).clamp(0.0, 1.0);
    let proj = (a.0 + ab.0 * t, a.1 + ab.1 * t);
    // Perpendicular offset based on a quick deterministic hash of the
    // projected point — substitute for a real Simplex meander to
    // avoid pulling another noise field per query. Same shape /
    // smoothness as a hash-noise interpolation.
    let amp = (seg.width * MEANDER_AMP_PER_WIDTH).min(MAX_MEANDER_AMP);
    // Sample a smooth 1D noise along the segment's parametric `t`:
    // hash adjacent buckets and linearly interpolate.
    //
    // `along_world` is the arc-length from `seg.from` to the projected point.
    // Since proj = a + t*ab and t is clamped to [0,1], this equals t * len —
    // one sqrt instead of the previous `hypot` (which hid an extra sqrt).
    let along_world = t * len;
    let bucket = (along_world / 8.0).floor() as i32;
    let frac = along_world / 8.0 - bucket as f32;
    let h0 = crate::worldgen::hash::mix_range(seed, &[seg.from.0, seg.from.1, bucket], -1.0, 1.0);
    let h1 =
        crate::worldgen::hash::mix_range(seed, &[seg.from.0, seg.from.1, bucket + 1], -1.0, 1.0);
    let offset = (h0 * (1.0 - frac) + h1 * frac) * amp;
    // Perpendicular unit vector.
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
#[path = "hydrology_tests.rs"]
mod tests;
