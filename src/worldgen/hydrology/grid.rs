//! D8 direction encoding and the `Grid` computational primitive used by
//! both the fine and macro hydrology passes.
//!
//! The `Grid` struct drives the three-phase algorithm:
//! 1. `sink_fill` — Planchon-Darboux priority-queue depression filling.
//! 2. `compute_flow` — D8 single-flow-direction on the filled DEM.
//! 3. `compute_acc` — O(N) topological-sort flow accumulation.
//!
//! See `docs/book/content/part-3-region-build/3.4-hydrology.mdx` for the
//! algorithm walkthrough.

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
pub(super) struct Grid {
    /// Number of cells along one edge.
    pub(super) n: usize,
    /// Heightmap samples at cell centers. Length = `n * n`.
    pub(super) h: Vec<i16>,
    /// Filled heightmap (Planchon-Darboux output). Length = `n * n`.
    pub(super) h_fill: Vec<i16>,
    /// D8 flow direction per cell. `DIR_NONE` until `compute_flow` runs.
    pub(super) flow_dir: Vec<u8>,
    /// Flow accumulation. `0` until `compute_acc` runs (cells start
    /// with `1` + any injected trunk units).
    pub(super) flow_acc: Vec<u32>,
    /// Trunk injection (in fine-cell units) added to starting
    /// accumulation. Only the fine pass uses this.
    pub(super) trunk_injection: Vec<u32>,
    /// PR 1: per-cell inbound direction from a cached neighbour at
    /// the corresponding window-edge cell. `DIR_NONE` (8) = no hint.
    /// When set, `compute_flow` forbids picking the OPPOSITE direction
    /// (which would form a 2-cycle across the seam).
    pub(super) inbound_dir: Vec<u8>,
    /// PR 1: per-cell additional starting accumulation supplied by a
    /// cached neighbour. Added to `compute_acc`'s `1 + trunk_injection`
    /// initial value so a fat river keeps its magnitude across seams.
    pub(super) inbound_acc: Vec<u32>,
}

impl Grid {
    #[inline]
    pub(super) fn idx(&self, ix: usize, iz: usize) -> usize {
        iz * self.n + ix
    }

    #[inline]
    pub(super) fn in_bounds(&self, ix: i32, iz: i32) -> bool {
        ix >= 0 && iz >= 0 && (ix as usize) < self.n && (iz as usize) < self.n
    }

    /// Priority-queue Planchon-Darboux sink fill.
    ///
    /// Works like water pouring onto a landscape: start from the
    /// boundaries (edges of the computation window always drain to the
    /// edge — they're guaranteed outlets), then process cells in order of
    /// increasing elevation. Each interior cell is forced to be at least as
    /// high as the lowest drain-accessible surface, ensuring no closed
    /// sinks remain. Uses a min-heap so cells are always processed
    /// lowest-first, giving the "water rises from below" intuition.
    ///
    /// After `sink_fill`, every cell has a non-strictly-decreasing path to
    /// the window boundary. Cells whose `h_fill > h` were the bottoms of
    /// closed basins — they are the future lake interiors.
    ///
    /// Reference: Planchon & Darboux (2002), "A fast, simple and versatile
    /// algorithm to fill the depressions of digital elevation models".
    /// Our variant uses 0 for epsilon (integer heights; flat plateaus are
    /// not a problem at our chunk scale).
    pub(super) fn sink_fill(&mut self) {
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

    /// Compute D8 flow direction on the *filled* heightmap.
    ///
    /// D8 flow direction: each cell drains to whichever of its 8
    /// neighbours (N, NE, E, SE, S, SW, W, NW) lies at the steepest
    /// downhill slope. Slope is measured as `height_drop / distance` so
    /// a 1-block cardinal drop (distance 1) correctly beats a 1-block
    /// diagonal drop (distance √2). Cells with no downhill neighbour
    /// (flat or local high-points after sink fill — lake rims) stay as
    /// `DIR_NONE` until an outflow direction is determined.
    ///
    /// This is the standard D8 algorithm from O'Callaghan & Mark (1984),
    /// "The extraction of drainage networks from digital elevation data".
    /// D8's single-flow-direction model occasionally produces
    /// bifurcation artefacts at flat plateaus, but those are rare in
    /// our integer heightmaps.
    pub(super) fn compute_flow(&mut self) {
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
    ///
    /// Walk upstream-to-downstream: each cell's accumulation equals 1
    /// (itself) plus any injected trunk units plus the sum of all cells
    /// that drain into it. Since we process in topological order (highest
    /// filled elevation first), every upstream contributor is already
    /// counted when we reach a cell. After this pass, `flow_acc[c]` is the
    /// total upstream drainage area (in fine cells) flowing through `c`.
    ///
    /// Cells with `flow_acc >= RIVER_THRESH` become rivers; width follows
    /// a power-law `clamp(sqrt(acc) * RIVER_WIDTH_SCALE, MIN, MAX)`.
    pub(super) fn compute_acc(&mut self) {
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
