//! Per-chunk region query context: [`ChunkRegions`].
//!
//! [`ChunkRegions`] is a 3 × 3 grid of pre-fetched fine regions centered
//! on the chunk being generated. It is built once at the top of
//! `Generator::fill_chunk` and then used for all per-column queries
//! (valley carve, lake rim, cave intersection, river cell) — avoiding
//! cache-mutex traffic on every column.
//!
//! ### Why 3 × 3?
//!
//! A chunk is 32 blocks wide; a fine region is 512 blocks wide. A chunk
//! therefore touches at most 4 distinct fine regions (at chunk corners).
//! The 3 × 3 grid is the minimal axis-aligned box of regions that always
//! fully contains the chunk and a 1-cell halo (needed by `lake_rim_at`'s
//! 8-neighbour search). Pre-fetching all nine regions up-front means the
//! per-column hot path is pure arithmetic — no mutex locks, no cache
//! misses.
//!
//! ### Access pattern
//!
//! `Generator::gather_chunk_regions` builds a `ChunkRegions` using the
//! Generator's fine-region cache. The `ChunkRegions` is then passed
//! into `fill_chunk`'s body and into the tree-placement pass.
//! It is intentionally NOT stored on `Generator` because it is
//! chunk-specific and must be rebuilt for every call to `fill_chunk`.
//!
//! See `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md` §4.1.

use crate::voxel::block::Block;
use crate::worldgen::tuning::{FINE_CELL, MOUTH_FLARE_MULT};
use crate::worldgen::{fluid, hydrology, region};

/// Pre-fetched 3 × 3 grid of fine regions centered on a chunk.
///
/// Built once per `fill_chunk` call by `Generator::gather_chunk_regions`.
/// All per-column queries (valley carve, lake rim, cave intersection,
/// river cell) use this grid instead of hitting the region cache per-call.
pub(crate) struct ChunkRegions {
    /// The region coordinate at the center of the 3 × 3 grid.
    pub(crate) center: region::RegionCoord,
    /// `grid[dz + 1][dx + 1]` is the fine region at offset `(dx, dz)` from
    /// `center`. Every slot is `Some` — `gather_chunk_regions` builds any
    /// missing region before populating the grid.
    pub(crate) grid: [[Option<std::sync::Arc<region::FineRegion>>; 3]; 3],
}

impl ChunkRegions {
    /// Look up the fine region containing world coords `(wx, wz)` inside
    /// the pre-fetched 3 × 3 grid.
    ///
    /// Returns `None` if `(wx, wz)` lies outside the gathered grid, which
    /// should not happen for any column inside the chunk that triggered the
    /// `gather_chunk_regions` call. Treated as a defensive no-op (the caller
    /// skips the query and returns 0 / `None` instead of panicking).
    pub(crate) fn region_at(&self, wx: i32, wz: i32) -> Option<&region::FineRegion> {
        let c = region::RegionCoord::containing(wx, wz);
        let dx = c.x - self.center.x + 1;
        let dz = c.z - self.center.z + 1;
        if dx < 0 || dz < 0 || dx >= 3 || dz >= 3 {
            return None;
        }
        self.grid[dz as usize][dx as usize].as_deref()
    }

    /// Lake rim Y at world `(wx, wz)`, or `None` if this column sits outside
    /// any known lake basin.
    ///
    /// Searches the column's own fine cell and its 8 cardinal + diagonal
    /// neighbour cells (each `FINE_CELL` blocks away) and returns the
    /// highest rim found. The 1-cell halo search is what makes lake water
    /// reach the shore cleanly: a cell *adjacent* to a lake still sees the
    /// lake's rim and fills to it, preventing terrace steps at cell
    /// boundaries.
    pub(crate) fn lake_rim_at(&self, wx: i32, wz: i32) -> Option<i32> {
        let mut best: Option<i32> = None;
        for dz in -1..=1i32 {
            for dx in -1..=1i32 {
                let nx = wx + dx * FINE_CELL;
                let nz = wz + dz * FINE_CELL;
                if let Some(region) = self.region_at(nx, nz) {
                    if let Some(rim) = hydrology::lake_rim_at(nx, nz, region) {
                        best = Some(best.map_or(rim, |b| b.max(rim)));
                    }
                }
            }
        }
        best
    }

    /// Cave systems whose AABB intersects the box `[chunk_min, chunk_max]`.
    ///
    /// Called once at the top of `fill_chunk` to build the per-chunk
    /// cave-system list. The list is then passed into the per-voxel cave
    /// SDF loop so the inner loop iterates only the systems that could
    /// actually contribute to this chunk.
    pub(crate) fn cave_systems_intersecting(
        &self,
        chunk_min: glam::IVec3,
        chunk_max: glam::IVec3,
    ) -> Vec<&region::CaveSystem> {
        let mut out = Vec::new();
        for row in &self.grid {
            for slot in row {
                if let Some(r) = slot {
                    for sys in &r.cave_systems {
                        if sys.overlaps_box(chunk_min, chunk_max) {
                            out.push(sys);
                        }
                    }
                }
            }
        }
        out
    }

    /// Cave pools whose ellipsoid AABB intersects `[chunk_min, chunk_max]`.
    ///
    /// Called once per `fill_chunk` alongside `cave_systems_intersecting`
    /// to pre-filter the pool list before the per-voxel loop.
    pub(crate) fn cave_pools_intersecting(
        &self,
        chunk_min: glam::IVec3,
        chunk_max: glam::IVec3,
    ) -> Vec<&region::CavePool> {
        let mut out = Vec::new();
        for row in &self.grid {
            for slot in row {
                if let Some(r) = slot {
                    for pool in &r.cave_pools {
                        // AABB derived from ellipsoid half-extents.
                        let px = pool.center.x as i32;
                        let py = pool.center.y as i32;
                        let pz = pool.center.z as i32;
                        let rx = pool.radii.x.ceil() as i32;
                        let ry = pool.radii.y.ceil() as i32;
                        let rz = pool.radii.z.ceil() as i32;
                        if px + rx < chunk_min.x
                            || px - rx > chunk_max.x
                            || py + ry < chunk_min.y
                            || py - ry > chunk_max.y
                            || pz + rz < chunk_min.z
                            || pz - rz > chunk_max.z
                        {
                            continue;
                        }
                        out.push(pool);
                    }
                }
            }
        }
        out
    }

    /// Valley carve depth at world column `(wx, wz)`.
    ///
    /// Iterates the river segments in the column's primary region and its
    /// 8 neighbours (clipped to the 3 × 3 grid). Per-column cost is
    /// O(segments visible from the column) — typically a few dozen.
    ///
    /// Returns 0.0 if `(wx, wz)` is outside the gathered grid.
    pub(crate) fn valley_carve(&self, wx: i32, wz: i32, seed: u64) -> f32 {
        let c = region::RegionCoord::containing(wx, wz);
        let center_dx = c.x - self.center.x + 1;
        let center_dz = c.z - self.center.z + 1;
        if center_dx < 0 || center_dz < 0 || center_dx >= 3 || center_dz >= 3 {
            // Column outside the gathered grid — should not happen in
            // practice; return 0.0 (no carve) rather than panicking.
            return 0.0;
        }
        let primary = self.grid[center_dz as usize][center_dx as usize]
            .as_deref()
            .expect("3x3 grid is always populated");
        let neighbour_regions = self.gather_neighbours(center_dx, center_dz);
        hydrology::valley_carve(wx, wz, primary, &neighbour_regions, seed)
    }

    /// Precompute a 32 × 32 valley-depth grid for the chunk whose
    /// origin column is at world `(origin_wx, origin_wz)`.
    ///
    /// Segment-first pass with AABB culling — much faster than calling
    /// `valley_carve` 1024 times per chunk. The hydrology module does
    /// the per-segment → per-cell projection work.
    pub(crate) fn valley_grid(
        &self,
        origin_wx: i32,
        origin_wz: i32,
        seed: u64,
    ) -> [[f32; 32]; 32] {
        let c = region::RegionCoord::containing(origin_wx, origin_wz);
        let center_dx = c.x - self.center.x + 1;
        let center_dz = c.z - self.center.z + 1;
        if center_dx < 0 || center_dz < 0 || center_dx >= 3 || center_dz >= 3 {
            return [[0.0; 32]; 32];
        }
        let primary = self.grid[center_dz as usize][center_dx as usize]
            .as_deref()
            .expect("3x3 grid is always populated");
        let neighbour_regions = self.gather_neighbours(center_dx, center_dz);
        hydrology::valley_grid(origin_wx, origin_wz, primary, &neighbour_regions, seed)
    }

    /// Build the `[Option<&FineRegion>; 8]` neighbour array used by the
    /// hydrology helpers. The 8 slots correspond to the 8-connected
    /// neighbours of grid cell `(center_dx, center_dz)` in the 3 × 3 grid,
    /// in clockwise order starting from North (0,-1).
    fn gather_neighbours(
        &self,
        center_dx: i32,
        center_dz: i32,
    ) -> [Option<&region::FineRegion>; 8] {
        let nbr_offsets: [(i32, i32); 8] = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ];
        let mut out: [Option<&region::FineRegion>; 8] = [None; 8];
        for (i, (ox, oz)) in nbr_offsets.iter().enumerate() {
            let nx = center_dx + ox;
            let nz = center_dz + oz;
            if nx < 0 || nz < 0 || nx >= 3 || nz >= 3 {
                continue;
            }
            out[i] = self.grid[nz as usize][nx as usize].as_deref();
        }
        out
    }

    /// Build the per-chunk river grid: for each column `(x, z)` in
    /// `[0..32) × [0..32)`, look up the riverbed cell (if any) that
    /// claims this world column. Returns an array indexed `z * 32 + x`.
    pub(crate) fn river_grid(
        &self,
        origin_wx: i32,
        origin_wz: i32,
        seed: u64,
    ) -> [Option<fluid::FluidCell>; 32 * 32] {
        let mut out = [None; 32 * 32];
        for z in 0..32 {
            for x in 0..32 {
                out[z * 32 + x] =
                    self.river_cell_at(origin_wx + x as i32, origin_wz + z as i32, seed);
            }
        }
        out
    }

    /// The highest-priority river cell at world column `(wx, wz)`, or
    /// `None` if no river segment claims this column.
    ///
    /// Priority: the closest segment's perpendicular distance wins when
    /// multiple segments overlap. Waterfalls are skipped (they don't fill
    /// water voxels, they just carve into the terrain).
    pub(crate) fn river_cell_at(&self, wx: i32, wz: i32, seed: u64) -> Option<fluid::FluidCell> {
        let c = region::RegionCoord::containing(wx, wz);
        let center_dx = c.x - self.center.x + 1;
        let center_dz = c.z - self.center.z + 1;
        if center_dx < 0 || center_dz < 0 || center_dx >= 3 || center_dz >= 3 {
            return None;
        }
        let primary = self.grid[center_dz as usize][center_dx as usize]
            .as_deref()
            .expect("3x3 grid is always populated");
        let neighbour_regions = self.gather_neighbours(center_dx, center_dz);

        let mut best: Option<(f32, fluid::FluidCell)> = None;
        // Segments narrower than this threshold look like hairline cracks
        // on screen; clamp to MIN_VISIBLE_RIVER_WIDTH so every river that
        // made it to the segment list is wide enough to see.
        const MIN_VISIBLE_RIVER_WIDTH: f32 = 7.0;

        hydrology::for_each_segment(primary, &neighbour_regions, |seg| {
            if seg.kind == region::RiverSegmentKind::Waterfall {
                return;
            }
            let width = if seg.mouth {
                seg.width * MOUTH_FLARE_MULT
            } else {
                seg.width
            }
            .max(MIN_VISIBLE_RIVER_WIDTH);
            let d = hydrology::perpendicular_distance(wx, wz, seg, seed);
            let half_w = width * 0.5;
            if d > half_w {
                return;
            }
            let cell = fluid::FluidCell {
                kind: fluid::FluidBodyKind::River,
                block: Block::Water,
                surface_y: seg.water_y,
                bed_y: seg.bed_y,
                reason: fluid::FluidReason::RiverChannel,
            };
            if best.is_none_or(|(best_d, _)| d < best_d) {
                best = Some((d, cell));
            }
        });
        best.map(|(_, cell)| cell)
    }
}
