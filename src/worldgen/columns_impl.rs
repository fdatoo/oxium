//! Per-column terrain evaluation: [`Generator::column_data`] and the
//! hot-path [`Generator::column_data_with`].
//!
//! These methods are part of `Generator`'s inherent impl but live here so
//! `mod.rs` stays focused on the Generator struct definition and the
//! region-level wiring that feeds every chunk. Column evaluation is a
//! self-contained concern: noise samples → height → biome → `ColumnData`.
//!
//! Called from `fill_chunk_impl.rs` (per-column hot path), `trees_impl.rs`
//! (tree surface-height lookup), `probe_impl.rs` (visualizer), and
//! `mod_tests.rs`.

use noise::NoiseFn;

use super::Generator;
use crate::voxel::coords::ChunkCoord;
use crate::worldgen::climate;
use crate::worldgen::columns::ColumnData;
use crate::worldgen::pipeline::ChunkRegions;
use crate::worldgen::region::RegionCoord;
use crate::worldgen::tuning::{CAVE_FLOOR_Y, FINE_REGION_SIZE, MAX_TERRAIN_Y, SEA_LEVEL};

impl Generator {
    /// Evaluate per-column terrain, biome, and water-surface data for
    /// `(wx, wz)`. Gathers the 3 × 3 region neighbourhood inline.
    ///
    /// Convenience wrapper around [`Self::column_data_with`] �� use at call
    /// sites outside the chunk-fill hot path (tests, tree placement, probe)
    /// where the extra cache lookups are acceptable.
    pub fn column_data(&self, wx: i32, wz: i32) -> ColumnData {
        let coord = RegionCoord::containing(wx, wz);
        let chunk_origin = ChunkCoord(glam::IVec3::new(
            coord.x * (FINE_REGION_SIZE / 32),
            0,
            coord.z * (FINE_REGION_SIZE / 32),
        ));
        let regions = self.gather_chunk_regions(chunk_origin);
        self.column_data_with(wx, wz, &regions, None)
    }

    /// Per-column terrain decisions using pre-fetched regions.
    ///
    /// The per-column hot path inside `fill_chunk` calls this version so
    /// we don't pay 9 mutex-protected cache lookups per column.
    ///
    /// `precomputed_carve` is an optional already-computed valley-carve
    /// depth for this column. Pass `Some(depth)` when calling from
    /// `fill_chunk` (where the depth grid was built once for the whole
    /// chunk via `ChunkRegions::valley_grid`). Pass `None` at other call
    /// sites (e.g. `probe_column`) to fall back to the per-column
    /// `valley_carve` path.
    pub(super) fn column_data_with(
        &self,
        wx: i32,
        wz: i32,
        regions: &ChunkRegions,
        precomputed_carve: Option<f32>,
    ) -> ColumnData {
        let cfg = self.config.load();
        // PR 3: spline-driven heightmap. h_pre is now the surface Y
        // derived from the climate-spline `offset_spline`, NOT the
        // old plate-mosaic shelf+ridge+warpedFBM formula.
        let h_pre =
            self.heightmap
                .h_pre(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density);
        // Cliff = high slope + no stencil sample dipping below sea.
        // The pre-PR-3 CLIFF_MIN_HEIGHT gate is gone (the spline
        // already places mountains far from the coast by design).
        let is_cliff =
            self.heightmap
                .is_cliff(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density);

        // Valley carve over the chunk's pre-fetched 3 × 3 region
        // neighbourhood. Operates on the spline-derived h_pre (PR 3
        // interface change — same shape as before, just a different
        // h_pre source).
        //
        // Hot path: `fill_chunk` precomputes the whole 32×32 depth grid
        // once (segment-first with AABB culling) and passes the
        // per-column result here. Other callers pass `None` and fall back
        // to the full per-column O(segments) path.
        let carve = precomputed_carve.unwrap_or_else(|| regions.valley_carve(wx, wz, self.seed));
        let height = (h_pre - carve).clamp((CAVE_FLOOR_Y + 8) as f32, MAX_TERRAIN_Y as f32) as i32;

        // PR 4: 6D climate sample + R-tree biome lookup with
        // per-block hash-Voronoi jitter for organic borders.
        let (jx, jz) = climate::voronoi_jitter_offset(self.seed, wx, height, wz);
        let qwx = wx + jx;
        let qwz = wz + jz;
        let xz_jitter = [qwx as f64, qwz as f64];

        let temperature = self.temperature_map.get(xz_jitter) as f32;
        let humidity = self.humidity_map.get(xz_jitter) as f32;
        // Continentalness, terrain_shape, ridges_pv from the same
        // climate sampler that drives the heightmap splines.
        let (c, s, _pv, _look) =
            self.heightmap
                .climate(self.seed, qwx as f32, qwz as f32, &cfg.climate);
        // Weirdness — independent mid-frequency Fbm.
        let weirdness =
            (self.weirdness_noise.get(xz_jitter) as f32) * cfg.biomes.weirdness_amplitude;
        // Depth axis: normalized world-Y of the column's surface.
        let depth_t = (height as f32 - cfg.density.y_min as f32)
            / (cfg.density.y_max - cfg.density.y_min) as f32;
        let depth = 1.0 - 2.0 * depth_t; // +1 at world floor, -1 at world top

        let target = climate::TargetPoint::new(temperature, humidity, c, s, depth, weirdness);
        let biome = self.biome_list.lookup(&target);
        // `desertness` is kept on ColumnData for the legacy sand
        // transition heuristic in fill_chunk. Derived from the
        // (now-quantized) humidity + temperature pair: high
        // temperature × low humidity == high desertness.
        let desertness = (temperature * 0.5) - humidity * 0.5;

        // Plate-driven ocean predicate. Uses continentalness `c` from
        // the climate call above (biome-jitter offset is a few blocks —
        // negligible at the 1024-block plate scale). Any oceanic-plate
        // column whose terrain is at or below sea level is classified as
        // ocean; inland sub-sea depressions on continental plates are
        // classified as lake or dry pit, never ocean.
        let is_ocean = c < 0.0 && height <= SEA_LEVEL;
        // Unified water surface Y. Lake rim is filtered to ≥ height+1
        // so shore columns that sit exactly one block below the lake surface
        // are correctly submerged (≥ height+2 left a 1-voxel exposed water
        // face at the waterline). Step 5 guarantees all interior lake columns
        // have height ≤ rim−3 (MIN_LAKE_BED_DROP), so they still pass; the
        // looser threshold only newly admits the one-block rim zone.
        // River priority is patched in by `fill_chunk` after the river_grid
        // is built.
        let water_surface_y = regions
            .lake_rim_at(wx, wz)
            .filter(|&rim| rim > height)
            .or_else(|| is_ocean.then_some(SEA_LEVEL));
        ColumnData {
            height,
            h_pre,
            is_cliff,
            desertness,
            biome,
            water_surface_y,
        }
    }
}
