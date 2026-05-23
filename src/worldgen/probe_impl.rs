//! Probe and visualizer debug methods on [`Generator`].
//!
//! These four methods (`probe_column`, `paint_column`, `sample_stage`,
//! `evaluate_density_breakdown`) are read-only, have no side-effects, and
//! are only called from the visualizer binary (`worldgen_viz`) and the probe
//! panel. Keeping them here rather than inline in `mod.rs` lets the
//! chunk-fill hot path stay compact without 550 lines of debug tooling
//! between §3 (column data) and §5 (chunk fill).
//!
//! All Generator fields are `pub(crate)`, so this child module can access
//! them directly. Private helper methods (`column_data_with`,
//! `gather_chunk_regions`, `build_fine_region`) are also accessible because
//! child modules can see parent-module private items in Rust.

use crate::voxel::block::Block;
use crate::voxel::coords::ChunkCoord;
use crate::worldgen::biome::Biome;
use crate::worldgen::density::math::{slide, smooth_plate_contribution};
use crate::worldgen::tuning::{
    CAVE_BAND_MIDDLE, CAVE_BAND_SHALLOW, CAVE_FLOOR_Y, CAVE_SURFACE_BUFFER, FINE_CELL,
    FINE_REGION_SIZE, SEA_LEVEL, SURFACE_BAND,
};
use crate::worldgen::{caves, density, fluid, plates, probe, region, surface};
// NoiseFn provides the `.get()` method on `Fbm<Simplex>` fields.
use super::Generator;
use noise::NoiseFn;

impl Generator {
    /// Snapshot every pipeline value computed for this column. Used by
    /// the viz column probe. Read-only, byte-stable per `(seed, wx, wz)`.
    pub fn probe_column(&self, wx: i32, wz: i32) -> probe::ColumnProbe {
        let cfg = self.config.load();

        // Reuse column_data for the values it already produces.
        let col = self.column_data(wx, wz);

        // Plate + continentalness. The smooth blend matches what
        // climate() feeds the offset spline; the raw 2-nearest
        // signed_continentalness has a step discontinuity at
        // second-rank-flip lines and is no longer authoritative.
        let plate = plates::plate_at(self.seed, wx, wz);
        let (continentalness, _) = smooth_plate_contribution(self.seed, wx, wz, &cfg.climate);

        // Pre-carve height.
        let h_pre =
            self.heightmap
                .h_pre(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density);

        // Slope.
        let slope =
            self.heightmap
                .slope_at(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density);

        // Valley carve: gather regions the same way column_data does, then
        // call the ChunkRegions valley_carve.
        let coord = region::RegionCoord::containing(wx, wz);
        // Match the exact pattern from column_data: FINE_REGION_SIZE / 32
        let chunk_origin = ChunkCoord(glam::IVec3::new(
            coord.x * (FINE_REGION_SIZE / 32),
            0,
            coord.z * (FINE_REGION_SIZE / 32),
        ));
        let regions = self.gather_chunk_regions(chunk_origin);
        let valley_carve = regions.valley_carve(wx, wz, self.seed);

        // Climate noise at this column (no Voronoi jitter for probe — raw noise).
        let xz = [wx as f64, wz as f64];
        let temperature = self.temperature_map.get(xz) as f32;
        let humidity = self.humidity_map.get(xz) as f32;
        let weirdness = (self.weirdness_noise.get(xz) as f32) * cfg.biomes.weirdness_amplitude;

        // Flow accumulation: look up the fine region and read the cell.
        // Granularity is FINE_CELL blocks (not per-column); adjacent columns
        // inside the same cell share a flow_accum value.
        let fine_region =
            region::get_fine(&self.fine_cache, coord, || self.build_fine_region(coord));
        let flow_accum = {
            let (ox, oz) = coord.origin();
            let lx = wx - ox;
            let lz = wz - oz;
            let ix = (lx / FINE_CELL).clamp(0, FINE_REGION_SIZE / FINE_CELL - 1);
            let iz = (lz / FINE_CELL).clamp(0, FINE_REGION_SIZE / FINE_CELL - 1);
            let idx = region::FineRegion::cell_index(ix, iz);
            fine_region.flow_acc[idx]
        };
        let river_cell = regions.river_cell_at(wx, wz, self.seed);

        // Aquifer: the nearest cell at sea-level for this column. The
        // cell's `fluid` is always `Block::Water` or `Block::Lava` —
        // the per-voxel `Substance::Density|Block(_)` resolution is
        // unrelated and only matters during chunk fill.
        let acell = self.aquifer.cell_for_column(wx, wz);

        // Cave systems intersecting this column's XZ coords across the
        // 3×3 region neighbourhood.
        let cave_systems_count = {
            let mut n = 0usize;
            for row in &regions.grid {
                for slot in row {
                    if let Some(r) = slot {
                        for sys in &r.cave_systems {
                            // Check only the XZ plane — count systems
                            // that *might* touch this column regardless of Y.
                            if sys.bb_min.x <= wx
                                && wx <= sys.bb_max.x
                                && sys.bb_min.z <= wz
                                && wz <= sys.bb_max.z
                            {
                                n += 1;
                            }
                        }
                    }
                }
            }
            n
        };

        // Derive unified water_surface_y for the probe, applying the same
        // river-wins-over-lake/ocean priority as fill_chunk.
        let probe_water_surface_y = river_cell
            .filter(|r| r.surface_y > col.height)
            .map(|r| r.surface_y)
            .or(col.water_surface_y);
        probe::ColumnProbe {
            wx,
            wz,
            plate,
            continentalness,
            h_pre,
            valley_carve,
            h_target: col.height,
            is_cliff: col.is_cliff,
            slope,
            temperature,
            humidity,
            desertness: col.desertness,
            weirdness,
            biome: col.biome,
            flow_accum,
            river_water_y: river_cell.map(|cell| cell.surface_y),
            river_bed_y: river_cell.map(|cell| cell.bed_y),
            water_surface_y: probe_water_surface_y,
            aquifer_y_top: acell.y_top,
            aquifer_fluid: acell.fluid,
            cave_systems_count,
        }
    }

    /// Lean per-column snapshot for viz paint passes — strictly the
    /// fields the per-face paint hook reads, no aquifer / cave /
    /// hydrology fields. Cheap enough to populate for all 1024 columns
    /// in a chunk before meshing.
    pub fn paint_column(&self, wx: i32, wz: i32) -> probe::PaintColumn {
        let cfg = self.config.load();
        let col = self.column_data(wx, wz);
        let plate = plates::plate_at(self.seed, wx, wz);
        let h_pre =
            self.heightmap
                .h_pre(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density);
        let slope =
            self.heightmap
                .slope_at(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density);
        probe::PaintColumn {
            biome: col.biome,
            plate_id: plate.a.id,
            h_pre,
            h_target: col.height,
            slope,
        }
    }

    /// Return a single f32 scalar for the given `stage` at world column
    /// `(wx, wz)`. Used by the overlay map to colour each pixel.
    ///
    /// Each arm calls the *minimum* code needed for that stage — no arm
    /// routes through `probe_column` (which does the full pipeline).
    pub fn sample_stage(&self, stage: probe::Stage, wx: i32, wz: i32) -> f32 {
        use probe::Stage;
        match stage {
            Stage::Continentalness => {
                let cfg = self.config.load();
                let (c, _) = smooth_plate_contribution(self.seed, wx, wz, &cfg.climate);
                c
            }
            Stage::PlateId => {
                let plate = plates::plate_at(self.seed, wx, wz);
                // Hash the plate cell coords against the world seed for a
                // stable per-plate hue that is independent of spatial
                // position within the plate.
                // `plate.a` is the closest (primary) plate at this column.
                crate::worldgen::hash::mix_unit(self.seed, &[plate.a.id.cell_x, plate.a.id.cell_z])
            }
            Stage::Temperature => {
                let xz = [wx as f64, wz as f64];
                self.temperature_map.get(xz) as f32
            }
            Stage::Humidity => {
                let xz = [wx as f64, wz as f64];
                self.humidity_map.get(xz) as f32
            }
            Stage::Desertness => self.column_data(wx, wz).desertness,
            Stage::Weirdness => {
                let cfg = self.config.load();
                let xz = [wx as f64, wz as f64];
                (self.weirdness_noise.get(xz) as f32) * cfg.biomes.weirdness_amplitude
            }
            Stage::HPre => {
                let cfg = self.config.load();
                self.heightmap
                    .h_pre(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density)
            }
            Stage::ValleyCarve => {
                let coord = region::RegionCoord::containing(wx, wz);
                let chunk_origin = ChunkCoord(glam::IVec3::new(
                    coord.x * (FINE_REGION_SIZE / 32),
                    0,
                    coord.z * (FINE_REGION_SIZE / 32),
                ));
                let regions = self.gather_chunk_regions(chunk_origin);
                regions.valley_carve(wx, wz, self.seed)
            }
            Stage::HTarget => self.column_data(wx, wz).height as f32,
            Stage::FlowAccum => {
                let coord = region::RegionCoord::containing(wx, wz);
                let fine_region =
                    region::get_fine(&self.fine_cache, coord, || self.build_fine_region(coord));
                let (ox, oz) = coord.origin();
                let lx = wx - ox;
                let lz = wz - oz;
                let ix = (lx / FINE_CELL).clamp(0, FINE_REGION_SIZE / FINE_CELL - 1);
                let iz = (lz / FINE_CELL).clamp(0, FINE_REGION_SIZE / FINE_CELL - 1);
                let idx = region::FineRegion::cell_index(ix, iz);
                fine_region.flow_acc[idx] as f32
            }
            Stage::RiverWaterSurface => {
                let coord = region::RegionCoord::containing(wx, wz);
                let chunk_origin = ChunkCoord(glam::IVec3::new(
                    coord.x * (FINE_REGION_SIZE / 32),
                    0,
                    coord.z * (FINE_REGION_SIZE / 32),
                ));
                let regions = self.gather_chunk_regions(chunk_origin);
                regions
                    .river_cell_at(wx, wz, self.seed)
                    .map_or(0.0, |cell| cell.surface_y as f32)
            }
            Stage::RiverBed => {
                let coord = region::RegionCoord::containing(wx, wz);
                let chunk_origin = ChunkCoord(glam::IVec3::new(
                    coord.x * (FINE_REGION_SIZE / 32),
                    0,
                    coord.z * (FINE_REGION_SIZE / 32),
                ));
                let regions = self.gather_chunk_regions(chunk_origin);
                regions
                    .river_cell_at(wx, wz, self.seed)
                    .map_or(0.0, |cell| cell.bed_y as f32)
            }
            Stage::LakeRim => self
                .column_data(wx, wz)
                .water_surface_y
                .map_or(0.0, |y| y as f32),
            Stage::WaterSurfaceY => self
                .column_data(wx, wz)
                .water_surface_y
                .map_or(0.0, |y| y as f32),
            Stage::BiomeId => {
                let biome = self.column_data(wx, wz).biome;
                // Biome has no #[repr], so we use a hand-written mapping that
                // is stable across all variants.
                // Return a hash-derived value in [0, 1) rather than a raw
                // integer, so the categorical colormap's fract() normalization
                // produces a distinct hue per biome (same trick as PlateId).
                let idx: i32 = match biome {
                    Biome::Tundra => 0,
                    Biome::SnowyForest => 1,
                    Biome::Plains => 2,
                    Biome::Forest => 3,
                    Biome::Desert => 4,
                    Biome::Tropical => 5,
                };
                crate::worldgen::hash::mix_unit(self.seed, &[idx, 0xB10E5_u32 as i32])
            }
            Stage::AquiferY => {
                let acell = self.aquifer.cell_for_column(wx, wz);
                acell.y_top as f32
            }
            Stage::AquiferSubstance => {
                let acell = self.aquifer.cell_for_column(wx, wz);
                match acell.fluid {
                    Block::Lava => 1.0,
                    _ => 0.0,
                }
            }
        }
    }

    /// Per-voxel density decomposition. Returns each contribution
    /// separately plus the final composed value and resolved block,
    /// matching as closely as possible what `fill_chunk` produces for
    /// the same voxel.
    ///
    /// **Important divergence from `fill_chunk`:** `fill_chunk` uses a
    /// 9×9×9 corner lattice + trilerp (`CellEvaluator`) for the base
    /// density. This method evaluates the density graph **exactly** at
    /// `(wx, wy, wz)` (no trilerp). The two paths agree everywhere
    /// except near voxels right at the density=0 threshold inside a
    /// cell's interior, where the trilerp approximation can straddle
    /// the sign boundary differently. The block field will occasionally
    /// disagree with `fill_chunk` at those boundary voxels.
    ///
    /// The `depth_below_surface` tracker needed for surface-block
    /// selection is seeded by a top-down scan that uses the **exact**
    /// evaluator (same caveat applies).
    ///
    /// Used by the viz probe panel's "sliding y" section.
    pub fn evaluate_density_breakdown(&self, wx: i32, wy: i32, wz: i32) -> probe::DensityBreakdown {
        let cfg_arc = self.config_snapshot();
        let cfg = &*cfg_arc;

        // --- Column geometry ---
        let col = self.column_data(wx, wz);
        let height = col.height;

        // --- Gather regions and cave systems ---
        let coord = region::RegionCoord::containing(wx, wz);
        let chunk_origin = ChunkCoord(glam::IVec3::new(
            coord.x * (FINE_REGION_SIZE / 32),
            0,
            coord.z * (FINE_REGION_SIZE / 32),
        ));
        let regions = self.gather_chunk_regions(chunk_origin);
        // Collect cave systems whose bounding box contains (wx, wy, wz).
        // We use a generous Y range (±256) so systems above and below
        // the voxel (which can have chambers that extend) are included.
        let probe_min = glam::IVec3::new(wx, wy - 256, wz);
        let probe_max = glam::IVec3::new(wx, wy + 256, wz);
        let cave_systems = regions.cave_systems_intersecting(probe_min, probe_max);

        // --- Density graph: exact evaluation (no trilerp) ---
        let graph = density::build_default_tree(&cfg.climate, &cfg.density);
        let (cc, sc, rc, _) = self
            .heightmap
            .climate(self.seed, wx as f32, wz as f32, &cfg.climate);
        let climate = density::ColumnClimate {
            continentalness: cc,
            terrain_shape: sc,
            ridges_pv: rc,
        };

        // Raw pre-slide graph value, then apply slide.
        let pre_slide = graph.evaluate(wx, wy, wz, climate, &self.density, &cfg.density);
        let raw_density = slide(pre_slide, wy, &cfg.density);

        // Extract `bias` (y-gradient term) and `base_3d` (3D noise term)
        // for the breakdown. These match the leaves of `build_default_tree`:
        //   y_gradient = amp * (1 - 2*(wy - y_min)/(y_max - y_min))
        //   base_3d    = density.evaluate_base_3d(wx, wy, wz, cfg)
        let t = (wy - cfg.density.y_min) as f32 / (cfg.density.y_max - cfg.density.y_min) as f32;
        let bias = cfg.density.y_gradient_amplitude * (1.0 - 2.0 * t);
        let base_3d = self.density.evaluate_base_3d(wx, wy, wz, &cfg.density);

        // --- Cave contributions (matching fill_chunk gate logic) ---
        let approx_depth = height - wy;

        // Graph-cave SDF + entrance SDF. Chambers/trunks gated at wy <= height;
        // entrance SDF extended by SURFACE_BAND to match fill_chunk logic.
        let mut cave_sdf_val = 0.0_f32;
        if !cave_systems.is_empty() && wy > CAVE_FLOOR_Y && wy <= height {
            if approx_depth > CAVE_SURFACE_BUFFER {
                cave_sdf_val = cave_sdf_val.max(caves::cave_sdf(wx, wy, wz, &cave_systems));
                cave_sdf_val = cave_sdf_val.max(caves::trunks_sdf(
                    wx,
                    wy,
                    wz,
                    &cave_systems,
                    self.seed,
                    cfg.cave.trunk_r,
                    cfg.cave.trunk_prob,
                ));
            }
        }
        if !cave_systems.is_empty() && wy > CAVE_FLOOR_Y && wy <= height + SURFACE_BAND {
            cave_sdf_val = cave_sdf_val.max(caves::entrance_sdf(wx, wy, wz, &cave_systems));
        }
        // Identify which cave system (if any) the probe voxel sits inside,
        // for the probe panel's style / band display rows.
        let (probe_cave_style, probe_cave_band) = cave_systems
            .iter()
            .find(|sys| caves::cave_sdf(wx, wy, wz, &[sys]) > 0.0)
            .map(|sys| {
                let style_name: &'static str = match sys.style {
                    caves::CaveStyle::Cathedral => "Cathedral",
                    caves::CaveStyle::Warren => "Warren",
                    caves::CaveStyle::Slot => "Slot",
                    caves::CaveStyle::Sump => "Sump",
                    caves::CaveStyle::Karst => "Karst",
                };
                let cy = (sys.bb_min.y + sys.bb_max.y) / 2;
                let band: &'static str = if cy >= CAVE_BAND_SHALLOW.0 {
                    "shallow"
                } else if cy >= CAVE_BAND_MIDDLE.0 {
                    "middle"
                } else {
                    "deep"
                };
                (Some(style_name), Some(band))
            })
            .unwrap_or((None, None));

        // Noise carvers (cheese) — same gate. `cheese_contribution`
        // also takes `raw_density` (gates a density-aware cap).
        let cheese = if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
            caves::cheese_contribution(wx, wy, wz, raw_density, &self.noise_carvers, &cfg.cave)
        } else {
            0.0
        };
        let pillar = if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
            caves::pillar_contribution(wx, wy, wz, &self.noise_carvers, &cfg.cave)
        } else {
            0.0
        };
        // Terasology ambient carver — same surface buffer + floor gate.
        let probe_surface_y =
            self.heightmap
                .h_pre(self.seed, wx as f32, wz as f32, &cfg.climate, &cfg.density);
        let tera = if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
            caves::terasology_ambient(wx, wy, wz, &self.noise_carvers, &cfg.cave, probe_surface_y)
        } else {
            0.0
        };

        // Compose to a signed final density the same way fill_chunk
        // does: start from `raw_density`, then `smin()` in each cave
        // carver's signed contribution. Graph cave SDFs are positive
        // intensities, so they're applied as `smin(..., -sdf, k)`. `cheese` is
        // signed (includes the cave_layer² term). Pillars apply last
        // via `max()`.
        //
        // NB: the procedural carver (`carver.rs`) mask isn't included
        // here — it operates per-chunk and isn't cheap to query at
        // a single voxel. The probe is informative, not authoritative;
        // the chunk fill is the ground truth.
        let mut final_density = raw_density;
        if cave_sdf_val > 0.0 {
            final_density = caves::smin(final_density, -cave_sdf_val, cfg.cave.smin_k);
        }
        if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
            final_density = caves::smin(final_density, cheese, cfg.cave.smin_k);
        }
        if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
            final_density = caves::smin(final_density, tera, cfg.cave.smin_k);
        }
        if pillar > 0.0 {
            final_density = final_density.max(pillar);
        }
        let solid = final_density > 0.0;

        // --- Block resolution ---
        let mut fluid_reason = None;
        let block = if !solid {
            let river = regions.river_cell_at(wx, wz, self.seed);
            if let Some(cell) = river {
                if wy >= cell.bed_y && wy <= cell.surface_y {
                    fluid_reason = Some(cell.reason);
                    Block::Water
                } else {
                    Block::Air
                }
            } else if let Some(wsurf) = col.water_surface_y {
                // Unified water surface covers ocean, lake, and river-
                // flooded columns in order of priority.
                if wy >= height && wy <= wsurf {
                    fluid_reason = Some(fluid::FluidReason::OceanConnected);
                    Block::Water
                } else {
                    Block::Air
                }
            } else {
                Block::Air
            }
        } else {
            // Solid: need depth_below_surface for the surface rule.
            // Scan from above (h_target + a few blocks) down to wy,
            // counting consecutive solid blocks using the same exact
            // evaluator (not trilerp). This mirrors the fill_chunk
            // top-down scan within a single column.
            let scan_top = (height + 8).max(wy + 1);
            let mut depth_below_surface: Option<i32> = None;
            // Seed from one block above scan_top (mirrors fill_chunk's
            // "above_chunk_top" seeding, but applied per-voxel here).
            {
                let above_pre =
                    graph.evaluate(wx, scan_top, wz, climate, &self.density, &cfg.density);
                let above_density = slide(above_pre, scan_top, &cfg.density);
                if above_density > 0.0 {
                    depth_below_surface = Some(4);
                }
            }
            for scan_y in (wy..=scan_top - 1).rev() {
                let scan_pre = graph.evaluate(wx, scan_y, wz, climate, &self.density, &cfg.density);
                let scan_density = slide(scan_pre, scan_y, &cfg.density);
                // Cave carving at scan_y changes whether a voxel appears solid.
                let scan_approx_depth = height - scan_y;
                let mut scan_cave = 0.0_f32;
                if !cave_systems.is_empty() && scan_y > CAVE_FLOOR_Y && scan_y <= height {
                    if scan_approx_depth > CAVE_SURFACE_BUFFER {
                        scan_cave = scan_cave.max(caves::cave_sdf(wx, scan_y, wz, &cave_systems));
                        scan_cave = scan_cave.max(caves::trunks_sdf(
                            wx,
                            scan_y,
                            wz,
                            &cave_systems,
                            self.seed,
                            cfg.cave.trunk_r,
                            cfg.cave.trunk_prob,
                        ));
                    }
                    scan_cave = scan_cave.max(caves::entrance_sdf(wx, scan_y, wz, &cave_systems));
                }
                if scan_approx_depth > CAVE_SURFACE_BUFFER && scan_y > CAVE_FLOOR_Y {
                    scan_cave = scan_cave.max(caves::cheese_contribution(
                        wx,
                        scan_y,
                        wz,
                        scan_density,
                        &self.noise_carvers,
                        &cfg.cave,
                    ));
                }
                if scan_approx_depth > CAVE_SURFACE_BUFFER && scan_y > CAVE_FLOOR_Y {
                    let scan_tera = caves::terasology_ambient(
                        wx,
                        scan_y,
                        wz,
                        &self.noise_carvers,
                        &cfg.cave,
                        probe_surface_y,
                    );
                    scan_cave = scan_cave.max(scan_tera);
                }
                let scan_pillar =
                    if scan_approx_depth > CAVE_SURFACE_BUFFER && scan_y > CAVE_FLOOR_Y {
                        caves::pillar_contribution(wx, scan_y, wz, &self.noise_carvers, &cfg.cave)
                    } else {
                        0.0
                    };
                let scan_dfc = if scan_cave > 0.0 {
                    scan_density.min(1.0)
                } else {
                    scan_density
                };
                let scan_solid = (scan_dfc - scan_cave + scan_pillar) > 0.0;
                if scan_solid {
                    depth_below_surface = Some(depth_below_surface.map(|d| d + 1).unwrap_or(0));
                } else {
                    depth_below_surface = None;
                }
            }
            let depth = depth_below_surface.map(|d| d + 1).unwrap_or(0);
            let surf_ctx = surface::SurfaceContext {
                wx,
                wy,
                wz,
                h_target: height,
                biome: col.biome,
                is_cliff: col.is_cliff,
                desertness: col.desertness,
                depth_below_surface: depth,
                water_surface_y: col.water_surface_y,
                seed: self.seed,
                cfg: &cfg,
                sea_level: SEA_LEVEL,
            };
            cfg.surface.apply(&surf_ctx).unwrap_or(Block::Stone)
        };

        probe::DensityBreakdown {
            wx,
            wy,
            wz,
            bias,
            base_3d,
            cave_sdf: cave_sdf_val,
            cheese,
            tera,
            pillar,
            final_density,
            block,
            fluid_reason,
            cave_style: probe_cave_style,
            cave_band: probe_cave_band,
        }
    }
}
