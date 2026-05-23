//! Chunk fill pipeline — [`Generator::fill_chunk`] and [`Generator::light_inputs_for_chunk`].
//!
//! Lives in a separate file from `mod.rs` so the chunk-generation algorithm
//! can be read end-to-end without scrolling past construction, region
//! helpers, or debug tooling. All Generator fields are `pub(crate)`, and
//! private helper methods (`gather_chunk_regions`, `column_data_with`,
//! `build_carver_mask`, `add_trees`) are accessible because child modules
//! can see parent-module private items in Rust.
//!
//! See `docs/book/content/part-4-chunk-fill/` for the full algorithm walk-
//! through and `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`
//! §4 for the region-prefetch design.

use super::Generator;
use crate::voxel::block::Block;
use crate::voxel::chunk::{ChunkLightInputs, DenseChunk};
use crate::voxel::coords::{CHUNK_DIM_U, ChunkCoord, LocalPos};
use crate::worldgen::density::math::slide;
use crate::worldgen::surface_fixer;
use crate::worldgen::tuning::{
    CAVE_FLOOR_Y, CAVE_SDF_INTENSITY, CAVE_SURFACE_BUFFER, SEA_LEVEL, SURFACE_BAND,
};
use crate::worldgen::{caves, density, fluid, surface};
use glam::UVec3;

impl Generator {
    /// Generate `coord`'s contents into `out`. Pure with respect to
    /// `(seed, coord)`.
    pub fn fill_chunk(&self, coord: ChunkCoord, out: &mut DenseChunk) {
        let origin = coord.origin().0;
        // Pre-fetch the regions overlapping this chunk plus their
        // neighbour halos. A chunk (32 blocks) is smaller than a
        // region (512 blocks), so it touches at most 4 distinct
        // regions; we cover the worst case by fetching the regions
        // containing each chunk corner and union-ing their
        // neighbour rings. Doing this up-front means the per-column
        // `column_data` path is just cheap noise evaluation and
        // already-cached region reads — no mutex traffic per column.
        let regions = self.gather_chunk_regions(coord);
        // Pre-collect every cave system whose bounding box intersects
        // this chunk so the per-cell cave SDF query iterates a short
        // list rather than walking the full region's system list.
        let chunk_max = origin + glam::IVec3::splat(CHUNK_DIM_U as i32);
        let cave_systems = regions.cave_systems_intersecting(origin, chunk_max);
        let cave_pools = regions.cave_pools_intersecting(origin, chunk_max);
        // Snapshot the hot-reloadable config once at the top of this
        // chunk and reuse for every voxel — keeps each chunk
        // deterministic even if a file watcher swaps mid-generation.
        let cfg = self.config_snapshot();

        // Procedural carver: rasterise nearby chunks' tunnels into
        // a per-chunk boolean mask once. Per-voxel test is then O(1)
        // mask lookup. See `carver.rs`.
        let carver_mask = self.build_carver_mask(coord);
        let dim = CHUNK_DIM_U as usize;

        // PR 5: cell-grid evaluator. Builds a 9x9x9 corner lattice
        // of pre-slide density values for this chunk; per-voxel
        // density is the trilerp of the 8 surrounding corners. ~730
        // expensive density evaluations per chunk instead of 32768
        // (≈45× speedup on the per-voxel hot path).
        let graph = density::build_default_tree(&cfg.climate, &cfg.density);
        let evaluator = density::CellEvaluator::new(
            &graph,
            &self.density,
            &cfg.density,
            (origin.x, origin.y, origin.z),
            |wx, wz| {
                let (c, s, pv, _) =
                    self.heightmap
                        .climate(self.seed, wx as f32, wz as f32, &cfg.climate);
                density::ColumnClimate {
                    continentalness: c,
                    terrain_shape: s,
                    ridges_pv: pv,
                }
            },
        );
        // Same trick for the noise-carver layers: build a 9³ corner
        // lattice once and trilerp per voxel. Roughly 13 FBM samples
        // per voxel become 13 per corner — a ~45× reduction in
        // Simplex calls inside the inner loop.
        let carver_eval = caves::CarverEvaluator::new(&self.noise_carvers, &cfg.cave, origin);

        // Precompute valley-carve depths for all 32×32 columns in one
        // segment-first pass. AABB culling means only columns actually
        // within a river valley pay the perpendicular_distance cost.
        // This eliminates the O(1024 × N_segments) per-column call to
        // valley_carve, replacing it with O(N_segments × affected_columns).
        let valley_depth_grid = regions.valley_grid(origin.x, origin.z, self.seed);
        let river_grid = regions.river_grid(origin.x, origin.z, self.seed);
        let mut columns = Vec::with_capacity(dim * dim);
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let wx = origin.x + x as i32;
                let wz = origin.z + z as i32;
                columns.push(self.column_data_with(
                    wx,
                    wz,
                    &regions,
                    Some(valley_depth_grid[z as usize][x as usize]),
                ));
            }
        }
        // Patch river water_surface_y. Rivers are authoritative: they override
        // lake/ocean regardless of terrain height. The river grid is only
        // available after `valley_depth_grid` is built, so this runs after the
        // columns loop rather than inside `column_data_with`.
        for z in 0..CHUNK_DIM_U as usize {
            for x in 0..CHUNK_DIM_U as usize {
                let idx = z * dim + x;
                if let Some(river) = river_grid[idx] {
                    columns[idx].water_surface_y = Some(river.surface_y);
                }
            }
        }

        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let wx = origin.x + x as i32;
                let wz = origin.z + z as i32;
                let col = columns[z as usize * dim + x as usize];
                let height = col.height;
                // h_pre is the pre-carve surface Y, used by the tera
                // surface-suppression depth term. Computed once per
                // XZ column so the inner y-loop pays no noise cost.
                let surface_y = col.h_pre;

                // PR A: density-based top-down scan. The "surface" is
                // wherever density transitions from negative (air) to
                // positive (solid) — found block-by-block, not at a
                // fixed `height`. State across the y-loop:
                //   * `depth_below_surface` = None when the current
                //     voxel is air; Some(d) when we're `d` blocks
                //     into solid after the most recent air→solid
                //     transition (so depth=0 is the topmost solid
                //     block of an exposed surface).
                // PR 5: seed `depth_below_surface` from the voxel one
                // above the chunk's top via the cell evaluator's
                // *non-interpolated* boundary corner. The chunk
                // boundary itself is a corner so we evaluate the
                // graph directly there (the trilerp would otherwise
                // need an extra corner row outside the chunk).
                let above_chunk_top_wy = origin.y + CHUNK_DIM_U as i32;
                let above_climate = {
                    let (c, s, pv, _) =
                        self.heightmap
                            .climate(self.seed, wx as f32, wz as f32, &cfg.climate);
                    density::ColumnClimate {
                        continentalness: c,
                        terrain_shape: s,
                        ridges_pv: pv,
                    }
                };
                let above_density_pre_slide = graph.evaluate(
                    wx,
                    above_chunk_top_wy,
                    wz,
                    above_climate,
                    &self.density,
                    &cfg.density,
                );
                let above_density =
                    slide(above_density_pre_slide, above_chunk_top_wy, &cfg.density);
                let mut depth_below_surface: Option<i32> =
                    if above_density > 0.0 { Some(4) } else { None };
                for y in (0..CHUNK_DIM_U).rev() {
                    let wy = origin.y + y as i32;
                    let local = LocalPos(UVec3::new(x, y, z));

                    let approx_depth = height - wy;
                    // PR 5: density via cell-grid corner sampling +
                    // trilerp. Slide is post-interp (it's cheap and
                    // varies per-voxel-y; baking it into corners
                    // would interact poorly with the trilerp at the
                    // slide boundary).
                    let raw_density = slide(evaluator.evaluate(wx, wy, wz), wy, &cfg.density);

                    // Signed-density cave composition.
                    //
                    // Each cave layer returns a *signed* density —
                    // negative values bias toward air, positive
                    // toward solid. We start from `raw_density` (the
                    // base terrain density) and pull it down via
                    // `min(...)` whenever any cave layer goes
                    // negative. A `max(..., pillars)` at the end
                    // refills carved voxels where pillars apply.
                    //
                    // Layers, in order applied:
                    //   1. Graph cave SDF        (negated → signed)
                    //   2. Graph trunks SDF      (negated → signed)
                    //   3. Graph entrance SDF    (negated → signed)
                    //   4. Cheese                (signed, surface-suppressed)
                    //   5. Terasology ambient    (signed, depth-driven 2-noise)
                    //   6. MC carver mask        (hard carve)
                    //   7. Pillars               (positive, refill stone)
                    //
                    // Spaghetti + cheese only run above the
                    // `underground_density_threshold` — below that
                    // (the surface band) only the graph layers
                    // carve, preserving the heightmap cap except at
                    // deliberate entrances.

                    let mut composed = raw_density;

                    // Graph carvers: SDF is positive in [0, intensity].
                    // Negate and smin so a positive SDF pulls density
                    // toward (or below) zero. smin(k>0) additionally
                    // blends nearly-touching cave volumes together.
                    if !cave_systems.is_empty()
                        && wy > CAVE_FLOOR_Y
                        && wy <= height
                        && approx_depth > CAVE_SURFACE_BUFFER
                    {
                        let sdf = caves::cave_sdf(wx, wy, wz, &cave_systems);
                        if sdf > 0.0 {
                            composed = caves::smin(composed, -sdf, cfg.cave.smin_k);
                        }
                        let trunk_sdf = caves::trunks_sdf(
                            wx,
                            wy,
                            wz,
                            &cave_systems,
                            self.seed,
                            cfg.cave.trunk_r,
                            cfg.cave.trunk_prob,
                        );
                        if trunk_sdf > 0.0 {
                            composed = caves::smin(composed, -trunk_sdf, cfg.cave.smin_k);
                        }
                    }
                    // Entrance SDF (sinkholes, skylights, cliff mouths) gets its
                    // own gate extended by SURFACE_BAND so the shaft carves through
                    // any 3D-density bump above h_pre and doesn't leave floating
                    // terrain islands over the entrance opening.
                    if !cave_systems.is_empty() && wy > CAVE_FLOOR_Y && wy <= height + SURFACE_BAND
                    {
                        let ent = caves::entrance_sdf(wx, wy, wz, &cave_systems);
                        if ent > 0.0 {
                            composed = caves::smin(composed, -ent, cfg.cave.smin_k);
                        }
                    }
                    // Noise carvers: only deeper than the underground
                    // density threshold.
                    if approx_depth > CAVE_SURFACE_BUFFER
                        && wy > CAVE_FLOOR_Y
                        && raw_density >= cfg.cave.underground_density_threshold
                    {
                        let cheese = carver_eval.cheese_at(wx, wy, wz, raw_density, &cfg.cave);
                        composed = caves::smin(composed, cheese, cfg.cave.smin_k);
                    }

                    // Terasology ambient carver: depth-driven 2-noise
                    // cave layer. Same surface buffer + floor gate as
                    // cheese.
                    // Tera intentionally skips the underground_density_threshold gate;
                    // its own freq_reduction provides surface suppression.
                    if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
                        let tera =
                            carver_eval.terasology_ambient_at(wx, wy, wz, &cfg.cave, surface_y);
                        composed = caves::smin(composed, tera, cfg.cave.smin_k);
                    }

                    // MC-style procedural carver mask. Hard carve to
                    // air; pillars below can refill if present
                    // (matches MC's `MAX(..., pillars)` semantics).
                    if wy > CAVE_FLOOR_Y {
                        let lx = (wx - origin.x) as usize;
                        let ly = (wy - origin.y) as usize;
                        let lz = (wz - origin.z) as usize;
                        let idx = lx + dim * ly + dim * dim * lz;
                        if carver_mask[idx] {
                            composed = caves::smin(composed, -CAVE_SDF_INTENSITY, cfg.cave.smin_k);
                        }
                    }

                    // Pillars: positive density component refilling
                    // any carved voxel where pillars are present.
                    if approx_depth > CAVE_SURFACE_BUFFER && wy > CAVE_FLOOR_Y {
                        let pillar = carver_eval.pillar_at(wx, wy, wz, &cfg.cave);
                        if pillar > 0.0 {
                            composed = composed.max(pillar);
                        }
                    }

                    let solid = composed > 0.0;

                    // Force-flood: voxels above the terrain floor and at or
                    // below water_surface_y must be Air so apply_surface_fluids
                    // can stamp Water into them. This eliminates "3D bumps
                    // through water" and "covered rivers / lakes" by removing
                    // any solid density that density evaluation placed inside
                    // the intended water column.
                    if let Some(wsurf) = col.water_surface_y
                        && wy > col.height
                        && wy <= wsurf
                    {
                        depth_below_surface = None;
                        out.set(local, Block::Air);
                        continue;
                    }

                    let block = if !solid {
                        depth_below_surface = None;
                        Block::Air
                    } else {
                        // Solid — `depth` counts blocks below the most
                        // recent air→solid transition. PR 6 delegates
                        // the surface/sub-surface block decision to
                        // the data-driven rule tree
                        // (`SurfaceSystem::surface_block`); the
                        // `WithinSurfaceBand` condition replaces the
                        // pre-PR-6 inline near_surface gate.
                        let depth = depth_below_surface.map(|d| d + 1).unwrap_or(0);
                        depth_below_surface = Some(depth);
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
                    out.set(local, block);
                }
            }
        }

        let fluid_planner = fluid::FluidPlanner::new(self.seed, coord);
        fluid_planner.apply_surface_fluids(out, &columns);
        fluid_planner.apply_cave_pools(out, &cave_pools);

        // Vertical-run clamp + surface block fixer: cap air shafts,
        // remove cave-ceiling grass artefacts, and stamp the climate-correct
        // surface block on cave floors that breach the heightmap.
        // All three passes are in `surface_fixer` to keep this function
        // focused on the primary generation sequence.
        surface_fixer::apply_surface_post_passes(out, &columns, origin, &cfg, self.seed);

        // After the terrain pass, lay trees on top. Cross-chunk trees
        // (whose trunks live in a neighbouring chunk but whose leaves
        // overlap this one) are placed too, because we scan every
        // cell in a `TREE_MARGIN`-block ring around the chunk.
        self.add_trees(coord, out, &regions);
    }

    /// Compute the light-input surface map for `coord`. Called by the
    /// lighting engine to determine the topmost-opaque-block Y per XZ
    /// column, used to initialize sky light.
    pub fn light_inputs_for_chunk(
        &self,
        coord: ChunkCoord,
        dense: &DenseChunk,
        registry: &crate::voxel::block::BlockRegistry,
    ) -> ChunkLightInputs {
        let origin = coord.origin().0;
        let regions = self.gather_chunk_regions(coord);
        let valley_depth_grid = regions.valley_grid(origin.x, origin.z, self.seed);
        let dim = CHUNK_DIM_U as usize;
        let mut surfaces = [0i32; 32 * 32];
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let wx = origin.x + x as i32;
                let wz = origin.z + z as i32;
                let col = self.column_data_with(
                    wx,
                    wz,
                    &regions,
                    Some(valley_depth_grid[z as usize][x as usize]),
                );
                surfaces[z as usize * dim + x as usize] = col.height;
            }
        }

        ChunkLightInputs::from_dense_with_surface(dense, coord, registry, |x, z| {
            Some(surfaces[z as usize * dim + x as usize])
        })
    }
}
