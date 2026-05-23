//! Generator tree-placement methods.
//!
//! These three methods (`add_trees`, `tree_in_cell_with_regions`,
//! `stamp_tree`) are part of [`Generator`]'s inherent impl but live here
//! so the chunk-fill file stays focused on terrain carving. The static
//! helpers (`Tree` struct, `tree_hash`, `try_set_air`) stay in `trees.rs`.
//!
//! Tree placement is deterministic in `(seed, cell_x, cell_z)`:
//! every chunk that overlaps a tree's blocks will write the same voxels,
//! so there are no cross-chunk-boundary seams or missing leaves.

use super::Generator;
use crate::voxel::block::Block;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::ChunkCoord;
use crate::worldgen::biome::TreeKind;
use crate::worldgen::pipeline::ChunkRegions;
use crate::worldgen::trees::{Tree, tree_hash, try_set_air};
use crate::worldgen::tuning::{SEA_LEVEL, SNOW_LINE, SURFACE_BAND, TREE_CELL_SIZE, TREE_MARGIN};

impl Generator {
    // ── §6 Tree placement ─────────────────────────────────────────────────
    // These methods stay in mod.rs because they access private Generator
    // fields (self.seed, self.heightmap, self.density). Free-standing
    // helpers (Tree struct, try_set_air, tree_hash) live in trees.rs.

    /// Place all trees whose blocks could overlap `coord`'s chunk
    /// volume. Each tree is deterministic in `(seed, cell_x, cell_z)`,
    /// so every chunk that touches the tree writes the same blocks —
    /// no double-placement and no missing slices at chunk boundaries.
    /// Called from `fill_chunk_impl` — visible as `pub(crate)` so sibling
    /// child modules of `worldgen` can call it.
    pub(crate) fn add_trees(
        &self,
        coord: ChunkCoord,
        out: &mut DenseChunk,
        regions: &ChunkRegions,
    ) {
        let chunk_origin = coord.origin().0;
        let cmin = chunk_origin;
        let cmax = chunk_origin + glam::IVec3::splat(crate::voxel::coords::CHUNK_DIM);
        // Cells whose interior could spill into the chunk's extended
        // bounds, allowing for tree-block radius around the cell.
        let xmin = cmin.x - TREE_MARGIN;
        let xmax = cmax.x + TREE_MARGIN;
        let zmin = cmin.z - TREE_MARGIN;
        let zmax = cmax.z + TREE_MARGIN;
        let cell_xmin = xmin.div_euclid(TREE_CELL_SIZE);
        let cell_xmax = (xmax - 1).div_euclid(TREE_CELL_SIZE);
        let cell_zmin = zmin.div_euclid(TREE_CELL_SIZE);
        let cell_zmax = (zmax - 1).div_euclid(TREE_CELL_SIZE);
        for cell_x in cell_xmin..=cell_xmax {
            for cell_z in cell_zmin..=cell_zmax {
                if let Some(tree) = self.tree_in_cell_with_regions(cell_x, cell_z, regions) {
                    self.stamp_tree(tree, coord, out);
                }
            }
        }
    }

    /// Return the tree (if any) belonging to the `(cell_x, cell_z)` tree
    /// cell. Determined entirely by `(seed, cell coords)` so adjacent
    /// chunks agree on which trees exist.
    fn tree_in_cell_with_regions(
        &self,
        cell_x: i32,
        cell_z: i32,
        regions: &ChunkRegions,
    ) -> Option<Tree> {
        let wx = cell_x * TREE_CELL_SIZE + (tree_hash(self.seed, cell_x, cell_z, 1) % 6) as i32 + 1;
        let wz = cell_z * TREE_CELL_SIZE + (tree_hash(self.seed, cell_x, cell_z, 2) % 6) as i32 + 1;
        let col = self.column_data_with(wx, wz, regions, None);

        // Trees don't grow on cliffs (bare stone), above the alpine
        // snow line, or where the column is submerged under a lake.
        if col.is_cliff {
            return None;
        }
        if col.height >= SNOW_LINE {
            return None;
        }
        // Lake veto: if this column sits below a lake's water surface,
        // no tree (even palms can't grow underwater).
        // Water veto: no trees in submerged columns (ocean, lake, river).
        if col.water_surface_y.is_some() {
            return None;
        }

        // Sand-surface veto. The beach band runs `[SEA_LEVEL - 1,
        // SEA_LEVEL + 2]` and surface material in that band is Sand;
        // oaks don't grow on sand. Palms *do*, but rarely — they're
        // the iconic tropical-beach silhouette.
        let on_beach = col.height >= SEA_LEVEL - 1 && col.height <= SEA_LEVEL + 2;
        let kind = col.biome.tree_kind();
        if on_beach && kind != TreeKind::Palm {
            return None;
        }

        let rate = col.biome.tree_rate_percentile()?;
        // Palms on the beach are rarer — divide their rate by 4 so
        // tropical beaches read as scattered palms, not dense palm
        // forests.
        let effective_rate = if on_beach { rate / 4 } else { rate };
        let roll = tree_hash(self.seed, cell_x, cell_z, 0) % 100;
        if roll >= effective_rate {
            return None;
        }

        // Find the actual topmost solid block for this column. With
        // 3D density the surface can sit up to ±SURFACE_BAND from
        // `col.height`; use a top-down density walk so the tree's
        // trunk lands on the real surface, not the heightmap target.
        let cfg = self.config_snapshot();
        // Sample the climate triple at this column so the topmost-solid
        // search uses the same spline outputs the chunk fill does.
        let (cc, sc, rc, _) = self
            .heightmap
            .climate(self.seed, wx as f32, wz as f32, &cfg.climate);
        let offset = cfg.climate.offset_spline.evaluate(cc, sc, rc);
        let factor = cfg.climate.factor_spline.evaluate(cc, sc, rc);
        let jagged = cfg.climate.jaggedness_spline.evaluate(cc, sc, rc);
        let height = self
            .density
            .topmost_solid(
                col.height as f32,
                wx,
                wz,
                col.height + SURFACE_BAND + 2,
                offset,
                factor,
                jagged,
                &cfg.density,
            )
            .unwrap_or(col.height);
        let trunk_h = match kind {
            TreeKind::Oak => 4 + (tree_hash(self.seed, cell_x, cell_z, 3) % 3) as i32,
            TreeKind::Palm => 7 + (tree_hash(self.seed, cell_x, cell_z, 3) % 3) as i32,
        };
        Some(Tree {
            wx,
            wz,
            base_y: height,
            trunk_h,
            kind,
        })
    }

    /// Write the trunk + leaf blocks of `tree` into `out`. Blocks whose
    /// world coordinates fall outside this chunk are silently ignored
    /// (the neighbouring chunk's call to `stamp_tree` writes them
    /// instead). Existing non-air voxels are preserved so the trunk
    /// doesn't carve through hills.
    fn stamp_tree(&self, tree: Tree, coord: ChunkCoord, out: &mut DenseChunk) {
        // Trunk: vertical column of Wood blocks above the surface.
        for dy in 1..=tree.trunk_h {
            try_set_air(coord, out, tree.wx, tree.base_y + dy, tree.wz, Block::Wood);
        }
        let top_y = tree.base_y + tree.trunk_h;
        match tree.kind {
            TreeKind::Oak => {
                // Round canopy. Radius² uses 6 so corner cells drop
                // out and the silhouette stays roughly spherical.
                for dy in -1..=2 {
                    for dz in -2..=2 {
                        for dx in -2..=2 {
                            let r2 = dx * dx + dy * dy + dz * dz;
                            if r2 > 6 {
                                continue;
                            }
                            try_set_air(
                                coord,
                                out,
                                tree.wx + dx,
                                top_y + dy,
                                tree.wz + dz,
                                Block::Leaves,
                            );
                        }
                    }
                }
            }
            TreeKind::Palm => {
                // Spreading-fronds canopy: 4–6 horizontal arms, one
                // block thick, radiating from the trunk top. Each arm
                // is a straight line of 3 blocks; a small +1y cap
                // sits at the centre.
                try_set_air(coord, out, tree.wx, top_y + 1, tree.wz, Block::Leaves);
                let arm_count = 5; // five fronds, evenly spaced
                for a in 0..arm_count {
                    let theta = a as f32 * std::f32::consts::TAU / arm_count as f32;
                    for step in 1..=3i32 {
                        let dx = (theta.cos() * step as f32).round() as i32;
                        let dz = (theta.sin() * step as f32).round() as i32;
                        // Fronds droop: outer tip is 1 block lower
                        // than the trunk top.
                        let dy = if step >= 3 { -1 } else { 0 };
                        try_set_air(
                            coord,
                            out,
                            tree.wx + dx,
                            top_y + dy,
                            tree.wz + dz,
                            Block::Leaves,
                        );
                    }
                }
            }
        }
    }
}
