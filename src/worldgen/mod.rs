//! Procedural world generation: a pure `(seed, ChunkCoord) -> DenseChunk` map.
//!
//! "Pure" matters: terrain output must depend only on the seed and chunk
//! coordinate so chunks can be regenerated from disk-free state and so unit
//! tests can pin output with golden hashes.
//!
//! The v0 algorithm is intentionally simple:
//!
//! 1. **Heightmap.** 2D fractal Brownian motion gives each `(x, z)` column a
//!    surface height in `[BASE - AMPL, BASE + AMPL]` blocks.
//! 2. **Layers.** Top block is grass (or sand near water); the next three
//!    are dirt; everything below is stone.
//! 3. **Caves.** A 3D fractal-Simplex noise sample > `CAVE_THRESH` (in
//!    absolute value) carves the block into air. Caves only carve below the
//!    top 3 blocks of dirt so the surface stays intact.
//! 4. **Sea level.** Any air at or below `SEA_LEVEL` becomes water.
//!
//! Adding biomes/structures (caves of differing styles, ore veins, trees)
//! happens in v0.2 by layering more passes on top of this baseline.

use crate::voxel::block::Block;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{ChunkCoord, LocalPos, CHUNK_DIM_U};
use glam::UVec3;
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

/// World-space Y at which the sea surface sits. Blocks above this with no
/// solid above turn into air; air below this turns into water.
pub const SEA_LEVEL: i32 = 62;
/// Mean terrain height above world `Y = 0`.
const BASE_HEIGHT: f32 = 64.0;
/// Peak-to-peak amplitude of the heightmap (a column can be `BASE ± AMPL`).
const AMPLITUDE: f32 = 24.0;
/// Absolute-value threshold above which the 3D cave noise carves out a block.
/// Lower values → more cave; higher values → fewer / smaller caves.
const CAVE_THRESH: f64 = 0.55;

/// World is partitioned into `CELL_SIZE × CELL_SIZE` (XZ) tree cells.
/// Each cell rolls a deterministic hash to decide whether it contains a
/// tree (and where in the cell). 8 blocks per cell + ~35 % spawn rate
/// gives a forest density of roughly one tree per 180 blocks² — enough
/// that hills look wooded without filling every meadow.
const TREE_CELL_SIZE: i32 = 8;
/// Maximum world-space radius (XZ + Y above surface) a tree's blocks can
/// occupy. Used to decide which neighbouring tree cells could spill
/// blocks into the chunk currently being generated.
const TREE_MARGIN: i32 = 5;

/// Pre-built noise fields for one world seed.
///
/// The struct exists mainly so the noise fields are constructed *once*: the
/// `Fbm` builder is comparatively expensive, and chunk generation calls
/// `get` thousands of times per chunk.
pub struct Generator {
    height_noise: Fbm<Simplex>,
    cave_noise: Fbm<Simplex>,
    seed: u64,
}

impl Generator {
    /// Build a `Generator` with the given world seed.
    ///
    /// `height_noise` and `cave_noise` are seeded with slightly different
    /// seeds (the cave seed is `seed + 1`) so they don't produce correlated
    /// patterns.
    pub fn new(seed: u64) -> Self {
        // Heightmap noise: 4 octaves, ~96-block period at octave 0. Persistence
        // 0.5 means each successive octave contributes half as much amplitude.
        let height_noise = Fbm::<Simplex>::new(seed as u32)
            .set_octaves(4)
            .set_frequency(1.0 / 96.0)
            .set_persistence(0.5);
        // Cave noise: tighter frequency (24-block period) gives twistier
        // caves; slightly higher persistence keeps mid-frequency detail.
        let cave_noise = Fbm::<Simplex>::new(seed.wrapping_add(1) as u32)
            .set_octaves(3)
            .set_frequency(1.0 / 24.0)
            .set_persistence(0.55);
        Self {
            height_noise,
            cave_noise,
            seed,
        }
    }

    /// Return the world seed this generator was constructed with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Generate `coord`'s contents into `out`. Pure with respect to
    /// `(seed, coord)`.
    pub fn fill_chunk(&self, coord: ChunkCoord, out: &mut DenseChunk) {
        let origin = coord.origin().0;
        for z in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let wx = origin.x + x as i32;
                let wz = origin.z + z as i32;
                let h_val = self.height_noise.get([wx as f64, wz as f64]) as f32;
                let height = (BASE_HEIGHT + h_val * AMPLITUDE) as i32;

                for y in 0..CHUNK_DIM_U {
                    let wy = origin.y + y as i32;
                    let local = LocalPos(UVec3::new(x, y, z));

                    let block = if wy > height {
                        // Above the terrain surface.
                        if wy <= SEA_LEVEL {
                            Block::Water
                        } else {
                            Block::Air
                        }
                    } else {
                        // Below or at the surface.
                        let depth = height - wy;
                        // Caves: only carve below the top 3 dirt rows so the
                        // surface look stays intact. Air above sea level;
                        // water below it (flood the cave).
                        let cave = depth > 3
                            && self
                                .cave_noise
                                .get([wx as f64, wy as f64, wz as f64])
                                .abs()
                                > CAVE_THRESH;
                        if cave {
                            if wy <= SEA_LEVEL {
                                Block::Water
                            } else {
                                Block::Air
                            }
                        } else if depth == 0 {
                            // Surface block: grass everywhere unless we're
                            // right at or below sea level — then sand.
                            if height <= SEA_LEVEL + 1 {
                                Block::Sand
                            } else {
                                Block::Grass
                            }
                        } else if depth <= 3 {
                            Block::Dirt
                        } else {
                            Block::Stone
                        }
                    };
                    out.set(local, block);
                }
            }
        }
        // After the terrain pass, lay trees on top. Cross-chunk trees
        // (whose trunks live in a neighbouring chunk but whose leaves
        // overlap this one) are placed too, because we scan every
        // cell in a `TREE_MARGIN`-block ring around the chunk.
        self.add_trees(coord, out);
    }

    /// Re-derive the height noise's vertical pick for a single column.
    /// Cheaper than running `fill_chunk` when all we need is a surface y.
    fn column_height(&self, wx: i32, wz: i32) -> i32 {
        let h = self.height_noise.get([wx as f64, wz as f64]) as f32;
        (BASE_HEIGHT + h * AMPLITUDE) as i32
    }

    /// Place all trees whose blocks could overlap `coord`'s chunk
    /// volume. Each tree is deterministic in `(seed, cell_x, cell_z)`,
    /// so every chunk that touches the tree writes the same blocks —
    /// no double-placement and no missing slices at chunk boundaries.
    fn add_trees(&self, coord: ChunkCoord, out: &mut DenseChunk) {
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
                if let Some(tree) = self.tree_in_cell(cell_x, cell_z) {
                    self.stamp_tree(tree, coord, out);
                }
            }
        }
    }

    /// Return the tree (if any) belonging to the `(cell_x, cell_z)` tree
    /// cell. Determined entirely by `(seed, cell coords)` so adjacent
    /// chunks agree on which trees exist.
    fn tree_in_cell(&self, cell_x: i32, cell_z: i32) -> Option<Tree> {
        // Roll #0: does this cell have a tree at all?
        let roll = tree_hash(self.seed, cell_x, cell_z, 0) % 100;
        if roll < 65 {
            return None;
        }
        // Roll #1, #2: tree's XZ offset inside the cell. Inset by 1 so
        // the trunk never lands exactly on a cell boundary.
        let off_x = (tree_hash(self.seed, cell_x, cell_z, 1) % 6) as i32 + 1;
        let off_z = (tree_hash(self.seed, cell_x, cell_z, 2) % 6) as i32 + 1;
        let wx = cell_x * TREE_CELL_SIZE + off_x;
        let wz = cell_z * TREE_CELL_SIZE + off_z;
        // Trees only grow on grass — above sea level and not on
        // sand-tipped islands. (height <= SEA_LEVEL produces sand.)
        let height = self.column_height(wx, wz);
        if height <= SEA_LEVEL + 1 {
            return None;
        }
        // Roll #3: trunk height in 4..=6 blocks.
        let trunk_h = 4 + (tree_hash(self.seed, cell_x, cell_z, 3) % 3) as i32;
        Some(Tree {
            wx,
            wz,
            base_y: height,
            trunk_h,
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
        // Leaves: a thick disc + slight cap around the top of the trunk.
        let top_y = tree.base_y + tree.trunk_h;
        // Round canopy. Radius² uses 6 so the corner cells are dropped
        // and the silhouette stays roughly spherical instead of cubic.
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
}

/// Tree placement metadata for one cell.
#[derive(Debug, Clone, Copy)]
struct Tree {
    /// World-space X coordinate of the trunk.
    wx: i32,
    /// World-space Z coordinate of the trunk.
    wz: i32,
    /// World-space Y of the surface block under the trunk (the trunk
    /// itself starts at `base_y + 1`).
    base_y: i32,
    /// Number of Wood blocks above the surface, inclusive.
    trunk_h: i32,
}

/// Write `b` at world coords `(wx, wy, wz)` if they fall inside
/// `coord`'s 32³ volume *and* the existing block is Air. Both
/// conditions are required so a tree's trunk doesn't cut through
/// hills and adjacent chunks' calls don't overwrite each other.
fn try_set_air(
    coord: ChunkCoord,
    out: &mut DenseChunk,
    wx: i32,
    wy: i32,
    wz: i32,
    b: Block,
) {
    use crate::voxel::coords::CHUNK_DIM;
    let chunk_origin = coord.origin().0;
    let lx = wx - chunk_origin.x;
    let ly = wy - chunk_origin.y;
    let lz = wz - chunk_origin.z;
    if lx < 0 || ly < 0 || lz < 0 || lx >= CHUNK_DIM || ly >= CHUNK_DIM || lz >= CHUNK_DIM {
        return;
    }
    let lp = LocalPos(UVec3::new(lx as u32, ly as u32, lz as u32));
    if out.get(lp) != Block::Air {
        return;
    }
    out.set(lp, b);
}

/// Deterministic mixer: `(seed, x, z, salt) → u32`. Uses the same
/// xor-shift / golden-ratio multiply pattern as the Wang/Mix hashes
/// commonly stamped into shader noise functions. Good enough for
/// tree placement; not cryptographic.
fn tree_hash(seed: u64, x: i32, z: i32, salt: u32) -> u32 {
    let mut h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (x as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h = h.rotate_left(13);
    h ^= (z as i64 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h = h.rotate_left(17);
    h ^= (salt as u64).wrapping_mul(0xCC9E_2D51_1B87_3593);
    ((h ^ (h >> 33)) as u32) ^ ((h >> 16) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Reduce a chunk to a single u64 hash — easier than comparing full
    /// 32 KB arrays in test failure messages.
    fn hash_chunk(c: &DenseChunk) -> u64 {
        let mut h = DefaultHasher::new();
        for b in c.blocks.iter() {
            (*b as u16).hash(&mut h);
        }
        h.finish()
    }

    #[test]
    fn fill_is_deterministic() {
        let g = Generator::new(42);
        let mut a = DenseChunk::empty();
        let mut b = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::ZERO), &mut a);
        g.fill_chunk(ChunkCoord(IVec3::ZERO), &mut b);
        assert_eq!(hash_chunk(&a), hash_chunk(&b));
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = DenseChunk::empty();
        let mut b = DenseChunk::empty();
        Generator::new(1).fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut a);
        Generator::new(2).fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut b);
        assert_ne!(hash_chunk(&a), hash_chunk(&b));
    }

    /// Golden test: locks the generator's output for a known seed/chunk.
    /// First run prints the actual hash; update `GOLDEN_42_002` once and
    /// future runs catch unintentional behavioural drift.
    #[test]
    fn golden_seed42_chunk_0_2_0() {
        // Hash captured after Task 39's visual confirmation; updated
        // again after the v0.1.10 tree-placement pass changed chunk
        // contents. Locks here so any future change to noise/parameters
        // surfaces immediately.
        const GOLDEN_42_002: u64 = 0x20E3_B24B_BDCE_1F0C;
        let g = Generator::new(42);
        let mut c = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut c);
        let actual = hash_chunk(&c);
        if GOLDEN_42_002 == 0xDEAD_BEEF_DEAD_BEEF {
            println!("UPDATE GOLDEN_42_002 to: 0x{:016X}", actual);
        } else {
            assert_eq!(actual, GOLDEN_42_002, "worldgen output changed");
        }
    }

    #[test]
    fn chunk_at_sea_level_has_water_or_solid() {
        // Sanity: the column-wise terrain must place *something* in any
        // sea-level chunk — either solid (under-water terrain) or water
        // (above-terrain flood).
        let g = Generator::new(42);
        let mut c = DenseChunk::empty();
        g.fill_chunk(ChunkCoord(IVec3::new(0, 2, 0)), &mut c);
        let has_non_air = c.blocks.iter().any(|&b| b != Block::Air);
        assert!(has_non_air);
    }
}
