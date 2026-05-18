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
    }
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
        // Hash captured after Task 39's visual confirmation; lock here so
        // any future change to noise/parameters surfaces immediately.
        const GOLDEN_42_002: u64 = 0x879DF2E78400E716;
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
