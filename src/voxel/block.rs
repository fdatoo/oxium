//! Block kinds and their per-kind metadata.
//!
//! [`Block`] is a `#[repr(u16)]` enum — fixing the underlying representation
//! means we can safely serialize it and cast it to `usize` for array lookup
//! into [`BlockRegistry`].
//!
//! [`BlockInfo`] holds the *static* properties of a block kind: whether it
//! collides, blocks light, emits light, what colour to draw it. These are
//! looked up by the mesher, the lighting pass, and the physics sweep.
//!
//! The registry is *constructed once at startup* and treated as immutable
//! thereafter — see [`BlockRegistry::new`]. v0 has no data-driven block
//! definitions; adding modding/data files in the future would mean changing
//! `new` into a "load from TOML" function.

use serde::{Deserialize, Serialize};

/// Every distinct kind of block in the world. v0 ships with this small set;
/// new variants are appended only (the discriminants are part of the on-disk
/// wire format via the paletted chunk).
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Block {
    /// Nothing — transparent, no collision. The only block whose ID is fixed
    /// (it's used as the "absent" sentinel throughout the engine).
    Air = 0,
    Stone,
    Dirt,
    Grass,
    Sand,
    Water,
    Wood,
    Leaves,
    Torch,
}

/// Number of distinct [`Block`] variants. Updated when adding new blocks.
pub const BLOCK_COUNT: usize = 9;

impl Block {
    /// Inverse of `Block as u16`: returns the variant whose discriminant
    /// matches `v`, or `None` if `v` is out of range. Cheap and exhaustive
    /// so it stays correct as new variants are added.
    pub fn from_repr(v: u16) -> Option<Block> {
        use Block::*;
        Some(match v {
            0 => Air,
            1 => Stone,
            2 => Dirt,
            3 => Grass,
            4 => Sand,
            5 => Water,
            6 => Wood,
            7 => Leaves,
            8 => Torch,
            _ => return None,
        })
    }
}

/// Per-block static properties. Cheap to copy; the registry stores these
/// inline in a fixed-size array indexed by `Block as usize`.
#[derive(Debug, Clone, Copy)]
pub struct BlockInfo {
    /// True if the block resists player movement (used by physics sweep).
    pub solid: bool,
    /// True if light cannot pass through. Used by the BFS flood fill and
    /// by the mesher's face-culling test.
    pub opaque: bool,
    /// Block-light emission, 0..15. Non-zero values seed the block-light BFS.
    pub emission: u8,
    /// Side-face RGBA colour. Stored as f32 to keep arithmetic clean;
    /// converted to bytes when emitting vertex data.
    pub color: [f32; 4],
    /// Optional separate top-face colour. Set for grass (green top, brown
    /// sides). `None` means the top is drawn with `color`.
    pub top_color: Option<[f32; 4]>,
}

/// Fixed-size lookup from [`Block`] to [`BlockInfo`].
pub struct BlockRegistry {
    infos: [BlockInfo; BLOCK_COUNT],
}

impl BlockRegistry {
    /// Build the canonical v0 registry. The unfilled slots are pre-seeded
    /// with a bright magenta "missing-info" sentinel so an out-of-order block
    /// variant immediately shows up visually.
    pub fn new() -> Self {
        use Block::*;
        let mut infos = [BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [1.0, 0.0, 1.0, 1.0],
            top_color: None,
        }; BLOCK_COUNT];

        infos[Air as usize] = BlockInfo {
            solid: false,
            opaque: false,
            emission: 0,
            color: [0.0; 4],
            top_color: None,
        };
        infos[Stone as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [0.50, 0.50, 0.52, 1.0],
            top_color: None,
        };
        infos[Dirt as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [0.55, 0.40, 0.25, 1.0],
            top_color: None,
        };
        infos[Grass as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [0.55, 0.40, 0.25, 1.0],
            top_color: Some([0.40, 0.70, 0.30, 1.0]),
        };
        infos[Sand as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [0.85, 0.78, 0.55, 1.0],
            top_color: None,
        };
        infos[Water as usize] = BlockInfo {
            // Water is non-solid (player wades through) and non-opaque (light
            // passes, with extra falloff cost handled by the lighting BFS).
            solid: false,
            opaque: false,
            emission: 0,
            color: [0.20, 0.45, 0.80, 0.55],
            top_color: None,
        };
        infos[Wood as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [0.40, 0.28, 0.18, 1.0],
            top_color: None,
        };
        infos[Leaves as usize] = BlockInfo {
            // Leaves block movement but not light — light filters through.
            solid: true,
            opaque: false,
            emission: 0,
            color: [0.20, 0.55, 0.25, 1.0],
            top_color: None,
        };
        infos[Torch as usize] = BlockInfo {
            // Bright emitter, non-solid (you walk through them, M-style).
            solid: false,
            opaque: false,
            emission: 13,
            color: [1.0, 0.80, 0.30, 1.0],
            top_color: None,
        };

        Self { infos }
    }

    /// Look up the static info for a block. Indexing is `O(1)` and never
    /// allocates — every kind has a pre-filled slot.
    #[inline]
    pub fn info(&self, b: Block) -> &BlockInfo {
        &self.infos[b as usize]
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_is_non_opaque_non_solid() {
        let r = BlockRegistry::new();
        let air = r.info(Block::Air);
        assert!(!air.opaque);
        assert!(!air.solid);
    }

    #[test]
    fn torch_emits_light() {
        let r = BlockRegistry::new();
        assert!(r.info(Block::Torch).emission > 0);
    }

    #[test]
    fn water_is_translucent_non_solid() {
        let r = BlockRegistry::new();
        let w = r.info(Block::Water);
        assert!(!w.opaque);
        assert!(!w.solid);
        assert!(w.color[3] < 1.0, "water alpha should be < 1");
    }

    #[test]
    fn grass_has_distinct_top() {
        let r = BlockRegistry::new();
        assert!(r.info(Block::Grass).top_color.is_some());
    }
}
