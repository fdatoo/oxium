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

use crate::mesher::Face;
use serde::{Deserialize, Serialize};

/// One named entry in the rendering atlas. The discriminant doubles as
/// the **tile index** the mesher writes into each vertex — it MUST stay
/// in lockstep with the file order in `render::atlas::tile_files` so
/// atlas slot N really does hold the texture at index N.
///
/// `Tile` lives in the library-side `voxel::block` module (not in
/// `render`) so [`BlockInfo`] can reference it without the library
/// depending on `wgpu`. The render layer keeps the actual atlas image.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Stone = 0,
    Dirt = 1,
    GrassTop = 2,
    GrassSide = 3,
    Sand = 4,
    OakLog = 5,
    OakLogTop = 6,
    OakLeaves = 7,
    WaterStill = 8,
    Snow = 9,
    LavaStill = 10,
}

impl Tile {
    /// The byte-sized index the vertex format carries. Mirrors
    /// `self as u8` but expressed as a method so callers don't sprinkle
    /// `as` casts around.
    pub fn index(self) -> u8 {
        self as u8
    }
}

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
    /// Snow surface block for cold biomes. Reuses the grass_top
    /// grayscale tile with a near-white tint so we don't need a
    /// separate asset; the colour gives it a slightly cool cast
    /// against the warmer dirt sides below.
    Snow,
    /// Liquid lava. Hot, bright, non-solid, emits block-light; placed
    /// only by deep aquifers (see `worldgen::aquifer`).
    Lava,
}

/// Number of distinct [`Block`] variants. Updated when adding new blocks.
pub const BLOCK_COUNT: usize = 11;

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
            9 => Snow,
            10 => Lava,
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
    ///
    /// With textures enabled, this is also the **tint** the fragment
    /// shader multiplies the sampled texel by — so grayscale tiles like
    /// `grass_block_top` come out green when `color` is green, and
    /// `stone.png` stays neutral when `color` is white.
    pub color: [f32; 4],
    /// Optional separate top-face colour / tint. Set for grass (green
    /// top, brown sides). `None` means the top is drawn with `color`.
    pub top_color: Option<[f32; 4]>,
    /// Atlas tile to sample on the ±X / ±Z (side) faces. `None` means
    /// this block has no texture binding — the shader falls back to a
    /// solid `color` fill, used for `Air` and `Torch` today.
    pub tile_side: Option<Tile>,
    /// Atlas tile for the +Y (top) face. `None` ⇒ use `tile_side`.
    pub tile_top: Option<Tile>,
    /// Atlas tile for the −Y (bottom) face. `None` ⇒ use `tile_side`.
    pub tile_bottom: Option<Tile>,
}

impl BlockInfo {
    /// Resolve which atlas tile should be sampled on the given face.
    /// Returns `None` for untextured blocks (the mesher emits 0 in that
    /// case and the shader uses the vertex colour directly).
    pub fn tile_for_face(&self, face: Face) -> Option<Tile> {
        match face {
            Face::PosY => self.tile_top.or(self.tile_side),
            Face::NegY => self.tile_bottom.or(self.tile_side),
            _ => self.tile_side,
        }
    }
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
        // Magenta "missing info" sentinel for any slot the explicit
        // list below forgets to fill.
        let mut infos = [BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [1.0, 0.0, 1.0, 1.0],
            top_color: None,
            tile_side: None,
            tile_top: None,
            tile_bottom: None,
        }; BLOCK_COUNT];

        infos[Air as usize] = BlockInfo {
            solid: false,
            opaque: false,
            emission: 0,
            color: [0.0; 4],
            top_color: None,
            tile_side: None,
            tile_top: None,
            tile_bottom: None,
        };
        infos[Stone as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            // Stone texture is already coloured — neutral white tint
            // leaves it unmodified.
            color: [1.0, 1.0, 1.0, 1.0],
            top_color: None,
            tile_side: Some(Tile::Stone),
            tile_top: None,
            tile_bottom: None,
        };
        infos[Dirt as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [1.0, 1.0, 1.0, 1.0],
            top_color: None,
            tile_side: Some(Tile::Dirt),
            tile_top: None,
            tile_bottom: None,
        };
        infos[Grass as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            // Sides and bottom use neutral white tint over the
            // pre-coloured dirt / grass_side textures…
            color: [1.0, 1.0, 1.0, 1.0],
            // …but `grass_block_top.png` ships **grayscale** so the
            // engine can tint it per-biome. Pre-multiplication
            // brightness: `grass_block_top.png` averages around 0.55
            // brightness, so the final lit colour is roughly `tint *
            // 0.55 * shade`. Using a vivid green tint here gives us
            // headroom for the ACES tonemap to compress without the
            // grass reading olive/gray — the earlier `[0.49, 0.78,
            // 0.32]` tint was too red-shifted and came out muddy.
            top_color: Some([0.32, 0.95, 0.28, 1.0]),
            tile_side: Some(Tile::GrassSide),
            tile_top: Some(Tile::GrassTop),
            tile_bottom: Some(Tile::Dirt),
        };
        infos[Sand as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [1.0, 1.0, 1.0, 1.0],
            top_color: None,
            tile_side: Some(Tile::Sand),
            tile_top: None,
            tile_bottom: None,
        };
        infos[Water as usize] = BlockInfo {
            // Water is non-solid (player wades through) and non-opaque (light
            // passes, with extra falloff cost handled by the lighting BFS).
            solid: false,
            opaque: false,
            emission: 0,
            // Blue tint over the grayscale ripple texture; alpha < 1
            // keeps the shader's water-shimmer code path active.
            color: [0.38, 0.62, 0.95, 0.78],
            top_color: None,
            tile_side: Some(Tile::WaterStill),
            tile_top: None,
            tile_bottom: None,
        };
        infos[Wood as usize] = BlockInfo {
            solid: true,
            opaque: true,
            emission: 0,
            color: [1.0, 1.0, 1.0, 1.0],
            top_color: None,
            // Bark on the sides, concentric rings on top + bottom.
            tile_side: Some(Tile::OakLog),
            tile_top: Some(Tile::OakLogTop),
            tile_bottom: Some(Tile::OakLogTop),
        };
        infos[Leaves as usize] = BlockInfo {
            // Leaves block movement but not light — light filters through.
            solid: true,
            opaque: false,
            emission: 0,
            // Grayscale leaves texture tinted to a leafy green.
            color: [0.40, 0.72, 0.30, 1.0],
            top_color: None,
            tile_side: Some(Tile::OakLeaves),
            tile_top: None,
            tile_bottom: None,
        };
        infos[Torch as usize] = BlockInfo {
            // Bright emitter, non-solid (you walk through them, M-style).
            solid: false,
            opaque: false,
            emission: 13,
            color: [1.0, 0.80, 0.30, 1.0],
            top_color: None,
            tile_side: None,
            tile_top: None,
            tile_bottom: None,
        };
        infos[Lava as usize] = BlockInfo {
            // Lava is non-solid (entities sink/burn) and non-opaque
            // (the glow leaks through). Emits maximum block-light so
            // pools and aquifer rooms light themselves.
            solid: false,
            opaque: false,
            emission: 15,
            // Neutral white tint — the lava_still.png is already
            // pre-coloured fiery orange. Alpha < 1 keeps it in the
            // translucent pipeline like water (so the shader can pick
            // up the emission/shimmer path).
            color: [1.0, 0.95, 0.85, 0.95],
            top_color: None,
            tile_side: Some(Tile::LavaStill),
            tile_top: None,
            tile_bottom: None,
        };
        infos[Snow as usize] = BlockInfo {
            // Snow blanket: dedicated cool-white pixel-noise texture
            // (`snow.png`). The tint is neutral white so the
            // texture's own pre-baked colour wins — earlier versions
            // re-used `grass_block_top.png` with a near-white tint,
            // but that texture's grayscale midtones average around
            // 0.6, so even a near-white tint multiplied by ~0.6 read
            // as medium gray instead of snow.
            solid: true,
            opaque: true,
            emission: 0,
            color: [1.0, 1.0, 1.0, 1.0],
            top_color: None,
            tile_side: Some(Tile::Snow),
            tile_top: None,
            tile_bottom: None,
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
