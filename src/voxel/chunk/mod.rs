//! Chunk storage and per-chunk runtime metadata.
//!
//! A chunk is a 32 x 32 x 32 block cube. The engine uses two chunk forms:
//!
//! - [`DenseChunk`] is the hot, unpacked working form for generation, meshing,
//!   lighting, and edits.
//! - [`PalettedChunk`] is the compressed in-memory/on-disk form stored in the
//!   world map and region files.
//!
//! Lighting metadata also lives here because both worldgen and persistence need
//! to carry enough information for the relight worker to produce stable results
//! even when neighbouring chunks stream in later.
//!
//! ### Submodule layout
//!
//! | Submodule      | Contents                                             |
//! |----------------|------------------------------------------------------|
//! | `dense`        | unpacked block/light arrays                          |
//! | `palette`      | palette-compressed chunk storage and legacy upgrade  |
//! | `lighting`     | light packing helpers and `ChunkLightInputs`         |
//! | `light_volume` | 33^3 GPU light-volume blob builder                   |
//! | `meta`         | chunk lifecycle, dirty flags, light state            |
//! | `neighbors`    | six-face dense-neighbour view                        |
//!
//! Public items are re-exported so existing callers can continue importing
//! from `crate::voxel::chunk::*`.

pub mod dense;
pub mod light_volume;
pub mod lighting;
pub mod meta;
pub mod neighbors;
pub mod palette;

/// Number of voxels in one chunk: 32 x 32 x 32 = 32,768.
pub const CHUNK_VOL: usize = 32 * 32 * 32;

/// Number of columns in one chunk footprint: 32 x 32 = 1,024.
pub const CHUNK_AREA: usize = 32 * 32;

pub use dense::DenseChunk;
pub use light_volume::build_light_volume_blob;
pub use lighting::{
    ChunkLightInputs, LIGHT_INPUT_UNKNOWN_Y, LightLevel, PackedRgbLight, pack_rgb, rgb_brightness,
    unpack_rgb,
};
pub use meta::{ChunkDirty, ChunkMeta, ChunkState, FaceMask, LightState};
pub use neighbors::Neighbors;
pub use palette::PalettedChunk;
pub(crate) use palette::PalettedChunkV1;

#[cfg(test)]
mod tests;
