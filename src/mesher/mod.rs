//! Voxel chunk meshing: convert dense chunks into GPU-ready triangle buffers.
//!
//! The live path is [`greedy::mesh_greedy`]: it scans visible faces into 2D
//! masks and merges adjacent same-appearance cells into larger quads. That
//! keeps chunk geometry small while preserving per-face texture, AO, and
//! lighting data. Water top faces are emitted at one-block resolution so the
//! water shader can displace them without chunk-sized sheets.
//!
//! ### Module layout
//!
//! | Module   | Responsibility                                             |
//! |----------|------------------------------------------------------------|
//! | `types`  | public `Face`, `Vertex`, `ChunkMesh`, texture sentinel     |
//! | `greedy` | production mesher: masks, merge pass, AO-aware emission    |
//! | `naive`  | simple one-quad-per-visible-face reference implementation  |
//! | `lod`    | downsampled chunk representation and coarse mesh builder   |
//! | `ao`     | ambient-occlusion corner sampling                          |
//!
//! The naive mesher stays as a debug fallback and test reference: a greedy
//! mesh and a naive mesh should be visually equivalent, even if their vertex
//! counts differ.

pub mod ao;
pub mod greedy;
pub mod lod;
pub mod naive;
pub mod types;

pub use types::{ChunkMesh, Face, UNTEXTURED_TILE, Vertex};
