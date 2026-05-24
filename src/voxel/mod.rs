//! Voxel domain: blocks, chunk storage, coordinates, world state, and raycast.
//!
//! A "voxel" is the volumetric analogue of a pixel — a 1×1×1 cube at integer
//! coordinates. The world is partitioned into 32³ chunks (32_768 voxels each)
//! to keep individual mesh/load/save units bounded.
//!
//! ### Module layout
//!
//! | Module    | Responsibility                                             |
//! |-----------|------------------------------------------------------------|
//! | `block`   | block enum, render/physics/light metadata, registry        |
//! | `coords`  | `BlockPos`, `ChunkCoord`, `LocalPos`, conversion rules     |
//! | `chunk`   | dense/paletted storage plus chunk lifecycle metadata       |
//! | `world`   | sparse loaded-chunk map and edit/dirty propagation         |
//! | `raycast` | grid DDA block picking                                     |
//! | `packed`  | 4-bit packed array used by paletted chunks                 |
//!
//! Coordinate spaces are explicit domain types. Negative chunk coordinates are
//! valid; use `BlockPos::to_chunk()` and `BlockPos::to_local()` rather than
//! hand-rolled division so Euclidean wrapping stays correct.

pub mod block;
pub mod chunk;
pub mod coords;
pub mod packed;
pub mod raycast;
pub mod world;
