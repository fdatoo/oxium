//! Voxel world: blocks, chunks, coordinates, raycast.
//!
//! A "voxel" is the volumetric analogue of a pixel — a 1×1×1 cube at integer
//! coordinates. The world is partitioned into 32³ chunks (32_768 voxels each)
//! to keep individual mesh/load/save units bounded.

pub mod block;
pub mod coords;
