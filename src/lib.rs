//! oxium — single-player voxel sandbox.
//!
//! This file exposes the *pure* engine modules — those that don't depend on
//! `winit` or a live `wgpu` device — as a library so integration tests and
//! external tools can construct a World, run lighting, mesh chunks, etc.,
//! without bringing up a window.
//!
//! The binary side of the crate (windowing, ECS scheduling, GPU rendering)
//! lives in `src/main.rs` plus the `app`, `ecs`, and `render` modules and
//! is **not** re-exported here. Those modules pull in the entire
//! `winit`/`wgpu` stack and we deliberately keep the library footprint
//! small.

pub mod jobs;
pub mod lighting;
pub mod mesher;
pub mod persistence;
pub mod physics;
pub mod viz_render;
pub mod voxel;
pub mod worldgen;
