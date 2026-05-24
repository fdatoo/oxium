//! oxium — single-player voxel sandbox.
//!
//! This crate is split into two surfaces:
//!
//! **Pure engine modules** (no `winit`/`wgpu` dependency at the module level):
//! `command`, `jobs`, `lighting`, `mesher`, `persistence`, `physics`,
//! `viz_render`, `voxel`, `worldgen`. Integration tests and pure tooling
//! (e.g. `oxium-probe inspect`) can import only these without bringing up a
//! window or GPU device.
//!
//! **Windowed engine modules** (depend on `winit`/`wgpu`): `app`, `ecs`,
//! `input_engine`, `profiler`, `render`, `ui`. Exposed here so the
//! `oxium-probe capture` subcommand can drive the renderer without
//! duplicating the modules. These modules must not be imported by pure
//! library consumers — bring them in only from binaries or integration tests
//! that boot a full render context.

// ── Pure modules (no windowing dependency) ────────────────────────────────────
pub mod command;
pub mod jobs;
pub mod lighting;
pub mod mesher;
pub mod persistence;
pub mod physics;
pub mod viz_render;
pub mod voxel;
pub mod worldgen;

// ── Windowed modules (winit + wgpu) ──────────────────────────────────────────
// These are re-exported from main.rs via `pub use oxium::{app, ecs, …}` so
// that `crate::*` paths inside those modules continue to resolve correctly.
pub mod app;
pub mod ecs;
pub mod input_engine;
pub mod profiler;
pub mod render;
pub mod ui;
