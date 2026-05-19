//! Render pipelines: one per material/topology kind.
//!
//! Adding a new pipeline (sky, water, cursor highlight, HUD) means adding a
//! submodule here. Keeping each pipeline in its own file makes shader/layout
//! changes local.

pub mod cursor;
pub mod hud;
pub mod opaque;
pub mod sky;
