//! Per-frame systems. The schedule is wired up in `crate::app::AppState::step`.
//!
//! Each system is a free function that takes a `&mut GameEcs` plus any
//! transient context it needs (input buffer, delta time, renderer handle).
//! Keeping them as plain functions sidesteps the complexity of `hecs`'s
//! optional automatic schedule and is the friendliest layout to debug.

pub mod input;
pub mod interaction;
pub mod mesh_upload;
pub mod movement;
pub mod physics;
pub mod render;
pub mod time_of_day;
pub mod world_stream;
