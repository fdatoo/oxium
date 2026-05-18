//! Pulls camera state from the player entity and asks the [`Renderer`] to
//! draw the frame.
//!
//! This is a thin "glue" system — almost all rendering logic lives in the
//! `render` module. By design: keep ECS-side code small and stateless so
//! that the *renderer* can grow more pipelines/passes without ECS systems
//! turning into render-state ceremonies.

use crate::ecs::components::{Camera, Position};
use crate::ecs::GameEcs;
use crate::render::Renderer;

/// Read the player's `Position` + `Camera`, compute the eye point, sample
/// the sun state, and invoke `Renderer::render`. Propagates the surface
/// error (e.g. swapchain out-of-date) up to the caller.
pub fn render(ecs: &GameEcs, renderer: &Renderer) -> Result<(), wgpu::SurfaceError> {
    let mut q = ecs
        .world
        .query_one::<(&Position, &Camera)>(ecs.player)
        .unwrap();
    let (pos, cam) = q.get().unwrap();
    let eye = pos.0 + cam.eye_offset;
    let (sun_dir, intensity) = crate::ecs::systems::time_of_day::sun_state(ecs);
    renderer.render(eye, cam.yaw, cam.pitch, sun_dir, intensity)
}
