//! Pulls camera state from the player entity and asks the [`Renderer`] to
//! draw the frame.
//!
//! This is a thin "glue" system — almost all rendering logic lives in the
//! `render` module. By design: keep ECS-side code small and stateless so
//! that the *renderer* can grow more pipelines/passes without ECS systems
//! turning into render-state ceremonies.

use crate::ecs::components::{Camera, CursorTarget, Position};
use crate::ecs::GameEcs;
use crate::render::Renderer;

/// Push the cursor target into the renderer and then draw a frame.
/// The renderer needs `&mut self` for the cursor write — the caller is
/// expected to already hold a `&mut Renderer`.
pub fn render(
    ecs: &GameEcs,
    renderer: &mut Renderer,
    time: f32,
) -> Result<(), wgpu::SurfaceError> {
    let target = ecs
        .world
        .query_one::<&CursorTarget>(ecs.player)
        .unwrap()
        .get()
        .copied()
        .unwrap_or_default();
    renderer.set_cursor(target.hit.map(|(b, _)| b));

    let mut q = ecs
        .world
        .query_one::<(&Position, &Camera)>(ecs.player)
        .unwrap();
    let (pos, cam) = q.get().unwrap();
    let eye = pos.0 + cam.eye_offset;
    let (sun_dir, intensity) = crate::ecs::systems::time_of_day::sun_state(ecs);
    renderer.render(eye, cam.yaw, cam.pitch, sun_dir, intensity, time)
}
