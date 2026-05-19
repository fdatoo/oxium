//! Pulls camera state from the player entity and asks the [`Renderer`] to
//! draw the frame.
//!
//! This is a thin "glue" system — almost all rendering logic lives in the
//! `render` module. By design: keep ECS-side code small and stateless so
//! that the *renderer* can grow more pipelines/passes without ECS systems
//! turning into render-state ceremonies.

use crate::app::PerfSnapshot;
use crate::ecs::components::{Camera, CursorTarget, Position, Selected};
use crate::ecs::GameEcs;
use crate::render::hud::{build_hud, HOTBAR_BLOCKS};
use crate::render::Renderer;
use crate::voxel::block::BlockRegistry;

/// Push the cursor target into the renderer and then draw a frame.
/// The renderer needs `&mut self` for the cursor write — the caller is
/// expected to already hold a `&mut Renderer`.
pub fn render(
    ecs: &GameEcs,
    renderer: &mut Renderer,
    registry: &BlockRegistry,
    fps: f32,
    time: f32,
    perf: &PerfSnapshot,
    ui: &crate::ui::Ui,
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
        .query_one::<(&Position, &Camera, &Selected)>(ecs.player)
        .unwrap();
    let (pos, cam, selected) = q.get().unwrap();
    let eye = pos.0 + cam.eye_offset;
    let (sun_dir, intensity) = crate::ecs::systems::time_of_day::sun_state(ecs);

    // Selected slot = index in the hotbar of the player's current
    // block. The hotbar enumerates blocks in fixed order; we look up
    // by matching the `Selected` block against `HOTBAR_BLOCKS`.
    let selected_slot = HOTBAR_BLOCKS
        .iter()
        .position(|b| *b == Some(selected.0))
        .unwrap_or(0);

    let (sw, sh) = renderer.framebuffer_size();
    let mut hud = build_hud((sw, sh), fps, eye, selected_slot, registry, perf);
    ui.draw_overlay((sw, sh), &mut hud);

    renderer.render(
        eye,
        cam.yaw,
        cam.pitch,
        sun_dir,
        intensity,
        time,
        Some(&hud),
    )
}
