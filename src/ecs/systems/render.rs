//! Pulls camera state from the player entity and asks the [`Renderer`] to
//! draw the frame.
//!
//! This is a thin "glue" system — almost all rendering logic lives in the
//! `render` module. By design: keep ECS-side code small and stateless so
//! that the *renderer* can grow more pipelines/passes without ECS systems
//! turning into render-state ceremonies.

use crate::app::PerfSnapshot;
use crate::ecs::components::{Camera, CursorTarget, Position, Selected, Sun, TimeOfDay};
use crate::ecs::GameEcs;
use crate::render::hud::{build_hud, SkyProbe, WorldDebug, HOTBAR_BLOCKS};
use crate::render::Renderer;
use crate::voxel::block::BlockRegistry;
use crate::voxel::coords::{BlockPos, ChunkCoord, LocalPos};
use crate::voxel::world::{ChunkSlot, World};
use crate::worldgen::Generator;
use glam::{IVec3, UVec3};

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
    generator: &Generator,
    world: &World,
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
    // Walk-bob: small `sin(phase)` oscillation on top of the eye
    // offset. The Y term has 2× the rate (a full head-bob cycle
    // = two foot-falls); the X term wobbles at the foot-fall rate
    // to give the slight side-to-side sway real walking has.
    // Amplitudes kept small — anything taller would feel motion-
    // sicky in first-person. Phase advances only when the player is
    // walking on the ground (see movement.rs), so flight + idle +
    // mid-air all stay perfectly steady.
    let bob_y = (cam.bob_phase * 2.0).sin() * 0.07;
    let bob_x = cam.bob_phase.sin() * 0.05;
    // Apply the bob in world space along the camera's right axis so
    // the side-sway always reads as "side-to-side" regardless of
    // facing.
    let (sy, cy) = cam.yaw.sin_cos();
    let right = glam::Vec3::new(-sy, 0.0, cy);
    let eye = pos.0 + cam.eye_offset + right * bob_x + glam::Vec3::Y * bob_y;
    let (sun_dir, intensity) = crate::ecs::systems::time_of_day::sun_state(ecs);

    // Selected slot = index in the hotbar of the player's current
    // block. The hotbar enumerates blocks in fixed order; we look up
    // by matching the `Selected` block against `HOTBAR_BLOCKS`.
    let selected_slot = HOTBAR_BLOCKS
        .iter()
        .position(|b| *b == Some(selected.0))
        .unwrap_or(0);

    let time_of_day = ecs
        .world
        .query::<(&Sun, &TimeOfDay)>()
        .iter()
        .next()
        .map(|(_, (_, tod))| tod.t)
        .unwrap_or(0.5);
    let probe = generator.probe_column(eye.x as i32, eye.z as i32);
    let sky = sky_probe(world, eye);
    let world_debug = WorldDebug {
        seed: generator.seed(),
        time_of_day,
        yaw: cam.yaw,
        pitch: cam.pitch,
        probe: &probe,
        sky,
    };

    let (sw, sh) = renderer.framebuffer_size();
    let mut hud = build_hud((sw, sh), fps, eye, selected_slot, registry, perf, Some(&world_debug));
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

/// Read `sky_light` around the eye for the debug HUD. Reads straight
/// from the `PalettedChunk` (4-bit packed) so no decompression is
/// needed — the per-frame cost is ~32 packed-array gets per call.
fn sky_probe(world: &World, eye: glam::Vec3) -> SkyProbe {
    let block_pos = BlockPos(IVec3::new(
        eye.x.floor() as i32,
        eye.y.floor() as i32,
        eye.z.floor() as i32,
    ));
    let eye_chunk = block_pos.to_chunk();
    let eye_local = block_pos.to_local();

    let mut probe = SkyProbe {
        eye_chunk: eye_chunk.0,
        at_eye: None,
        column_hex: None,
        above_bottom: None,
    };

    if let Some(ChunkSlot::Stored { data, .. }) = world.chunks.get(&eye_chunk) {
        probe.at_eye = Some(data.sky_light.get(eye_local.to_index()));
        let mut col = String::with_capacity(32);
        for y in 0..32u32 {
            let idx = LocalPos(UVec3::new(eye_local.0.x, y, eye_local.0.z)).to_index();
            let v = data.sky_light.get(idx);
            col.push(std::char::from_digit(v as u32, 16).unwrap_or('?'));
        }
        probe.column_hex = Some(col);
    }

    let above = ChunkCoord(eye_chunk.0 + IVec3::new(0, 1, 0));
    if let Some(ChunkSlot::Stored { data, .. }) = world.chunks.get(&above) {
        let idx = LocalPos(UVec3::new(eye_local.0.x, 0, eye_local.0.z)).to_index();
        probe.above_bottom = Some(data.sky_light.get(idx));
    }

    probe
}
