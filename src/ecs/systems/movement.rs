//! Player movement: turn the per-frame `PlayerInput` into world-space motion.
//!
//! M2 ships only **flight**: velocity is set directly from `wishdir × speed`,
//! ignoring gravity, friction, and collision. M6 adds the walking model
//! (ground accel + friction + jump + gravity) and inserts the AABB-vs-voxel
//! sweep right after this system in the schedule.
//!
//! Sprint doubles the speed regardless of mode. The "wishdir" is *local* to
//! the camera basis: `+z = forward`, `+x = right`, `+y = up`.

use crate::ecs::components::{Camera, Movement, MovementMode, PlayerInput, Position, Velocity};
use crate::ecs::GameEcs;
use glam::Vec3;

/// Drive `Position` + `Velocity` from `PlayerInput` and `Camera`.
///
/// `dt` is the wall-clock delta since the previous tick, in seconds. The
/// caller (`AppState::step`) clamps it to ≤100 ms so frame hitches don't
/// teleport the player half a chunk away.
pub fn movement(ecs: &mut GameEcs, dt: f32) {
    let mut q = ecs
        .world
        .query_one::<(
            &Camera,
            &PlayerInput,
            &Movement,
            &mut Position,
            &mut Velocity,
        )>(ecs.player)
        .unwrap();
    let (cam, input, mov, pos, vel) = q.get().unwrap();

    // Camera basis (right-handed, +Y up):
    //   forward_3d  — full 3D look direction, used by Fly mode.
    //   forward_horiz — XZ-plane projection, used by Walk so look-down
    //                   doesn't drag the player into the ground.
    //   right       — perpendicular to forward in XZ plane.
    let (sy, cy) = cam.yaw.sin_cos();
    let (sp, cp) = cam.pitch.sin_cos();
    let forward_3d = Vec3::new(cy * cp, sp, sy * cp).normalize_or_zero();
    let forward_horiz = Vec3::new(cy, 0.0, sy).normalize_or_zero();
    // 90° to the left of `forward_horiz`, in the XZ plane. This is the
    // "strafe right" axis, regardless of where the camera is looking.
    let right = Vec3::new(-sy, 0.0, cy).normalize_or_zero();

    let speed = if input.sprint { mov.speed * 2.0 } else { mov.speed };

    match mov.mode {
        MovementMode::Fly => {
            // Forward goes along the *full* 3D camera direction so looking
            // up + W flies you up. The wishdir.y stick (Space/Shift) adds
            // pure vertical movement on top.
            let forward = if input.wishdir.z != 0.0 {
                forward_3d
            } else {
                Vec3::ZERO
            };
            let wish = right * input.wishdir.x
                + forward * input.wishdir.z
                + Vec3::Y * input.wishdir.y;
            vel.0 = wish.normalize_or_zero() * speed;
            pos.0 += vel.0 * dt;
        }
        MovementMode::Walk => {
            // M2 stub: behave like flight but on the XZ-plane forward only,
            // so the milestone is fun to fly + walk-toggle test even before
            // M6 lands the real walking simulation.
            let wish = right * input.wishdir.x
                + forward_horiz * input.wishdir.z
                + Vec3::Y * input.wishdir.y;
            vel.0 = wish.normalize_or_zero() * speed;
            pos.0 += vel.0 * dt;
        }
    }
}
