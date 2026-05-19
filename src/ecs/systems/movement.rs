//! Player movement: turn the per-frame `PlayerInput` into world-space motion.
//!
//! Sprint doubles the speed regardless of mode. The "wishdir" is *local* to
//! the camera basis: `+z = forward`, `+x = right`, `+y = up`.
//!
//! Integration in both modes happens in [`crate::ecs::systems::physics`] via
//! `sweep_player`; this system only writes the desired velocity. Fly mode
//! skips gravity (so you float when not pressing keys) but still respects
//! voxel collision, matching creative-flight conventions.

use crate::ecs::components::{Camera, Movement, MovementMode, PlayerInput, Velocity};
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
        .query_one::<(&Camera, &PlayerInput, &Movement, &mut Velocity)>(ecs.player)
        .unwrap();
    let (cam, input, mov, vel) = q.get().unwrap();

    // Camera basis (right-handed, +Y up):
    //   forward_horiz — XZ-plane projection. Both Walk and Fly use this
    //                   so look-down doesn't drag the player along Y.
    //   right       — perpendicular to forward in XZ plane.
    let (sy, cy) = cam.yaw.sin_cos();
    let forward_horiz = Vec3::new(cy, 0.0, sy).normalize_or_zero();
    // 90° to the left of `forward_horiz`, in the XZ plane. This is the
    // "strafe right" axis, regardless of where the camera is looking.
    let right = Vec3::new(-sy, 0.0, cy).normalize_or_zero();

    let base_speed = if input.sprint { mov.speed * 2.0 } else { mov.speed };

    match mov.mode {
        MovementMode::Fly => {
            // Horizontal-only forward (XZ projection) so looking up/down
            // doesn't pull the player along the camera's Y axis — only
            // Space (wishdir.y > 0) and Shift (wishdir.y < 0) move you
            // vertically. Integration + collision are both done by
            // `physics::physics` (which calls sweep_player for Fly the
            // same way it does for Walk, just without accumulating
            // gravity).
            //
            // Fly gets a fixed speed multiplier on top of `mov.speed`
            // so creative-mode movement feels distinctly faster than
            // walking, especially with sprint stacked on top.
            const FLY_SPEED_MULT: f32 = 3.0;
            let speed = base_speed * FLY_SPEED_MULT;
            let wish = right * input.wishdir.x
                + forward_horiz * input.wishdir.z
                + Vec3::Y * input.wishdir.y;
            vel.0 = wish.normalize_or_zero() * speed;
        }
        MovementMode::Walk => {
            // Quake-style ground accel + friction.
            //
            // The horizontal velocity moves toward `wish × speed`, gated
            // by a per-frame acceleration cap so changes feel responsive
            // without snapping. Y velocity is *not* touched here — the
            // physics system applies gravity and the swept-AABB collision.
            use crate::physics::sweep::GRAVITY;

            let wish = right * input.wishdir.x + forward_horiz * input.wishdir.z;
            let target = wish.normalize_or_zero() * base_speed;
            let horiz = Vec3::new(vel.0.x, 0.0, vel.0.z);
            // Move toward target by at most `accel * dt` units per frame.
            let accel: f32 = 60.0;
            let mut new_horiz = horiz + (target - horiz).clamp_length_max(accel * dt);
            // Friction when no horizontal input is given (the "stop tap"
            // is what makes WASD release feel snappy).
            if input.wishdir.x.abs() < 1e-4 && input.wishdir.z.abs() < 1e-4 {
                let friction: f32 = 12.0;
                new_horiz *= (1.0 - friction * dt).max(0.0);
            }
            vel.0.x = new_horiz.x;
            vel.0.z = new_horiz.z;

            // Gravity. Jumping is detected by the physics system once
            // `grounded` is known (otherwise we'd double-jump in air).
            vel.0.y += GRAVITY * dt;
            // Integration is handled by `physics::physics` — don't `pos.0
            // += vel.0 * dt` here, or the player will move twice.
        }
    }
}
