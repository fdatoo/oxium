//! Player movement: turn the per-frame `PlayerInput` into world-space motion.
//!
//! Sprint doubles the speed regardless of mode. The "wishdir" is *local* to
//! the camera basis: `+z = forward`, `+x = right`, `+y = up`.
//!
//! Integration in both modes happens in [`crate::ecs::systems::physics`] via
//! `sweep_player`; this system only writes the desired velocity. Fly mode
//! skips gravity but still respects voxel collision, matching creative-flight
//! conventions. It accelerates toward the desired velocity and applies drag
//! with no input, so motion has controlled momentum instead of snap starts.
//!
//! Walk-bob: advances `Camera.bob_phase` while the player is walking on
//! the ground. The renderer reads the phase to apply a small head-sway
//! offset. Fly mode, falling, and standing still hold the phase fixed.

use crate::ecs::GameEcs;
use crate::ecs::components::{Camera, Grounded, Movement, MovementMode, PlayerInput, Velocity};
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
            &mut Camera,
            &PlayerInput,
            &Movement,
            &mut Velocity,
            &Grounded,
        )>(ecs.player)
        .unwrap();
    let (cam, input, mov, vel, grounded) = q.get().unwrap();

    // Camera basis (right-handed, +Y up):
    //   forward_horiz — XZ-plane projection. Both Walk and Fly use this
    //                   so look-down doesn't drag the player along Y.
    //   right       — perpendicular to forward in XZ plane.
    let (sy, cy) = cam.yaw.sin_cos();
    let forward_horiz = Vec3::new(cy, 0.0, sy).normalize_or_zero();
    // 90° to the left of `forward_horiz`, in the XZ plane. This is the
    // "strafe right" axis, regardless of where the camera is looking.
    let right = Vec3::new(-sy, 0.0, cy).normalize_or_zero();

    let base_speed = if input.sprint {
        mov.speed * 2.0
    } else {
        mov.speed
    };

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
            const FLY_ACCEL: f32 = 42.0;
            const FLY_DRAG: f32 = 8.0;
            let speed = base_speed * FLY_SPEED_MULT;
            let wish = right * input.wishdir.x
                + forward_horiz * input.wishdir.z
                + Vec3::Y * input.wishdir.y;
            if wish.length_squared() > 1e-6 {
                let target = wish.normalize() * speed;
                vel.0 += (target - vel.0).clamp_length_max(FLY_ACCEL * dt);
            } else {
                vel.0 *= (1.0 - FLY_DRAG * dt).max(0.0);
                if vel.0.length_squared() < 1e-4 {
                    vel.0 = Vec3::ZERO;
                }
            }
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

    // Walk-bob phase. Advances only when the player is on foot AND on
    // the ground AND actually moving horizontally — Fly mode, falling,
    // and standing still all hold the phase fixed.  Advancement rate
    // is proportional to horizontal speed so a sprint produces a
    // faster head-bob than a stroll. Held phase (when not walking)
    // keeps the eye centred so there's no oscillation while flying.
    let horiz_speed = Vec3::new(vel.0.x, 0.0, vel.0.z).length();
    let walking_on_ground =
        matches!(mov.mode, MovementMode::Walk) && grounded.0 && horiz_speed > 0.5;
    if walking_on_ground {
        // 1.6 rad per metre walked → about one full sin cycle every
        // ~4 m of travel, which matches a comfortable footstep
        // cadence at the player's 5 m/s walk speed.
        cam.bob_phase += horiz_speed * 1.6 * dt;
        // Wrap to keep the float from drifting toward precision loss
        // over very long sessions.
        if cam.bob_phase > std::f32::consts::TAU * 32.0 {
            cam.bob_phase -= std::f32::consts::TAU * 32.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::{MovementMode, PlayerInput, Velocity};
    use glam::Vec3;

    #[test]
    fn fly_accelerates_toward_target_instead_of_snapping() {
        let mut ecs = GameEcs::new(Vec3::new(0.0, 64.0, 0.0));
        {
            let mut q = ecs
                .world
                .query_one::<(&mut Movement, &mut PlayerInput)>(ecs.player)
                .unwrap();
            let (movement, input) = q.get().unwrap();
            movement.mode = MovementMode::Fly;
            input.wishdir.z = 1.0;
        }

        movement(&mut ecs, 1.0 / 60.0);
        let speed = {
            let mut q = ecs.world.query_one::<&Velocity>(ecs.player).unwrap();
            q.get().unwrap().0.length()
        };
        assert!(speed > 0.0);
        assert!(speed < 15.0, "fly velocity should ramp, not snap to max");

        {
            let mut q = ecs.world.query_one::<&mut PlayerInput>(ecs.player).unwrap();
            q.get().unwrap().wishdir = Vec3::ZERO;
        }
        movement(&mut ecs, 1.0 / 60.0);
        let decayed = {
            let mut q = ecs.world.query_one::<&Velocity>(ecs.player).unwrap();
            q.get().unwrap().0.length()
        };
        assert!(decayed < speed, "fly velocity should decay with no input");
    }
}
