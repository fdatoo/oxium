//! Apply gravity + collision to the player and trigger jumps.
//!
//! Runs *after* the movement system (which sets the desired velocity) and
//! *before* world streaming (which reads the player's position to decide
//! which chunks to load).
//!
//! In `Walk` mode we delegate to the axis-by-axis swept AABB and respect
//! the `grounded` flag for jumping. In `Fly` mode we trivially integrate
//! position from velocity — flight is a debug convenience and bypasses
//! collision so the developer can poke around inside terrain.

use crate::ecs::components::{
    Aabb, Grounded, Movement, MovementMode, PlayerInput, Position, Velocity,
};
use crate::ecs::GameEcs;
use crate::physics::sweep::sweep_player;
use crate::voxel::world::World;

/// Integrate the player one tick. `world` is read-only — collision tests
/// look at currently-loaded chunks; pending chunks are treated as solid
/// inside `sweep_player`.
pub fn physics(ecs: &mut GameEcs, world: &World, dt: f32) {
    let mut q = ecs
        .world
        .query_one::<(
            &Aabb,
            &Movement,
            &PlayerInput,
            &mut Position,
            &mut Velocity,
            &mut Grounded,
        )>(ecs.player)
        .unwrap();
    let (aabb, mov, input, pos, vel, grounded) = q.get().unwrap();

    // Apply jump if the player is grounded and currently walking.
    // We reset `grounded` immediately so the same press doesn't trigger
    // a second jump while the swept collision hasn't yet decided whether
    // the feet are still touching.
    if matches!(mov.mode, MovementMode::Walk) && input.jump && grounded.0 {
        vel.0.y = mov.jump_v;
        grounded.0 = false;
    }

    match mov.mode {
        MovementMode::Walk => {
            let res = sweep_player(world, pos.0, aabb.half, vel.0, dt);
            pos.0 = res.pos;
            vel.0 = res.vel;
            grounded.0 = res.grounded;
        }
        MovementMode::Fly => {
            // No collision in fly mode — integrate directly. Movement
            // system already set vel.0 to wish × speed.
            pos.0 += vel.0 * dt;
        }
    }
}
