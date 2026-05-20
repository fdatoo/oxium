//! Apply gravity + collision to the player and trigger jumps.
//!
//! Runs *after* the movement system (which sets the desired velocity) and
//! *before* world streaming (which reads the player's position to decide
//! which chunks to load).
//!
//! Both `Walk` and `Fly` use the axis-by-axis swept AABB (`sweep_player`)
//! so the player can't phase through blocks in either mode; the only
//! difference is that `Walk` accumulates gravity in the movement system
//! and Fly does not. `grounded` is meaningful in Walk (controls jump);
//! the same flag is still computed in Fly so a future "respect-ground"
//! cosmetic could read it, but no current system does.

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
            &mut Movement,
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

    // Noclip: fly mode + the /noclip flag → skip the swept collision
    // and integrate position directly. Grounded is always false in this
    // mode (no contact tests run).
    if matches!(mov.mode, MovementMode::Fly) && mov.noclip {
        pos.0 += vel.0 * dt;
        grounded.0 = false;
        return;
    }

    let res = sweep_player(world, pos.0, aabb.half, vel.0, dt);
    pos.0 = res.pos;
    vel.0 = res.vel;
    grounded.0 = res.grounded;

    // Auto-land: holding Shift while flying onto solid ground drops the
    // player out of fly mode so they immediately start walking. The
    // grounded test ensures we only switch when the foot actually
    // touched a block this tick — pressing Shift mid-air just descends.
    if matches!(mov.mode, MovementMode::Fly) && input.wishdir.y < 0.0 && grounded.0 {
        mov.mode = MovementMode::Walk;
    }
}
