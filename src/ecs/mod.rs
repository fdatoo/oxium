//! Game-entity ECS (player, camera, sun, cursor). Chunks live in `voxel::World`.
//!
//! We use `hecs` — a tiny archetypal ECS — and a hand-written per-frame
//! schedule (no macros, no automatic parallelism). The chunk grid is
//! intentionally *not* an ECS resource: voxel data is too cache-hot and
//! co-accessed by jobs running off the main thread, so it lives in its own
//! [`crate::voxel::World`] container.

pub mod components;
pub mod systems;

use hecs::Entity;

/// Container for the game's ECS state.
///
/// We wrap `hecs::World` so that the spawn ritual (creating the player
/// with all the right components) lives in exactly one place and so the
/// player [`Entity`] handle is always cached — there's only one and many
/// systems need it.
pub struct GameEcs {
    pub world: hecs::World,
    /// Handle to the single player entity. Stored so systems don't have to
    /// scan for the `Player` marker every frame.
    pub player: Entity,
}

impl GameEcs {
    /// Build the ECS with a single player entity at `spawn_pos`.
    ///
    /// Movement defaults to **flight** so M2 has something fun to test before
    /// physics arrives in M6.
    pub fn new(spawn_pos: glam::Vec3) -> Self {
        use components::*;
        let mut world = hecs::World::new();
        let player = world.spawn((
            Player,
            Position(spawn_pos),
            Velocity(glam::Vec3::ZERO),
            // 0.6 × 1.8 × 0.6 m AABB — standard "humanoid" voxel-game
            // dimensions. half-extents = (0.3, 0.9, 0.3).
            Aabb {
                half: glam::Vec3::new(0.3, 0.9, 0.3),
            },
            Movement {
                // Walk by default — the player experiences gravity + the
                // 5 m/s walk speed from M6 onward. F toggles to flight
                // for free movement when exploring.
                mode: MovementMode::Walk,
                speed: 5.0,
                jump_v: 8.4,
                noclip: false,
            },
            Camera {
                yaw: 0.0,
                pitch: 0.0,
                fov: 70f32.to_radians(),
                // Eye is 1.6 m above the feet — a touch below the AABB top.
                eye_offset: glam::Vec3::new(0.0, 1.6, 0.0),
                bob_phase: 0.0,
            },
            PlayerInput::default(),
            Grounded::default(),
            Selected::default(),
            CursorTarget::default(),
        ));
        // Sun entity holds the single TimeOfDay state queried by render +
        // sky systems. Living in the ECS means future features (multiple
        // suns, scripted skies) just spawn another entity.
        world.spawn((Sun, TimeOfDay::default()));
        Self { world, player }
    }
}
