//! ECS components used by game-entity systems.
//!
//! Components are plain data — they describe *state* but not behavior;
//! systems (in `ecs::systems`) read or write them every frame.
//!
//! By convention each component is a thin newtype or struct of `Copy` data;
//! storing them by value in `hecs` means no allocator pressure per entity.

use glam::Vec3;

/// World-space position. For the player this is the *feet* position — the
/// AABB's lower-Y face sits at `Position.0.y`.
#[derive(Debug, Clone, Copy)]
pub struct Position(pub Vec3);

/// World-space linear velocity, in blocks/second.
#[derive(Debug, Clone, Copy)]
pub struct Velocity(pub Vec3);

/// Axis-aligned bounding box, centered on the owning entity's [`Position`].
/// `half` is the half-extents in each axis.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub half: Vec3,
}

/// How the entity converts its [`PlayerInput::wishdir`] into movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovementMode {
    /// Subject to gravity, jumps, AABB collision. Activated in M6.
    Walk,
    /// Free movement: wishdir × speed, no collision, no gravity. The dev
    /// tool that lets us inspect any chunk regardless of physics state.
    Fly,
}

/// Movement tuning parameters. Per-entity rather than global so future
/// mobs/cameras can pick their own speeds.
#[derive(Debug, Clone, Copy)]
pub struct Movement {
    pub mode: MovementMode,
    /// Base ground speed, m/s. Sprint doubles this.
    pub speed: f32,
    /// Upward velocity imparted by a jump press. `2 g h ≈ v²` → ~8.4 m/s
    /// for a 1.25-block jump under our 28 m/s² gravity.
    pub jump_v: f32,
}

/// View-camera state. Attached to the player; M5+ may attach it to other
/// entities (cinematic cameras, mob viewpoints).
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Rotation around the world `+Y` axis, in radians. `yaw = 0` looks
    /// toward `+X`.
    pub yaw: f32,
    /// Up-down rotation, radians, clamped to ±89°.
    pub pitch: f32,
    /// Vertical field-of-view, radians. The renderer currently hardcodes
    /// 70° at the call site; this field is reserved for a future "FOV
    /// slider" setting and lives here so the player's preferred FOV
    /// travels with the entity.
    #[allow(dead_code)]
    pub fov: f32,
    /// Translation from [`Position`] (feet) to the eye/lens. Player default
    /// is `(0, 1.6, 0)` — eye-height in blocks.
    pub eye_offset: Vec3,
}

/// Inputs gathered from the player this frame. Mostly direct copies of
/// keyboard/mouse state, but with edge-triggered flags for action presses
/// (so a single click reliably produces exactly one place/break attempt).
#[derive(Debug, Clone, Copy, Default)]
pub struct PlayerInput {
    /// Local-space desired motion: `+x = strafe right`, `+y = up`,
    /// `+z = forward`. Cleared and rebuilt each frame.
    pub wishdir: Vec3,
    pub jump: bool,
    pub sprint: bool,
    /// Edge-triggered (consumed per frame): right-mouse button pressed.
    pub place: bool,
    /// Edge-triggered (consumed per frame): left-mouse button pressed.
    pub break_: bool,
    /// Edge-triggered: `F` was tapped this frame to toggle walk/fly.
    /// Currently the input system applies the toggle directly to the
    /// `Movement` component, so this flag isn't consumed elsewhere —
    /// kept for a future scriptable input rebind UI.
    #[allow(dead_code)]
    pub toggle_mode: bool,
}

/// True when the entity's feet touched a solid this frame. Used to gate
/// jumping (no double jumps) and to switch the friction model.
#[derive(Debug, Clone, Copy, Default)]
pub struct Grounded(pub bool);

/// Zero-sized marker on the single player entity. Useful for queries
/// (`Query::<&Player>` finds the player and nothing else).
pub struct Player;

/// Currently-selected block for the "place" action. Cycled via the 1–7
/// number-row keys.
#[derive(Debug, Clone, Copy)]
pub struct Selected(pub crate::voxel::block::Block);
impl Default for Selected {
    fn default() -> Self {
        Self(crate::voxel::block::Block::Stone)
    }
}

/// What the cursor (centre-screen raycast) is pointed at, if anything.
/// Refreshed each frame by the interaction system; consumed by the
/// renderer for the wireframe highlight.
#[derive(Debug, Clone, Copy, Default)]
pub struct CursorTarget {
    pub hit: Option<(crate::voxel::coords::BlockPos, crate::mesher::Face)>,
}

/// Marker on the sun entity that carries the [`TimeOfDay`] state.
pub struct Sun;

/// Wall-clock state of the simulated day/night cycle.
///
/// `t` is a 0..1 fraction of a full day. By convention `t = 0.0` is
/// midnight, `0.25` is sunrise, `0.5` is noon, `0.75` is sunset. The
/// renderer's sun direction and procedural sky are derived from `t`.
#[derive(Debug, Clone, Copy)]
pub struct TimeOfDay {
    pub t: f32,
    /// Real-time seconds in a full simulated day. Smaller = faster cycle.
    pub day_length: f32,
}

impl Default for TimeOfDay {
    fn default() -> Self {
        // Start a bit after sunrise — sun is ~45° above the horizon in
        // the +X direction so the default `yaw=0, pitch=0` camera has
        // it visible up-and-ahead, and intensity is already ~0.8 so the
        // world is well lit. A 10-minute day keeps the cycle visible
        // during play sessions.
        Self {
            t: 0.125,
            day_length: 600.0,
        }
    }
}
