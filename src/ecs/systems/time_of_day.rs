//! Drive the simulated day/night cycle and derive the sun's world-space
//! direction + brightness from it.
//!
//! [`advance`] integrates `TimeOfDay::t` once per frame. [`sun_state`]
//! reads the current state and returns `(sun_dir, intensity)` for the
//! renderer + sky pipeline to consume.
//!
//! At `t = 0.25` the sun has just risen; at `t = 0.5` it's directly
//! overhead; at `t = 0.75` it's setting. Between `t = 0.75` and `t = 1.0`
//! (sunset → midnight) and from `0.0` to `0.25` (midnight → sunrise) the
//! sun is below the horizon and `intensity` clamps to 0.

use crate::ecs::components::{Sun, TimeOfDay};
use crate::ecs::GameEcs;

/// Step `TimeOfDay::t` forward by `dt` seconds, wrapping at 1.0.
pub fn advance(ecs: &mut GameEcs, dt: f32) {
    for (_, tod) in ecs.world.query::<&mut TimeOfDay>().iter() {
        tod.t = (tod.t + dt / tod.day_length).rem_euclid(1.0);
    }
}

/// Read the sun entity's `TimeOfDay` and convert it into a unit sun
/// direction + scalar intensity.
///
/// Returns `([1, 0, 0], 0)` if no sun entity is in the world — safe
/// defaults that produce a static, dark scene.
pub fn sun_state(ecs: &GameEcs) -> ([f32; 3], f32) {
    let mut sun_dir = [1.0, 0.0, 0.0];
    let mut intensity = 0.0;
    for (_, (_, tod)) in ecs.world.query::<(&Sun, &TimeOfDay)>().iter() {
        // The sun sweeps a circle in the XY plane. t=0 puts the sun at
        // +X (technically below horizon as sin(0)=0, but the intensity
        // clamp handles that).
        //
        // The Z component is held at 0 so the sun stays in the X/Y
        // plane. A previous "slight Z tilt for visual warmth" (0.2)
        // produced a permanent +Z bias that visibly brightened +Z-
        // facing walls relative to -Z-facing walls at every time of
        // day — two walls both "parallel to the sun" ended up with
        // 0.43 vs 0.286 wrap values from the n·L kink at zero.
        let angle = tod.t * std::f32::consts::TAU;
        let (s, c) = angle.sin_cos();
        sun_dir = [c, s, 0.0];
        // Intensity = clamp(sin(angle) + 0.1, 0, 1).
        // The +0.1 gives a tiny pre-dawn / post-dusk twilight glow so the
        // world doesn't pop suddenly from black to lit at the horizon.
        intensity = (s + 0.1).clamp(0.0, 1.0);
    }
    (sun_dir, intensity)
}
