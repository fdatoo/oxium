//! Translates raw winit events into `PlayerInput` + camera deltas.
//!
//! The flow is:
//!
//! 1. Main thread (in `main.rs`) routes winit events into [`InputBuf`].
//! 2. Once per frame, [`apply_input`] turns the buffered state into changes
//!    on the player entity's `Camera`, `PlayerInput`, and `Movement`.
//! 3. The buffer's "this-frame" flags are cleared by
//!    [`InputBuf::clear_per_frame`] after the systems consume them.
//!
//! This split — buffer → apply → clear — keeps the asynchronous OS event
//! arrival decoupled from the deterministic per-frame logic.

use crate::ecs::components::{Camera, MovementMode, Movement, PlayerInput};
use crate::ecs::GameEcs;
use glam::Vec3;
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

/// Per-frame buffer of input deltas. The main thread accumulates state here
/// in winit callbacks; [`apply_input`] drains it once per tick.
#[derive(Default, Debug)]
pub struct InputBuf {
    /// Accumulated mouse-X delta this frame (raw, pre-sensitivity).
    pub mouse_dx: f32,
    /// Accumulated mouse-Y delta this frame.
    pub mouse_dy: f32,
    /// Keys currently held down. Persists across frames until release.
    pub keys_down: std::collections::HashSet<KeyCode>,
    /// Keys that transitioned from up→down *this frame only*. Edge-triggered;
    /// cleared every `clear_per_frame`.
    pub key_pressed_this_frame: std::collections::HashSet<KeyCode>,
    /// Left mouse pressed during this frame (edge-triggered).
    pub lmb_pressed: bool,
    /// Right mouse pressed during this frame (edge-triggered).
    pub rmb_pressed: bool,
}

impl InputBuf {
    /// Reset the per-frame counters/edge flags. Called *after* a frame's
    /// systems have consumed them.
    pub fn clear_per_frame(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.key_pressed_this_frame.clear();
        self.lmb_pressed = false;
        self.rmb_pressed = false;
    }

    /// Add a raw mouse-motion delta from winit's `DeviceEvent::MouseMotion`.
    /// We accumulate (rather than overwrite) so multiple events between
    /// frames don't get lost.
    pub fn on_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_dx += dx as f32;
        self.mouse_dy += dy as f32;
    }

    /// Record a key press/release. New presses also fill the
    /// `key_pressed_this_frame` set so callers can detect edge events.
    pub fn on_key(&mut self, key: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                // `insert` returns true only on transition from absent to
                // present — exactly the "new press this frame" semantic.
                if self.keys_down.insert(key) {
                    self.key_pressed_this_frame.insert(key);
                }
            }
            ElementState::Released => {
                self.keys_down.remove(&key);
            }
        }
    }

    /// Record a mouse-button press. We only care about pressed transitions
    /// for v0 (no held-down detection needed yet).
    pub fn on_mouse_button(&mut self, btn: MouseButton, state: ElementState) {
        if state == ElementState::Pressed {
            match btn {
                MouseButton::Left => self.lmb_pressed = true,
                MouseButton::Right => self.rmb_pressed = true,
                _ => {}
            }
        }
    }
}

/// Mouse sensitivity factor — radians per pixel of raw mouse delta. Picked
/// for a typical 1000 DPI mouse; would graduate to a setting in v0.2.
const MOUSE_SENS: f32 = 0.0025;

/// Drive the player's `Camera`, `PlayerInput`, and `Movement` components
/// from the current [`InputBuf`].
pub fn apply_input(ecs: &mut GameEcs, buf: &InputBuf) {
    // Single composite query: cheap with hecs's archetype storage.
    let mut q = ecs
        .world
        .query_one::<(&mut Camera, &mut PlayerInput, &mut Movement)>(ecs.player)
        .unwrap();
    let (cam, input, movement) = q.get().unwrap();

    // Yaw turns the camera around +Y; sign matches "moving the mouse right
    // turns right". Pitch *subtracts* dy because raw mouse Y grows downward.
    cam.yaw += buf.mouse_dx * MOUSE_SENS;
    cam.pitch -= buf.mouse_dy * MOUSE_SENS;
    // Clamp slightly under ±90° to avoid degenerate look-at vectors.
    cam.pitch = cam.pitch.clamp(-1.553, 1.553);

    let mut wd = Vec3::ZERO;
    if buf.keys_down.contains(&KeyCode::KeyW) {
        wd.z += 1.0;
    }
    if buf.keys_down.contains(&KeyCode::KeyS) {
        wd.z -= 1.0;
    }
    if buf.keys_down.contains(&KeyCode::KeyD) {
        wd.x += 1.0;
    }
    if buf.keys_down.contains(&KeyCode::KeyA) {
        wd.x -= 1.0;
    }
    // Vertical wishdir is only honored in Fly mode; the movement system
    // ignores `wd.y` while walking (jump press is a separate signal).
    if buf.keys_down.contains(&KeyCode::Space) {
        wd.y += 1.0;
    }
    if buf.keys_down.contains(&KeyCode::ShiftLeft) {
        wd.y -= 1.0;
    }
    input.wishdir = wd;
    input.jump = buf.keys_down.contains(&KeyCode::Space);
    input.sprint = buf.keys_down.contains(&KeyCode::ControlLeft);

    // Edge-triggered: `F` toggles between Walk and Fly.
    if buf.key_pressed_this_frame.contains(&KeyCode::KeyF) {
        movement.mode = match movement.mode {
            MovementMode::Walk => MovementMode::Fly,
            MovementMode::Fly => MovementMode::Walk,
        };
    }

    input.break_ = buf.lmb_pressed;
    input.place = buf.rmb_pressed;
}
