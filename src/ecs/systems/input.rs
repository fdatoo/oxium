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

use crate::ecs::GameEcs;
use crate::ecs::components::{Camera, Movement, MovementMode, PlayerInput};
use glam::Vec3;
use std::time::{Duration, Instant};
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
    /// Signed scroll-wheel delta this frame (positive = wheel up =
    /// previous hotbar slot, matching Minecraft's convention).
    pub scroll_delta: f32,
    /// Continuous held state (set on press, cleared on release).
    pub lmb_down: bool,
    pub rmb_down: bool,
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
        self.scroll_delta = 0.0;
    }

    /// Add a scroll-wheel delta. winit reports lines on a desktop
    /// mouse and pixels on a trackpad; we accumulate the signed
    /// magnitude either way and let `apply_input` decide on a step.
    pub fn on_scroll(&mut self, lines: f32) {
        self.scroll_delta += lines;
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

    /// Record a mouse-button press or release. Edge flags are set on press;
    /// continuous held state is set on press and cleared on release.
    pub fn on_mouse_button(&mut self, btn: MouseButton, state: ElementState) {
        match (btn, state) {
            (MouseButton::Left, ElementState::Pressed) => {
                self.lmb_pressed = true;
                self.lmb_down = true;
            }
            (MouseButton::Left, ElementState::Released) => {
                self.lmb_down = false;
            }
            (MouseButton::Right, ElementState::Pressed) => {
                self.rmb_pressed = true;
                self.rmb_down = true;
            }
            (MouseButton::Right, ElementState::Released) => {
                self.rmb_down = false;
            }
            _ => {}
        }
    }
}

/// Cross-frame input bookkeeping. Held in `AppState` so it survives
/// across `apply_input` calls.
#[derive(Debug, Default)]
pub struct InputState {
    pub last_break_at: Option<Instant>,
    pub last_place_at: Option<Instant>,
    pub last_space_press_at: Option<Instant>,
}

const ACTION_REPEAT: Duration = Duration::from_millis(200);

/// Mouse sensitivity factor — radians per pixel of raw mouse delta. Picked
/// for a typical 1000 DPI mouse; would graduate to a setting in v0.2.
const MOUSE_SENS: f32 = 0.0025;

/// Drive the player's `Camera`, `PlayerInput`, and `Movement` components
/// from the current [`InputBuf`].
pub fn apply_input(ecs: &mut GameEcs, buf: &InputBuf, state: &mut InputState) {
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

    // Double-tap Space within 280 ms also toggles Walk ⇄ Fly.
    const DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(280);
    if buf.key_pressed_this_frame.contains(&KeyCode::Space) {
        let now = Instant::now();
        if state
            .last_space_press_at
            .is_some_and(|t| now - t <= DOUBLE_TAP_WINDOW)
        {
            movement.mode = match movement.mode {
                MovementMode::Walk => MovementMode::Fly,
                MovementMode::Fly => MovementMode::Walk,
            };
            state.last_space_press_at = None;
        } else {
            state.last_space_press_at = Some(now);
        }
    }

    // Hold-to-act with `ACTION_REPEAT` cooldown. Press fires immediately
    // (edge flag), then while still held, fires once every cooldown.
    // Release resets the timer so the *next* tap fires immediately too.
    let now = Instant::now();
    let break_fired = buf.lmb_pressed
        || (buf.lmb_down && state.last_break_at.is_none_or(|t| now - t >= ACTION_REPEAT));
    input.break_ = break_fired;
    if break_fired {
        state.last_break_at = Some(now);
    }
    if !buf.lmb_down {
        state.last_break_at = None;
    }

    let place_fired = buf.rmb_pressed
        || (buf.rmb_down && state.last_place_at.is_none_or(|t| now - t >= ACTION_REPEAT));
    input.place = place_fired;
    if place_fired {
        state.last_place_at = Some(now);
    }
    if !buf.rmb_down {
        state.last_place_at = None;
    }

    // Number-row 1..8 + scroll wheel cycle the currently-selected
    // block. Held in its own query so the borrow above can release
    // before we touch a different component on the same entity.
    //
    // The hotbar layout is the single source of truth for which block
    // each slot holds — `apply_input` just steps the *slot index* and
    // resolves the block via `HOTBAR_BLOCKS`. That keeps the input
    // path and the HUD's icon rendering in sync without a second
    // ordering list to maintain.
    drop(q);
    use crate::render::hud::HOTBAR_BLOCKS;
    let digit_keys = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    let mut sq = ecs
        .world
        .query_one::<&mut crate::ecs::components::Selected>(ecs.player)
        .unwrap();
    let sel = sq.get().unwrap();

    // 1..9 keys: jump straight to that slot (only the slots whose
    // block is `Some` actually change the selection — empty slot 9
    // is ignored).
    for (i, k) in digit_keys.iter().enumerate() {
        if buf.key_pressed_this_frame.contains(k)
            && let Some(block) = HOTBAR_BLOCKS[i]
        {
            sel.0 = block;
        }
    }

    // Scroll wheel: each notch (winit reports ~1.0 per detent on a
    // typical mouse) steps the selection forward or backward by one
    // slot. Wraps at the ends. Empty slots are skipped so the scroll
    // never lands on a "nothing to place" state.
    if buf.scroll_delta.abs() >= 0.5 {
        let cur = HOTBAR_BLOCKS
            .iter()
            .position(|b| *b == Some(sel.0))
            .unwrap_or(0) as i32;
        let dir = if buf.scroll_delta > 0.0 { -1 } else { 1 };
        let n = HOTBAR_BLOCKS.len() as i32;
        // Step until we land on a non-empty slot — at most `n` steps,
        // and since there's always at least one populated slot we
        // always terminate.
        let mut next = cur;
        for _ in 0..n {
            next = (next + dir).rem_euclid(n);
            if let Some(block) = HOTBAR_BLOCKS[next as usize] {
                sel.0 = block;
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::GameEcs;
    use glam::Vec3;
    use std::thread::sleep;
    use winit::keyboard::KeyCode;

    fn ecs() -> GameEcs {
        GameEcs::new(Vec3::new(0.0, 64.0, 0.0))
    }

    fn read_pi(e: &mut GameEcs) -> crate::ecs::components::PlayerInput {
        let mut q = e
            .world
            .query_one::<&crate::ecs::components::PlayerInput>(e.player)
            .unwrap();
        *q.get().unwrap()
    }

    #[test]
    fn lmb_press_fires_break_once() {
        let mut e = ecs();
        let mut buf = InputBuf::default();
        let mut st = InputState::default();
        buf.lmb_pressed = true;
        buf.lmb_down = true;
        apply_input(&mut e, &buf, &mut st);
        let pi = read_pi(&mut e);
        assert!(pi.break_);
    }

    #[test]
    fn lmb_held_repeats_after_cooldown() {
        let mut e = ecs();
        let mut buf = InputBuf::default();
        let mut st = InputState::default();
        buf.lmb_pressed = true;
        buf.lmb_down = true;
        apply_input(&mut e, &buf, &mut st);
        // Clear edge flag (frame boundary), hold stays.
        buf.lmb_pressed = false;
        apply_input(&mut e, &buf, &mut st);
        let pi = read_pi(&mut e);
        assert!(!pi.break_, "should not fire within cooldown window");
        // Wait past cooldown.
        sleep(ACTION_REPEAT + Duration::from_millis(20));
        apply_input(&mut e, &buf, &mut st);
        let pi = read_pi(&mut e);
        assert!(pi.break_, "should fire after cooldown");
    }

    #[test]
    fn lmb_release_resets_timer() {
        let mut e = ecs();
        let mut buf = InputBuf::default();
        let mut st = InputState::default();
        buf.lmb_pressed = true;
        buf.lmb_down = true;
        apply_input(&mut e, &buf, &mut st);
        buf.lmb_pressed = false;
        buf.lmb_down = false;
        apply_input(&mut e, &buf, &mut st);
        assert!(st.last_break_at.is_none());
    }

    #[test]
    fn double_tap_space_toggles_fly() {
        let mut e = ecs();
        let mut buf = InputBuf::default();
        let mut st = InputState::default();
        buf.key_pressed_this_frame.insert(KeyCode::Space);
        buf.keys_down.insert(KeyCode::Space);
        apply_input(&mut e, &buf, &mut st);
        let mv = e
            .world
            .query_one::<&crate::ecs::components::Movement>(e.player)
            .unwrap()
            .get()
            .unwrap()
            .mode;
        assert_eq!(mv, crate::ecs::components::MovementMode::Walk);

        buf.key_pressed_this_frame.clear();
        buf.key_pressed_this_frame.insert(KeyCode::Space);
        apply_input(&mut e, &buf, &mut st);
        let mv = e
            .world
            .query_one::<&crate::ecs::components::Movement>(e.player)
            .unwrap()
            .get()
            .unwrap()
            .mode;
        assert_eq!(mv, crate::ecs::components::MovementMode::Fly);
    }

    #[test]
    fn slow_space_presses_do_not_toggle() {
        let mut e = ecs();
        let mut buf = InputBuf::default();
        let mut st = InputState::default();
        buf.key_pressed_this_frame.insert(KeyCode::Space);
        apply_input(&mut e, &buf, &mut st);
        sleep(Duration::from_millis(350));
        buf.key_pressed_this_frame.clear();
        buf.key_pressed_this_frame.insert(KeyCode::Space);
        apply_input(&mut e, &buf, &mut st);
        let mv = e
            .world
            .query_one::<&crate::ecs::components::Movement>(e.player)
            .unwrap()
            .get()
            .unwrap()
            .mode;
        assert_eq!(mv, crate::ecs::components::MovementMode::Walk);
    }

    #[test]
    fn f_still_toggles_fly() {
        let mut e = ecs();
        let mut buf = InputBuf::default();
        let mut st = InputState::default();
        buf.key_pressed_this_frame.insert(KeyCode::KeyF);
        apply_input(&mut e, &buf, &mut st);
        let mv = e
            .world
            .query_one::<&crate::ecs::components::Movement>(e.player)
            .unwrap()
            .get()
            .unwrap()
            .mode;
        assert_eq!(mv, crate::ecs::components::MovementMode::Fly);
    }
}
