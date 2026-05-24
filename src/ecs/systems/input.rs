//! Translates resolved input actions into `PlayerInput` + camera deltas.
//!
//! The flow is:
//!
//! 1. Main thread (in `main.rs`) routes winit events into `InputEngine`.
//! 2. Once per frame, `InputEngine` resolves context-aware actions.
//! 3. [`apply_input`] turns those actions into changes on the player entity's
//!    `Camera`, `PlayerInput`, and `Movement`.
//!
//! This split — buffer → apply → clear — keeps the asynchronous OS event
//! arrival decoupled from the deterministic per-frame logic.

use crate::ecs::GameEcs;
use crate::ecs::components::{Camera, Movement, MovementMode, PlayerInput};
use crate::input_engine::{ActionState, InputAction};
use glam::Vec3;
use std::time::{Duration, Instant};

/// Cross-frame input bookkeeping. Held in `AppState` so it survives
/// across `apply_input` calls.
#[derive(Debug, Default)]
pub struct InputState {
    pub last_break_at: Option<Instant>,
    pub last_place_at: Option<Instant>,
    pub last_space_press_at: Option<Instant>,
}

const ACTION_REPEAT: Duration = Duration::from_millis(200);

/// Drive the player's `Camera`, `PlayerInput`, and `Movement` components
/// from the current [`ActionState`].
pub fn apply_input(ecs: &mut GameEcs, actions: &ActionState, state: &mut InputState) {
    // Single composite query: cheap with hecs's archetype storage.
    let mut q = ecs
        .world
        .query_one::<(&mut Camera, &mut PlayerInput, &mut Movement)>(ecs.player)
        .unwrap();
    let (cam, input, movement) = q.get().unwrap();

    // Yaw turns the camera around +Y; sign matches "moving the mouse right
    // turns right". Pitch *subtracts* dy because raw mouse Y grows downward.
    cam.yaw += actions.look_delta.x;
    cam.pitch -= actions.look_delta.y;
    // Clamp slightly under ±90° to avoid degenerate look-at vectors.
    cam.pitch = cam.pitch.clamp(-1.553, 1.553);

    let mut wd = Vec3::ZERO;
    if actions.down(InputAction::MoveForward) {
        wd.z += 1.0;
    }
    if actions.down(InputAction::MoveBack) {
        wd.z -= 1.0;
    }
    if actions.down(InputAction::MoveRight) {
        wd.x += 1.0;
    }
    if actions.down(InputAction::MoveLeft) {
        wd.x -= 1.0;
    }
    // Vertical wishdir is only honored in Fly mode; the movement system
    // ignores `wd.y` while walking (jump press is a separate signal).
    if actions.down(InputAction::Jump) {
        wd.y += 1.0;
    }
    if actions.down(InputAction::Descend) {
        wd.y -= 1.0;
    }
    input.wishdir = wd;
    input.jump = actions.down(InputAction::Jump);
    input.sprint = actions.down(InputAction::Sprint);

    // Edge-triggered: `F` toggles between Walk and Fly.
    if actions.pressed(InputAction::ToggleFly) {
        movement.mode = match movement.mode {
            MovementMode::Walk => MovementMode::Fly,
            MovementMode::Fly => MovementMode::Walk,
        };
    }

    // Double-tap Space within 280 ms also toggles Walk ⇄ Fly.
    const DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(280);
    if actions.pressed(InputAction::Jump) {
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
    let break_down = actions.down(InputAction::Break);
    let break_fired = actions.pressed(InputAction::Break)
        || (break_down && state.last_break_at.is_none_or(|t| now - t >= ACTION_REPEAT));
    input.break_ = break_fired;
    if break_fired {
        state.last_break_at = Some(now);
    }
    if !break_down {
        state.last_break_at = None;
    }

    let place_down = actions.down(InputAction::Place);
    let place_fired = actions.pressed(InputAction::Place)
        || (place_down && state.last_place_at.is_none_or(|t| now - t >= ACTION_REPEAT));
    input.place = place_fired;
    if place_fired {
        state.last_place_at = Some(now);
    }
    if !place_down {
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
    let mut sq = ecs
        .world
        .query_one::<&mut crate::ecs::components::Selected>(ecs.player)
        .unwrap();
    let sel = sq.get().unwrap();

    // 1..9 keys: jump straight to that slot (only the slots whose
    // block is `Some` actually change the selection — empty slot 9
    // is ignored).
    if let Some(i) = actions.selected_slot
        && let Some(block) = HOTBAR_BLOCKS[i]
    {
        sel.0 = block;
    }

    // Scroll wheel: each notch (winit reports ~1.0 per detent on a
    // typical mouse) steps the selection forward or backward by one
    // slot. Wraps at the ends. Empty slots are skipped so the scroll
    // never lands on a "nothing to place" state.
    if actions.hotbar_step != 0 {
        let cur = HOTBAR_BLOCKS
            .iter()
            .position(|b| *b == Some(sel.0))
            .unwrap_or(0) as i32;
        let dir = actions.hotbar_step.signum();
        let n = HOTBAR_BLOCKS.len() as i32;
        // Step until we land on a non-empty slot — at most `n` steps,
        // and since there's always at least one populated slot we
        // always terminate.
        let mut next = cur;
        for _ in 0..actions.hotbar_step.abs() {
            for _ in 0..n {
                next = (next + dir).rem_euclid(n);
                if let Some(block) = HOTBAR_BLOCKS[next as usize] {
                    sel.0 = block;
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::GameEcs;
    use crate::input_engine::InputAction;
    use glam::Vec3;
    use std::thread::sleep;

    fn actions() -> ActionState {
        ActionState::default()
    }

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
        let mut actions = actions();
        let mut st = InputState::default();
        actions.set_pressed(InputAction::Break);
        actions.set_down(InputAction::Break);
        apply_input(&mut e, &actions, &mut st);
        let pi = read_pi(&mut e);
        assert!(pi.break_);
    }

    #[test]
    fn lmb_held_repeats_after_cooldown() {
        let mut e = ecs();
        let mut actions = actions();
        let mut st = InputState::default();
        actions.set_pressed(InputAction::Break);
        actions.set_down(InputAction::Break);
        apply_input(&mut e, &actions, &mut st);
        // Clear edge flag (frame boundary), hold stays.
        actions.pressed.clear();
        apply_input(&mut e, &actions, &mut st);
        let pi = read_pi(&mut e);
        assert!(!pi.break_, "should not fire within cooldown window");
        // Wait past cooldown.
        sleep(ACTION_REPEAT + Duration::from_millis(20));
        apply_input(&mut e, &actions, &mut st);
        let pi = read_pi(&mut e);
        assert!(pi.break_, "should fire after cooldown");
    }

    #[test]
    fn lmb_release_resets_timer() {
        let mut e = ecs();
        let mut actions = actions();
        let mut st = InputState::default();
        actions.set_pressed(InputAction::Break);
        actions.set_down(InputAction::Break);
        apply_input(&mut e, &actions, &mut st);
        actions.pressed.clear();
        actions.down.clear();
        apply_input(&mut e, &actions, &mut st);
        assert!(st.last_break_at.is_none());
    }

    #[test]
    fn double_tap_space_toggles_fly() {
        let mut e = ecs();
        let mut actions = actions();
        let mut st = InputState::default();
        actions.set_pressed(InputAction::Jump);
        actions.set_down(InputAction::Jump);
        apply_input(&mut e, &actions, &mut st);
        let mv = e
            .world
            .query_one::<&crate::ecs::components::Movement>(e.player)
            .unwrap()
            .get()
            .unwrap()
            .mode;
        assert_eq!(mv, crate::ecs::components::MovementMode::Walk);

        actions.pressed.clear();
        actions.set_pressed(InputAction::Jump);
        apply_input(&mut e, &actions, &mut st);
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
        let mut actions = actions();
        let mut st = InputState::default();
        actions.set_pressed(InputAction::Jump);
        apply_input(&mut e, &actions, &mut st);
        sleep(Duration::from_millis(350));
        actions.pressed.clear();
        actions.set_pressed(InputAction::Jump);
        apply_input(&mut e, &actions, &mut st);
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
        let mut actions = actions();
        let mut st = InputState::default();
        actions.set_pressed(InputAction::ToggleFly);
        apply_input(&mut e, &actions, &mut st);
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
