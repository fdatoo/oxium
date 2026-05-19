//! Pause menu + chat/commands UI. Pure data — no winit/wgpu handles.
//! See `docs/superpowers/specs/2026-05-19-ui-pause-chat-design.md`.

pub mod chat;
pub mod commands;
pub mod effect;
pub mod input;
pub mod menu;
pub mod render;
pub mod state;

use std::collections::VecDeque;

use winit::event::ElementState;
use winit::keyboard::KeyCode;

use crate::ui::chat::{ChatInput, ChatLog};
use crate::ui::commands::Registry;
use crate::ui::effect::UiEffect;
use crate::ui::input::InputDisposition;
use crate::ui::state::{MenuNav, UiState};

pub struct Ui {
    pub state: UiState,
    pub log: ChatLog,
    pub commands: Registry,
    /// Drained once per frame by `AppState::step`.
    effects: VecDeque<UiEffect>,
    /// Set when a state transition crossed the `Playing` boundary;
    /// `main.rs` reads + clears this to (re-)grab the cursor.
    pub cursor_state_changed: bool,
}

impl Default for Ui {
    fn default() -> Self {
        Self::new()
    }
}

impl Ui {
    pub fn new() -> Self {
        Self {
            state: UiState::Playing,
            log: ChatLog::new(),
            commands: Registry::builtin(),
            effects: VecDeque::new(),
            cursor_state_changed: false,
        }
    }

    pub fn is_playing(&self) -> bool {
        matches!(self.state, UiState::Playing)
    }

    /// Advance any UI time-keeping. Currently a no-op; reserved for
    /// future caret-blink accumulators if we ever need them.
    pub fn tick(&mut self, _dt: f32) {}

    /// Drain queued effects; caller (AppState) applies them.
    pub fn drain_effects(&mut self) -> Vec<UiEffect> {
        self.effects.drain(..).collect()
    }

    pub fn draw_overlay(&self, screen_px: (u32, u32), frame: &mut crate::render::hud::HudFrame) {
        render::draw_overlay(self, screen_px, frame);
    }

    pub(crate) fn push_effect(&mut self, e: UiEffect) {
        self.effects.push_back(e);
    }

    /// Route a key event. Returns `Consumed` if the UI handled it (in
    /// which case `main.rs` does NOT forward the key to `InputBuf`),
    /// `Forward` otherwise.
    pub fn on_key(&mut self, code: KeyCode, state: ElementState, text: Option<&str>) -> InputDisposition {
        // Always forward release events so `InputBuf` can clean up its
        // `keys_down` set. A press → pause → release sequence (W held
        // when the player Escs, then W released while paused) must not
        // leave W stuck as "down" or apply_input will phantom-walk on
        // resume.
        if state != ElementState::Pressed {
            return InputDisposition::Forward;
        }

        // Toggle keys: Esc / T / Slash. These are handled regardless of
        // current state because they cross between states.
        let was_playing = self.is_playing();
        let handled = self.handle_toggle_key(code, text);
        if handled {
            if was_playing != self.is_playing() {
                self.cursor_state_changed = true;
            }
            return InputDisposition::Consumed;
        }

        // Non-toggle keys: in Playing, forward to InputBuf. In Paused or
        // Chat, the UI consumes (later tasks add menu nav + chat input).
        if self.is_playing() {
            InputDisposition::Forward
        } else {
            self.consume_in_ui(code, text);
            InputDisposition::Consumed
        }
    }

    fn handle_toggle_key(&mut self, code: KeyCode, _text: Option<&str>) -> bool {
        match (&self.state, code) {
            (UiState::Playing, KeyCode::Escape) => {
                self.state = UiState::Paused { menu: MenuNav::Top { hovered: 0 } };
                true
            }
            (UiState::Playing, KeyCode::KeyT) => {
                self.state = UiState::Chat { input: ChatInput::new(""), prefilled_slash: false };
                true
            }
            (UiState::Playing, KeyCode::Slash) => {
                self.state = UiState::Chat { input: ChatInput::new("/"), prefilled_slash: true };
                true
            }
            (UiState::Paused { menu: MenuNav::Top { .. } }, KeyCode::Escape) => {
                self.state = UiState::Playing;
                true
            }
            (UiState::Paused { menu: MenuNav::Settings }, KeyCode::Escape) => {
                self.state = UiState::Paused { menu: MenuNav::Top { hovered: 0 } };
                true
            }
            (UiState::Chat { .. }, KeyCode::Escape) => {
                self.state = UiState::Playing;
                true
            }
            _ => false,
        }
    }

    /// Handle non-toggle keys while the UI is open. Stub — later tasks
    /// fill in menu nav and chat editing.
    fn consume_in_ui(&mut self, _code: KeyCode, _text: Option<&str>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::event::ElementState::Pressed;
    use winit::keyboard::KeyCode;

    #[test]
    fn new_starts_in_playing_state() {
        let ui = Ui::new();
        assert!(ui.is_playing());
    }

    #[test]
    fn drain_effects_returns_then_clears() {
        let mut ui = Ui::new();
        ui.push_effect(UiEffect::Save);
        ui.push_effect(UiEffect::Quit);
        let drained = ui.drain_effects();
        assert_eq!(drained, vec![UiEffect::Save, UiEffect::Quit]);
        assert!(ui.drain_effects().is_empty());
    }

    #[test]
    fn esc_in_playing_opens_pause_menu() {
        let mut ui = Ui::new();
        let d = ui.on_key(KeyCode::Escape, Pressed, None);
        assert_eq!(d, InputDisposition::Consumed);
        assert!(matches!(ui.state, UiState::Paused { menu: MenuNav::Top { hovered: 0 } }));
        assert!(ui.cursor_state_changed);
    }

    #[test]
    fn esc_in_paused_top_resumes() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None); // enter pause
        ui.cursor_state_changed = false;          // reset flag
        let d = ui.on_key(KeyCode::Escape, Pressed, None);
        assert_eq!(d, InputDisposition::Consumed);
        assert!(ui.is_playing());
        assert!(ui.cursor_state_changed);
    }

    #[test]
    fn t_opens_chat_empty() {
        let mut ui = Ui::new();
        let d = ui.on_key(KeyCode::KeyT, Pressed, None);
        assert_eq!(d, InputDisposition::Consumed);
        match &ui.state {
            UiState::Chat { input, prefilled_slash } => {
                assert_eq!(input.buf, "");
                assert!(!prefilled_slash);
            }
            _ => panic!("expected Chat state"),
        }
    }

    #[test]
    fn slash_opens_chat_with_slash() {
        let mut ui = Ui::new();
        let d = ui.on_key(KeyCode::Slash, Pressed, None);
        assert_eq!(d, InputDisposition::Consumed);
        match &ui.state {
            UiState::Chat { input, prefilled_slash } => {
                assert_eq!(input.buf, "/");
                assert_eq!(input.cursor, 1);
                assert!(prefilled_slash);
            }
            _ => panic!("expected Chat state"),
        }
    }

    #[test]
    fn movement_keys_forward_while_playing() {
        let mut ui = Ui::new();
        let d = ui.on_key(KeyCode::KeyW, Pressed, None);
        assert_eq!(d, InputDisposition::Forward);
    }

    #[test]
    fn movement_keys_consumed_while_paused() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        let d = ui.on_key(KeyCode::KeyW, Pressed, None);
        assert_eq!(d, InputDisposition::Consumed);
    }
}
