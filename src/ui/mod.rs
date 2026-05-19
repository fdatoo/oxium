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

use crate::ui::chat::ChatLog;
use crate::ui::commands::Registry;
use crate::ui::effect::UiEffect;
use crate::ui::state::UiState;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
