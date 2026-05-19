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
use crate::ui::menu::TOP_MENU;
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
    /// Set to `true` when a `UiEffect::Quit` is drained, so the event
    /// loop can call `event_loop.exit()` from outside the step.
    pub wants_quit: bool,
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
            wants_quit: false,
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

    /// Handle non-toggle keys while the UI is open.
    fn consume_in_ui(&mut self, code: KeyCode, text: Option<&str>) {
        match &mut self.state {
            UiState::Paused { menu: MenuNav::Top { hovered } } => {
                let action = match code {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        *hovered = (*hovered + TOP_MENU.len() - 1) % TOP_MENU.len();
                        None
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        *hovered = (*hovered + 1) % TOP_MENU.len();
                        None
                    }
                    KeyCode::Enter | KeyCode::Space | KeyCode::NumpadEnter => {
                        Some(TOP_MENU[*hovered].activate())
                    }
                    _ => None,
                };
                if let Some(a) = action {
                    self.apply_menu_action(a);
                }
            }
            UiState::Paused { menu: MenuNav::Settings } => {}
            UiState::Chat { input, .. } => {
                let submitted: Option<String> = match code {
                    KeyCode::Backspace  => { input.backspace(); None }
                    KeyCode::Delete     => { input.delete_forward(); None }
                    KeyCode::ArrowLeft  => { input.move_left(); None }
                    KeyCode::ArrowRight => { input.move_right(); None }
                    KeyCode::Home       => { input.move_home(); None }
                    KeyCode::End        => { input.move_end(); None }
                    KeyCode::ArrowUp    => { input.history_prev(); None }
                    KeyCode::ArrowDown  => { input.history_next(); None }
                    KeyCode::Enter | KeyCode::NumpadEnter => Some(input.submit()),
                    _ => {
                        if let Some(t) = text {
                            if !t.chars().any(|c| c.is_control()) {
                                input.insert_text(t);
                            }
                        }
                        None
                    }
                };
                if let Some(line) = submitted {
                    self.submit_chat(&line);
                    self.state = UiState::Playing;
                    self.cursor_state_changed = true;
                }
            }
            UiState::Playing => {}
        }
    }

    fn apply_menu_action(&mut self, action: crate::ui::menu::MenuAction) {
        use crate::ui::menu::MenuAction;
        match action {
            MenuAction::Resume => {
                self.state = UiState::Playing;
                self.cursor_state_changed = true;
            }
            MenuAction::Save => {
                self.push_effect(UiEffect::Save);
                self.log.push_system("Saved.");
            }
            MenuAction::OpenSettings => {
                self.state = UiState::Paused { menu: MenuNav::Settings };
            }
            MenuAction::BackToTop => {
                self.state = UiState::Paused { menu: MenuNav::Top { hovered: 0 } };
            }
            MenuAction::Quit => {
                self.push_effect(UiEffect::Quit);
            }
        }
    }

    fn submit_chat(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() { return }
        if !line.starts_with('/') {
            self.log.push_player(line);
            return;
        }
        self.log.push_echo(line);

        // Special-case /help so it can walk the registry, which
        // Command::run can't access.
        if line == "/help" {
            for c in self.commands.all() {
                self.log.push_system(c.help());
            }
            return;
        }

        match self.commands.dispatch(line) {
            Ok(effs) => {
                for e in effs {
                    match e {
                        UiEffect::PostMessage(msg) => self.log.push_system(msg),
                        UiEffect::ClearChat => self.log.clear(),
                        other => self.push_effect(other),
                    }
                }
            }
            Err(msg) => self.log.push_error(msg),
        }
    }

    pub fn on_mouse_button(&mut self, button: winit::event::MouseButton, state: ElementState) {
        if state != ElementState::Pressed { return }
        if button != winit::event::MouseButton::Left { return }
        let action = match &self.state {
            UiState::Paused { menu: MenuNav::Top { hovered } } => {
                Some(TOP_MENU[*hovered].activate())
            }
            _ => None,
        };
        if let Some(a) = action {
            self.apply_menu_action(a);
        }
    }

    pub fn on_mouse_move(&mut self, x: f32, y: f32, screen_px: (u32, u32)) {
        let UiState::Paused { menu: MenuNav::Top { hovered } } = &mut self.state else { return };
        for i in 0..TOP_MENU.len() {
            let (rx, ry, rw, rh) = crate::ui::render::top_menu_item_rect(i, screen_px);
            if x >= rx && x < rx + rw && y >= ry && y < ry + rh {
                *hovered = i;
                return;
            }
        }
    }
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

    #[test]
    fn arrow_down_advances_hover() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        ui.on_key(KeyCode::ArrowDown, Pressed, None);
        match ui.state {
            UiState::Paused { menu: MenuNav::Top { hovered } } => assert_eq!(hovered, 1),
            _ => panic!("expected paused top"),
        }
    }

    #[test]
    fn arrow_up_wraps_at_top() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        ui.on_key(KeyCode::ArrowUp, Pressed, None);
        match ui.state {
            UiState::Paused { menu: MenuNav::Top { hovered } } => {
                assert_eq!(hovered, crate::ui::menu::TOP_MENU.len() - 1);
            }
            _ => panic!("expected paused top"),
        }
    }

    #[test]
    fn enter_on_resume_returns_to_playing() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        ui.on_key(KeyCode::Enter, Pressed, None);
        assert!(ui.is_playing());
    }

    #[test]
    fn enter_on_quit_emits_effect() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        for _ in 0..3 { ui.on_key(KeyCode::ArrowDown, Pressed, None); }
        ui.on_key(KeyCode::Enter, Pressed, None);
        let effs = ui.drain_effects();
        assert!(effs.contains(&UiEffect::Quit));
    }

    #[test]
    fn enter_on_save_emits_effect_and_stays() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        ui.on_key(KeyCode::ArrowDown, Pressed, None);
        ui.on_key(KeyCode::Enter, Pressed, None);
        assert!(matches!(ui.state, UiState::Paused { .. }));
        let effs = ui.drain_effects();
        assert!(effs.contains(&UiEffect::Save));
    }

    #[test]
    fn enter_on_settings_opens_sub_menu() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        for _ in 0..2 { ui.on_key(KeyCode::ArrowDown, Pressed, None); }
        ui.on_key(KeyCode::Enter, Pressed, None);
        assert!(matches!(ui.state, UiState::Paused { menu: MenuNav::Settings }));
    }

    #[test]
    fn esc_from_settings_goes_to_top_not_play() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        for _ in 0..2 { ui.on_key(KeyCode::ArrowDown, Pressed, None); }
        ui.on_key(KeyCode::Enter, Pressed, None);
        ui.on_key(KeyCode::Escape, Pressed, None);
        assert!(matches!(ui.state, UiState::Paused { menu: MenuNav::Top { hovered: 0 } }));
    }
}
