//! Pause menu + chat/commands UI. Pure data — no winit/wgpu handles.
//! See `docs/superpowers/specs/2026-05-19-ui-pause-chat-design.md`.

pub mod chat;
pub mod commands;
pub mod effect;
#[cfg(test)]
pub mod input;
pub mod menu;
pub mod render;
pub mod state;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use winit::event::ElementState;
use winit::keyboard::{KeyCode, ModifiersState};

use crate::input_engine::{ActionState, InputAction, InputMode};
use crate::ui::chat::{ChatInput, ChatLog};
use crate::ui::commands::Registry;
use crate::ui::effect::UiEffect;
#[cfg(test)]
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
    chat_history: VecDeque<String>,
    chat_drag_anchor: Option<usize>,
    last_chat_click: Option<(Instant, usize)>,
}

pub trait ClipboardAccess {
    fn get_text(&mut self) -> Result<String, ClipboardError>;
    fn set_text(&mut self, text: &str) -> Result<(), ClipboardError>;
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ClipboardError {
    #[error("clipboard unavailable: {0}")]
    Unavailable(String),
    #[error("clipboard read failed: {0}")]
    Read(String),
    #[error("clipboard write failed: {0}")]
    Write(String),
}

pub struct Clipboard {
    inner: Option<Result<arboard::Clipboard, ClipboardError>>,
}

impl Clipboard {
    pub fn new() -> Self {
        Self { inner: None }
    }

    fn clipboard(&mut self) -> Result<&mut arboard::Clipboard, ClipboardError> {
        if self.inner.is_none() {
            self.inner = Some(
                arboard::Clipboard::new()
                    .map_err(|err| ClipboardError::Unavailable(err.to_string())),
            );
        }
        match self.inner.as_mut().expect("clipboard init state is set") {
            Ok(clipboard) => Ok(clipboard),
            Err(err) => Err(err.clone()),
        }
    }
}

impl Default for Clipboard {
    fn default() -> Self {
        Self::new()
    }
}

impl ClipboardAccess for Clipboard {
    fn get_text(&mut self) -> Result<String, ClipboardError> {
        self.clipboard()?
            .get_text()
            .map_err(|err| ClipboardError::Read(err.to_string()))
    }

    fn set_text(&mut self, text: &str) -> Result<(), ClipboardError> {
        self.clipboard()?
            .set_text(text.to_string())
            .map_err(|err| ClipboardError::Write(err.to_string()))
    }
}

fn primary_modifier(modifiers: ModifiersState) -> bool {
    if cfg!(target_os = "macos") {
        modifiers.super_key()
    } else {
        modifiers.control_key()
    }
}

fn line_nav_modifier(modifiers: ModifiersState) -> bool {
    primary_modifier(modifiers)
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
            chat_history: VecDeque::new(),
            chat_drag_anchor: None,
            last_chat_click: None,
        }
    }

    fn chat_input(&self, prefill: &str) -> ChatInput {
        let mut input = ChatInput::new(prefill);
        input.history = self.chat_history.clone();
        input
    }

    pub fn is_playing(&self) -> bool {
        matches!(self.state, UiState::Playing)
    }

    pub fn should_step_game(&self) -> bool {
        !matches!(self.state, UiState::Paused { .. })
    }

    pub fn input_mode(&self) -> InputMode {
        match self.state {
            UiState::Playing => InputMode::Gameplay,
            UiState::Paused { .. } => InputMode::PausedMenu,
            UiState::Chat { .. } => InputMode::Chat,
        }
    }

    pub fn is_chat_open(&self) -> bool {
        matches!(self.state, UiState::Chat { .. })
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

    pub fn apply_actions(&mut self, actions: &ActionState) {
        let was_playing = self.is_playing();
        let was_chat = matches!(self.state, UiState::Chat { .. });
        match &mut self.state {
            UiState::Playing => {
                if actions.pressed(InputAction::Pause) {
                    self.state = UiState::Paused {
                        menu: MenuNav::Top { hovered: 0 },
                    };
                } else if actions.pressed(InputAction::OpenChat) {
                    self.state = UiState::Chat {
                        input: self.chat_input(""),
                    };
                } else if actions.pressed(InputAction::OpenCommandChat) {
                    self.state = UiState::Chat {
                        input: self.chat_input("/"),
                    };
                }
            }
            UiState::Paused {
                menu: MenuNav::Top { hovered },
            } => {
                if actions.pressed(InputAction::Pause) {
                    self.state = UiState::Playing;
                } else {
                    let action = if actions.pressed(InputAction::MenuUp) {
                        *hovered = (*hovered + TOP_MENU.len() - 1) % TOP_MENU.len();
                        None
                    } else if actions.pressed(InputAction::MenuDown) {
                        *hovered = (*hovered + 1) % TOP_MENU.len();
                        None
                    } else if actions.pressed(InputAction::MenuAccept) {
                        Some(TOP_MENU[*hovered].activate())
                    } else {
                        None
                    };
                    if let Some(action) = action {
                        self.apply_menu_action(action);
                    }
                }
            }
            UiState::Paused {
                menu: MenuNav::Settings,
            } => {
                if actions.pressed(InputAction::Pause) {
                    self.state = UiState::Paused {
                        menu: MenuNav::Top { hovered: 0 },
                    };
                }
            }
            UiState::Chat { input } if was_chat => {
                if actions.pressed(InputAction::Pause) {
                    self.state = UiState::Playing;
                } else {
                    if actions.pressed(InputAction::ChatBackspace) {
                        input.backspace();
                    }
                    if actions.pressed(InputAction::ChatDelete) {
                        input.delete_forward();
                    }
                    if actions.pressed(InputAction::ChatLeft) {
                        input.move_left();
                    }
                    if actions.pressed(InputAction::ChatRight) {
                        input.move_right();
                    }
                    if actions.pressed(InputAction::ChatHome) {
                        input.move_home();
                    }
                    if actions.pressed(InputAction::ChatEnd) {
                        input.move_end();
                    }
                    if actions.pressed(InputAction::ChatHistoryPrev) {
                        input.history_prev();
                    }
                    if actions.pressed(InputAction::ChatHistoryNext) {
                        input.history_next();
                    }
                    if actions.pressed(InputAction::ChatComplete) {
                        Self::apply_chat_completion(&self.commands, input);
                    }
                    for text in &actions.text {
                        input.insert_text(text);
                    }
                    if actions.pressed(InputAction::ChatSubmit) {
                        let line = input.submit();
                        self.chat_history = input.history.clone();
                        if !self.submit_chat(&line) {
                            self.state = UiState::Playing;
                        }
                    }
                }
            }
            UiState::Chat { .. } => {}
        }
        if was_playing != self.is_playing() {
            self.cursor_state_changed = true;
        }
    }

    pub(crate) fn push_effect(&mut self, e: UiEffect) {
        self.effects.push_back(e);
    }

    pub fn on_chat_ime_commit(&mut self, text: &str) -> bool {
        let UiState::Chat { input } = &mut self.state else {
            return false;
        };
        input.insert_text(text);
        true
    }

    pub fn on_chat_key(
        &mut self,
        code: KeyCode,
        text: Option<&str>,
        modifiers: ModifiersState,
        clipboard: &mut dyn ClipboardAccess,
    ) -> bool {
        let mut submitted = None;
        let mut close_chat = false;
        let mut clipboard_error = None;
        let mut chat_history = None;

        {
            let UiState::Chat { input } = &mut self.state else {
                return false;
            };

            let primary = primary_modifier(modifiers);
            let line_nav = line_nav_modifier(modifiers);
            let extend = modifiers.shift_key();
            let word_nav = modifiers.alt_key();

            match code {
                KeyCode::Escape => close_chat = true,
                KeyCode::KeyA if primary => input.select_all(),
                KeyCode::KeyC if primary => {
                    if let Some(range) = input.selected_range()
                        && let Err(err) = clipboard.set_text(&input.buf[range])
                    {
                        clipboard_error = Some(format!("Clipboard copy failed: {err}"));
                    }
                }
                KeyCode::KeyX if primary => {
                    if let Some(range) = input.selected_range() {
                        match clipboard.set_text(&input.buf[range]) {
                            Ok(()) => input.delete_forward(),
                            Err(err) => {
                                clipboard_error = Some(format!("Clipboard cut failed: {err}"))
                            }
                        }
                    }
                }
                KeyCode::KeyV if primary => match clipboard.get_text() {
                    Ok(text) => input.replace_selection(&text),
                    Err(err) => clipboard_error = Some(format!("Clipboard paste failed: {err}")),
                },
                KeyCode::Backspace if word_nav => input.delete_word_backward(),
                KeyCode::Delete if word_nav => input.delete_word_forward(),
                KeyCode::Backspace => input.delete_backward(),
                KeyCode::Delete => input.delete_forward(),
                KeyCode::ArrowLeft if word_nav => input.move_word_left(extend),
                KeyCode::ArrowRight if word_nav => input.move_word_right(extend),
                KeyCode::ArrowLeft if line_nav => input.move_home_ext(extend),
                KeyCode::ArrowRight if line_nav => input.move_end_ext(extend),
                KeyCode::ArrowLeft => input.move_left_ext(extend),
                KeyCode::ArrowRight => input.move_right_ext(extend),
                KeyCode::Home => input.move_home_ext(extend),
                KeyCode::End => input.move_end_ext(extend),
                KeyCode::ArrowUp => input.history_prev(),
                KeyCode::ArrowDown => input.history_next(),
                KeyCode::Tab => Self::apply_chat_completion(&self.commands, input),
                KeyCode::Enter | KeyCode::NumpadEnter => {
                    submitted = Some(input.submit());
                    chat_history = Some(input.history.clone());
                }
                _ if primary => {}
                _ => {
                    if let Some(text) = text
                        && !text.chars().any(|ch| ch.is_control())
                    {
                        input.insert_text(text);
                    }
                }
            }
        }

        if let Some(err) = clipboard_error {
            self.log.push_error(err);
        }
        if let Some(history) = chat_history {
            self.chat_history = history;
        }
        if let Some(line) = submitted {
            close_chat = !self.submit_chat(&line);
        }
        if close_chat {
            self.state = UiState::Playing;
            self.chat_drag_anchor = None;
            self.cursor_state_changed = true;
        }
        true
    }

    /// Route a key event. Returns `Consumed` if the UI handled it (in
    /// which case `main.rs` does NOT forward the key to `InputBuf`),
    /// `Forward` otherwise.
    #[cfg(test)]
    pub fn on_key(
        &mut self,
        code: KeyCode,
        state: ElementState,
        text: Option<&str>,
    ) -> InputDisposition {
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

    #[cfg(test)]
    fn handle_toggle_key(&mut self, code: KeyCode, _text: Option<&str>) -> bool {
        match (&self.state, code) {
            (UiState::Playing, KeyCode::Escape) => {
                self.state = UiState::Paused {
                    menu: MenuNav::Top { hovered: 0 },
                };
                true
            }
            (UiState::Playing, KeyCode::KeyT) => {
                self.state = UiState::Chat {
                    input: self.chat_input(""),
                };
                true
            }
            (UiState::Playing, KeyCode::Slash) => {
                self.state = UiState::Chat {
                    input: self.chat_input("/"),
                };
                true
            }
            (
                UiState::Paused {
                    menu: MenuNav::Top { .. },
                },
                KeyCode::Escape,
            ) => {
                self.state = UiState::Playing;
                true
            }
            (
                UiState::Paused {
                    menu: MenuNav::Settings,
                },
                KeyCode::Escape,
            ) => {
                self.state = UiState::Paused {
                    menu: MenuNav::Top { hovered: 0 },
                };
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
    #[cfg(test)]
    fn consume_in_ui(&mut self, code: KeyCode, text: Option<&str>) {
        match &mut self.state {
            UiState::Paused {
                menu: MenuNav::Top { hovered },
            } => {
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
            UiState::Paused {
                menu: MenuNav::Settings,
            } => {}
            UiState::Chat { input, .. } => {
                let submitted: Option<String> = match code {
                    KeyCode::Backspace => {
                        input.backspace();
                        None
                    }
                    KeyCode::Delete => {
                        input.delete_forward();
                        None
                    }
                    KeyCode::ArrowLeft => {
                        input.move_left();
                        None
                    }
                    KeyCode::ArrowRight => {
                        input.move_right();
                        None
                    }
                    KeyCode::Home => {
                        input.move_home();
                        None
                    }
                    KeyCode::End => {
                        input.move_end();
                        None
                    }
                    KeyCode::ArrowUp => {
                        input.history_prev();
                        None
                    }
                    KeyCode::ArrowDown => {
                        input.history_next();
                        None
                    }
                    KeyCode::Tab => {
                        Self::apply_chat_completion(&self.commands, input);
                        None
                    }
                    KeyCode::Enter | KeyCode::NumpadEnter => {
                        let line = input.submit();
                        self.chat_history = input.history.clone();
                        Some(line)
                    }
                    _ => {
                        if let Some(t) = text
                            && !t.chars().any(|c| c.is_control())
                        {
                            input.insert_text(t);
                        }
                        None
                    }
                };
                if let Some(line) = submitted {
                    if !self.submit_chat(&line) {
                        self.state = UiState::Playing;
                        self.cursor_state_changed = true;
                    }
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
                self.state = UiState::Paused {
                    menu: MenuNav::Settings,
                };
            }
            MenuAction::Quit => {
                self.push_effect(UiEffect::Quit);
            }
        }
    }

    fn submit_chat(&mut self, line: &str) -> bool {
        let line = line.trim();
        if line.is_empty() {
            return false;
        }
        if !line.starts_with('/') {
            self.log.push_player(line);
            return false;
        }
        let keep_open = command_name(line) == Some("help");
        self.log.push_echo(line);

        match self.commands.dispatch(line) {
            Ok(effs) => {
                for e in effs {
                    match e {
                        UiEffect::PostMessage(msg) => self.log.push_system(msg),
                        UiEffect::ClearChat => {
                            self.log.clear();
                            self.log.push_system("Chat cleared.");
                        }
                        other => self.push_effect(other),
                    }
                }
            }
            Err(err) => self.log.push_error(err.message),
        }
        keep_open
    }

    fn apply_chat_completion(commands: &Registry, input: &mut ChatInput) {
        let completions = if input.completion.entries.is_empty() {
            commands.complete(&input.buf, input.cursor)
        } else {
            crate::command::CompletionList {
                range: input
                    .completion
                    .range
                    .clone()
                    .unwrap_or(input.cursor..input.cursor),
                entries: input.completion.entries.clone(),
            }
        };
        input.apply_completion(completions);
    }

    pub fn on_mouse_button(&mut self, button: winit::event::MouseButton, state: ElementState) {
        if state != ElementState::Pressed {
            return;
        }
        if button != winit::event::MouseButton::Left {
            return;
        }
        let action = match &self.state {
            UiState::Paused {
                menu: MenuNav::Top { hovered },
            } => Some(TOP_MENU[*hovered].activate()),
            _ => None,
        };
        if let Some(a) = action {
            self.apply_menu_action(a);
        }
    }

    pub fn on_chat_mouse_down(&mut self, x: f32, y: f32, screen_px: (u32, u32)) -> bool {
        let UiState::Chat { input } = &mut self.state else {
            return false;
        };
        let layout = crate::ui::render::chat_layout(screen_px);
        let in_input = x >= layout.input_x
            && x < layout.input_x + layout.input_w
            && y >= layout.input_y
            && y < layout.input_y + layout.input_h;
        if !in_input {
            input.set_cursor(input.cursor, false);
            self.chat_drag_anchor = None;
            return true;
        }

        let byte = crate::ui::render::chat_input_byte_at(input, &layout, x);
        let now = Instant::now();
        let double_click = self.last_chat_click.is_some_and(|(then, last)| {
            now.duration_since(then) <= Duration::from_millis(400) && last == byte
        });
        self.last_chat_click = Some((now, byte));

        if double_click {
            let range = input.word_range_at(byte);
            input.selection_anchor = Some(range.start);
            input.cursor = range.end;
            self.chat_drag_anchor = None;
        } else {
            input.set_cursor(byte, false);
            input.selection_anchor = Some(byte);
            self.chat_drag_anchor = Some(byte);
        }
        true
    }

    pub fn on_chat_mouse_move(&mut self, x: f32, screen_px: (u32, u32)) -> bool {
        if self.chat_drag_anchor.is_none() {
            return false;
        }
        let UiState::Chat { input } = &mut self.state else {
            self.chat_drag_anchor = None;
            return false;
        };
        let layout = crate::ui::render::chat_layout(screen_px);
        let byte = crate::ui::render::chat_input_byte_at(input, &layout, x);
        input.set_cursor(byte, true);
        true
    }

    pub fn on_chat_mouse_up(&mut self) -> bool {
        let was_dragging = self.chat_drag_anchor.take().is_some();
        if let UiState::Chat { input } = &mut self.state
            && input.selected_range().is_none()
        {
            input.selection_anchor = None;
        }
        was_dragging
    }

    pub fn on_mouse_move(&mut self, x: f32, y: f32, screen_px: (u32, u32)) {
        let UiState::Paused {
            menu: MenuNav::Top { hovered },
        } = &mut self.state
        else {
            return;
        };
        for i in 0..TOP_MENU.len() {
            let (rx, ry, rw, rh) = crate::ui::render::top_menu_item_rect(i, screen_px);
            if x >= rx && x < rx + rw && y >= ry && y < ry + rh {
                *hovered = i;
                return;
            }
        }
    }
}

fn command_name(line: &str) -> Option<&str> {
    line.trim().strip_prefix('/')?.split_whitespace().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input_engine::{ActionState, InputAction};
    use winit::event::ElementState::Pressed;
    use winit::keyboard::{KeyCode, ModifiersState};

    #[derive(Default)]
    struct TestClipboard {
        text: String,
        fail_read: bool,
        fail_write: bool,
    }

    impl ClipboardAccess for TestClipboard {
        fn get_text(&mut self) -> Result<String, ClipboardError> {
            if self.fail_read {
                Err(ClipboardError::Read("read failed".to_string()))
            } else {
                Ok(self.text.clone())
            }
        }

        fn set_text(&mut self, text: &str) -> Result<(), ClipboardError> {
            if self.fail_write {
                Err(ClipboardError::Write("write failed".to_string()))
            } else {
                self.text = text.to_string();
                Ok(())
            }
        }
    }

    fn actions(pressed: &[InputAction], text: &[&str]) -> ActionState {
        let mut actions = ActionState::default();
        for action in pressed {
            actions.set_pressed(*action);
        }
        actions.text = text.iter().map(|s| s.to_string()).collect();
        actions
    }

    fn primary_mod() -> ModifiersState {
        if cfg!(target_os = "macos") {
            ModifiersState::SUPER
        } else {
            ModifiersState::CONTROL
        }
    }

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
        assert!(matches!(
            ui.state,
            UiState::Paused {
                menu: MenuNav::Top { hovered: 0 }
            }
        ));
        assert!(ui.cursor_state_changed);
    }

    #[test]
    fn pause_action_in_playing_opens_pause_menu() {
        let mut ui = Ui::new();
        ui.apply_actions(&actions(&[InputAction::Pause], &[]));
        assert!(matches!(ui.state, UiState::Paused { .. }));
        assert!(ui.cursor_state_changed);
    }

    #[test]
    fn open_command_chat_action_does_not_duplicate_slash_text() {
        let mut ui = Ui::new();
        ui.apply_actions(&actions(&[InputAction::OpenCommandChat], &["/"]));
        match &ui.state {
            UiState::Chat { input } => {
                assert_eq!(input.buf, "/");
                assert_eq!(input.cursor, 1);
            }
            _ => panic!("expected chat"),
        }
    }

    #[test]
    fn chat_complete_cycles_inline() {
        let mut ui = Ui::new();
        ui.apply_actions(&actions(&[InputAction::OpenCommandChat], &[]));
        ui.apply_actions(&actions(&[], &["t"]));
        ui.apply_actions(&actions(&[InputAction::ChatComplete], &[]));
        match &ui.state {
            UiState::Chat { input } => {
                assert_eq!(input.buf, "/teleport");
                assert_eq!(input.cursor, "/teleport".len());
                assert_eq!(input.completion.entries.len(), 3);
            }
            _ => panic!("expected chat"),
        }
        ui.apply_actions(&actions(&[InputAction::ChatComplete], &[]));
        match &ui.state {
            UiState::Chat { input } => {
                assert_eq!(input.buf, "/time");
                assert_eq!(input.cursor, "/time".len());
            }
            _ => panic!("expected chat"),
        }
    }

    #[test]
    fn live_hint_does_not_log_while_typing() {
        let mut ui = Ui::new();
        ui.apply_actions(&actions(&[InputAction::OpenCommandChat], &[]));
        ui.apply_actions(&actions(&[], &["tp 1 nope"]));
        let hint = match &ui.state {
            UiState::Chat { input } => ui.commands.hint(&input.buf, input.cursor).unwrap(),
            _ => panic!("expected chat"),
        };
        assert!(hint.is_error);
        assert_eq!(ui.log.len(), 0);
    }

    #[test]
    fn chat_action_inserts_text_only_when_chat_was_already_open() {
        let mut ui = Ui::new();
        ui.state = UiState::Chat {
            input: ChatInput::new(""),
        };
        ui.apply_actions(&actions(&[], &["hello"]));
        match &ui.state {
            UiState::Chat { input } => assert_eq!(input.buf, "hello"),
            _ => panic!("expected chat"),
        }
    }

    #[test]
    fn chat_history_survives_reopen() {
        let mut ui = Ui::new();
        ui.apply_actions(&actions(&[InputAction::OpenCommandChat], &[]));
        ui.apply_actions(&actions(&[], &["fly"]));
        ui.apply_actions(&actions(&[InputAction::ChatSubmit], &[]));
        assert!(ui.is_playing());

        ui.apply_actions(&actions(&[InputAction::OpenChat], &[]));
        ui.apply_actions(&actions(&[InputAction::ChatHistoryPrev], &[]));
        match &ui.state {
            UiState::Chat { input } => assert_eq!(input.buf, "/fly"),
            _ => panic!("expected chat"),
        }
    }

    #[test]
    fn chat_does_not_pause_game_step() {
        let mut ui = Ui::new();
        assert!(ui.should_step_game());
        ui.apply_actions(&actions(&[InputAction::OpenChat], &[]));
        assert!(!ui.is_playing());
        assert!(ui.should_step_game());

        let mut paused = Ui::new();
        paused.apply_actions(&actions(&[InputAction::Pause], &[]));
        assert!(!paused.should_step_game());
    }

    #[test]
    fn help_command_keeps_chat_open_with_header() {
        let mut ui = Ui::new();
        ui.apply_actions(&actions(&[InputAction::OpenCommandChat], &[]));
        ui.apply_actions(&actions(&[], &["help"]));
        ui.apply_actions(&actions(&[InputAction::ChatSubmit], &[]));

        assert!(matches!(ui.state, UiState::Chat { .. }));
        let lines: Vec<_> = ui.log.iter().map(|line| line.text.as_str()).collect();
        assert!(lines.contains(&"Commands:"));
        assert!(
            lines
                .iter()
                .any(|line| *line == "/tp <x> <y> <z> - teleport the player")
        );
    }

    #[test]
    fn chat_option_backspace_deletes_previous_word() {
        let mut ui = Ui::new();
        ui.state = UiState::Chat {
            input: ChatInput::new("/teleport home"),
        };
        let mut clipboard = TestClipboard::default();
        ui.on_chat_key(
            KeyCode::Backspace,
            None,
            ModifiersState::ALT,
            &mut clipboard,
        );
        match &ui.state {
            UiState::Chat { input } => assert_eq!(input.buf, "/teleport "),
            _ => panic!("expected chat"),
        }
    }

    #[test]
    fn chat_primary_clipboard_shortcuts_copy_cut_paste() {
        let mut ui = Ui::new();
        ui.state = UiState::Chat {
            input: ChatInput::new("alpha beta"),
        };
        let mut clipboard = TestClipboard::default();
        if let UiState::Chat { input } = &mut ui.state {
            input.set_cursor(6, false);
            input.move_word_right(true);
        }
        ui.on_chat_key(KeyCode::KeyC, None, primary_mod(), &mut clipboard);
        assert_eq!(clipboard.text, "beta");
        ui.on_chat_key(KeyCode::KeyX, None, primary_mod(), &mut clipboard);
        match &ui.state {
            UiState::Chat { input } => assert_eq!(input.buf, "alpha "),
            _ => panic!("expected chat"),
        }
        clipboard.text = "gamma".to_string();
        ui.on_chat_key(KeyCode::KeyV, None, primary_mod(), &mut clipboard);
        match &ui.state {
            UiState::Chat { input } => assert_eq!(input.buf, "alpha gamma"),
            _ => panic!("expected chat"),
        }
    }

    #[test]
    fn chat_clipboard_errors_are_logged() {
        let mut ui = Ui::new();
        ui.state = UiState::Chat {
            input: ChatInput::new("alpha"),
        };
        let mut clipboard = TestClipboard {
            fail_read: true,
            ..TestClipboard::default()
        };
        ui.on_chat_key(KeyCode::KeyV, None, primary_mod(), &mut clipboard);
        assert_eq!(ui.log.len(), 1);
    }

    #[test]
    fn chat_mouse_click_drag_and_double_click_select_text() {
        let mut ui = Ui::new();
        ui.state = UiState::Chat {
            input: ChatInput::new("alpha beta"),
        };
        let screen = (1280, 720);
        let layout = crate::ui::render::chat_layout(screen);
        let glyph_w = crate::render::font::cell_w() as f32 * layout.text_scale;
        let y = layout.input_y + 1.0;

        ui.on_chat_mouse_down(layout.input_x + glyph_w * 4.2, y, screen);
        ui.on_chat_mouse_up();
        match &ui.state {
            UiState::Chat { input } => assert_eq!(input.cursor, 2),
            _ => panic!("expected chat"),
        }

        ui.on_chat_mouse_down(layout.input_x + glyph_w * 2.0, y, screen);
        ui.on_chat_mouse_move(layout.input_x + glyph_w * 7.0, screen);
        ui.on_chat_mouse_up();
        match &ui.state {
            UiState::Chat { input } => assert_eq!(input.selected_range(), Some(0..5)),
            _ => panic!("expected chat"),
        }

        let beta_x = layout.input_x + glyph_w * 8.2;
        ui.on_chat_mouse_down(beta_x, y, screen);
        ui.on_chat_mouse_up();
        ui.on_chat_mouse_down(beta_x, y, screen);
        ui.on_chat_mouse_up();
        match &ui.state {
            UiState::Chat { input } => assert_eq!(input.selected_range(), Some(6..10)),
            _ => panic!("expected chat"),
        }
    }

    #[test]
    fn esc_in_paused_top_resumes() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None); // enter pause
        ui.cursor_state_changed = false; // reset flag
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
            UiState::Chat { input } => {
                assert_eq!(input.buf, "");
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
            UiState::Chat { input } => {
                assert_eq!(input.buf, "/");
                assert_eq!(input.cursor, 1);
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
            UiState::Paused {
                menu: MenuNav::Top { hovered },
            } => assert_eq!(hovered, 1),
            _ => panic!("expected paused top"),
        }
    }

    #[test]
    fn arrow_up_wraps_at_top() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        ui.on_key(KeyCode::ArrowUp, Pressed, None);
        match ui.state {
            UiState::Paused {
                menu: MenuNav::Top { hovered },
            } => {
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
        for _ in 0..3 {
            ui.on_key(KeyCode::ArrowDown, Pressed, None);
        }
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
        for _ in 0..2 {
            ui.on_key(KeyCode::ArrowDown, Pressed, None);
        }
        ui.on_key(KeyCode::Enter, Pressed, None);
        assert!(matches!(
            ui.state,
            UiState::Paused {
                menu: MenuNav::Settings
            }
        ));
    }

    #[test]
    fn esc_from_settings_goes_to_top_not_play() {
        let mut ui = Ui::new();
        ui.on_key(KeyCode::Escape, Pressed, None);
        for _ in 0..2 {
            ui.on_key(KeyCode::ArrowDown, Pressed, None);
        }
        ui.on_key(KeyCode::Enter, Pressed, None);
        ui.on_key(KeyCode::Escape, Pressed, None);
        assert!(matches!(
            ui.state,
            UiState::Paused {
                menu: MenuNav::Top { hovered: 0 }
            }
        ));
    }
}
