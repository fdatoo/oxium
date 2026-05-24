//! Action-oriented input engine for keyboard/mouse gameplay and UI routing.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, bail};
use arc_swap::ArcSwap;
use glam::Vec2;
use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};
use serde::{Deserialize, Serialize};
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputContext {
    Gameplay,
    PausedMenu,
    Chat,
    Debug,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Gameplay,
    PausedMenu,
    Chat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputAction {
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    Jump,
    Descend,
    Sprint,
    Break,
    Place,
    PickBlock,
    ToggleFly,
    SelectSlot1,
    SelectSlot2,
    SelectSlot3,
    SelectSlot4,
    SelectSlot5,
    SelectSlot6,
    SelectSlot7,
    SelectSlot8,
    SelectSlot9,
    CycleHotbar,
    Pause,
    OpenChat,
    OpenCommandChat,
    MenuUp,
    MenuDown,
    MenuAccept,
    ChatBackspace,
    ChatDelete,
    ChatLeft,
    ChatRight,
    ChatHome,
    ChatEnd,
    ChatHistoryPrev,
    ChatHistoryNext,
    ChatComplete,
    ChatSubmit,
    ToggleDebugContext,
    ToggleFullbright,
}

impl InputAction {
    pub fn selected_slot(self) -> Option<usize> {
        match self {
            Self::SelectSlot1 => Some(0),
            Self::SelectSlot2 => Some(1),
            Self::SelectSlot3 => Some(2),
            Self::SelectSlot4 => Some(3),
            Self::SelectSlot5 => Some(4),
            Self::SelectSlot6 => Some(5),
            Self::SelectSlot7 => Some(6),
            Self::SelectSlot8 => Some(7),
            Self::SelectSlot9 => Some(8),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputBinding {
    Key(KeyName),
    Mouse(MouseButtonName),
    ScrollY,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MouseButtonName {
    Left,
    Right,
    Middle,
}

impl MouseButtonName {
    fn to_winit(self) -> MouseButton {
        match self {
            Self::Left => MouseButton::Left,
            Self::Right => MouseButton::Right,
            Self::Middle => MouseButton::Middle,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyName {
    KeyW,
    KeyA,
    KeyS,
    KeyD,
    KeyF,
    KeyP,
    KeyB,
    KeyT,
    Slash,
    Escape,
    Space,
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    F3,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Enter,
    NumpadEnter,
    Backspace,
    Delete,
    Home,
    End,
    Tab,
}

impl KeyName {
    fn to_winit(self) -> KeyCode {
        match self {
            Self::KeyW => KeyCode::KeyW,
            Self::KeyA => KeyCode::KeyA,
            Self::KeyS => KeyCode::KeyS,
            Self::KeyD => KeyCode::KeyD,
            Self::KeyF => KeyCode::KeyF,
            Self::KeyP => KeyCode::KeyP,
            Self::KeyB => KeyCode::KeyB,
            Self::KeyT => KeyCode::KeyT,
            Self::Slash => KeyCode::Slash,
            Self::Escape => KeyCode::Escape,
            Self::Space => KeyCode::Space,
            Self::ShiftLeft => KeyCode::ShiftLeft,
            Self::ShiftRight => KeyCode::ShiftRight,
            Self::ControlLeft => KeyCode::ControlLeft,
            Self::ControlRight => KeyCode::ControlRight,
            Self::Digit1 => KeyCode::Digit1,
            Self::Digit2 => KeyCode::Digit2,
            Self::Digit3 => KeyCode::Digit3,
            Self::Digit4 => KeyCode::Digit4,
            Self::Digit5 => KeyCode::Digit5,
            Self::Digit6 => KeyCode::Digit6,
            Self::Digit7 => KeyCode::Digit7,
            Self::Digit8 => KeyCode::Digit8,
            Self::Digit9 => KeyCode::Digit9,
            Self::F3 => KeyCode::F3,
            Self::ArrowUp => KeyCode::ArrowUp,
            Self::ArrowDown => KeyCode::ArrowDown,
            Self::ArrowLeft => KeyCode::ArrowLeft,
            Self::ArrowRight => KeyCode::ArrowRight,
            Self::Enter => KeyCode::Enter,
            Self::NumpadEnter => KeyCode::NumpadEnter,
            Self::Backspace => KeyCode::Backspace,
            Self::Delete => KeyCode::Delete,
            Self::Home => KeyCode::Home,
            Self::End => KeyCode::End,
            Self::Tab => KeyCode::Tab,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingConfig {
    pub context: InputContext,
    pub action: InputAction,
    pub inputs: Vec<InputBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    pub mouse_sensitivity: f32,
    pub scroll_units_per_step: f32,
    pub bindings: Vec<BindingConfig>,
}

impl InputConfig {
    pub fn from_ron_file(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = ron::from_str(&text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    #[cfg(test)]
    pub fn bundled_default() -> anyhow::Result<Self> {
        let path = default_config_path();
        Self::from_ron_file(&path)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.mouse_sensitivity.is_finite() || self.mouse_sensitivity <= 0.0 {
            bail!("mouse_sensitivity must be finite and > 0");
        }
        if !self.scroll_units_per_step.is_finite() || self.scroll_units_per_step <= 0.0 {
            bail!("scroll_units_per_step must be finite and > 0");
        }

        let mut seen: HashMap<(InputContext, InputBinding), InputAction> = HashMap::new();
        for binding in &self.bindings {
            if binding.inputs.is_empty() {
                bail!("{:?}/{:?} has no inputs", binding.context, binding.action);
            }
            for input in &binding.inputs {
                if let Some(prev) = seen.insert((binding.context, input.clone()), binding.action) {
                    bail!(
                        "{:?}/{:?} is already bound to {:?}",
                        binding.context,
                        input,
                        prev
                    );
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct InputConfigSnapshot {
    pub config: InputConfig,
    pub generation: u64,
}

#[derive(Clone)]
pub struct InputConfigHolder(Arc<ArcSwap<InputConfigSnapshot>>);

impl InputConfigHolder {
    pub fn new(initial: InputConfig) -> Self {
        Self(Arc::new(ArcSwap::new(Arc::new(InputConfigSnapshot {
            config: initial,
            generation: 0,
        }))))
    }

    pub fn load(&self) -> Arc<InputConfigSnapshot> {
        self.0.load_full()
    }

    pub fn swap(&self, config: InputConfig) {
        let generation = self.0.load().generation.wrapping_add(1);
        self.0
            .store(Arc::new(InputConfigSnapshot { config, generation }));
    }
}

pub fn default_config_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("input")
        .join("default.ron")
}

pub fn spawn_watcher(
    path: PathBuf,
    holder: InputConfigHolder,
) -> anyhow::Result<Debouncer<notify_debouncer_mini::notify::RecommendedWatcher>> {
    let watch_path = path.clone();
    let mut debouncer = new_debouncer(
        Duration::from_millis(300),
        move |res: DebounceEventResult| match res {
            Ok(_events) => match InputConfig::from_ron_file(&watch_path) {
                Ok(cfg) => {
                    log::info!("input config reloaded from {:?}", watch_path);
                    holder.swap(cfg);
                }
                Err(e) => {
                    log::error!(
                        "input config reload failed ({:?}): {} - keeping previous",
                        watch_path,
                        e
                    );
                }
            },
            Err(e) => log::error!("input watcher error: {:?}", e),
        },
    )?;
    debouncer.watcher().watch(
        &path,
        notify_debouncer_mini::notify::RecursiveMode::NonRecursive,
    )?;
    Ok(debouncer)
}

#[derive(Default, Debug)]
pub struct RawInputState {
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub scroll_delta: f32,
    pub text: Vec<String>,
    pub keys_down: HashSet<KeyCode>,
    pub key_pressed_this_frame: HashSet<KeyCode>,
    pub key_released_this_frame: HashSet<KeyCode>,
    pub mouse_down: HashSet<MouseButton>,
    pub mouse_pressed_this_frame: HashSet<MouseButton>,
    pub mouse_released_this_frame: HashSet<MouseButton>,
}

impl RawInputState {
    pub fn clear_frame(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.scroll_delta = 0.0;
        self.text.clear();
        self.key_pressed_this_frame.clear();
        self.key_released_this_frame.clear();
        self.mouse_pressed_this_frame.clear();
        self.mouse_released_this_frame.clear();
    }

    pub fn clear_all(&mut self) {
        self.clear_frame();
        self.keys_down.clear();
        self.mouse_down.clear();
    }

    pub fn on_key(&mut self, key: KeyCode, state: ElementState, text: Option<&str>) {
        match state {
            ElementState::Pressed => {
                if self.keys_down.insert(key) {
                    self.key_pressed_this_frame.insert(key);
                }
                if let Some(text) = text
                    && !text.chars().any(|c| c.is_control())
                {
                    self.text.push(text.to_string());
                }
            }
            ElementState::Released => {
                if self.keys_down.remove(&key) {
                    self.key_released_this_frame.insert(key);
                }
            }
        }
    }

    pub fn on_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if self.mouse_down.insert(button) {
                    self.mouse_pressed_this_frame.insert(button);
                }
            }
            ElementState::Released => {
                if self.mouse_down.remove(&button) {
                    self.mouse_released_this_frame.insert(button);
                }
            }
        }
    }

    pub fn on_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_dx += dx as f32;
        self.mouse_dy += dy as f32;
    }

    pub fn on_scroll(&mut self, lines: f32) {
        self.scroll_delta += lines;
    }
}

#[derive(Debug, Default)]
pub struct ActionState {
    pub(crate) pressed: HashSet<InputAction>,
    pub(crate) down: HashSet<InputAction>,
    pub move_axis: Vec2,
    pub look_delta: Vec2,
    pub hotbar_step: i32,
    pub selected_slot: Option<usize>,
    pub text: Vec<String>,
}

impl ActionState {
    pub fn pressed(&self, action: InputAction) -> bool {
        self.pressed.contains(&action)
    }

    pub fn down(&self, action: InputAction) -> bool {
        self.down.contains(&action)
    }

    pub(crate) fn set_pressed(&mut self, action: InputAction) {
        self.pressed.insert(action);
    }

    pub(crate) fn set_down(&mut self, action: InputAction) {
        self.down.insert(action);
    }
}

pub struct InputEngine {
    raw: RawInputState,
    config: InputConfigHolder,
    seen_generation: u64,
    scroll_accumulator: f32,
    debug_enabled: bool,
}

impl InputEngine {
    pub fn new(config: InputConfigHolder) -> Self {
        let seen_generation = config.load().generation;
        Self {
            raw: RawInputState::default(),
            config,
            seen_generation,
            scroll_accumulator: 0.0,
            debug_enabled: false,
        }
    }

    pub fn on_key(&mut self, key: KeyCode, state: ElementState, text: Option<&str>) {
        self.raw.on_key(key, state, text);
    }

    pub fn on_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        self.raw.on_mouse_button(button, state);
    }

    pub fn on_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.raw.on_mouse_motion(dx, dy);
    }

    pub fn on_scroll(&mut self, lines: f32) {
        self.raw.on_scroll(lines);
    }

    pub fn clear_frame(&mut self) {
        self.raw.clear_frame();
    }

    pub fn clear_all(&mut self) {
        self.raw.clear_all();
        self.scroll_accumulator = 0.0;
    }

    pub fn debug_enabled(&self) -> bool {
        self.debug_enabled
    }

    pub fn resolve(&mut self, mode: InputMode) -> ActionState {
        let snapshot = self.config.load();
        if snapshot.generation != self.seen_generation {
            self.seen_generation = snapshot.generation;
            self.clear_all();
        }

        let mut actions = ActionState {
            text: self.raw.text.clone(),
            ..ActionState::default()
        };
        if matches!(mode, InputMode::Gameplay) {
            let sens = snapshot.config.mouse_sensitivity;
            actions.look_delta = Vec2::new(self.raw.mouse_dx * sens, self.raw.mouse_dy * sens);
        }

        let contexts = self.active_contexts(mode);
        let mut consumed = HashSet::new();
        for context in contexts {
            for binding in snapshot
                .config
                .bindings
                .iter()
                .filter(|binding| binding.context == context)
            {
                for input in &binding.inputs {
                    if matches!(input, InputBinding::ScrollY) {
                        continue;
                    }
                    if consumed.contains(input) {
                        continue;
                    }
                    let pressed = self.binding_pressed(input);
                    let down = self.binding_down(input);
                    if pressed || down {
                        consumed.insert(input.clone());
                    }
                    if pressed {
                        actions.set_pressed(binding.action);
                        if let Some(slot) = binding.action.selected_slot() {
                            actions.selected_slot = Some(slot);
                        }
                    }
                    if down {
                        actions.set_down(binding.action);
                    }
                }
            }
        }

        if actions.down(InputAction::MoveRight) {
            actions.move_axis.x += 1.0;
        }
        if actions.down(InputAction::MoveLeft) {
            actions.move_axis.x -= 1.0;
        }
        if actions.down(InputAction::MoveForward) {
            actions.move_axis.y += 1.0;
        }
        if actions.down(InputAction::MoveBack) {
            actions.move_axis.y -= 1.0;
        }

        if self.scroll_action_enabled(&snapshot.config, &consumed, mode) {
            self.scroll_accumulator += self.raw.scroll_delta;
            let units = snapshot.config.scroll_units_per_step;
            let steps = (self.scroll_accumulator / units).trunc() as i32;
            if steps != 0 {
                actions.hotbar_step = -steps;
                self.scroll_accumulator -= steps as f32 * units;
            }
        }

        if actions.pressed(InputAction::ToggleDebugContext) {
            self.debug_enabled = !self.debug_enabled;
        }

        actions
    }

    fn active_contexts(&self, mode: InputMode) -> Vec<InputContext> {
        match mode {
            InputMode::Gameplay if self.debug_enabled => {
                vec![InputContext::Debug, InputContext::Gameplay]
            }
            InputMode::Gameplay => vec![InputContext::Gameplay],
            InputMode::PausedMenu => vec![InputContext::PausedMenu],
            InputMode::Chat => vec![InputContext::Chat],
        }
    }

    fn binding_pressed(&self, input: &InputBinding) -> bool {
        match input {
            InputBinding::Key(key) => self.raw.key_pressed_this_frame.contains(&key.to_winit()),
            InputBinding::Mouse(button) => self
                .raw
                .mouse_pressed_this_frame
                .contains(&button.to_winit()),
            InputBinding::ScrollY => self.raw.scroll_delta.abs() > 0.0,
        }
    }

    fn binding_down(&self, input: &InputBinding) -> bool {
        match input {
            InputBinding::Key(key) => self.raw.keys_down.contains(&key.to_winit()),
            InputBinding::Mouse(button) => self.raw.mouse_down.contains(&button.to_winit()),
            InputBinding::ScrollY => false,
        }
    }

    fn scroll_action_enabled(
        &self,
        config: &InputConfig,
        consumed: &HashSet<InputBinding>,
        mode: InputMode,
    ) -> bool {
        if !matches!(mode, InputMode::Gameplay) || consumed.contains(&InputBinding::ScrollY) {
            return false;
        }
        self.active_contexts(mode).into_iter().any(|context| {
            config.bindings.iter().any(|binding| {
                binding.context == context
                    && binding.action == InputAction::CycleHotbar
                    && binding.inputs.contains(&InputBinding::ScrollY)
            })
        })
    }
}

pub fn load_default_holder() -> anyhow::Result<(InputConfigHolder, PathBuf)> {
    let path = default_config_path();
    let cfg = InputConfig::from_ron_file(&path)
        .with_context(|| format!("failed to load input config {}", path.display()))?;
    Ok((InputConfigHolder::new(cfg), path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> InputEngine {
        let cfg = InputConfig::bundled_default().unwrap();
        InputEngine::new(InputConfigHolder::new(cfg))
    }

    #[test]
    fn bundled_default_loads_and_validates() {
        let cfg = InputConfig::bundled_default().expect("input config must parse");
        cfg.validate().unwrap();
    }

    #[test]
    fn key_press_edge_only_fires_once() {
        let mut input = RawInputState::default();
        input.on_key(KeyCode::KeyW, ElementState::Pressed, None);
        input.on_key(KeyCode::KeyW, ElementState::Pressed, None);
        assert!(input.keys_down.contains(&KeyCode::KeyW));
        assert!(input.key_pressed_this_frame.contains(&KeyCode::KeyW));
        assert_eq!(input.key_pressed_this_frame.len(), 1);
    }

    #[test]
    fn clear_frame_preserves_held_but_clear_all_drops_it() {
        let mut input = RawInputState::default();
        input.on_key(KeyCode::KeyW, ElementState::Pressed, None);
        input.on_mouse_button(MouseButton::Left, ElementState::Pressed);
        input.on_mouse_motion(3.0, 4.0);
        input.on_scroll(1.0);
        input.clear_frame();
        assert!(input.keys_down.contains(&KeyCode::KeyW));
        assert!(input.mouse_down.contains(&MouseButton::Left));
        assert_eq!(input.mouse_dx, 0.0);
        assert_eq!(input.scroll_delta, 0.0);
        input.clear_all();
        assert!(input.keys_down.is_empty());
        assert!(input.mouse_down.is_empty());
    }

    #[test]
    fn config_generation_change_clears_held_state() {
        let cfg = InputConfig::bundled_default().unwrap();
        let holder = InputConfigHolder::new(cfg.clone());
        let mut engine = InputEngine::new(holder.clone());
        engine.on_key(KeyCode::KeyW, ElementState::Pressed, None);
        holder.swap(cfg);

        let actions = engine.resolve(InputMode::Gameplay);
        assert!(!actions.down(InputAction::MoveForward));
    }

    #[test]
    fn gameplay_context_resolves_movement() {
        let mut engine = engine();
        engine.on_key(KeyCode::KeyW, ElementState::Pressed, None);
        let actions = engine.resolve(InputMode::Gameplay);
        assert!(actions.down(InputAction::MoveForward));
        assert_eq!(actions.move_axis.y, 1.0);
    }

    #[test]
    fn pick_block_defaults_to_p_and_middle_mouse() {
        let mut engine = engine();
        engine.on_key(KeyCode::KeyP, ElementState::Pressed, None);
        let actions = engine.resolve(InputMode::Gameplay);
        assert!(actions.pressed(InputAction::PickBlock));

        engine.clear_all();
        engine.on_mouse_button(MouseButton::Middle, ElementState::Pressed);
        let actions = engine.resolve(InputMode::Gameplay);
        assert!(actions.pressed(InputAction::PickBlock));
    }

    #[test]
    fn paused_context_suppresses_gameplay() {
        let mut engine = engine();
        engine.on_key(KeyCode::KeyW, ElementState::Pressed, None);
        let actions = engine.resolve(InputMode::PausedMenu);
        assert!(!actions.down(InputAction::MoveForward));
        assert_eq!(actions.move_axis, Vec2::ZERO);
    }

    #[test]
    fn debug_context_is_toggled_by_f3() {
        let mut engine = engine();
        engine.on_key(KeyCode::F3, ElementState::Pressed, None);
        let actions = engine.resolve(InputMode::Gameplay);
        assert!(actions.pressed(InputAction::ToggleDebugContext));
        assert!(engine.debug_enabled());

        engine.clear_frame();
        engine.on_key(KeyCode::KeyB, ElementState::Pressed, None);
        let actions = engine.resolve(InputMode::Gameplay);
        assert!(actions.pressed(InputAction::ToggleFullbright));
    }

    #[test]
    fn scroll_accumulates_into_steps() {
        let mut engine = engine();
        engine.on_scroll(0.4);
        let actions = engine.resolve(InputMode::Gameplay);
        assert_eq!(actions.hotbar_step, 0);
        engine.clear_frame();

        engine.on_scroll(0.6);
        let actions = engine.resolve(InputMode::Gameplay);
        assert_eq!(actions.hotbar_step, -1);
    }

    #[test]
    fn duplicate_binding_rejected() {
        let cfg = InputConfig {
            mouse_sensitivity: 0.0025,
            scroll_units_per_step: 1.0,
            bindings: vec![
                BindingConfig {
                    context: InputContext::Gameplay,
                    action: InputAction::Jump,
                    inputs: vec![InputBinding::Key(KeyName::Space)],
                },
                BindingConfig {
                    context: InputContext::Gameplay,
                    action: InputAction::OpenChat,
                    inputs: vec![InputBinding::Key(KeyName::Space)],
                },
            ],
        };
        assert!(cfg.validate().is_err());
    }
}
