//! UI → game intents. The only coupling from `ui` back to the rest of
//! `AppState`. `AppState::step` drains these once per frame and applies
//! them, so commands and menu actions stay free of direct game-state
//! references.

use glam::Vec3;

#[derive(Debug, Clone, PartialEq)]
pub enum UiEffect {
    /// Exit the application. `main.rs` calls `event_loop.exit()` after
    /// the step drains this; the existing `Drop` flush path still runs.
    Quit,
    /// Force an autosave by calling `AppState::flush_modified()`.
    Save,
    /// Set the player's `Position`.
    Teleport(Vec3),
    /// Set the singleton `TimeOfDay.t` (clamped 0..=1 at emit site).
    SetTime(f32),
    /// Toggle `Movement.mode` Walk⇄Fly.
    ToggleFly,
    /// Toggle `Movement.noclip`. Only takes effect in Fly mode;
    /// posts a "noclip ON/OFF" line to the chat.
    ToggleNoclip,
    /// Append a `LineKind::System` line to the chat log.
    PostMessage(String),
    /// Empty the chat log.
    ClearChat,
}
