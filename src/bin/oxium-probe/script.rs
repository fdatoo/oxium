//! Scripted player input driver for `oxium-probe capture --mode sim`.
//!
//! A script is a JSON array of [`ScriptStep`] objects. Each step fires at a
//! particular simulation time (`at_s` seconds since the burst started) and may
//! press or release named [`InputAction`] keys, or apply a one-frame camera
//! rotation. Steps are applied in `at_s` order; multiple steps with the same
//! `at_s` all fire in the same tick.
//!
//! The driver validates that every action name in the file maps to a known
//! [`InputAction`] at load time — if a name is misspelled the error surfaces
//! before the renderer is even started.
//!
//! # Script format
//!
//! ```json
//! [
//!   { "at_s": 0.0, "press":   ["MoveForward"] },
//!   { "at_s": 0.5, "press":   ["Jump"] },
//!   { "at_s": 0.6, "release": ["Jump"] },
//!   { "at_s": 2.0, "release": ["MoveForward"], "look_delta_deg": [10.0, -5.0] }
//! ]
//! ```
//!
//! `look_delta_deg` is `[yaw_delta_deg, pitch_delta_deg]`. Positive yaw turns
//! the camera right; positive pitch looks up. The delta is injected as a
//! single-frame synthetic mouse event and does not accumulate across ticks.

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;
use winit::event::ElementState;

use oxium::input_engine::{InputAction, InputEngine};

// ── Public types ──────────────────────────────────────────────────────────────

/// One timed event in a probe input script.
#[derive(Debug, Deserialize)]
pub struct ScriptStep {
    /// Simulation time (seconds since burst start) at which this step fires.
    pub at_s: f32,
    /// Actions to press (begin holding) at this time.
    #[serde(default)]
    pub press: Vec<String>,
    /// Actions to release (stop holding) at this time.
    #[serde(default)]
    pub release: Vec<String>,
    /// Optional one-frame camera rotation `[yaw_delta_deg, pitch_delta_deg]`.
    /// Positive yaw = turn right; positive pitch = look up.
    pub look_delta_deg: Option<[f32; 2]>,
}

/// Drives scripted input during a sim-mode burst.
///
/// Created by [`ScriptDriver::load`], then ticked each sim step via
/// [`ScriptDriver::tick`]. The driver fires every step whose `at_s <= sim_time`
/// in order, advancing an internal cursor so each step fires exactly once.
pub struct ScriptDriver {
    /// Steps sorted ascending by `at_s`. Pre-parsed at load time.
    steps: Vec<ParsedStep>,
    /// Index of the next step that hasn't fired yet.
    cursor: usize,
}

// ── Internal ──────────────────────────────────────────────────────────────────

/// A step with action names already resolved to `InputAction` variants.
struct ParsedStep {
    at_s: f32,
    press: Vec<InputAction>,
    release: Vec<InputAction>,
    look_delta_rad: Option<(f32, f32)>,
}

impl ScriptDriver {
    /// Load and validate a script from a JSON file.
    ///
    /// All action name strings are resolved to [`InputAction`] at this point.
    /// Returns an error immediately if the file is missing, malformed, or
    /// contains an unrecognised action name — before the renderer starts.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read script file {:?}", path))?;
        let raw_steps: Vec<ScriptStep> = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse script file {:?} as JSON", path))?;

        let mut steps: Vec<ParsedStep> = raw_steps
            .into_iter()
            .map(|s| {
                let press = s
                    .press
                    .iter()
                    .map(|name| parse_action(name))
                    .collect::<anyhow::Result<Vec<_>>>()
                    .with_context(|| {
                        format!("in step at_s={}: bad 'press' action", s.at_s)
                    })?;
                let release = s
                    .release
                    .iter()
                    .map(|name| parse_action(name))
                    .collect::<anyhow::Result<Vec<_>>>()
                    .with_context(|| {
                        format!("in step at_s={}: bad 'release' action", s.at_s)
                    })?;
                let look_delta_rad = s.look_delta_deg.map(|[yaw, pitch]| {
                    (yaw.to_radians(), pitch.to_radians())
                });
                Ok(ParsedStep {
                    at_s: s.at_s,
                    press,
                    release,
                    look_delta_rad,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Sort ascending so the cursor scan is O(k) per tick.
        steps.sort_by(|a, b| a.at_s.partial_cmp(&b.at_s).unwrap_or(std::cmp::Ordering::Equal));

        Ok(Self { steps, cursor: 0 })
    }

    /// Fire all steps whose `at_s <= sim_time`.
    ///
    /// Injects synthetic key events and camera deltas into `engine`. Call this
    /// once per sim tick, *before* `state.step_with_dt`, so the injected input
    /// is consumed by the movement system in the same frame.
    pub fn tick(&mut self, engine: &mut InputEngine, sim_time: f32) {
        while self.cursor < self.steps.len()
            && self.steps[self.cursor].at_s <= sim_time
        {
            let step = &self.steps[self.cursor];

            for &action in &step.press {
                engine.inject_action(action, ElementState::Pressed);
            }
            for &action in &step.release {
                engine.inject_action(action, ElementState::Released);
            }
            if let Some((yaw_rad, pitch_rad)) = step.look_delta_rad {
                engine.inject_look(yaw_rad, pitch_rad);
            }

            self.cursor += 1;
        }
    }
}

// ── Action name parser ────────────────────────────────────────────────────────

/// Parse a string action name (e.g. `"MoveForward"`) into an [`InputAction`].
///
/// The recognised names match the variant names of `InputAction` exactly,
/// case-sensitive. Returns an error with the unrecognised name so the load-time
/// validation gives a useful diagnostic.
pub fn parse_action(name: &str) -> anyhow::Result<InputAction> {
    match name {
        "MoveForward" => Ok(InputAction::MoveForward),
        "MoveBack" => Ok(InputAction::MoveBack),
        "MoveLeft" => Ok(InputAction::MoveLeft),
        "MoveRight" => Ok(InputAction::MoveRight),
        "Jump" => Ok(InputAction::Jump),
        "Descend" => Ok(InputAction::Descend),
        "Sprint" => Ok(InputAction::Sprint),
        "Break" => Ok(InputAction::Break),
        "Place" => Ok(InputAction::Place),
        "PickBlock" => Ok(InputAction::PickBlock),
        "ToggleFly" => Ok(InputAction::ToggleFly),
        "SelectSlot1" => Ok(InputAction::SelectSlot1),
        "SelectSlot2" => Ok(InputAction::SelectSlot2),
        "SelectSlot3" => Ok(InputAction::SelectSlot3),
        "SelectSlot4" => Ok(InputAction::SelectSlot4),
        "SelectSlot5" => Ok(InputAction::SelectSlot5),
        "SelectSlot6" => Ok(InputAction::SelectSlot6),
        "SelectSlot7" => Ok(InputAction::SelectSlot7),
        "SelectSlot8" => Ok(InputAction::SelectSlot8),
        "SelectSlot9" => Ok(InputAction::SelectSlot9),
        "CycleHotbar" => Ok(InputAction::CycleHotbar),
        "Pause" => Ok(InputAction::Pause),
        "OpenChat" => Ok(InputAction::OpenChat),
        "OpenCommandChat" => Ok(InputAction::OpenCommandChat),
        "MenuUp" => Ok(InputAction::MenuUp),
        "MenuDown" => Ok(InputAction::MenuDown),
        "MenuAccept" => Ok(InputAction::MenuAccept),
        "ChatBackspace" => Ok(InputAction::ChatBackspace),
        "ChatDelete" => Ok(InputAction::ChatDelete),
        "ChatLeft" => Ok(InputAction::ChatLeft),
        "ChatRight" => Ok(InputAction::ChatRight),
        "ChatHome" => Ok(InputAction::ChatHome),
        "ChatEnd" => Ok(InputAction::ChatEnd),
        "ChatHistoryPrev" => Ok(InputAction::ChatHistoryPrev),
        "ChatHistoryNext" => Ok(InputAction::ChatHistoryNext),
        "ChatComplete" => Ok(InputAction::ChatComplete),
        "ChatSubmit" => Ok(InputAction::ChatSubmit),
        "ToggleDebugContext" => Ok(InputAction::ToggleDebugContext),
        "ToggleFullbright" => Ok(InputAction::ToggleFullbright),
        other => anyhow::bail!(
            "unknown action {:?}; valid names: MoveForward, MoveBack, MoveLeft, MoveRight, \
             Jump, Descend, Sprint, Break, Place, PickBlock, ToggleFly, SelectSlot1..9, \
             CycleHotbar, Pause, OpenChat, OpenCommandChat, MenuUp, MenuDown, MenuAccept, \
             ChatBackspace, ChatDelete, ChatLeft, ChatRight, ChatHome, ChatEnd, \
             ChatHistoryPrev, ChatHistoryNext, ChatComplete, ChatSubmit, \
             ToggleDebugContext, ToggleFullbright",
            other
        ),
    }
}
