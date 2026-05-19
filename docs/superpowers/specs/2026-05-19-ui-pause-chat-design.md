# UI: pause menu + chat/commands — Design Spec

**Date:** 2026-05-19
**Status:** Approved, ready for implementation planning
**Worktree / branch:** `worktree-ui-pause-chat`

## Summary

Add a UI layer to oxium with two user-visible features:

1. A pause menu (Esc) with **Resume / Save now / Settings / Quit to desktop**.
2. A chat console (T, or `/` to prefill a slash) supporting free-text notes and the slash commands `/tp`, `/time`, `/fly`, `/save`, `/help`, `/clear`. Chat is single-player only — messages are session-local self-notes, not networked.

The work is bundled with two input-ergonomics fixes that touch the same files: **hold-to-place/break** and **double-tap-Space to toggle fly**.

## Scope

**In scope (v1):**

- New `src/ui/` module owning UI state, chat log, command registry.
- Inverted input routing: UI sees winit events first, returns `Consumed` or `Forward`.
- Full freeze on pause (skip game schedule, render last frame with overlay).
- Pause menu with keyboard + mouse navigation.
- Chat log with idle fade during play, full view in chat mode.
- Single-line chat input with history, no IME/clipboard/selection.
- Six slash commands wired through a `UiEffect` outflow channel.
- Hold-to-place/break with 200 ms repeat rate.
- Double-tap-Space toggles fly (additive to F).

**Out of scope (v1):**

- Settings screen contents (placeholder only; "(coming soon)").
- Networked chat / multiplayer.
- Chat persistence to disk (session-only).
- Multi-line input, IME, clipboard, text selection.
- Localization.
- Inventory, hotbar configuration UI, mod menus.

## Architecture

### Module layout

```
src/ui/
├── mod.rs            # pub struct Ui — top-level state machine + frame entry
├── state.rs          # enum UiState { Playing, Paused { menu }, Chat { input, prefilled_slash } }
├── input.rs          # UiInput pump: consumes raw winit events, returns InputDisposition
├── menu.rs           # pause menu layout + navigation (items, hover, MenuAction enum)
├── chat.rs           # ChatLog (ring buffer w/ fade timestamps) + ChatInput (text field)
├── commands.rs       # Command trait + dispatcher + the six builtins
└── render.rs         # build_overlay(&Ui, ...) — uses existing render::hud primitives
```

`AppState` gains one field: `pub ui: Ui`.

`InputBuf` and `apply_input` live in the existing `src/ecs/systems/input.rs`; we add a sibling `InputState` struct in the same file to hold cross-frame input bookkeeping (action cooldowns, double-tap timestamp).

### Boundaries

- `Ui` owns **no wgpu or winit-handle types** (no `Device`, no `Window`, no `EventLoop`). It accepts winit's input enums (`KeyCode`, `ElementState`) by value at its boundary — that's a deliberate trade for not having to author a parallel input-event vocabulary. Renderer-side code lives in `ui::render`, which calls into `render::hud::{HudFrame::push_text, HudBatch::push_rect}` only.
- The game subsystems (`ecs`, `voxel`, `render`, `physics`, `worldgen`) stay UI-unaware — they don't know whether the game is paused. `AppState::step` is the only place that consults `ui.is_playing()`.
- The only coupling from UI back to game state is `UiEffect`: a per-frame outflow queue of intents (`Quit`, `Save`, `Teleport(Vec3)`, `SetTime(f32)`, `ToggleFly`, `PostMessage(String)`, `ClearChat`) that `AppState::step` drains and applies.

## UiState lifecycle

```rust
pub enum UiState {
    Playing,
    Paused  { menu: MenuNav },
    Chat    { input: ChatInput, prefilled_slash: bool },
}
```

Transitions (single source of truth: `Ui::handle_toggle_key`):

| From                       | Key            | To                                          |
|----------------------------|----------------|---------------------------------------------|
| `Playing`                  | `Esc`          | `Paused { menu: top }`                      |
| `Playing`                  | `T`            | `Chat { input: "", false }`                 |
| `Playing`                  | `/`            | `Chat { input: "/", true }`                 |
| `Paused { menu: top }`     | `Esc`          | `Playing` (same as Resume)                  |
| `Paused { menu: settings }`| `Esc`          | `Paused { menu: top }` (one step back)      |
| `Paused`                   | menu→Resume    | `Playing`                                   |
| `Paused`                   | menu→SaveNow   | (emit `UiEffect::Save`, stay)               |
| `Paused`                   | menu→Settings  | `Paused { menu: settings }`                 |
| `Paused`                   | menu→Quit      | (emit `UiEffect::Quit`)                     |
| `Chat`                     | `Esc`          | `Playing` (discard input)                   |
| `Chat`                     | `Enter`        | `Playing` after submit                      |

On entering `Paused` or `Chat`: release cursor grab + show cursor. On returning to `Playing`: re-grab + hide. Cursor manipulation lives in `main.rs` because the window handle does; `Ui` signals the transition via a `cursor_state_changed: bool` flag the caller checks.

## Input routing

`main.rs` inverts today's flow. Pseudo-Rust for the `KeyboardInput` branch:

```rust
match state.ui.on_key(code, ke.state, ke.text.as_deref()) {
    InputDisposition::Consumed => {}
    InputDisposition::Forward  => state.input_buf.on_key(code, ke.state),
}
```

`Ui::on_key` returns:

- **`Consumed`** when the UI is `Paused` or `Chat`, OR when the key is a UI-toggle key (`Esc`, `T`, `/`) regardless of state.
- **`Forward`** otherwise — i.e., during `Playing` for non-toggle keys.

Same pattern for `MouseInput` (button press/release) and `MouseWheel`: consumed in `Paused`/`Chat`, forwarded in `Playing`.

`MouseMotion` (from `device_event`) is **always forwarded** into `InputBuf`, but `apply_input` only reads it when `ui.is_playing()`. This keeps the camera-locked feeling without `Ui` needing to know about the device event channel.

Chat text uses winit's `KeyEvent::text` (the cooked Unicode string) so shifted symbols come through correctly. Backspace, arrow keys, Home/End are handled in `ChatInput::on_key` by `KeyCode`.

The existing `KeyCode::Escape → event_loop.exit()` in `main.rs:257-261` is **removed**. Exit now flows through the pause-menu Quit item via `UiEffect::Quit`, which `main.rs` handles by calling `event_loop.exit()` after `AppState::step`. The existing `Drop` flush path still runs.

## Input ergonomics (bundled fixes)

These touch `ecs/systems/input.rs` and ship in the same worktree as separate commits.

### Hold-to-place/break

`InputBuf` gains held-state fields alongside the existing edge flags:

```rust
pub lmb_down: bool,   // currently held
pub rmb_down: bool,
// `lmb_pressed` / `rmb_pressed` (edge) remain — used to fire the *first* action
// immediately on press, before the cooldown timer arms.
```

New sibling on `AppState`:

```rust
pub struct InputState {
    last_break_at:        Option<Instant>,
    last_place_at:        Option<Instant>,
    last_space_press_at:  Option<Instant>,   // for double-tap fly
}
```

`apply_input` replaces today's `input.break_ = buf.lmb_pressed` with:

```rust
const ACTION_REPEAT: Duration = Duration::from_millis(200);
let now = Instant::now();
input.break_ = buf.lmb_pressed
    || (buf.lmb_down
        && state.last_break_at.map_or(true, |t| now - t >= ACTION_REPEAT));
if input.break_ { state.last_break_at = Some(now); }
// Symmetric for `input.place` / `rmb`.
```

`interaction.rs` does not change. The repeat rate is one tunable constant.

### Double-tap-Space → fly toggle

Additive to F. In `apply_input`:

```rust
const DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(280);
if buf.key_pressed_this_frame.contains(&KeyCode::Space) {
    let now = Instant::now();
    if state.last_space_press_at.map_or(false, |t| now - t <= DOUBLE_TAP_WINDOW) {
        movement.mode = match movement.mode { Walk => Fly, Fly => Walk };
        state.last_space_press_at = None;   // reset so triple-tap doesn't re-toggle
    } else {
        state.last_space_press_at = Some(now);
    }
}
```

In Walk mode, single Space still triggers jump on press; double-tap = "jump once, then toggle to Fly mid-air". In Fly mode, Space is vertical up; double-tap toggles back to Walk.

## Pause-menu rendering

Layout, in pixel space:

```
┌───────────── full framebuffer ─────────────┐
│                                            │
│         (game frame, dimmed by full-       │
│          screen 0x00000080 rect)           │
│                                            │
│            ┌──── 360 × 280 ────┐           │
│            │      PAUSED       │  title    │
│            │                   │           │
│            │ ▸ Resume          │  hovered  │
│            │   Save now        │           │
│            │   Settings        │           │
│            │   Quit to desktop │           │
│            └───────────────────┘           │
│                                            │
└────────────────────────────────────────────┘
```

- Dim: one `HudBatch::push_rect` with alpha 0x80 on the icons batch (renders before text — same ordering as the debug overlay backdrop at `render/hud.rs:224`).
- Panel: one rect + four text lines + a chevron in front of the hovered item.
- Title at scale 4 (20×28 px glyphs); items at scale 3 (matches debug overlay).
- All rendering through `HudFrame::push_text` / `HudBatch::push_rect`. No new pipeline, no new uniforms.

**Navigation** — `MenuNav { items: &'static [MenuItem; 4], hovered: usize }`:

- `Up`/`Down` (and `W`/`S`) move `hovered`; wraps at the ends.
- `Enter` (or `Space`) activates the hovered item.
- Mouse: `Ui::on_mouse_move` hit-tests each row's pixel rect and updates `hovered`. LMB on a row activates.
- `Esc` activates Resume when on the top menu; in the Settings sub-menu, `Esc` goes back to the top menu (see lifecycle table).

`MenuItem` is `Resume | SaveNow | Settings | Quit`. Activation either transitions state, emits a `UiEffect`, or both. Settings opens a `MenuNav::Settings` variant showing one line "(coming soon)"; Back/Esc returns to top.

## Chat log + input field

### ChatLog

Fixed-capacity ring buffer of `ChatLine { text: String, posted_at: Instant, kind: LineKind }`. Capacity 64; oldest evicted on push. `LineKind` is `Player | System | CommandEcho | CommandError`, driving the tint (white / yellow / cyan / red).

**Display states:**

| Mode      | Visible lines              | Backdrop    | Input field |
|-----------|----------------------------|-------------|-------------|
| `Playing` | last 5, fade after 8 s     | none        | hidden      |
| `Chat`    | last 12, no fade           | 0x000000A0  | visible     |
| `Paused`  | none                       | n/a         | hidden      |

In `Playing`, per-line alpha is `1.0` for the first 6 s after `posted_at`, then linearly fades to `0.0` over the next 2 s; lines past 8 s are skipped. The alpha feeds into the existing `color: [u8; 4]` field on `HudVertex` — no shader change.

Anchored bottom-left, above where the hotbar renders.

### ChatInput

```rust
pub struct ChatInput {
    buf: String,
    cursor: usize,                 // byte offset; always on a char boundary
    history: VecDeque<String>,     // last 32 submitted lines
    history_pos: Option<usize>,    // None when typing, Some(i) when recalling
}
```

Editing keys handled in `ChatInput::on_key`:

- `Backspace` / `Delete` — remove prev/next grapheme.
- `Left` / `Right` — move cursor by grapheme.
- `Home` / `End` — jump to start/end.
- `Up` / `Down` — cycle `history`.
- `Enter` — submit; returns the text, clears `buf`, pushes to history.

Printable text comes from winit's `KeyEvent::text`, inserted at `cursor`. No multi-line, no selection, no clipboard.

Rendered as `> {buf_before_cursor}|{buf_after_cursor}` where `|` is a caret toggled every 0.5 s using `start_time.elapsed()`.

## Commands & dispatcher

```rust
pub trait Command {
    fn name(&self) -> &'static str;
    fn help(&self) -> &'static str;
    fn run(&self, args: &[&str]) -> Result<Vec<UiEffect>, String>;
}
```

Registry is `Vec<Box<dyn Command>>` built once in `Ui::new()`. Lookup is linear — six entries.

**Submit flow** (`Ui::submit_chat`):

1. Echo the raw input as `CommandEcho` (cyan) if it begins with `/`, else `Player` (white).
2. If first char ≠ `/`, log as a note and stop.
3. Tokenize on whitespace, find the command by exact-match name, dispatch.
4. Push each returned `UiEffect` onto the outflow queue; push any returned `PostMessage` effects as `System` lines.
5. Unknown command or parse error → red `CommandError` line.

**The six builtins:**

| Command       | Args        | Behaviour |
|---------------|-------------|-----------|
| `/tp x y z`   | three floats| Emits `UiEffect::Teleport(Vec3)`. `AppState` writes to the player's `Position` component. |
| `/time t`     | one `[0,1]` | Clamps to `[0,1]`; emits `UiEffect::SetTime(f32)`. `AppState` writes to `TimeOfDay.t` (same path `main.rs:217-225` uses). |
| `/fly`        | none        | Emits `UiEffect::ToggleFly`. `AppState` toggles `Movement.mode`. |
| `/save`       | none        | Emits `UiEffect::Save`. `AppState` calls `flush_modified()`. |
| `/help`       | none        | Pure-local; walks the registry and pushes a `System` line per command. |
| `/clear`      | none        | Emits `UiEffect::ClearChat`; `Ui` handles it next frame. |

Adding a new command that needs game state in the future = one variant on `UiEffect` + one match arm in `AppState::step`. Adding one that's pure-UI = no `AppState` change at all.

## Pause integration with the schedule

`AppState::step` becomes:

```rust
let now = Instant::now();
let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
self.last_tick = now;
self.fps_meter.record(dt);

// Always feed UI a frame tick (drives caret blink + drops stale chat
// effects). Time-since-posted for chat fade is read from each line's
// own `Instant` so the UI doesn't carry a separate accumulator.
self.ui.tick(dt);

if self.ui.is_playing() {
    // ... existing input → movement → physics → interaction → world_stream
    //     → drain_jobs → drain_persistence → relight_pump → autosave ...
}

// Drain UI effects regardless of state (so menu actions work while paused).
for eff in self.ui.drain_effects() {
    self.apply_ui_effect(eff);   // mutates Position/TimeOfDay/Movement, calls flush_modified, etc.
}

// Render is unconditional. The HUD builder now also draws the UI overlay
// on top of the existing FPS/XYZ/hotbar layout.
//  ...existing render() call, with hud built via build_hud() + Ui::draw_overlay(&mut hud)
```

Background workers (`Jobs`, `Persistence`) keep running while paused — their inboxes simply stop being drained. When the player unpauses, the next `step()` consumes whatever queued up. This matches "full freeze" semantics without needing to pause threads.

## Testing & verification

- **Unit tests** (`ui::chat::tests`, `ui::commands::tests`):
  - `ChatInput` editing: insert / backspace at start, end, and across multi-byte chars; cursor moves by grapheme not byte.
  - `ChatLog` ring eviction at capacity; fade alpha at `t = 0, 5.99, 6.01, 7, 8.01`.
  - Command parsing: `/tp 1 2 abc` → error, `/time -0.1` → clamped to 0.0, `/time 1.5` → clamped to 1.0, `/unknown` → unknown-command error.
- **State-transition tests** (`ui::tests`):
  - Drive `Ui::on_key` with synthetic key events through `Playing → Paused → Playing` and `Playing → Chat → Playing`.
  - Assert `cursor_state_changed` is set on each transition into/out of `Playing`.
  - Assert `MouseMotion` is forwarded in `Playing` and that `apply_input` consumes/ignores it correctly when `ui.is_playing()` is false.
- **Input ergonomics tests** (`ecs::systems::input::tests`):
  - With `lmb_down = true`, `apply_input` fires `input.break_` immediately, then after 200 ms, then 200 ms after that; not in between.
  - Two `Space` presses within 280 ms toggle `Movement.mode`; two presses 300 ms apart do not.
  - F still toggles fly (regression).
- **Integration / visual** (existing screenshot path):
  - Add hidden `--ui paused` / `--ui chat` flags to `main.rs` that put the UI in that state before the screenshot frame.
  - Commit reference PNGs under `tests/snapshots/ui_*.png`. The existing screenshot test infrastructure compares against them.
- **Manual** (called out in the plan, not automatable here):
  - Pause-then-unpause does not drop input events or stick keys (e.g. W held across the pause boundary).
  - Cursor grab releases on pause/chat and re-acquires on resume.
  - Hold-to-mine feels right at 200 ms (open to tuning).
  - Double-tap-Space-to-fly in Walk mode produces "jump then toggle"; not jarring.

## File-touch summary

- **New:** `src/ui/{mod,state,input,menu,chat,commands,render}.rs`.
- **Modified:** `src/app.rs` (UI field, schedule branch, effect drain, render wiring); `src/main.rs` (event routing inversion, cursor grab on state change, UiEffect::Quit handling, `--ui` test flag); `src/ecs/systems/input.rs` (held button state, InputState, hold-to-act, double-tap fly); `src/lib.rs` and binary crate root for `pub mod ui;`.
- **No changes:** `voxel/`, `worldgen/`, `lighting/`, `mesher/`, `physics/`, `render/pipelines/`, shaders. The renderer's HUD module gains no new public API — `ui::render` uses what's already there.
