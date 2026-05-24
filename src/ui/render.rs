//! UI overlay rendering. Adds pause-menu + chat geometry on top of the
//! existing HUD frame built by `render::hud::build_hud`. Coordinates
//! are pixels with origin at the top-left of the framebuffer.
//!
//! All overlay sizes scale with window height via [`ui_scale`] so the
//! pause panel and chat block read the same on a 720p window as on a
//! 4K one. The single design height is 720 px; everything else multi-
//! plies through.

use crate::render::hud::HudFrame;
use crate::ui::Ui;
use crate::ui::menu::TOP_MENU;
use crate::ui::state::{MenuNav, UiState};

const DIM_COLOR: [u8; 4] = [0, 0, 0, 0x80];
const PANEL_BG: [u8; 4] = [20, 20, 24, 0xE0];
const TEXT_WHITE: [u8; 4] = [255, 255, 255, 255];
const TEXT_DIM: [u8; 4] = [180, 180, 180, 255];
const HOVER: [u8; 4] = [255, 220, 120, 255];

// Base values are calibrated for a 720 px-tall framebuffer; the actual
// rendered size is `BASE_* * ui_scale(screen_h)`.
const BASE_PANEL_W: f32 = 440.0;
const BASE_PANEL_H: f32 = 280.0;
const BASE_TITLE_SCALE: f32 = 4.0;
const BASE_ITEM_SCALE: f32 = 2.5;
const BASE_ITEM_LINE_H: f32 = 28.0;
const BASE_TITLE_PAD_Y: f32 = 32.0;
const BASE_ITEMS_TOP_Y: f32 = 120.0;
const BASE_ITEMS_LEFT_X: f32 = 60.0;

const BASE_CHAT_PAD: f32 = 10.0;
const CHAT_VISIBLE_PLAYING: usize = 4;
const CHAT_VISIBLE_OPEN: usize = 7;
const CHAT_FADE_START_SEC: f32 = 6.0;
const CHAT_FADE_END_SEC: f32 = 8.0;

/// UI scale factor. 1.0 at a 720 px-tall framebuffer; doubles by 1440 px;
/// clamped so on a 360-px-tall window the overlay doesn't shrink past
/// readability and on a 2160-px-tall window the title isn't comically
/// huge.
fn ui_scale(screen_h: f32) -> f32 {
    (screen_h / 720.0).clamp(0.75, 2.5)
}

/// Resolved chat panel layout for rendering and hit testing.
#[derive(Debug, Clone, Copy)]
pub struct ChatLayout {
    pub panel_x: f32,
    pub panel_y: f32,
    pub panel_w: f32,
    pub panel_h: f32,
    pub line_h: f32,
    pub text_scale: f32,
    pub input_x: f32,
    pub input_y: f32,
    pub input_w: f32,
    pub input_h: f32,
    pub hint_y: f32,
}

pub fn chat_layout(screen_px: (u32, u32)) -> ChatLayout {
    let (sw, sh) = (screen_px.0 as f32, screen_px.1 as f32);
    let s = crate::render::hud::hud_scale(sh);
    let hotbar = crate::render::hud::hotbar_layout(screen_px);
    let pad = BASE_CHAT_PAD * s;
    let text_scale = 2.0 * s;
    let glyph_w = crate::render::font::cell_w() as f32 * text_scale;
    let glyph_h = crate::render::font::glyph_h() as f32 * text_scale;
    let line_h = glyph_h + 2.0 * s;
    let target_w = glyph_w * 56.0 + pad * 2.0;
    let max_w = (sw * 0.55).min(sw - 28.0 * s);
    let panel_w = target_w.min(max_w).max(240.0 * s);
    let panel_h = line_h * (CHAT_VISIBLE_OPEN as f32 + 2.0) + pad * 2.0;
    let panel_x = 14.0 * s;
    let panel_y = hotbar.y - 10.0 * s - panel_h;
    let input_x = panel_x + pad;
    let input_y = panel_y + panel_h - pad - line_h;
    ChatLayout {
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        line_h,
        text_scale,
        input_x,
        input_y,
        input_w: panel_w - pad * 2.0,
        input_h: line_h,
        hint_y: input_y - line_h,
    }
}

pub fn chat_input_byte_at(
    input: &crate::ui::chat::ChatInput,
    layout: &ChatLayout,
    x: f32,
) -> usize {
    let glyph_w = crate::render::font::cell_w() as f32 * layout.text_scale;
    let col = ((x - layout.input_x) / glyph_w).floor() as isize - 2;
    let col = col.clamp(0, input.buf.chars().count() as isize) as usize;
    byte_for_char_col(&input.buf, col)
}

fn byte_for_char_col(s: &str, col: usize) -> usize {
    if col == 0 {
        return 0;
    }
    s.char_indices().nth(col).map_or(s.len(), |(byte, _)| byte)
}

pub fn draw_overlay(ui: &Ui, screen_px: (u32, u32), frame: &mut HudFrame) {
    match &ui.state {
        UiState::Playing => draw_idle_log(ui, screen_px, frame),
        UiState::Paused { menu } => draw_pause(menu, screen_px, frame),
        UiState::Chat { input, .. } => draw_chat_open(ui, input, screen_px, frame),
    }
}

fn line_color(kind: crate::ui::chat::LineKind, alpha: u8) -> [u8; 4] {
    let [r, g, b] = match kind {
        crate::ui::chat::LineKind::Player => [255, 255, 255],
        crate::ui::chat::LineKind::System => [255, 220, 120],
        crate::ui::chat::LineKind::CommandEcho => [120, 220, 255],
        crate::ui::chat::LineKind::CommandError => [255, 100, 100],
    };
    [r, g, b, alpha]
}

fn draw_chat_open(
    ui: &Ui,
    input: &crate::ui::chat::ChatInput,
    screen_px: (u32, u32),
    frame: &mut HudFrame,
) {
    let sh = screen_px.1 as f32;
    let layout = chat_layout(screen_px);
    frame.icons.push_rect(
        layout.panel_x,
        layout.panel_y,
        layout.panel_w,
        layout.panel_h,
        [0, 0, 0, 0xA0],
    );

    let lines: Vec<_> = ui.log.iter().rev().take(CHAT_VISIBLE_OPEN).collect();
    for (i, line) in lines.iter().enumerate() {
        let y = layout.hint_y - layout.line_h * (i as f32 + 1.0);
        let color = line_color(line.kind, 255);
        frame.push_text(layout.input_x, y, &line.text, layout.text_scale, color);
    }

    frame.icons.push_rect(
        layout.panel_x,
        layout.input_y - 2.0 * crate::render::hud::hud_scale(sh),
        layout.panel_w,
        layout.input_h + 4.0 * crate::render::hud::hud_scale(sh),
        [10, 10, 12, 0xC0],
    );

    if let Some(range) = input.selected_range() {
        let cw = crate::render::font::cell_w() as f32;
        let gh = crate::render::font::glyph_h() as f32;
        let start_chars = input.buf[..range.start].chars().count();
        let end_chars = input.buf[..range.end].chars().count();
        let x = layout.input_x + (2 + start_chars) as f32 * cw * layout.text_scale;
        let w = (end_chars - start_chars) as f32 * cw * layout.text_scale;
        frame.icons.push_rect(
            x,
            layout.input_y,
            w,
            gh * layout.text_scale,
            [90, 140, 255, 120],
        );
    }

    let prompt = format!("> {}", input.buf);
    frame.push_text(
        layout.input_x,
        layout.input_y,
        &prompt,
        layout.text_scale,
        TEXT_WHITE,
    );
    if let Some((hint, color)) = chat_hint(ui, input) {
        frame.push_text(
            layout.input_x,
            layout.hint_y,
            &hint,
            layout.text_scale,
            color,
        );
    }

    let blink_on = {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        (t.as_millis() / 500).is_multiple_of(2)
    };
    if blink_on && input.selected_range().is_none() {
        let prefix_chars = 2 + input.buf[..input.cursor].chars().count();
        let cw = crate::render::font::cell_w() as f32;
        let gh = crate::render::font::glyph_h() as f32;
        let caret_x = layout.input_x + prefix_chars as f32 * cw * layout.text_scale;
        // Draw the caret as a one-cell-wide rectangle the height of a glyph.
        frame.icons.push_rect(
            caret_x,
            layout.input_y,
            (2.0 * crate::render::hud::hud_scale(sh)).max(1.0),
            gh * layout.text_scale,
            [255, 255, 255, 180],
        );
    }
}

fn chat_hint(ui: &Ui, input: &crate::ui::chat::ChatInput) -> Option<(String, [u8; 4])> {
    if !input.completion.entries.is_empty() {
        let parts: Vec<_> = input
            .completion
            .entries
            .iter()
            .take(5)
            .enumerate()
            .map(|(i, entry)| {
                if i == input.completion.selected {
                    format!("[{}]", entry.display)
                } else {
                    entry.display.clone()
                }
            })
            .collect();
        return Some((format!("tab: {}", parts.join("  ")), TEXT_DIM));
    }

    ui.commands.hint(&input.buf, input.cursor).map(|hint| {
        let color = if hint.is_error {
            [255, 100, 100, 255]
        } else {
            TEXT_DIM
        };
        (hint.message, color)
    })
}

fn draw_idle_log(ui: &Ui, screen_px: (u32, u32), frame: &mut HudFrame) {
    let (_, sh) = (screen_px.0 as f32, screen_px.1 as f32);
    let s = crate::render::hud::hud_scale(sh);
    let hotbar = crate::render::hud::hotbar_layout(screen_px);
    let pad = 14.0 * s;
    let text_scale = 2.0 * s;
    let line_h = crate::render::font::glyph_h() as f32 * text_scale + 2.0 * s;

    let now = std::time::Instant::now();
    let block_y = hotbar.y - 10.0 * s - line_h * CHAT_VISIBLE_PLAYING as f32;
    let mut drawn = 0usize;
    for line in ui.log.iter().rev() {
        if drawn >= CHAT_VISIBLE_PLAYING {
            break;
        }
        let age = now.duration_since(line.posted_at).as_secs_f32();
        if age > CHAT_FADE_END_SEC {
            continue;
        }
        let alpha = if age < CHAT_FADE_START_SEC {
            1.0
        } else {
            1.0 - (age - CHAT_FADE_START_SEC) / (CHAT_FADE_END_SEC - CHAT_FADE_START_SEC)
        };
        let alpha = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
        let y = block_y + line_h * (CHAT_VISIBLE_PLAYING - 1 - drawn) as f32;
        let color = line_color(line.kind, alpha);
        frame.push_text(pad, y, &line.text, text_scale, color);
        drawn += 1;
    }
}

/// Resolved pause-overlay layout for the current frame. Computed once
/// per draw + once per hit-test from the framebuffer dimensions; not
/// stored on `Ui`.
struct PauseLayout {
    panel_x: f32,
    panel_y: f32,
    panel_w: f32,
    panel_h: f32,
    title_scale: f32,
    title_pad_y: f32,
    item_scale: f32,
    item_cell_w: f32,
    item_line_h: f32,
    items_top: f32,
    items_left: f32,
}

fn pause_layout(screen_px: (u32, u32)) -> PauseLayout {
    let (sw, sh) = (screen_px.0 as f32, screen_px.1 as f32);
    let s = ui_scale(sh);
    let panel_w = BASE_PANEL_W * s;
    let panel_h = BASE_PANEL_H * s;
    let title_scale = BASE_TITLE_SCALE * s;
    let item_scale = BASE_ITEM_SCALE * s;
    let item_cell_w = crate::render::font::cell_w() as f32 * item_scale;
    let panel_x = (sw - panel_w) * 0.5;
    let panel_y = (sh - panel_h) * 0.5;
    PauseLayout {
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        title_scale,
        title_pad_y: BASE_TITLE_PAD_Y * s,
        item_scale,
        item_cell_w,
        item_line_h: BASE_ITEM_LINE_H * s,
        items_top: panel_y + BASE_ITEMS_TOP_Y * s,
        items_left: panel_x + BASE_ITEMS_LEFT_X * s,
    }
}

fn draw_pause(menu: &MenuNav, screen_px: (u32, u32), frame: &mut HudFrame) {
    let (sw, sh) = (screen_px.0 as f32, screen_px.1 as f32);
    frame.icons.push_rect(0.0, 0.0, sw, sh, DIM_COLOR);

    let l = pause_layout(screen_px);
    frame
        .icons
        .push_rect(l.panel_x, l.panel_y, l.panel_w, l.panel_h, PANEL_BG);

    match menu {
        MenuNav::Top { hovered } => draw_top_menu(*hovered, &l, frame),
        MenuNav::Settings => draw_settings(&l, frame),
    }
}

fn draw_top_menu(hovered: usize, l: &PauseLayout, frame: &mut HudFrame) {
    let title = "PAUSED";
    let title_w =
        title.chars().count() as f32 * crate::render::font::cell_w() as f32 * l.title_scale;
    let title_x = l.panel_x + (l.panel_w - title_w) * 0.5;
    frame.push_text(
        title_x,
        l.panel_y + l.title_pad_y,
        title,
        l.title_scale,
        TEXT_WHITE,
    );

    for (i, item) in TOP_MENU.iter().enumerate() {
        let y = l.items_top + i as f32 * l.item_line_h;
        let color = if i == hovered { HOVER } else { TEXT_WHITE };
        if i == hovered {
            frame.push_text(
                l.items_left - l.item_cell_w * 1.5,
                y,
                ">",
                l.item_scale,
                HOVER,
            );
        }
        frame.push_text(l.items_left, y, item.label(), l.item_scale, color);
    }
}

fn draw_settings(l: &PauseLayout, frame: &mut HudFrame) {
    let title = "SETTINGS";
    let title_w =
        title.chars().count() as f32 * crate::render::font::cell_w() as f32 * l.title_scale;
    let title_x = l.panel_x + (l.panel_w - title_w) * 0.5;
    frame.push_text(
        title_x,
        l.panel_y + l.title_pad_y,
        title,
        l.title_scale,
        TEXT_WHITE,
    );
    let body_x = l.items_left - l.item_cell_w * 1.5;
    let body_y = l.items_top;
    frame.push_text(body_x, body_y, "(coming soon)", l.item_scale, TEXT_DIM);
    frame.push_text(
        body_x,
        body_y + l.item_line_h * 2.0,
        "Esc - back",
        l.item_scale,
        TEXT_DIM,
    );
}

/// Pixel rect of the i-th top-menu item. Used by mouse hit-testing in
/// `Ui::on_mouse_move`. Returned as `(x, y, w, h)`.
pub fn top_menu_item_rect(i: usize, screen_px: (u32, u32)) -> (f32, f32, f32, f32) {
    let l = pause_layout(screen_px);
    let y = l.items_top + i as f32 * l.item_line_h;
    let width = l.panel_w - (l.items_left - l.panel_x) * 2.0;
    let pad_y = 4.0 * ui_scale(screen_px.1 as f32);
    (
        l.items_left - l.item_cell_w * 1.5,
        y - pad_y,
        width,
        l.item_line_h,
    )
}
