//! UI overlay rendering. Adds pause-menu + chat geometry on top of the
//! existing HUD frame built by `render::hud::build_hud`. Coordinates
//! are pixels with origin at the top-left of the framebuffer.

use crate::render::hud::HudFrame;
use crate::ui::menu::TOP_MENU;
use crate::ui::state::{MenuNav, UiState};
use crate::ui::Ui;

const DIM_COLOR: [u8; 4] = [0, 0, 0, 0x80];
const PANEL_BG: [u8; 4] = [20, 20, 24, 0xE0];
const TEXT_WHITE: [u8; 4] = [255, 255, 255, 255];
const TEXT_DIM:   [u8; 4] = [180, 180, 180, 255];
const HOVER:      [u8; 4] = [255, 220, 120, 255];

const PANEL_W: f32 = 360.0;
const PANEL_H: f32 = 280.0;
const TITLE_SCALE: f32 = 4.0;
const ITEM_SCALE:  f32 = 3.0;
const ITEM_LINE_H: f32 = 32.0;
const ITEM_CELL_W: f32 = crate::render::font::CELL_W as f32 * ITEM_SCALE;

const CHAT_PAD: f32 = 12.0;
const CHAT_LINE_H: f32 = 22.0;
const CHAT_SCALE: f32 = 2.0;
const CHAT_VISIBLE_PLAYING: usize = 5;
const CHAT_VISIBLE_OPEN: usize = 12;
const CHAT_FADE_START_SEC: f32 = 6.0;
const CHAT_FADE_END_SEC:   f32 = 8.0;

pub fn draw_overlay(ui: &Ui, screen_px: (u32, u32), frame: &mut HudFrame) {
    match &ui.state {
        UiState::Playing => draw_idle_log(ui, screen_px, frame),
        UiState::Paused { menu } => draw_pause(menu, screen_px, frame),
        UiState::Chat { input, .. } => {
            draw_chat_open(ui, input, screen_px, frame);
        }
    }
}

fn line_color(kind: crate::ui::chat::LineKind, alpha: u8) -> [u8; 4] {
    let [r, g, b] = match kind {
        crate::ui::chat::LineKind::Player        => [255, 255, 255],
        crate::ui::chat::LineKind::System        => [255, 220, 120],
        crate::ui::chat::LineKind::CommandEcho   => [120, 220, 255],
        crate::ui::chat::LineKind::CommandError  => [255, 100, 100],
    };
    [r, g, b, alpha]
}

fn draw_chat_open(
    ui: &Ui,
    input: &crate::ui::chat::ChatInput,
    screen_px: (u32, u32),
    frame: &mut HudFrame,
) {
    let (_, sh) = (screen_px.0 as f32, screen_px.1 as f32);
    let block_h = CHAT_LINE_H * (CHAT_VISIBLE_OPEN as f32 + 1.5);
    let block_y = sh - block_h - 80.0;
    frame.icons.push_rect(0.0, block_y, 480.0, block_h, [0, 0, 0, 0xA0]);

    let lines: Vec<_> = ui.log.iter().rev().take(CHAT_VISIBLE_OPEN).collect();
    for (i, line) in lines.iter().enumerate() {
        let y = block_y + block_h - CHAT_LINE_H * (i as f32 + 2.0);
        let color = line_color(line.kind, 255);
        frame.push_text(CHAT_PAD, y, &line.text, CHAT_SCALE, color);
    }

    let input_y = block_y + block_h - CHAT_LINE_H;
    let prompt = format!("> {}", input.buf);
    frame.push_text(CHAT_PAD, input_y, &prompt, CHAT_SCALE, [255, 255, 255, 255]);

    // Caret blink off the system clock (Ui has no central timer).
    let blink_on = {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        (t.as_millis() / 500) % 2 == 0
    };
    if blink_on {
        let prefix_chars = 2 + input.buf[..input.cursor].chars().count();
        let caret_x = CHAT_PAD + prefix_chars as f32
            * crate::render::font::CELL_W as f32 * CHAT_SCALE;
        let caret_w = crate::render::font::GLYPH_W as f32 * CHAT_SCALE;
        let caret_h = crate::render::font::GLYPH_H as f32 * CHAT_SCALE;
        frame.icons.push_rect(caret_x, input_y, caret_w, caret_h, [255, 255, 255, 180]);
    }
}

fn draw_idle_log(ui: &Ui, screen_px: (u32, u32), frame: &mut HudFrame) {
    let (_, sh) = (screen_px.0 as f32, screen_px.1 as f32);
    let now = std::time::Instant::now();
    let block_y = sh - CHAT_LINE_H * (CHAT_VISIBLE_PLAYING as f32) - 80.0;
    let mut drawn = 0usize;
    for line in ui.log.iter().rev() {
        if drawn >= CHAT_VISIBLE_PLAYING { break }
        let age = now.duration_since(line.posted_at).as_secs_f32();
        if age > CHAT_FADE_END_SEC { continue }
        let alpha = if age < CHAT_FADE_START_SEC {
            1.0
        } else {
            1.0 - (age - CHAT_FADE_START_SEC) / (CHAT_FADE_END_SEC - CHAT_FADE_START_SEC)
        };
        let alpha = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
        let y = block_y + CHAT_LINE_H * (CHAT_VISIBLE_PLAYING - 1 - drawn) as f32;
        let color = line_color(line.kind, alpha);
        frame.push_text(CHAT_PAD, y, &line.text, CHAT_SCALE, color);
        drawn += 1;
    }
}

fn draw_pause(menu: &MenuNav, screen_px: (u32, u32), frame: &mut HudFrame) {
    let (sw, sh) = (screen_px.0 as f32, screen_px.1 as f32);
    frame.icons.push_rect(0.0, 0.0, sw, sh, DIM_COLOR);

    let px = (sw - PANEL_W) * 0.5;
    let py = (sh - PANEL_H) * 0.5;
    frame.icons.push_rect(px, py, PANEL_W, PANEL_H, PANEL_BG);

    match menu {
        MenuNav::Top { hovered } => draw_top_menu(*hovered, px, py, frame),
        MenuNav::Settings        => draw_settings(px, py, frame),
    }
}

fn draw_top_menu(hovered: usize, px: f32, py: f32, frame: &mut HudFrame) {
    let title = "PAUSED";
    let title_w = title.chars().count() as f32
        * crate::render::font::CELL_W as f32 * TITLE_SCALE;
    let title_x = px + (PANEL_W - title_w) * 0.5;
    frame.push_text(title_x, py + 32.0, title, TITLE_SCALE, TEXT_WHITE);

    let items_top = py + 110.0;
    let items_left = px + 40.0;
    for (i, item) in TOP_MENU.iter().enumerate() {
        let y = items_top + i as f32 * ITEM_LINE_H;
        let color = if i == hovered { HOVER } else { TEXT_WHITE };
        if i == hovered {
            frame.push_text(items_left - ITEM_CELL_W * 2.0, y, ">", ITEM_SCALE, HOVER);
        }
        frame.push_text(items_left, y, item.label(), ITEM_SCALE, color);
    }
}

fn draw_settings(px: f32, py: f32, frame: &mut HudFrame) {
    let title = "SETTINGS";
    let title_w = title.chars().count() as f32
        * crate::render::font::CELL_W as f32 * TITLE_SCALE;
    let title_x = px + (PANEL_W - title_w) * 0.5;
    frame.push_text(title_x, py + 32.0, title, TITLE_SCALE, TEXT_WHITE);
    frame.push_text(px + 40.0, py + 130.0, "(coming soon)", ITEM_SCALE, TEXT_DIM);
    frame.push_text(px + 40.0, py + 130.0 + ITEM_LINE_H * 2.0,
                    "Esc - back", ITEM_SCALE, TEXT_DIM);
}

/// Pixel rect of the i-th top-menu item. Used by mouse hit-testing in
/// `Ui::on_mouse_move`. Returned as `(x, y, w, h)`.
pub fn top_menu_item_rect(i: usize, screen_px: (u32, u32)) -> (f32, f32, f32, f32) {
    let (sw, sh) = (screen_px.0 as f32, screen_px.1 as f32);
    let px = (sw - PANEL_W) * 0.5;
    let py = (sh - PANEL_H) * 0.5;
    let items_top = py + 110.0;
    let items_left = px + 40.0;
    let y = items_top + i as f32 * ITEM_LINE_H;
    (items_left - ITEM_CELL_W * 2.0, y - 4.0, PANEL_W - 80.0, ITEM_LINE_H)
}
