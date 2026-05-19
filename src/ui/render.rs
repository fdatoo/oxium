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

pub fn draw_overlay(ui: &Ui, screen_px: (u32, u32), frame: &mut HudFrame) {
    match &ui.state {
        UiState::Playing => {}
        UiState::Paused { menu } => draw_pause(menu, screen_px, frame),
        UiState::Chat { .. } => {} // Task 12.
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
