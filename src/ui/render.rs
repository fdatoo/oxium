//! UI overlay rendering. Adds pause-menu + chat geometry on top of the
//! existing HUD frame built by `render::hud::build_hud`.

use crate::render::hud::HudFrame;
use crate::ui::Ui;

pub fn draw_overlay(_ui: &Ui, _screen_px: (u32, u32), _frame: &mut HudFrame) {
    // Filled in by later tasks. Stub keeps `Ui::draw_overlay` callable.
}
