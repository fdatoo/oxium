//! 2D pan/zoom map of pipeline stages. Click-to-probe.

pub mod colormap;
pub mod stages;

use egui::{ColorImage, TextureHandle, TextureOptions, Ui};
use oxium::worldgen::probe::Stage;
use oxium::worldgen::Generator;

pub const MAP_SIZE_PX: usize = 256;

pub struct MapView {
    pub stage: Stage,
    pub center_wx: f32,
    pub center_wz: f32,
    pub blocks_per_pixel: f32,
    texture: Option<TextureHandle>,
    last_render_key: Option<RenderKey>,
}

#[derive(PartialEq)]
struct RenderKey {
    stage: Stage,
    center_wx: i32,
    center_wz: i32,
    blocks_per_pixel: i32,
    config_revision: u64,
}

impl MapView {
    pub fn new() -> Self {
        Self {
            stage: Stage::HTarget,
            center_wx: 0.0,
            center_wz: 0.0,
            blocks_per_pixel: 4.0,
            texture: None,
            last_render_key: None,
        }
    }

    pub fn set_stage(&mut self, stage: Stage) {
        if self.stage != stage {
            self.stage = stage;
            self.last_render_key = None;
        }
    }

    /// Translate `(px, py)` of the map into world `(wx, wz)`.
    pub fn pixel_to_world(&self, px: f32, py: f32) -> (i32, i32) {
        let half = MAP_SIZE_PX as f32 * 0.5;
        let wx = self.center_wx + (px - half) * self.blocks_per_pixel;
        let wz = self.center_wz + (py - half) * self.blocks_per_pixel;
        (wx.round() as i32, wz.round() as i32)
    }

    /// Translate world `(wx, wz)` into map pixel coords, if visible.
    pub fn world_to_pixel(&self, wx: i32, wz: i32) -> Option<(f32, f32)> {
        let half = MAP_SIZE_PX as f32 * 0.5;
        let px = half + (wx as f32 - self.center_wx) / self.blocks_per_pixel;
        let py = half + (wz as f32 - self.center_wz) / self.blocks_per_pixel;
        if px < 0.0 || px >= MAP_SIZE_PX as f32 || py < 0.0 || py >= MAP_SIZE_PX as f32 {
            None
        } else {
            Some((px, py))
        }
    }

    pub fn pan(&mut self, dx_px: f32, dy_px: f32) {
        self.center_wx -= dx_px * self.blocks_per_pixel;
        self.center_wz -= dy_px * self.blocks_per_pixel;
        self.last_render_key = None;
    }

    pub fn zoom(&mut self, factor: f32) {
        self.blocks_per_pixel = (self.blocks_per_pixel * factor).clamp(0.5, 64.0);
        self.last_render_key = None;
    }

    fn regenerate(&mut self, generator: &Generator, ctx: &egui::Context, revision: u64) {
        let key = RenderKey {
            stage: self.stage,
            center_wx: self.center_wx as i32,
            center_wz: self.center_wz as i32,
            blocks_per_pixel: self.blocks_per_pixel as i32,
            config_revision: revision,
        };
        if self.last_render_key.as_ref() == Some(&key) && self.texture.is_some() {
            return;
        }
        let mut pixels = vec![egui::Color32::TRANSPARENT; MAP_SIZE_PX * MAP_SIZE_PX];
        for py in 0..MAP_SIZE_PX {
            for px in 0..MAP_SIZE_PX {
                let (wx, wz) = self.pixel_to_world(px as f32, py as f32);
                let rgba = stages::render_pixel(generator, self.stage, wx, wz);
                pixels[py * MAP_SIZE_PX + px] = egui::Color32::from_rgba_premultiplied(
                    rgba[0], rgba[1], rgba[2], rgba[3],
                );
            }
        }
        let img = ColorImage { size: [MAP_SIZE_PX, MAP_SIZE_PX], pixels };
        let tex = ctx.load_texture("viz_overlay_map", img, TextureOptions::NEAREST);
        self.texture = Some(tex);
        self.last_render_key = Some(key);
    }

    /// Render the map widget. Returns `Some((wx, wz))` if the user
    /// clicked. Pan with drag (middle button); zoom with scroll.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        generator: &Generator,
        revision: u64,
    ) -> Option<(i32, i32)> {
        self.regenerate(generator, ui.ctx(), revision);

        // Stage dropdown
        egui::ComboBox::from_id_source("stage_combo")
            .selected_text(self.stage.label())
            .show_ui(ui, |ui| {
                for &s in Stage::ALL {
                    if ui.selectable_label(self.stage == s, s.label()).clicked() {
                        self.set_stage(s);
                    }
                }
            });

        let tex = self.texture.clone();
        let mut clicked = None;
        if let Some(tex) = tex {
            let size = egui::vec2(MAP_SIZE_PX as f32, MAP_SIZE_PX as f32);
            let resp = ui.add(egui::Image::new((tex.id(), size)).sense(egui::Sense::click_and_drag()));
            if resp.clicked() {
                if let Some(pos) = resp.interact_pointer_pos() {
                    let local = pos - resp.rect.left_top();
                    clicked = Some(self.pixel_to_world(local.x, local.y));
                }
            }
            if resp.dragged_by(egui::PointerButton::Middle) {
                let d = resp.drag_delta();
                self.pan(d.x, d.y);
            }
            if resp.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.1 {
                    let factor = if scroll > 0.0 { 0.9 } else { 1.1 };
                    self.zoom(factor);
                }
            }
        }
        ui.label(format!(
            "center ({:.0}, {:.0})  bpp {:.1}",
            self.center_wx, self.center_wz, self.blocks_per_pixel,
        ));
        clicked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_to_world_centers_at_center_wx() {
        let mut m = MapView::new();
        m.center_wx = 100.0;
        m.center_wz = 200.0;
        m.blocks_per_pixel = 4.0;
        let half = MAP_SIZE_PX as f32 * 0.5;
        let (wx, wz) = m.pixel_to_world(half, half);
        assert_eq!(wx, 100);
        assert_eq!(wz, 200);
    }

    #[test]
    fn world_to_pixel_inverse_of_pixel_to_world() {
        let mut m = MapView::new();
        m.center_wx = 32.0;
        m.center_wz = -64.0;
        m.blocks_per_pixel = 4.0;
        let (wx, wz) = m.pixel_to_world(64.0, 64.0);
        let (px, py) = m.world_to_pixel(wx, wz).unwrap();
        assert!((px - 64.0).abs() < 4.0);
        assert!((py - 64.0).abs() < 4.0);
    }

    #[test]
    fn pan_shifts_center_inversely_to_drag() {
        let mut m = MapView::new();
        m.center_wx = 0.0;
        m.pan(10.0, 0.0); // dragging right by 10 px → world center moves left
        assert!(m.center_wx < 0.0);
    }

    #[test]
    fn zoom_clamps_in_range() {
        let mut m = MapView::new();
        for _ in 0..100 { m.zoom(0.5); }
        assert!(m.blocks_per_pixel >= 0.5);
        for _ in 0..100 { m.zoom(2.0); }
        assert!(m.blocks_per_pixel <= 64.0);
    }
}
