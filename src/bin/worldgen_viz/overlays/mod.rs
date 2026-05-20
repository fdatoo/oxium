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
    /// clicked. Pan with drag (middle button); zoom with scroll. The
    /// `pinned` coord, if any, is drawn as a yellow crosshair so the
    /// 2D map and the (eventual) 3D viewport stay visually linked.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        generator: &Generator,
        revision: u64,
        pinned: Option<(i32, i32)>,
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

        // Per-stage description (helps if the user has no idea what
        // "h_pre" or "FlowAccum" means).
        ui.label(
            egui::RichText::new(self.stage.description())
                .small()
                .weak(),
        );

        let tex = self.texture.clone();
        let mut clicked = None;
        let mut hover_world: Option<(i32, i32)> = None;
        if let Some(tex) = tex {
            let size = egui::vec2(MAP_SIZE_PX as f32, MAP_SIZE_PX as f32);
            let resp = ui.add(
                egui::Image::new((tex.id(), size)).sense(egui::Sense::click_and_drag()),
            );
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
                if let Some(pos) = resp.hover_pos() {
                    let local = pos - resp.rect.left_top();
                    hover_world = Some(self.pixel_to_world(local.x, local.y));
                }
            }

            // Pinned-column crosshair: yellow lines bisecting the
            // pixel that the pinned column occupies.
            if let Some((pwx, pwz)) = pinned {
                if let Some((px, py)) = self.world_to_pixel(pwx, pwz) {
                    let p = resp.rect.left_top() + egui::vec2(px, py);
                    let painter = ui.painter_at(resp.rect);
                    let pin_color = egui::Color32::from_rgb(255, 220, 70);
                    painter.line_segment(
                        [
                            egui::pos2(resp.rect.left(), p.y),
                            egui::pos2(resp.rect.right(), p.y),
                        ],
                        egui::Stroke::new(1.0, pin_color),
                    );
                    painter.line_segment(
                        [
                            egui::pos2(p.x, resp.rect.top()),
                            egui::pos2(p.x, resp.rect.bottom()),
                        ],
                        egui::Stroke::new(1.0, pin_color),
                    );
                    painter.circle_stroke(p, 3.5, egui::Stroke::new(1.5, pin_color));
                }
            }

            // Hover-column cursor: thin cyan crosshair following the mouse.
            if let (Some((hwx, hwz)), true) = (hover_world, resp.hovered()) {
                if let Some((px, py)) = self.world_to_pixel(hwx, hwz) {
                    let p = resp.rect.left_top() + egui::vec2(px, py);
                    let painter = ui.painter_at(resp.rect);
                    let hover_color = egui::Color32::from_rgba_premultiplied(120, 220, 240, 180);
                    painter.line_segment(
                        [
                            egui::pos2(resp.rect.left(), p.y),
                            egui::pos2(resp.rect.right(), p.y),
                        ],
                        egui::Stroke::new(0.5, hover_color),
                    );
                    painter.line_segment(
                        [
                            egui::pos2(p.x, resp.rect.top()),
                            egui::pos2(p.x, resp.rect.bottom()),
                        ],
                        egui::Stroke::new(0.5, hover_color),
                    );
                }
            }
        }

        // Hover / center / zoom readout.
        ui.horizontal(|ui| {
            ui.label(format!(
                "center ({:.0}, {:.0})  bpp {:.1}",
                self.center_wx, self.center_wz, self.blocks_per_pixel,
            ));
        });
        if let Some((hwx, hwz)) = hover_world {
            let raw = generator.sample_stage(self.stage, hwx, hwz);
            ui.label(
                egui::RichText::new(format!(
                    "hover ({}, {}) → {} = {:.3}",
                    hwx, hwz, self.stage.label(), raw,
                ))
                .monospace()
                .small(),
            );
        }

        // Legend strip for scalar stages.
        legend(ui, self.stage);

        clicked
    }
}

/// Draw a legend strip for the current stage. Scalar stages get a
/// horizontal gradient with min/max labels; categorical stages get a
/// short text hint.
fn legend(ui: &mut Ui, stage: Stage) {
    use crate::overlays::stages;
    let strip_h = 12.0;
    let strip_w = MAP_SIZE_PX as f32;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(strip_w, strip_h), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    if let Some((lo, hi)) = stages::range(stage) {
        // Sample 64 stops across the gradient using the same colour
        // mapping as the map texture, so the legend matches.
        let stops = 64;
        let dx = strip_w / stops as f32;
        for i in 0..stops {
            let t = i as f32 / (stops - 1) as f32;
            let raw = lo + t * (hi - lo);
            let rgba = stages::pixel(stage, raw);
            let color = egui::Color32::from_rgba_premultiplied(rgba[0], rgba[1], rgba[2], rgba[3]);
            let x0 = rect.left() + i as f32 * dx;
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(x0, rect.top()), egui::vec2(dx + 0.5, strip_h)),
                0.0,
                color,
            );
        }
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(format!("{lo:.2}")).small().weak());
            ui.add_space(strip_w - 70.0);
            ui.label(egui::RichText::new(format!("{hi:.2}")).small().weak());
        });
    } else {
        // Categorical: paint a few sample hues so the user sees the
        // palette, no min/max labels.
        let samples: &[f32] = match stage {
            Stage::AquiferSubstance => &[0.0, 1.0],
            _ => &[0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
        };
        let cell_w = strip_w / samples.len() as f32;
        for (i, &v) in samples.iter().enumerate() {
            let rgba = stages::pixel(stage, v);
            let color = egui::Color32::from_rgba_premultiplied(rgba[0], rgba[1], rgba[2], rgba[3]);
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(rect.left() + i as f32 * cell_w, rect.top()),
                    egui::vec2(cell_w, strip_h),
                ),
                0.0,
                color,
            );
        }
        let hint = match stage {
            Stage::PlateId => "plate ID → hashed hue",
            Stage::BiomeId => "biome ID → hashed hue",
            Stage::AquiferSubstance => "blue = Water · orange = Lava",
            _ => "",
        };
        ui.label(egui::RichText::new(hint).small().weak());
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
