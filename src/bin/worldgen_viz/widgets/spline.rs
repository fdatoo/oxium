//! Custom egui widget that draws a CubicSpline curve and lets the
//! user drag knots with the mouse.

use egui::{Color32, Pos2, Sense, Stroke, Vec2, Widget};
use oxium::worldgen::spline::CubicSpline;

pub struct SplineEditor<'a> {
    spline: &'a mut CubicSpline,
    x_range: (f32, f32),
    y_range: (f32, f32),
    desired_size: Vec2,
}

impl<'a> SplineEditor<'a> {
    pub fn new(spline: &'a mut CubicSpline) -> Self {
        Self {
            spline,
            x_range: (-1.1, 1.1),
            y_range: (-1.5, 1.5),
            desired_size: Vec2::new(380.0, 220.0),
        }
    }
    #[allow(dead_code)]
    pub fn x_range(mut self, lo: f32, hi: f32) -> Self {
        self.x_range = (lo, hi);
        self
    }
    #[allow(dead_code)]
    pub fn y_range(mut self, lo: f32, hi: f32) -> Self {
        self.y_range = (lo, hi);
        self
    }
}

impl<'a> Widget for SplineEditor<'a> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let (rect, response) = ui.allocate_exact_size(self.desired_size, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let to_screen = |loc: f32, val: f32| -> Pos2 {
            let nx = (loc - self.x_range.0) / (self.x_range.1 - self.x_range.0);
            let ny = (val - self.y_range.0) / (self.y_range.1 - self.y_range.0);
            Pos2::new(
                rect.min.x + nx * rect.width(),
                rect.max.y - ny * rect.height(),
            )
        };
        let from_screen = |p: Pos2| -> (f32, f32) {
            let nx = (p.x - rect.min.x) / rect.width();
            let ny = (rect.max.y - p.y) / rect.height();
            (
                self.x_range.0 + nx * (self.x_range.1 - self.x_range.0),
                self.y_range.0 + ny * (self.y_range.1 - self.y_range.0),
            )
        };

        // Background grid.
        painter.rect_filled(rect, 4.0, Color32::from_rgb(20, 20, 26));
        for i in 0..=4 {
            let t = i as f32 / 4.0;
            let y = rect.min.y + t * rect.height();
            painter.line_segment(
                [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
                Stroke::new(0.5, Color32::from_gray(60)),
            );
        }

        // Draw the spline curve.
        let mut prev: Option<Pos2> = None;
        for i in 0..=120 {
            let t = i as f32 / 120.0;
            let x = self.x_range.0 + t * (self.x_range.1 - self.x_range.0);
            let y = self.spline.evaluate(x);
            let p = to_screen(x, y);
            if let Some(prev_p) = prev {
                painter.line_segment(
                    [prev_p, p],
                    Stroke::new(1.5, Color32::from_rgb(100, 200, 255)),
                );
            }
            prev = Some(p);
        }

        // Draw + drag knots.
        let mut response = response;
        if let CubicSpline::Multipoint(knots) = self.spline {
            let mut drag_target: Option<usize> = None;
            for (i, knot) in knots.iter().enumerate() {
                let p = to_screen(knot.loc, knot.val);
                let hit = response
                    .hover_pos()
                    .map(|hp| (hp - p).length() < 8.0)
                    .unwrap_or(false);
                let color = if hit {
                    Color32::YELLOW
                } else {
                    Color32::from_rgb(220, 180, 80)
                };
                painter.circle_filled(p, 5.0, color);
                if hit && response.dragged() {
                    drag_target = Some(i);
                }
            }
            if let Some(i) = drag_target
                && let Some(pos) = response.interact_pointer_pos()
            {
                let (lx, ly) = from_screen(pos);
                let new_loc = lx.clamp(self.x_range.0, self.x_range.1);
                let new_val = ly.clamp(self.y_range.0, self.y_range.1);
                if (knots[i].loc - new_loc).abs() > 1e-6 || (knots[i].val - new_val).abs() > 1e-6 {
                    knots[i].loc = new_loc;
                    knots[i].val = new_val;
                    // Without this, callers see `response.changed()`
                    // as false during drags and never trigger a
                    // regen — silently broken spline tuning.
                    response.mark_changed();
                }
            }
        }

        response
    }
}
