//! Recursive editor for `NestedSpline` (the worldgen tree spline that
//! powers climate-driven offset / factor / jaggedness).
//!
//! NestedSpline is a tree:
//!
//! ```text
//! depth 0 — outer knots vary continentalness (c)
//!     each knot's `val` is a sub-spline at depth 1
//! depth 1 — mid knots vary terrain_shape (s)
//!     each knot's `val` is a sub-spline at depth 2
//! depth 2 — inner knots vary ridges_pv (r)
//!     each knot's `val` is a sub-spline (typically Constant)
//! ```
//!
//! The editor renders this as a recursive tree of CollapsingHeader
//! widgets. At every Multipoint level it also draws a 1D curve plot
//! of `spline.evaluate(x, 0, 0)` — the value the spline produces as
//! its first axis varies with the deeper axes pinned to zero. Gives
//! a sense of the function shape without requiring a 3D plotter.

use egui::{Color32, Pos2, Sense, Stroke, Ui};
use oxium::worldgen::config::{NestedKnot, NestedSpline};

const AXIS_NAMES: [&str; 3] = ["continentalness (c)", "terrain_shape (s)", "ridges_pv (r)"];
const MAX_DEPTH: usize = AXIS_NAMES.len();
const VALUE_RANGE: (f32, f32) = (-2.0, 2.0);
const INPUT_RANGE: (f32, f32) = (-1.5, 1.5);

/// Show + edit a NestedSpline. `depth` is the level in the tree
/// (0 = outermost, varying continentalness). Returns true if the user
/// changed any value this frame.
pub fn show(ui: &mut Ui, spline: &mut NestedSpline, depth: usize) -> bool {
    let mut dirty = false;
    let axis = AXIS_NAMES.get(depth).copied().unwrap_or("(beyond inputs)");

    if matches!(spline, NestedSpline::Multipoint(_)) {
        draw_curve(ui, spline);
    }

    match spline {
        NestedSpline::Constant(v) => {
            ui.label(
                egui::RichText::new(format!("Constant({:.3}) over {}", v, axis))
                    .small()
                    .weak(),
            );
            dirty |= ui
                .add(egui::Slider::new(v, VALUE_RANGE.0..=VALUE_RANGE.1).text("value"))
                .on_hover_text("Sets this branch of the spline to a single value across all inputs.")
                .changed();
            // Only offer "convert to Multipoint" when there's a deeper
            // input axis to vary on. Beyond depth 2 (r), additional
            // knots evaluate against a default-0 input — pointless.
            if depth < MAX_DEPTH
                && ui
                    .button("→ convert to Multipoint")
                    .on_hover_text("Replace with a 2-knot spline you can shape against the next axis.")
                    .clicked()
            {
                let current = *v;
                *spline = NestedSpline::Multipoint(vec![
                    NestedKnot { loc: -1.0, val: NestedSpline::Constant(current), slope: 0.0 },
                    NestedKnot { loc: 1.0, val: NestedSpline::Constant(current), slope: 0.0 },
                ]);
                dirty = true;
            }
        }
        NestedSpline::Multipoint(knots) => {
            ui.label(
                egui::RichText::new(format!("Multipoint over {} — {} knots", axis, knots.len()))
                    .small()
                    .weak(),
            );

            let mut to_remove: Option<usize> = None;
            for (i, knot) in knots.iter_mut().enumerate() {
                let header = format!("Knot {i}: loc={:+.2}", knot.loc);
                let resp = egui::CollapsingHeader::new(header)
                    .id_source(("nested_knot", depth, i))
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            dirty |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut knot.loc,
                                        INPUT_RANGE.0..=INPUT_RANGE.1,
                                    )
                                    .text("loc"),
                                )
                                .on_hover_text(format!(
                                    "Position of this knot on the {axis} axis.",
                                ))
                                .changed();
                            dirty |= ui
                                .add(egui::Slider::new(&mut knot.slope, -3.0..=3.0).text("slope"))
                                .on_hover_text("Tangent at this knot — controls curvature into adjacent segments.")
                                .changed();
                            if ui
                                .small_button("✖")
                                .on_hover_text("Delete this knot")
                                .clicked()
                            {
                                to_remove = Some(i);
                            }
                        });
                        if depth + 1 < MAX_DEPTH {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Sub-spline over {}:",
                                    AXIS_NAMES[depth + 1]
                                ))
                                .small()
                                .weak(),
                            );
                        }
                        dirty |= show(ui, &mut knot.val, depth + 1);
                    });
                let _ = resp;
            }

            if let Some(i) = to_remove {
                knots.remove(i);
                if knots.is_empty() {
                    // Empty Multipoint would panic in evaluate_inner.
                    *spline = NestedSpline::Constant(0.0);
                }
                dirty = true;
            }

            ui.horizontal(|ui| {
                if ui
                    .button("+ knot")
                    .on_hover_text("Insert a knot at the midpoint of the current loc range.")
                    .clicked()
                {
                    if let NestedSpline::Multipoint(knots) = spline {
                        let new_loc = if knots.is_empty() {
                            0.0
                        } else {
                            let lo = knots.first().unwrap().loc;
                            let hi = knots.last().unwrap().loc;
                            (lo + hi) * 0.5
                        };
                        knots.push(NestedKnot {
                            loc: new_loc,
                            val: NestedSpline::Constant(0.0),
                            slope: 0.0,
                        });
                        knots.sort_by(|a, b| {
                            a.loc.partial_cmp(&b.loc).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    dirty = true;
                }
                if ui
                    .button("→ Constant(0.0)")
                    .on_hover_text("Collapse the entire spline at this level to a single constant.")
                    .clicked()
                {
                    *spline = NestedSpline::Constant(0.0);
                    dirty = true;
                }
            });
        }
    }
    dirty
}

/// Draw a 1D curve plot of `spline.evaluate(x, 0, 0)` over the input
/// range. Read-only — the editor below the plot does the mutation.
/// Knot positions overlay as yellow dots so you can see where each
/// outer knot lands on the curve.
fn draw_curve(ui: &mut Ui, spline: &NestedSpline) {
    let width = ui.available_width().min(280.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 64.0), Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 4.0, Color32::from_rgb(20, 20, 26));

    // y=0 reference line.
    let y0 = rect.min.y + 0.5 * rect.height();
    painter.line_segment(
        [Pos2::new(rect.min.x, y0), Pos2::new(rect.max.x, y0)],
        Stroke::new(0.5, Color32::from_gray(60)),
    );

    let value_to_y = |v: f32| -> f32 {
        let t = (v - VALUE_RANGE.0) / (VALUE_RANGE.1 - VALUE_RANGE.0);
        rect.max.y - t.clamp(0.0, 1.0) * rect.height()
    };
    let input_to_x = |x: f32| -> f32 {
        let t = (x - INPUT_RANGE.0) / (INPUT_RANGE.1 - INPUT_RANGE.0);
        rect.min.x + t * rect.width()
    };

    let n = 80;
    let mut prev: Option<Pos2> = None;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = INPUT_RANGE.0 + t * (INPUT_RANGE.1 - INPUT_RANGE.0);
        let y = spline.evaluate(x, 0.0, 0.0);
        let p = Pos2::new(input_to_x(x), value_to_y(y));
        if let Some(pp) = prev {
            painter.line_segment([pp, p], Stroke::new(1.5, Color32::from_rgb(100, 200, 255)));
        }
        prev = Some(p);
    }

    // Knot markers: outer knot positions on the curve, drawn at the
    // sub-spline's value at (0, 0, 0) so the marker tracks the curve.
    if let NestedSpline::Multipoint(knots) = spline {
        for knot in knots {
            let x = knot.loc.clamp(INPUT_RANGE.0, INPUT_RANGE.1);
            let v = knot.val.evaluate(0.0, 0.0, 0.0);
            let p = Pos2::new(input_to_x(x), value_to_y(v));
            painter.circle_filled(p, 3.5, Color32::from_rgb(255, 220, 70));
            painter.circle_stroke(p, 3.5, Stroke::new(1.0, Color32::from_rgb(120, 100, 30)));
        }
    }
}
