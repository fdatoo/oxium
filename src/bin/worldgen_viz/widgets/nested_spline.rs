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
        dirty |= draw_and_drag_curve(ui, spline);
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

/// Draw the 1D curve plot AND let the user drag the knot markers.
/// Returns true if a drag mutated the spline this frame.
///
/// Horizontal drag → `knot.loc`. Vertical drag → shifts `knot.val` by
/// the implied delta: if the sub-spline is a Constant, the constant
/// moves; if it's a sub-Multipoint, every inner Constant moves by the
/// same delta, preserving the relative shape of the sub-curve. That's
/// the only sensible interpretation of "drag this region up by 0.3"
/// when the knot's value isn't a scalar.
fn draw_and_drag_curve(ui: &mut Ui, spline: &mut NestedSpline) -> bool {
    let width = ui.available_width().min(280.0);
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(width, 64.0),
        Sense::click_and_drag(),
    );
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 4.0, Color32::from_rgb(20, 20, 26));

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
    let x_to_input = |px: f32| -> f32 {
        let t = (px - rect.min.x) / rect.width();
        INPUT_RANGE.0 + t * (INPUT_RANGE.1 - INPUT_RANGE.0)
    };
    let y_to_value = |py: f32| -> f32 {
        let t = (rect.max.y - py) / rect.height();
        VALUE_RANGE.0 + t * (VALUE_RANGE.1 - VALUE_RANGE.0)
    };

    // Curve trace.
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

    // Drag interaction — only meaningful for Multipoint.
    let mut dirty = false;
    let drag_id = response.id;
    if let NestedSpline::Multipoint(knots) = spline {
        // Find the knot closest to the pointer (within 12 px).
        let hovered_idx = response.hover_pos().and_then(|hp| {
            let mut best: Option<(usize, f32)> = None;
            for (i, knot) in knots.iter().enumerate() {
                let x = input_to_x(knot.loc.clamp(INPUT_RANGE.0, INPUT_RANGE.1));
                let y = value_to_y(knot.val.evaluate(0.0, 0.0, 0.0));
                let d = (hp - Pos2::new(x, y)).length();
                if d <= 12.0 && best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((i, d));
                }
            }
            best.map(|(i, _)| i)
        });

        // Start drag → stash the knot index in egui memory.
        if response.drag_started() {
            if let Some(i) = hovered_idx {
                ui.ctx().memory_mut(|m| m.data.insert_temp(drag_id, i));
            }
        }

        // During drag → update loc + shift val.
        if response.dragged() {
            let stashed: Option<usize> =
                ui.ctx().memory(|m| m.data.get_temp::<usize>(drag_id));
            if let (Some(idx), Some(pos)) = (stashed, response.interact_pointer_pos()) {
                if idx < knots.len() {
                    let new_loc = x_to_input(pos.x).clamp(INPUT_RANGE.0, INPUT_RANGE.1);
                    let new_val = y_to_value(pos.y).clamp(VALUE_RANGE.0, VALUE_RANGE.1);
                    let current_val = knots[idx].val.evaluate(0.0, 0.0, 0.0);
                    let delta = new_val - current_val;
                    if (knots[idx].loc - new_loc).abs() > 1e-6 || delta.abs() > 1e-6 {
                        knots[idx].loc = new_loc;
                        offset_nested(&mut knots[idx].val, delta);
                        dirty = true;
                    }
                }
            }
        }

        // End drag → resort by loc so the spline stays well-ordered;
        // clear the stashed index.
        if response.drag_stopped() {
            knots.sort_by(|a, b| a.loc.partial_cmp(&b.loc).unwrap_or(std::cmp::Ordering::Equal));
            ui.ctx().memory_mut(|m| m.data.remove::<usize>(drag_id));
        }

        // Knot markers — bright hover, even brighter while dragging.
        let dragging_idx: Option<usize> = if response.dragged() {
            ui.ctx().memory(|m| m.data.get_temp::<usize>(drag_id))
        } else {
            None
        };
        for (i, knot) in knots.iter().enumerate() {
            let x = input_to_x(knot.loc.clamp(INPUT_RANGE.0, INPUT_RANGE.1));
            let y = value_to_y(knot.val.evaluate(0.0, 0.0, 0.0));
            let p = Pos2::new(x, y);
            let (fill, ring, radius) = if dragging_idx == Some(i) {
                (Color32::from_rgb(255, 255, 120), Color32::from_rgb(200, 160, 30), 5.0)
            } else if hovered_idx == Some(i) {
                (Color32::from_rgb(255, 240, 100), Color32::from_rgb(160, 130, 40), 4.5)
            } else {
                (Color32::from_rgb(255, 220, 70), Color32::from_rgb(120, 100, 30), 3.5)
            };
            painter.circle_filled(p, radius, fill);
            painter.circle_stroke(p, radius, Stroke::new(1.0, ring));
        }
    }

    if response.hovered() && !response.dragged() {
        response.clone().on_hover_text("Drag a yellow knot to move it on (loc, val).");
    }

    dirty
}

/// Recursively offset every leaf Constant in `spline` by `delta`.
/// Preserves the shape of any sub-Multipoint by lifting all its
/// leaves the same amount.
fn offset_nested(spline: &mut NestedSpline, delta: f32) {
    match spline {
        NestedSpline::Constant(v) => *v += delta,
        NestedSpline::Multipoint(knots) => {
            for knot in knots {
                offset_nested(&mut knot.val, delta);
            }
        }
    }
}
