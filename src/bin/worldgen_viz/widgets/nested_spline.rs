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
//! Editing is direct-manipulation on the curve plot:
//! * Drag a knot → moves `loc` (horizontal) and shifts `val` by the
//!   corresponding delta (vertical — Constant leaves move, sub-
//!   Multipoint trees lift all leaves by the same amount).
//! * Drag either tangent handle → sets the knot's `slope`. Both
//!   handles share the slope so either gives the same result.
//! * Shift+click a knot → resets `slope` to 0.
//! * Click a knot → selects it. Below the curve, the selected knot
//!   gets a panel with explicit `loc`/`slope` readouts, a delete
//!   button, and a recursive sub-spline editor.
//!
//! The previous per-knot CollapsingHeader sections went away once
//! tangent dragging covered the slope axis — they were the last
//! thing the curve couldn't already express.

use egui::{Color32, Pos2, Sense, Stroke, Ui};
use oxium::worldgen::config::{NestedKnot, NestedSpline};

const AXIS_NAMES: [&str; 3] = ["continentalness (c)", "terrain_shape (s)", "ridges_pv (r)"];
const MAX_DEPTH: usize = AXIS_NAMES.len();
const VALUE_RANGE: (f32, f32) = (-1.5, 1.5);
const INPUT_RANGE: (f32, f32) = (-1.1, 1.1);
const PLOT_SIZE: (f32, f32) = (380.0, 160.0);
const KNOT_RADIUS_PX: f32 = 5.0;
const TANGENT_HANDLE_PX: f32 = 28.0;
const KNOT_HIT_RADIUS_PX: f32 = 8.0;
const HANDLE_HIT_RADIUS_PX: f32 = 6.0;
const CURVE_SAMPLES: usize = 120;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DragTarget {
    /// Dragging the knot itself — affects loc + val.
    Knot(usize),
    /// Dragging one of the tangent handles — affects slope.
    Tangent(usize),
}

/// Show + edit a NestedSpline. `depth` is the level in the tree
/// (0 = outermost, varying continentalness). Returns true if the user
/// changed any value this frame.
pub fn show(ui: &mut Ui, spline: &mut NestedSpline, depth: usize) -> bool {
    let mut dirty = false;
    let axis = AXIS_NAMES.get(depth).copied().unwrap_or("(beyond inputs)");

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
        NestedSpline::Multipoint(_) => {
            // Read knot count for the header before we hand the spline
            // off to the curve editor (which needs &mut).
            let knot_count = if let NestedSpline::Multipoint(k) = &*spline { k.len() } else { 0 };
            ui.label(
                egui::RichText::new(format!("Multipoint over {} — {} knots", axis, knot_count))
                    .small()
                    .weak(),
            );

            let (curve_dirty, selected_idx) = draw_and_edit_curve(ui, spline);
            dirty |= curve_dirty;

            // Selected-knot panel: explicit readouts + delete + recursive sub-editor.
            if let (Some(idx), NestedSpline::Multipoint(knots)) = (selected_idx, &mut *spline) {
                if idx < knots.len() {
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "Selected knot {idx}: loc={:+.3}, slope={:+.3}, val={:.3}",
                            knots[idx].loc,
                            knots[idx].slope,
                            knots[idx].val.evaluate(0.0, 0.0, 0.0),
                        ))
                        .small(),
                    );
                    let mut delete_requested = false;
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("✖ Delete knot")
                            .on_hover_text("Remove this knot from the spline.")
                            .clicked()
                        {
                            delete_requested = true;
                        }
                        if ui
                            .small_button("⟲ Reset slope")
                            .on_hover_text("Set this knot's slope to 0 (flat tangent).")
                            .clicked()
                        {
                            knots[idx].slope = 0.0;
                            dirty = true;
                        }
                    });
                    if delete_requested {
                        knots.remove(idx);
                        if knots.is_empty() {
                            *spline = NestedSpline::Constant(0.0);
                        }
                        dirty = true;
                        // Avoid recursing into a freed knot below.
                    } else {
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
                        dirty |= show(ui, &mut knots[idx].val, depth + 1);
                    }
                }
            } else if selected_idx.is_none() {
                ui.label(
                    egui::RichText::new("Click a knot to edit its sub-spline.")
                        .small()
                        .weak(),
                );
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

/// Draw the curve plot AND handle all direct-manipulation gestures
/// on it: knot drag, tangent drag, click-to-select, shift+click to
/// reset slope. Returns `(dirty, selected_knot_idx)`. Selection is
/// stashed in egui memory keyed on the response id so each recursive
/// level tracks its own selection.
fn draw_and_edit_curve(ui: &mut Ui, spline: &mut NestedSpline) -> (bool, Option<usize>) {
    let width = ui.available_width().min(PLOT_SIZE.0);
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(width, PLOT_SIZE.1),
        Sense::click_and_drag(),
    );
    let painter = ui.painter_at(rect);

    // Background + 5 horizontal grid lines — same styling as the
    // original `widgets::spline::SplineEditor` so all the spline
    // controls in the panel feel like one widget family.
    painter.rect_filled(rect, 4.0, Color32::from_rgb(20, 20, 26));
    for i in 0..=4 {
        let t = i as f32 / 4.0;
        let y = rect.min.y + t * rect.height();
        painter.line_segment(
            [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
            Stroke::new(0.5, Color32::from_gray(60)),
        );
    }

    let pixel_per_input = rect.width() / (INPUT_RANGE.1 - INPUT_RANGE.0);
    let pixel_per_value = rect.height() / (VALUE_RANGE.1 - VALUE_RANGE.0);

    let input_to_x = |x: f32| -> f32 {
        rect.min.x + (x - INPUT_RANGE.0) * pixel_per_input
    };
    let value_to_y = |v: f32| -> f32 {
        rect.max.y - (v - VALUE_RANGE.0) * pixel_per_value
    };
    let x_to_input = |px: f32| -> f32 {
        INPUT_RANGE.0 + (px - rect.min.x) / pixel_per_input
    };
    let y_to_value = |py: f32| -> f32 {
        VALUE_RANGE.0 + (rect.max.y - py) / pixel_per_value
    };

    // Curve trace — 120 samples, same blue stroke as the original
    // CubicSpline editor.
    let mut prev: Option<Pos2> = None;
    for i in 0..=CURVE_SAMPLES {
        let t = i as f32 / CURVE_SAMPLES as f32;
        let x = INPUT_RANGE.0 + t * (INPUT_RANGE.1 - INPUT_RANGE.0);
        let y = spline.evaluate(x, 0.0, 0.0);
        let p = Pos2::new(input_to_x(x), value_to_y(y).clamp(rect.min.y, rect.max.y));
        if let Some(pp) = prev {
            painter.line_segment([pp, p], Stroke::new(1.5, Color32::from_rgb(100, 200, 255)));
        }
        prev = Some(p);
    }

    let mut dirty = false;
    let widget_id = response.id;

    // Memory keys.
    let sel_id = widget_id.with("selected_knot");
    let drag_id = widget_id.with("drag_target");

    let mut selected: Option<usize> = ui.ctx().memory(|m| m.data.get_temp(sel_id));
    let mut drag_target: Option<DragTarget> = ui.ctx().memory(|m| m.data.get_temp(drag_id));

    // --- Hit testing + gestures (Multipoint only) ---
    if let NestedSpline::Multipoint(knots) = spline {
        // Compute pixel positions for each knot + tangent handle.
        let mut knot_px: Vec<Pos2> = Vec::with_capacity(knots.len());
        let mut left_handle_px: Vec<Pos2> = Vec::with_capacity(knots.len());
        let mut right_handle_px: Vec<Pos2> = Vec::with_capacity(knots.len());
        for knot in knots.iter() {
            let x = input_to_x(knot.loc.clamp(INPUT_RANGE.0, INPUT_RANGE.1));
            let y = value_to_y(knot.val.evaluate(0.0, 0.0, 0.0));
            let p = Pos2::new(x, y);
            // Tangent vector in pixel space: 1 input unit, `slope` value units.
            // y-axis is flipped in screen space, hence the negative.
            let tdx = pixel_per_input;
            let tdy = -knot.slope * pixel_per_value;
            let mag = (tdx * tdx + tdy * tdy).sqrt().max(1e-6);
            let dir = egui::vec2(tdx / mag, tdy / mag);
            knot_px.push(p);
            left_handle_px.push(p - dir * TANGENT_HANDLE_PX);
            right_handle_px.push(p + dir * TANGENT_HANDLE_PX);
        }

        // What's under the pointer right now?
        let hovered_target: Option<DragTarget> = response.hover_pos().and_then(|hp| {
            // Knots take priority over handles (they sit on top).
            let mut best: Option<(DragTarget, f32)> = None;
            for (i, &p) in knot_px.iter().enumerate() {
                let d = (hp - p).length();
                if d <= KNOT_HIT_RADIUS_PX && best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((DragTarget::Knot(i), d));
                }
            }
            if best.is_none() {
                for (i, &p) in left_handle_px.iter().enumerate().chain(right_handle_px.iter().enumerate()) {
                    let d = (hp - p).length();
                    if d <= HANDLE_HIT_RADIUS_PX && best.map_or(true, |(_, bd)| d < bd) {
                        best = Some((DragTarget::Tangent(i), d));
                    }
                }
            }
            best.map(|(t, _)| t)
        });

        // Scan raw input events for a primary-button RELEASE inside
        // the curve rect. Capturing modifiers at the release instant
        // is the only reliable way — `response.clicked()` doesn't
        // fire if the press introduced even a single pixel of drag,
        // which a real trackpad shift+click often does. We also use
        // press events to detect a plain click (so clicks that
        // weren't a drag still register), but the modifier check
        // happens against the release-event modifiers.
        let mut shift_release_in_rect: Option<Pos2> = None;
        let mut plain_release_in_rect: Option<Pos2> = None;
        ui.input(|i| {
            for ev in &i.events {
                if let egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers,
                } = ev
                {
                    if !rect.contains(*pos) {
                        continue;
                    }
                    if modifiers.shift {
                        shift_release_in_rect = Some(*pos);
                    } else {
                        plain_release_in_rect = Some(*pos);
                    }
                }
            }
        });

        // Hit-test a click position against the knot + tangent layout
        // we already computed.
        let hit_target = |pos: Pos2| -> Option<DragTarget> {
            let mut best: Option<(DragTarget, f32)> = None;
            for (i, &p) in knot_px.iter().enumerate() {
                let d = (pos - p).length();
                if d <= KNOT_HIT_RADIUS_PX && best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((DragTarget::Knot(i), d));
                }
            }
            if best.is_none() {
                for (i, &p) in left_handle_px.iter().enumerate() {
                    let d = (pos - p).length();
                    if d <= HANDLE_HIT_RADIUS_PX && best.map_or(true, |(_, bd)| d < bd) {
                        best = Some((DragTarget::Tangent(i), d));
                    }
                }
                for (i, &p) in right_handle_px.iter().enumerate() {
                    let d = (pos - p).length();
                    if d <= HANDLE_HIT_RADIUS_PX && best.map_or(true, |(_, bd)| d < bd) {
                        best = Some((DragTarget::Tangent(i), d));
                    }
                }
            }
            best.map(|(t, _)| t)
        };

        // Shift+release on a knot or handle → reset that knot's slope to 0.
        // Suppresses the plain-release "deselect" that would also fire.
        let mut consumed_release = false;
        if let Some(pos) = shift_release_in_rect {
            match hit_target(pos) {
                Some(DragTarget::Knot(i)) | Some(DragTarget::Tangent(i)) if i < knots.len() => {
                    knots[i].slope = 0.0;
                    dirty = true;
                    consumed_release = true;
                }
                _ => {}
            }
        }

        // Plain release: click-to-select on a knot, click-on-empty to deselect.
        // Uses response.clicked() AS WELL so a release without an event hit
        // (e.g. egui already consumed the event) still selects — and falls
        // back to the explicit event when response.clicked() is suppressed
        // by a micro-drag.
        if !consumed_release && (response.clicked() || plain_release_in_rect.is_some()) {
            let pos = response
                .interact_pointer_pos()
                .or(plain_release_in_rect)
                .unwrap_or(rect.center());
            match hit_target(pos) {
                Some(DragTarget::Knot(i)) => selected = Some(i),
                _ => selected = None,
            }
        }

        // Drag start → stash what we're dragging.
        if response.drag_started() {
            drag_target = hovered_target;
            if let Some(DragTarget::Knot(i)) = drag_target {
                selected = Some(i);
            }
        }

        // During drag → mutate.
        if response.dragged() {
            if let (Some(target), Some(pos)) = (drag_target, response.interact_pointer_pos()) {
                match target {
                    DragTarget::Knot(idx) if idx < knots.len() => {
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
                    DragTarget::Tangent(idx) if idx < knots.len() => {
                        // Pixel vector from knot to drag pos → slope.
                        let kp = knot_px[idx];
                        let dx_px = pos.x - kp.x;
                        let dy_px = pos.y - kp.y;
                        if dx_px.abs() > 1.0 {
                            let input_dx = dx_px / pixel_per_input;
                            let value_dy = -dy_px / pixel_per_value;
                            let new_slope = (value_dy / input_dx).clamp(-8.0, 8.0);
                            if (knots[idx].slope - new_slope).abs() > 1e-4 {
                                knots[idx].slope = new_slope;
                                dirty = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Drag end → resort by loc if we moved a knot, clear drag stash.
        if response.drag_stopped() {
            if matches!(drag_target, Some(DragTarget::Knot(_))) {
                knots.sort_by(|a, b| a.loc.partial_cmp(&b.loc).unwrap_or(std::cmp::Ordering::Equal));
            }
            drag_target = None;
        }

        // Tangent lines + handles (draw first so knots sit on top).
        // Tangent line uses a darker shade of the knot colour so it
        // reads as "part of the knot" rather than a separate widget.
        for i in 0..knots.len() {
            let lp = left_handle_px[i];
            let rp = right_handle_px[i];
            let is_active = matches!(drag_target, Some(DragTarget::Tangent(j)) if j == i)
                || matches!(hovered_target, Some(DragTarget::Tangent(j)) if j == i);
            let tangent_color = if is_active {
                Color32::YELLOW
            } else {
                Color32::from_rgba_premultiplied(160, 130, 40, 220)
            };
            painter.line_segment([lp, rp], Stroke::new(1.0, tangent_color));
            painter.circle_filled(lp, 2.5, tangent_color);
            painter.circle_filled(rp, 2.5, tangent_color);
        }

        // Knots — same colours as the original CubicSpline editor:
        // amber (220, 180, 80) at rest, bright YELLOW when hovered or
        // dragging. Selection gets a thin white ring so you can tell
        // which knot the inline panel is editing.
        for (i, &p) in knot_px.iter().enumerate() {
            let is_selected = selected == Some(i);
            let is_hot = matches!(hovered_target, Some(DragTarget::Knot(j)) if j == i)
                || matches!(drag_target, Some(DragTarget::Knot(j)) if j == i);
            let fill = if is_hot { Color32::YELLOW } else { Color32::from_rgb(220, 180, 80) };
            painter.circle_filled(p, KNOT_RADIUS_PX, fill);
            if is_selected {
                painter.circle_stroke(p, KNOT_RADIUS_PX + 1.0, Stroke::new(1.5, Color32::WHITE));
            }
        }

        if response.hovered() && !response.dragged() {
            response
                .clone()
                .on_hover_text("Drag knots to move them. Drag tangent endpoints to shape the slope. Shift+click resets slope to 0. Click empty space to deselect.");
        }
    }

    // Persist selection + drag target.
    ui.ctx().memory_mut(|m| {
        match selected {
            Some(i) => m.data.insert_temp(sel_id, i),
            None => m.data.remove::<usize>(sel_id),
        }
        match drag_target {
            Some(t) => m.data.insert_temp(drag_id, t),
            None => m.data.remove::<DragTarget>(drag_id),
        }
    });

    (dirty, selected)
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
