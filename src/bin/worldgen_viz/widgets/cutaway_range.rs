//! Horizontal range slider with two draggable clamps — sets the
//! min/max world-Y for the cutaway sweep. When looping is off the
//! upper clamp doubles as the cutaway position, so this one widget
//! replaces both the old single-value "max y" slider and a separate
//! "loop bounds" control.
//!
//! The playhead (only drawn while looping) shows where the sweep
//! currently sits within the clamp range.

use egui::{vec2, Color32, Pos2, Rect, Response, Sense, Stroke, Ui};

pub struct CutawayRangeResponse {
    pub response: Response,
    /// True if either clamp moved this frame. The caller uses this
    /// to know when to forward changes to debounced regen, etc.
    pub clamps_changed: bool,
}

/// Range track + two clamp handles + optional playhead.
///
/// `min` / `max` are the loop bounds (mutated in place). `playhead`
/// is `Some` when looping is on; the marker is drawn at that value.
/// When `playhead` is `None` the upper handle is highlighted in the
/// active colour so it reads as "this is the cutaway position".
pub fn cutaway_range(
    ui: &mut Ui,
    min: &mut f32,
    max: &mut f32,
    playhead: Option<f32>,
    range: std::ops::RangeInclusive<f32>,
) -> CutawayRangeResponse {
    let desired = vec2(ui.available_width().min(260.0).max(160.0), 26.0);
    let (rect, mut response) = ui.allocate_exact_size(desired, Sense::click_and_drag());

    let track_y = rect.center().y;
    let pad = 10.0;
    let track_left = rect.left() + pad;
    let track_right = rect.right() - pad;
    let track_w = track_right - track_left;
    let (r_lo, r_hi) = (*range.start(), *range.end());
    let span = (r_hi - r_lo).max(1e-6);

    let to_x = |v: f32| -> f32 { track_left + ((v - r_lo) / span) * track_w };
    let to_v = |x: f32| -> f32 { r_lo + ((x - track_left) / track_w).clamp(0.0, 1.0) * span };

    // Drag handling. We snap to whichever handle is closer on initial
    // click and stick with it until the drag ends — otherwise crossing
    // through the middle would hand the drag off mid-gesture, which
    // feels broken.
    let id = response.id;
    let mut clamps_changed = false;
    if response.drag_started() {
        if let Some(p) = response.interact_pointer_pos() {
            let d_min = (p.x - to_x(*min)).abs();
            let d_max = (p.x - to_x(*max)).abs();
            ui.memory_mut(|m| {
                m.data
                    .insert_temp::<bool>(id.with("dragging_max"), d_max <= d_min);
            });
        }
    }
    if response.dragged() {
        if let Some(p) = response.interact_pointer_pos() {
            let dragging_max = ui
                .memory(|m| m.data.get_temp::<bool>(id.with("dragging_max")))
                .unwrap_or(true);
            let v = to_v(p.x);
            if dragging_max {
                let nv = v.max(*min);
                if (nv - *max).abs() > f32::EPSILON {
                    *max = nv;
                    clamps_changed = true;
                }
            } else {
                let nv = v.min(*max);
                if (nv - *min).abs() > f32::EPSILON {
                    *min = nv;
                    clamps_changed = true;
                }
            }
        }
    }
    if clamps_changed {
        response.mark_changed();
    }

    // Paint.
    let painter = ui.painter_at(rect);
    // Inactive track.
    painter.rect_filled(
        Rect::from_min_max(
            Pos2::new(track_left, track_y - 3.0),
            Pos2::new(track_right, track_y + 3.0),
        ),
        2.0,
        Color32::from_gray(60),
    );
    // Active region between clamps.
    let min_x = to_x(*min);
    let max_x = to_x(*max);
    painter.rect_filled(
        Rect::from_min_max(
            Pos2::new(min_x, track_y - 3.0),
            Pos2::new(max_x, track_y + 3.0),
        ),
        2.0,
        Color32::from_rgb(70, 120, 180),
    );
    // Tick marks at 0 and 64 (sea level / typical surface) so the user
    // has a frame of reference without needing a number label.
    for tick_v in [0.0_f32, 64.0_f32] {
        if tick_v >= r_lo && tick_v <= r_hi {
            let x = to_x(tick_v);
            painter.line_segment(
                [
                    Pos2::new(x, track_y - 8.0),
                    Pos2::new(x, track_y + 8.0),
                ],
                Stroke::new(1.0, Color32::from_gray(100)),
            );
        }
    }
    // Playhead — drawn only when looping is on (playhead Some).
    if let Some(ph) = playhead {
        let x = to_x(ph.clamp(r_lo, r_hi));
        painter.line_segment(
            [
                Pos2::new(x, track_y - 10.0),
                Pos2::new(x, track_y + 10.0),
            ],
            Stroke::new(2.0, Color32::from_rgb(255, 180, 60)),
        );
    }
    // Clamp handles. When the playhead is hidden the upper handle is
    // the cutaway position, so we make it brighter to advertise that.
    let lo_color = Color32::from_gray(200);
    let hi_color = if playhead.is_some() {
        Color32::from_gray(200)
    } else {
        Color32::from_rgb(255, 200, 80)
    };
    painter.circle_filled(Pos2::new(min_x, track_y), 5.0, lo_color);
    painter.circle_filled(Pos2::new(max_x, track_y), 5.0, hi_color);

    CutawayRangeResponse {
        response,
        clamps_changed,
    }
}
