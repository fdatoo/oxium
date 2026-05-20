//! Dense Dashboard layout shell for the viz.

use crate::app::AppState;
use crate::session::CamKind;
use crate::widgets::cfg_panels::{
    biomes_panel, caves_panel, climate_panel, density_panel, graph_panel, preset_panel,
    surface_panel,
};
use egui::Context;
use oxium::worldgen::config::WorldgenConfig;

pub struct LayoutResult {
    pub dirty: bool,
    pub reset_camera: bool,
    pub force_regen: bool,
}

pub fn dashboard(ctx: &Context, app: &mut AppState) -> LayoutResult {
    let mut out = LayoutResult {
        dirty: false,
        reset_camera: false,
        force_regen: false,
    };

    // Top toolbar — paint mode, camera-kind toggle, regen button.
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Paint:");
            let prev_paint = app.session.paint;
            egui::ComboBox::from_id_source("paint_mode_combo")
                .selected_text(prev_paint.label())
                .show_ui(ui, |ui| {
                    for &m in crate::paint::PaintMode::ALL {
                        let resp = ui.selectable_label(prev_paint == m, m.label());
                        resp.clone().on_hover_text(m.description());
                        if resp.clicked() {
                            app.session.paint = m;
                        }
                    }
                });
            if app.session.paint != prev_paint {
                // Push the new mode into the streaming pipeline so the
                // next batch of jobs picks it up, and bump the
                // invalidator so the visible scene re-meshes.
                app.session.world.set_paint_mode(app.session.paint);
                app.session.invalidator.bump();
                out.dirty = true;
            }

            ui.separator();

            ui.label("Camera:");
            let is_fly = matches!(app.session.cam_kind, CamKind::Fly);
            if ui.selectable_label(is_fly, "Fly (WASD)").clicked() && !is_fly {
                app.session.toggle_camera();
            }
            if ui.selectable_label(!is_fly, "Orbit").clicked() && is_fly {
                app.session.toggle_camera();
            }

            ui.separator();

            if ui.button("Reset camera").clicked() {
                out.reset_camera = true;
            }
            if ui.button("[R] Regen").clicked() {
                out.force_regen = true;
            }
        });
    });

    // Left panel: config editors (today's panels, unchanged).
    egui::SidePanel::left("config_panel")
        .resizable(true)
        .default_width(380.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let cfg_arc = app.session.config.load();
                let mut cfg: WorldgenConfig = (*cfg_arc).clone();
                let mut local_dirty = false;
                local_dirty |= density_panel(ui, &mut cfg.density);
                ui.separator();
                local_dirty |= climate_panel(ui, &mut cfg.climate);
                ui.separator();
                local_dirty |= caves_panel(ui, &mut cfg);
                ui.separator();
                local_dirty |= biomes_panel(ui, &mut cfg.biomes);
                ui.separator();
                local_dirty |= surface_panel(ui, &mut cfg);
                ui.separator();
                local_dirty |= preset_panel(ui, &mut cfg);
                ui.separator();
                graph_panel(ui, &cfg.density);
                if local_dirty {
                    app.session.config.swap(cfg);
                    // queue() instead of bump() — debounces slider /
                    // spline drags so each tick of a drag doesn't
                    // trigger a wipe-and-refill. The actual bump fires
                    // ~200 ms after the user pauses, via Invalidator::tick().
                    app.session.invalidator.queue();
                    out.dirty = true;
                }
            });
        });

    // Right panel: overlay map + probe (PR 2).
    let generator = app.session.generator.clone();
    egui::SidePanel::right("probe_panel")
        .resizable(true)
        .default_width(360.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Map section
                ui.heading("Overlay map");
                let revision = app.session.invalidator.revision();
                let pinned = app.session.probe.pinned;
                let clicked = app.session.map.show(ui, &generator, revision, pinned);
                if let Some((wx, wz)) = clicked {
                    app.session.probe.pin(&generator, wx, wz);
                }
                ui.separator();

                // Probe section
                ui.heading("Probe");
                if let Some(snap) = app.session.probe.snapshot.clone() {
                    ui.horizontal(|ui| {
                        if ui.button("Unpin").clicked() {
                            app.session.probe.unpin();
                        }
                    });
                    let probe_y_before = app.session.probe.probe_y;
                    let mut y_local = probe_y_before;
                    ui.add(egui::Slider::new(&mut y_local, -64..=256).text("probe y"));
                    if y_local != probe_y_before {
                        app.session.probe.set_y(&generator, y_local);
                    }
                    crate::widgets::probe_table::show(
                        ui,
                        &snap,
                        app.session.probe.breakdown.as_ref(),
                        app.session.probe.probe_y,
                    );
                } else {
                    ui.label("Click the map to pin a column.");
                }
            });
        });

    // Bottom status bar.
    egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let pos = app.session.camera().position();
            ui.label(format!("pos ({:.0}, {:.0}, {:.0})", pos.x, pos.y, pos.z));
            ui.separator();
            let is_fly = matches!(app.session.cam_kind, CamKind::Fly);
            if is_fly {
                ui.label(format!("fly speed {:.0}", app.session.fly.speed));
                ui.separator();
            }
            ui.label(format!("seed {}", app.session.seed));
            ui.separator();
            ui.label(format!(
                "drawn {}/{}",
                app.scene_chunks_visible, app.scene_chunks_total,
            ))
            .on_hover_text(
                "Chunks passing frustum culling / total chunks held by the renderer.\n\
                 Chunks outside the camera frustum stay in memory but skip the draw call.",
            );
            ui.separator();
            ui.label(format!("in-flight: {}", app.session.world.in_flight_len()))
                .on_hover_text("Fill+mesh jobs queued on the rayon pool.");
            ui.separator();
            if let Some(remaining) = app.session.invalidator.pending_remaining() {
                ui.colored_label(
                    egui::Color32::from_rgb(230, 180, 80),
                    format!("edit pending ({} ms)", remaining.as_millis()),
                );
                ui.separator();
            } else if let Some(ms) = app.last_regen_ms {
                ui.label(format!("last regen: {:.0} ms", ms));
                ui.separator();
            }
            // Controls hint, right-aligned. Hover for the full keymap.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let short = if is_fly {
                    "WASD/QE · RMB look · MMB pan · Wheel dolly · F focus"
                } else {
                    "RMB orbit · MMB pan · Wheel zoom · F focus"
                };
                ui.label(egui::RichText::new(short).small().weak())
                    .on_hover_text(
                        "Camera & picking:\n\
                         · WASD : forward/back/strafe (fly cam)\n\
                         · QE : down/up (fly cam)\n\
                         · Shift : boost while moving\n\
                         · RMB drag : look (fly) / orbit (orbit)\n\
                         · MMB drag : pan parallel to view\n\
                         · Wheel : dolly forward (fly) / zoom (orbit)\n\
                         · Ctrl + Wheel : adjust fly speed\n\
                         · F : focus camera on pinned column\n\
                         · O : toggle fly / orbit\n\
                         · R : force regen\n\
                         · Left-click 3D : pin column",
                    );
            });
        });
    });

    // Central panel = transparent so the wgpu scene shows through.
    // We also draw the pinned-chunk wireframe here so it gets clipped
    // to the 3D viewport rect (no overlap with side panels).
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |ui| {
            if let Some((pwx, pwz)) = app.session.probe.pinned {
                let screen = ctx.screen_rect();
                let aspect = screen.width() / screen.height().max(1.0);
                let view_proj = app.session.camera().view_proj(aspect);
                let cfg = app.session.config.load();
                draw_pinned_chunk_outline(
                    ui,
                    screen,
                    view_proj,
                    pwx,
                    pwz,
                    cfg.density.y_min as f32,
                    cfg.density.y_max as f32,
                );
            }
        });

    out
}

/// Draw a yellow wireframe of the chunk that contains world column
/// `(pwx, pwz)` — 12 edges of the chunk's AABB (4 vertical posts, 4
/// edges along the top, 4 along the bottom). Each edge is sampled
/// and stitched with a polyline so perspective curvature is honoured
/// and segments behind the camera are dropped. Far more informative
/// than the previous single-column line: you can see at a glance
/// which 32-block slab the probe is in.
fn draw_pinned_chunk_outline(
    ui: &mut egui::Ui,
    screen: egui::Rect,
    view_proj: glam::Mat4,
    pwx: i32,
    pwz: i32,
    y_min: f32,
    y_max: f32,
) {
    use oxium::voxel::coords::CHUNK_DIM_U;
    let dim = CHUNK_DIM_U as i32;
    let cx = pwx.div_euclid(dim) * dim;
    let cz = pwz.div_euclid(dim) * dim;
    let x0 = cx as f32;
    let x1 = (cx + dim) as f32;
    let z0 = cz as f32;
    let z1 = (cz + dim) as f32;

    let painter = ui.painter();
    let stroke = egui::Stroke::new(
        2.0,
        egui::Color32::from_rgba_premultiplied(255, 220, 70, 220),
    );

    let v = glam::Vec3::new;
    // 12 edges of the chunk AABB.
    let edges: [(glam::Vec3, glam::Vec3); 12] = [
        // Verticals at the four XZ corners.
        (v(x0, y_min, z0), v(x0, y_max, z0)),
        (v(x1, y_min, z0), v(x1, y_max, z0)),
        (v(x0, y_min, z1), v(x0, y_max, z1)),
        (v(x1, y_min, z1), v(x1, y_max, z1)),
        // Bottom rectangle.
        (v(x0, y_min, z0), v(x1, y_min, z0)),
        (v(x1, y_min, z0), v(x1, y_min, z1)),
        (v(x1, y_min, z1), v(x0, y_min, z1)),
        (v(x0, y_min, z1), v(x0, y_min, z0)),
        // Top rectangle.
        (v(x0, y_max, z0), v(x1, y_max, z0)),
        (v(x1, y_max, z0), v(x1, y_max, z1)),
        (v(x1, y_max, z1), v(x0, y_max, z1)),
        (v(x0, y_max, z1), v(x0, y_max, z0)),
    ];

    for (a, b) in edges {
        draw_world_segment(&painter, screen, view_proj, a, b, stroke);
    }
}

/// Sample a world-space line segment, project each sample through
/// `view_proj` to screen space, draw a polyline. Drops samples behind
/// the camera so an edge that crosses the camera plane doesn't wrap
/// around the screen.
fn draw_world_segment(
    painter: &egui::Painter,
    screen: egui::Rect,
    view_proj: glam::Mat4,
    from: glam::Vec3,
    to: glam::Vec3,
    stroke: egui::Stroke,
) {
    let steps = 16;
    let mut prev: Option<egui::Pos2> = None;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let p = from.lerp(to, t);
        let clip = view_proj * glam::Vec4::new(p.x, p.y, p.z, 1.0);
        if clip.w <= 0.001 {
            prev = None;
            continue;
        }
        let nx = clip.x / clip.w;
        let ny = clip.y / clip.w;
        let sx = screen.left() + (nx + 1.0) * 0.5 * screen.width();
        let sy = screen.top() + (1.0 - ny) * 0.5 * screen.height();
        let sp = egui::pos2(sx, sy);
        if let Some(pp) = prev {
            painter.line_segment([pp, sp], stroke);
        }
        prev = Some(sp);
    }
}
