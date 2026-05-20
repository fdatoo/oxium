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
    // (Pinned-chunk highlight is rendered as a shader tint on the
    // mesh itself — see render/shader.wgsl.)
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |_ui| {});

    out
}
