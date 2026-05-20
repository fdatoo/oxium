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
            ui.selectable_value(&mut app.session.paint, crate::paint::PaintMode::Block, "Block");

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
                    app.session.invalidator.bump();
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
            ui.label(format!("seed {}", app.session.seed));
            ui.separator();
            ui.label(format!(
                "chunks meshed: {}",
                app.session.world.cached_mesh_coords().len()
            ));
            ui.separator();
            ui.label(format!("in-flight: {}", app.session.world.in_flight_len()));
            ui.separator();
            if let Some(ms) = app.last_regen_ms {
                ui.label(format!("last regen: {:.0} ms", ms));
            }
        });
    });

    // Central panel = transparent so the wgpu scene shows through.
    // We also draw the pinned-column highlight here so it gets clipped
    // to the 3D viewport rect (no overlap with side panels).
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |ui| {
            if let Some((pwx, pwz)) = app.session.probe.pinned {
                let screen = ctx.screen_rect();
                let aspect = screen.width() / screen.height().max(1.0);
                let view_proj = app.session.camera().view_proj(aspect);
                let cfg = app.session.config.load();
                let y_min = cfg.density.y_min as f32;
                let y_max = cfg.density.y_max as f32;
                draw_pinned_column(
                    ui,
                    screen,
                    view_proj,
                    pwx as f32 + 0.5,
                    pwz as f32 + 0.5,
                    y_min,
                    y_max,
                );
            }
        });

    out
}

/// Draw a yellow vertical line in screen space at world XZ `(wx, wz)`
/// spanning Y in `[y_min, y_max]`. Each segment is projected through
/// `view_proj` and stitched with a polyline so curvature from
/// perspective is honoured. Segments whose endpoints are behind the
/// camera are dropped (no spurious wrap-around across the screen).
fn draw_pinned_column(
    ui: &mut egui::Ui,
    screen: egui::Rect,
    view_proj: glam::Mat4,
    wx: f32,
    wz: f32,
    y_min: f32,
    y_max: f32,
) {
    let painter = ui.painter();
    let steps = 24;
    let stroke = egui::Stroke::new(
        2.0,
        egui::Color32::from_rgba_premultiplied(255, 220, 70, 200),
    );
    let mut prev: Option<egui::Pos2> = None;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let y = y_min + t * (y_max - y_min);
        let world = glam::Vec4::new(wx, y, wz, 1.0);
        let clip = view_proj * world;
        if clip.w <= 0.001 {
            prev = None;
            continue;
        }
        let nx = clip.x / clip.w;
        let ny = clip.y / clip.w;
        let sx = screen.left() + (nx + 1.0) * 0.5 * screen.width();
        let sy = screen.top() + (1.0 - ny) * 0.5 * screen.height();
        let p = egui::pos2(sx, sy);
        if let Some(pp) = prev {
            painter.line_segment([pp, p], stroke);
        }
        prev = Some(p);
    }
}
