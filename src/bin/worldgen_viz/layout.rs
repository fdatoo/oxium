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

    // Right panel: probe placeholder (PR 2 fills this).
    egui::SidePanel::right("probe_panel")
        .resizable(true)
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("Probe");
            ui.label("Click a column in the 3D view to inspect (PR 2).");
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
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |_ui| {});

    out
}
