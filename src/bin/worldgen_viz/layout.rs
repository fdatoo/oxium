//! Dense Dashboard layout shell for the viz.

use crate::app::AppState;
use crate::session::CamKind;
use crate::widgets::cfg_panels::{
    biomes_panel, caves_panel, climate_panel, density_panel, graph_panel, surface_panel,
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

    // Left panel: preset library + config editors.
    egui::SidePanel::left("config_panel")
        .resizable(true)
        .default_width(380.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Preset library — sits at the top so it's the first
                // thing the user sees. Loading a preset is a
                // wholesale config swap; subsequent panel edits work
                // on top of it.
                let cfg_arc = app.session.config.load();
                let mut cfg: WorldgenConfig = (*cfg_arc).clone();
                let mut preset_loaded = false;
                egui::CollapsingHeader::new("Presets")
                    .default_open(true)
                    .show(ui, |ui| {
                        preset_loaded = preset_library_section(ui, app, &mut cfg);
                    });
                ui.separator();

                let mut local_dirty = preset_loaded;
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
                graph_panel(ui, &cfg.density);
                if local_dirty {
                    app.session.config.swap(cfg);
                    // queue() instead of bump() for slider/spline
                    // drags. Preset loads are also queued for
                    // simplicity — the debounce window is short
                    // enough that the user perceives a load as
                    // immediate.
                    app.session.invalidator.queue();
                    out.dirty = true;
                }
            });
        });

    // Right panel: tabbed visualisations (Map / Cross-section) over a
    // persistent Probe inspector below. Tabs share the same input
    // (pinned column) and the same revision counter — switching tabs
    // is a zero-cost UI swap; the tab not currently shown still
    // holds its cached texture, no regen on switch-back.
    let generator = app.session.generator.clone();
    let revision = app.session.invalidator.revision();
    let pin_for_cross = app
        .session
        .probe
        .snapshot
        .as_ref()
        .map(|s| (s.wx, s.h_target, s.wz));

    egui::SidePanel::right("probe_panel")
        .resizable(true)
        .default_width(360.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Tab strip.
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(app.right_tab == crate::app::RightTab::Map, "🗺 Map")
                        .on_hover_text("2D top-down map with selectable pipeline stage overlay.")
                        .clicked()
                    {
                        app.right_tab = crate::app::RightTab::Map;
                    }
                    if ui
                        .selectable_label(
                            app.right_tab == crate::app::RightTab::CrossSection,
                            "✂ Cross-section",
                        )
                        .on_hover_text("Cut-plane heatmap through the pinned column.")
                        .clicked()
                    {
                        app.right_tab = crate::app::RightTab::CrossSection;
                    }
                });
                ui.separator();

                // Active tab body.
                match app.right_tab {
                    crate::app::RightTab::Map => {
                        let pinned = app.session.probe.pinned;
                        let clicked = app
                            .session
                            .map
                            .show(ui, &generator, revision, pinned);
                        if let Some((wx, wz)) = clicked {
                            app.session.probe.pin(&generator, wx, wz);
                        }
                    }
                    crate::app::RightTab::CrossSection => {
                        app.session
                            .cross
                            .show(ui, &generator, revision, pin_for_cross);
                    }
                }
                ui.separator();

                // Probe inspector — persistent across tabs because it's
                // the canonical "details about the pinned column" view.
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
                    ui.label("Click the map (or a column in the 3D view) to pin one.");
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

/// Preset library section. Lists every `*.ron` under
/// `assets/worldgen/presets/`, plus the bundled `default.ron` as
/// an immutable factory baseline. Returns `true` if the user just
/// loaded a preset (caller marks the config dirty so the regen
/// path picks it up).
fn preset_library_section(
    ui: &mut egui::Ui,
    app: &mut AppState,
    cfg: &mut WorldgenConfig,
) -> bool {
    use crate::preset;
    let mut loaded = false;

    // Always-available factory baseline.
    ui.horizontal(|ui| {
        if ui
            .button("⟲ Reload default")
            .on_hover_text("Replace the current config with the bundled assets/worldgen/default.ron.")
            .clicked()
        {
            if let Ok(new) = WorldgenConfig::bundled_default() {
                *cfg = new;
                app.presets.activate(None);
                loaded = true;
            }
        }
    });

    ui.separator();

    // Save-as input.
    ui.horizontal(|ui| {
        ui.label("Save as:");
        let resp = ui.add(
            egui::TextEdit::singleline(&mut app.presets.new_name)
                .hint_text("preset-name")
                .desired_width(180.0),
        );
        let _ = resp;
        let can_save = !app.presets.new_name.trim().is_empty();
        if ui
            .add_enabled(can_save, egui::Button::new("💾 Save"))
            .on_hover_text("Write the current config to assets/worldgen/presets/<name>.ron. Existing presets with the same name are overwritten.")
            .clicked()
        {
            match preset::save(&app.presets.new_name, cfg) {
                Ok(entry) => {
                    let name = entry.name.clone();
                    app.presets.refresh_entries();
                    app.presets.activate(Some(name));
                    app.presets.new_name.clear();
                }
                Err(e) => eprintln!("save preset: {e}"),
            }
        }
    });

    ui.separator();

    // Preset list.
    if app.presets.entries.is_empty() {
        ui.label(
            egui::RichText::new("No saved presets yet. Type a name above and click Save.")
                .small()
                .weak(),
        );
    } else {
        let active = app.presets.active_name.clone();
        let entries = app.presets.entries.clone();
        let mut to_load: Option<usize> = None;
        let mut to_delete: Option<usize> = None;
        for (i, entry) in entries.iter().enumerate() {
            ui.horizontal(|ui| {
                let is_active = active.as_deref() == Some(entry.name.as_str());
                let label = if is_active {
                    format!("● {}", entry.name)
                } else {
                    format!("○ {}", entry.name)
                };
                if ui
                    .selectable_label(is_active, label)
                    .on_hover_text("Click to activate (select for notes); use Load to apply its config.")
                    .clicked()
                {
                    app.presets.activate(Some(entry.name.clone()));
                }
                if ui
                    .small_button("▶ Load")
                    .on_hover_text("Replace the current config with this preset.")
                    .clicked()
                {
                    to_load = Some(i);
                }
                if ui
                    .small_button("✖")
                    .on_hover_text("Delete this preset and its notes file.")
                    .clicked()
                {
                    to_delete = Some(i);
                }
            });
        }
        if let Some(i) = to_load {
            match preset::load(&entries[i]) {
                Ok(new_cfg) => {
                    *cfg = new_cfg;
                    app.presets.activate(Some(entries[i].name.clone()));
                    loaded = true;
                }
                Err(e) => eprintln!("load preset {}: {e}", entries[i].name),
            }
        }
        if let Some(i) = to_delete {
            let _ = preset::delete(&entries[i]);
            app.presets.refresh_entries();
            if app.presets.active_name.as_deref() == Some(entries[i].name.as_str()) {
                app.presets.activate(None);
            }
        }
    }

    // Notes editor for the active preset.
    if let Some(name) = app.presets.active_name.clone() {
        ui.separator();
        ui.label(
            egui::RichText::new(format!("Notes for '{name}':"))
                .small()
                .weak(),
        );
        ui.add(
            egui::TextEdit::multiline(&mut app.presets.notes_buffer)
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .hint_text("Freeform notes — what works, what to try next."),
        );
        ui.horizontal(|ui| {
            let dirty = app.presets.notes_dirty();
            if ui
                .add_enabled(dirty, egui::Button::new("💾 Save notes"))
                .on_hover_text("Write notes to assets/worldgen/presets/<name>.notes.md.")
                .clicked()
            {
                if let Some(entry) = app.presets.entries.iter().find(|e| e.name == name).cloned() {
                    match preset::save_notes(&entry, &app.presets.notes_buffer) {
                        Ok(()) => app.presets.mark_notes_saved(),
                        Err(e) => eprintln!("save notes: {e}"),
                    }
                }
            }
            if dirty {
                ui.label(
                    egui::RichText::new("(unsaved)")
                        .small()
                        .color(egui::Color32::from_rgb(220, 180, 80)),
                );
            }
        });
    }

    loaded
}


