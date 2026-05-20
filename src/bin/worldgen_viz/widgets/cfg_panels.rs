//! egui panels for editing WorldgenConfig live.

use egui::Ui;
use oxium::worldgen::config::{DensityConfig, WorldgenConfig};

/// Returns true if the user changed any value (caller marks `dirty`).
pub fn density_panel(ui: &mut Ui, cfg: &mut DensityConfig) -> bool {
    let snapshot_y_min = cfg.y_min;
    let snapshot_y_max = cfg.y_max;
    let snapshot_y_grad = cfg.y_gradient_amplitude;
    let snapshot_comp = cfg.composition_scale;
    let snapshot_soft = cfg.above_surface_softening;
    let snapshot_factor = cfg.factor;
    let snapshot_period = cfg.base_3d_period;
    let snapshot_amp = cfg.base_3d_amplitude;
    let snapshot_y_scale = cfg.base_3d_y_scale;
    let snapshot_top_blocks = cfg.slide_top_blocks;
    let snapshot_top_tgt = cfg.slide_top_target;
    let snapshot_bot_blocks = cfg.slide_bottom_blocks;
    let snapshot_bot_tgt = cfg.slide_bottom_target;
    ui.heading("Density");
    ui.collapsing("World range", |ui| {
        ui.add(egui::Slider::new(&mut cfg.y_min, -256..=0).text("y_min"));
        ui.add(egui::Slider::new(&mut cfg.y_max, 0..=512).text("y_max"));
        ui.add(
            egui::Slider::new(&mut cfg.y_gradient_amplitude, 0.1..=5.0)
                .text("y_gradient_amplitude"),
        );
    });
    ui.collapsing("Composition", |ui| {
        ui.add(egui::Slider::new(&mut cfg.composition_scale, 0.5..=16.0).text("composition_scale"));
        ui.add(
            egui::Slider::new(&mut cfg.above_surface_softening, 0.0..=1.0)
                .text("above_surface_softening"),
        );
        ui.add(egui::Slider::new(&mut cfg.factor, 0.1..=10.0).text("factor"));
    });
    ui.collapsing("Base 3D noise", |ui| {
        ui.add(egui::Slider::new(&mut cfg.base_3d_period, 4.0..=128.0).text("base_3d_period"));
        ui.add(egui::Slider::new(&mut cfg.base_3d_amplitude, 0.0..=4.0).text("base_3d_amplitude"));
        ui.add(egui::Slider::new(&mut cfg.base_3d_y_scale, 0.1..=2.0).text("base_3d_y_scale"));
    });
    ui.collapsing("Slides", |ui| {
        ui.add(egui::Slider::new(&mut cfg.slide_top_blocks, 0..=64).text("slide_top_blocks"));
        ui.add(egui::Slider::new(&mut cfg.slide_top_target, -1.0..=1.0).text("slide_top_target"));
        ui.add(egui::Slider::new(&mut cfg.slide_bottom_blocks, 0..=64).text("slide_bottom_blocks"));
        ui.add(
            egui::Slider::new(&mut cfg.slide_bottom_target, -1.0..=1.0).text("slide_bottom_target"),
        );
    });
    ui.collapsing("Offset spline", |ui| {
        ui.add(crate::widgets::spline::SplineEditor::new(&mut cfg.offset_spline));
        if let oxium::worldgen::spline::CubicSpline::Multipoint(knots) = &cfg.offset_spline {
            ui.label(format!("{} knots", knots.len()));
        } else {
            ui.label(format!("{:?}", cfg.offset_spline));
        }
        if ui.button("Convert to Multipoint with 4 knots").clicked() {
            cfg.offset_spline = oxium::worldgen::spline::CubicSpline::Multipoint(vec![
                oxium::worldgen::spline::Knot { loc: -1.0, val: -0.5, slope: 0.0 },
                oxium::worldgen::spline::Knot { loc: -0.3, val: -0.2, slope: 0.0 },
                oxium::worldgen::spline::Knot { loc: 0.2, val: 0.1, slope: 0.0 },
                oxium::worldgen::spline::Knot { loc: 1.0, val: 0.4, slope: 0.0 },
            ]);
        }
    });
    cfg.y_min != snapshot_y_min
        || cfg.y_max != snapshot_y_max
        || (cfg.y_gradient_amplitude - snapshot_y_grad).abs() > 1e-6
        || (cfg.composition_scale - snapshot_comp).abs() > 1e-6
        || (cfg.above_surface_softening - snapshot_soft).abs() > 1e-6
        || (cfg.factor - snapshot_factor).abs() > 1e-6
        || (cfg.base_3d_period - snapshot_period).abs() > 1e-6
        || (cfg.base_3d_amplitude - snapshot_amp).abs() > 1e-6
        || (cfg.base_3d_y_scale - snapshot_y_scale).abs() > 1e-6
        || cfg.slide_top_blocks != snapshot_top_blocks
        || (cfg.slide_top_target - snapshot_top_tgt).abs() > 1e-6
        || cfg.slide_bottom_blocks != snapshot_bot_blocks
        || (cfg.slide_bottom_target - snapshot_bot_tgt).abs() > 1e-6
}

/// Save / load preset buttons. Returns true if the config was
/// replaced from disk (caller marks `dirty`).
pub fn preset_panel(ui: &mut Ui, cfg: &mut WorldgenConfig) -> bool {
    let mut loaded = false;
    ui.heading("Presets");
    if ui.button("Save current → assets/worldgen/scratch.ron").clicked() {
        if let Ok(s) = ron::ser::to_string_pretty(cfg, ron::ser::PrettyConfig::default()) {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("assets")
                .join("worldgen")
                .join("scratch.ron");
            if let Err(e) = std::fs::write(&path, s) {
                eprintln!("save preset: {e}");
            } else {
                eprintln!("saved to {path:?}");
            }
        }
    }
    if ui.button("Reload default.ron").clicked() {
        if let Ok(new) = WorldgenConfig::bundled_default() {
            *cfg = new;
            loaded = true;
        }
    }
    loaded
}
