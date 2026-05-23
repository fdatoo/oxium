//! Field-value table for the probe panel.

use egui::Ui;
use oxium::voxel::block::Block;
use oxium::worldgen::probe::{ColumnProbe, DensityBreakdown};

pub fn show(
    ui: &mut Ui,
    snapshot: &ColumnProbe,
    breakdown: Option<&DensityBreakdown>,
    probe_y: i32,
) {
    ui.heading(format!("Column ({}, {})", snapshot.wx, snapshot.wz));

    ui.collapsing("Geometry", |ui| {
        kv(ui, "plate primary", format!("{:?}", snapshot.plate.a.id));
        kv(ui, "plate secondary", format!("{:?}", snapshot.plate.b.id));
        kv(ui, "boundary_t", fmt(snapshot.plate.t));
        kv(ui, "continentalness", fmt(snapshot.continentalness));
        kv(ui, "h_pre", fmt(snapshot.h_pre));
        kv(ui, "valley_carve", fmt(snapshot.valley_carve));
        kv(ui, "h_target", snapshot.h_target.to_string());
        kv(ui, "is_cliff", snapshot.is_cliff.to_string());
        kv(ui, "slope", fmt(snapshot.slope));
    });

    ui.collapsing("Climate", |ui| {
        kv(ui, "temperature", fmt(snapshot.temperature));
        kv(ui, "humidity", fmt(snapshot.humidity));
        kv(ui, "desertness", fmt(snapshot.desertness));
        kv(ui, "weirdness", fmt(snapshot.weirdness));
        kv(ui, "biome", format!("{:?}", snapshot.biome));
    });

    ui.collapsing("Hydrology", |ui| {
        kv(ui, "flow_accum", snapshot.flow_accum.to_string());
        kv(
            ui,
            "river_water_y",
            snapshot
                .river_water_y
                .map_or_else(|| "—".to_string(), |y| y.to_string()),
        );
        kv(
            ui,
            "river_bed_y",
            snapshot
                .river_bed_y
                .map_or_else(|| "—".to_string(), |y| y.to_string()),
        );
        kv(
            ui,
            "water_surf_y",
            match snapshot.water_surface_y {
                Some(y) => y.to_string(),
                None => "—".to_string(),
            },
        );
    });

    ui.collapsing("Legacy aquifer", |ui| {
        kv(ui, "y_top", snapshot.aquifer_y_top.to_string());
        kv(
            ui,
            "fluid",
            match snapshot.aquifer_fluid {
                Block::Water => "Water".to_string(),
                Block::Lava => "Lava".to_string(),
                b => format!("{:?}", b),
            },
        );
    });

    ui.collapsing("Caves", |ui| {
        kv(
            ui,
            "intersecting systems",
            snapshot.cave_systems_count.to_string(),
        );
    });

    if let Some(b) = breakdown {
        ui.collapsing(format!("Density @ y={probe_y}"), |ui| {
            kv(ui, "bias", fmt(b.bias));
            kv(ui, "base_3d", fmt(b.base_3d));
            kv(ui, "cave_sdf", fmt(b.cave_sdf));
            kv(ui, "cheese", fmt(b.cheese));
            kv(ui, "tera", fmt(b.tera));
            kv(ui, "cave_style", b.cave_style.unwrap_or("none").to_string());
            kv(ui, "cave_band", b.cave_band.unwrap_or("none").to_string());
            kv(ui, "pillar", fmt(b.pillar));
            kv(ui, "final_density", fmt(b.final_density));
            kv(ui, "block", format!("{:?}", b.block));
            kv(ui, "fluid_reason", format!("{:?}", b.fluid_reason));
        });
    }
}

fn kv(ui: &mut Ui, label: &str, value: String) {
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));
        ui.label(value);
    });
}

fn fmt(v: f32) -> String {
    if v.abs() < 1e-3 || v.abs() > 1e4 {
        format!("{:.3e}", v)
    } else {
        format!("{:.3}", v)
    }
}
