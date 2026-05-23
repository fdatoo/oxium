//! egui panels for editing WorldgenConfig live. Every control returns
//! a `Response`; we OR `.changed()` into a local `dirty` so the caller
//! knows whether to swap the config in and trigger a regen. Each
//! control also carries an `.on_hover_text()` tooltip describing what
//! it actually controls — without those a tuning session is mostly
//! mystery-meat sliders.

use egui::{Response, Ui};
use oxium::worldgen::config::{DensityConfig, WorldgenConfig};

/// Helper: add a slider with a tooltip. Returns `Response.changed()`.
fn slider<Num: egui::emath::Numeric>(
    ui: &mut Ui,
    value: &mut Num,
    range: std::ops::RangeInclusive<Num>,
    text: &str,
    tooltip: &str,
) -> bool {
    add_with_tooltip(
        ui,
        |ui| ui.add(egui::Slider::new(value, range).text(text)),
        tooltip,
    )
    .changed()
}

fn add_with_tooltip<R: FnOnce(&mut Ui) -> Response>(
    ui: &mut Ui,
    add: R,
    tooltip: &str,
) -> Response {
    let r = add(ui);
    r.on_hover_text(tooltip)
}

pub fn density_panel(ui: &mut Ui, cfg: &mut DensityConfig) -> bool {
    let mut dirty = false;
    ui.heading("Density");

    ui.collapsing("World range", |ui| {
        dirty |= slider(
            ui, &mut cfg.y_min, -256..=0, "y_min",
            "Lowest world Y. At this depth the y-gradient term has full positive value (forces solid).",
        );
        dirty |= slider(
            ui, &mut cfg.y_max, 0..=512, "y_max",
            "Highest world Y. At this height the y-gradient term has full negative value (forces air).",
        );
        dirty |= slider(
            ui, &mut cfg.y_gradient_amplitude, 0.1..=5.0, "y_gradient_amplitude",
            "Magnitude of the world-Y gradient that turns density positive below the surface and negative above. Larger = sharper transition between solid floor and air sky.",
        );
    });

    ui.collapsing("Composition", |ui| {
        dirty |= slider(
            ui, &mut cfg.composition_scale, 0.5..=16.0, "composition_scale",
            "Multiplier on the `(depth + jagged) * factor` shaped term before adding base 3D noise. MC default is 4.",
        );
        dirty |= slider(
            ui, &mut cfg.above_surface_softening, 0.0..=1.0, "above_surface_softening",
            "How much the shaped term is dampened above the surface, where depth is negative. MC default is 0.25 — keeps mountains from getting wildly tall.",
        );
        dirty |= slider(
            ui, &mut cfg.factor, 0.1..=10.0, "factor",
            "Constant multiplier on the depth + jagged term. Higher → sharper surface transition (more cliff-like).",
        );
    });

    ui.collapsing("Base 3D noise", |ui| {
        dirty |= slider(
            ui, &mut cfg.base_3d_period, 4.0..=128.0, "base_3d_period",
            "Wavelength of the 3D base noise in blocks. Smaller → bumpier surface; larger → smoother.",
        );
        dirty |= slider(
            ui, &mut cfg.base_3d_amplitude, 0.0..=4.0, "base_3d_amplitude",
            "Strength of the 3D base noise contribution. 0 = pure heightmap; higher = more 3D variation, overhangs.",
        );
        dirty |= slider(
            ui, &mut cfg.base_3d_y_scale, 0.1..=2.0, "base_3d_y_scale",
            "Vertical stretch of the 3D base noise relative to XZ. 0.5 makes features twice as tall as wide (matches MC).",
        );
    });

    ui.collapsing("Slides", |ui| {
        dirty |= slider(
            ui, &mut cfg.slide_top_blocks, 0..=64, "slide_top_blocks",
            "Within this many blocks of `y_max`, density gets blended toward `slide_top_target`. Use a large negative target to force a sky cap.",
        );
        dirty |= slider(
            ui, &mut cfg.slide_top_target, -1.0..=1.0, "slide_top_target",
            "Target density value at the very top of the world. Negative forces air; positive forces solid.",
        );
        dirty |= slider(
            ui, &mut cfg.slide_bottom_blocks, 0..=64, "slide_bottom_blocks",
            "Within this many blocks of `y_min`, density gets blended toward `slide_bottom_target`. Use a large positive target to force a stone floor.",
        );
        dirty |= slider(
            ui, &mut cfg.slide_bottom_target, -1.0..=1.0, "slide_bottom_target",
            "Target density value at the very bottom of the world. Positive forces solid; negative forces air.",
        );
    });

    ui.collapsing("Offset spline", |ui| {
        let resp = ui.add(crate::widgets::spline::SplineEditor::new(&mut cfg.offset_spline));
        resp.clone().on_hover_text(
            "Per-(continentalness, slope, ridges) offset added to the depth term. Drag knots to tune."
        );
        dirty |= resp.changed();
        if let oxium::worldgen::spline::CubicSpline::Multipoint(knots) = &cfg.offset_spline {
            ui.label(format!("{} knots", knots.len()));
        } else {
            ui.label(format!("{:?}", cfg.offset_spline));
        }
        if add_with_tooltip(
            ui,
            |ui| ui.button("Convert to Multipoint with 4 knots"),
            "Replaces the current spline with a 4-knot multipoint version you can drag. Use this when the spline is currently a Constant.",
        )
        .clicked()
        {
            cfg.offset_spline = oxium::worldgen::spline::CubicSpline::Multipoint(vec![
                oxium::worldgen::spline::Knot { loc: -1.0, val: -0.5, slope: 0.0 },
                oxium::worldgen::spline::Knot { loc: -0.3, val: -0.2, slope: 0.0 },
                oxium::worldgen::spline::Knot { loc: 0.2, val: 0.1, slope: 0.0 },
                oxium::worldgen::spline::Knot { loc: 1.0, val: 0.4, slope: 0.0 },
            ]);
            dirty = true;
        }
    });

    dirty
}

pub fn climate_panel(ui: &mut Ui, cfg: &mut oxium::worldgen::config::ClimateConfig) -> bool {
    let mut dirty = false;
    ui.heading("Climate");
    ui.collapsing("Terrain shape noise", |ui| {
        dirty |= slider(
            ui, &mut cfg.terrain_shape_period, 100.0..=4000.0, "terrain_shape_period",
            "Wavelength of the broad terrain-shape Fbm in blocks. Sets the size of continents/mountain groups.",
        );
        dirty |= slider(
            ui, &mut cfg.terrain_shape_amplitude, 0.1..=4.0, "terrain_shape_amplitude",
            "Strength of the terrain-shape noise contribution. Higher → more dramatic continent-vs-ocean contrast.",
        );
    });
    ui.collapsing("Ridge noise", |ui| {
        dirty |= slider(
            ui,
            &mut cfg.ridges_period,
            50.0..=1000.0,
            "ridges_period",
            "Wavelength of the high-frequency ridge noise that drives jaggedness/peaks.",
        );
        dirty |= slider(
            ui,
            &mut cfg.ridges_amplitude,
            0.1..=4.0,
            "ridges_amplitude",
            "Strength of the ridge noise. Higher → more peaks and valleys.",
        );
    });
    ui.collapsing("Plate roughness bias", |ui| {
        ui.label(egui::RichText::new(
            "Per-plate mountain/flatness bias range. The Voronoi plate gets a random \
             value in this range that nudges its terrain_shape input.",
        ).small().weak());
        let (mut lo, mut hi) = cfg.plate_roughness_bias_range;
        dirty |= slider(
            ui, &mut lo, -1.0..=0.0, "min",
            "Lower bound of per-plate roughness bias. Plates rolled to this end produce flatter terrain.",
        );
        dirty |= slider(
            ui, &mut hi, 0.0..=1.0, "max",
            "Upper bound of per-plate roughness bias. Plates rolled to this end produce mountainous terrain.",
        );
        cfg.plate_roughness_bias_range = (lo, hi);
    });
    ui.collapsing("Offset spline (nested)", |ui| {
        dirty |= nested_spline_panel(
            ui,
            &mut cfg.offset_spline,
            "Offset spline. Nested over (continentalness, terrain_shape, ridges_pv); \
             output adds to the depth term so positive values push surface up, negative down.",
        );
    });
    ui.collapsing("Factor spline (nested)", |ui| {
        dirty |= nested_spline_panel(
            ui,
            &mut cfg.factor_spline,
            "Factor spline. Nested over (continentalness, terrain_shape, ridges_pv); \
             output multiplies the depth term. Higher → sharper surface transition.",
        );
    });
    ui.collapsing("Jaggedness spline (nested)", |ui| {
        dirty |= nested_spline_panel(
            ui,
            &mut cfg.jaggedness_spline,
            "Jaggedness spline. Nested over (continentalness, terrain_shape, ridges_pv); \
             output is added to depth via the ridges term — controls peak amplitude.",
        );
    });
    dirty
}

/// Editor for a NestedSpline. Delegates to the recursive widget in
/// `widgets::nested_spline`, which handles the three-level tree
/// (continentalness → terrain_shape → ridges_pv) with a curve preview
/// at each Multipoint level.
fn nested_spline_panel(
    ui: &mut Ui,
    spline: &mut oxium::worldgen::config::NestedSpline,
    tooltip: &str,
) -> bool {
    ui.label(egui::RichText::new(tooltip).small().weak());
    crate::widgets::nested_spline::show(ui, spline, 0)
}

pub fn caves_panel(ui: &mut Ui, cfg: &mut WorldgenConfig) -> bool {
    let mut dirty = false;
    let cave = &mut cfg.cave;
    ui.heading("Caves");
    ui.label(
        egui::RichText::new(
            "Noise channels (first_octave + amplitudes) are baked into the Generator \
             at startup — changing them needs a restart. Everything else here re-reads live.",
        )
        .small()
        .weak(),
    );

    ui.collapsing("Cheese (blobby pockets)", |ui| {
        dirty |= slider(
            ui, &mut cave.cheese_xz_scale, 0.1..=4.0, "cheese_xz_scale",
            "XZ stretch of the cheese FBM sample point. Larger → wider, smoother pockets; smaller → tighter, more chaotic.",
        );
        dirty |= slider(
            ui, &mut cave.cheese_offset, -1.0..=1.0, "cheese_offset",
            "Constant added to cheese noise. Positive biases toward solid (fewer caves); negative carves more aggressively.",
        );
        ui.label(egui::RichText::new("Surface suppression (raw_density-gated):").small().weak());
        dirty |= slider(
            ui, &mut cave.cheese_suppression_offset, -2.0..=2.0, "supp_offset",
            "Suppression base term. Higher = stronger anti-cheese force at the surface.",
        );
        dirty |= slider(
            ui, &mut cave.cheese_suppression_slope, -2.0..=2.0, "supp_slope",
            "How quickly suppression decays as raw_density rises. Negative is normal (deeper = less suppression).",
        );
        dirty |= slider(
            ui, &mut cave.cheese_suppression_min, -1.0..=1.0, "supp_min",
            "Clamp minimum for the suppression term.",
        );
        dirty |= slider(
            ui, &mut cave.cheese_suppression_max, -1.0..=2.0, "supp_max",
            "Clamp maximum for the suppression term.",
        );
        ui.label(egui::RichText::new(format!(
            "cheese channel: first_octave={} amplitudes={:?} (read-only)",
            cave.cheese.first_octave, cave.cheese.amplitudes,
        )).small().weak());
    });

    ui.collapsing("Pillars (refill stone inside caves)", |ui| {
        dirty |= slider(
            ui,
            &mut cave.pillar_xz_scale,
            0.1..=4.0,
            "pillar_xz_scale",
            "XZ stretch of the pillar noise. Larger → thicker, sparser pillars.",
        );
        dirty |= slider(
            ui,
            &mut cave.pillar_y_scale,
            0.1..=4.0,
            "pillar_y_scale",
            "Y stretch of the pillar noise. Smaller → straighter columns; larger → wobbly.",
        );
        dirty |= slider(
            ui,
            &mut cave.pillar_cutoff,
            0.0..=1.0,
            "pillar_cutoff",
            "Threshold the pillar noise must exceed to add material. Lower = more pillars.",
        );
        dirty |= slider(
            ui,
            &mut cave.pillar_intensity,
            0.0..=8.0,
            "pillar_intensity",
            "Strength of the pillar add-back. Higher = more solid refill inside caves.",
        );
    });

    ui.collapsing("Density threshold", |ui| {
        dirty |= slider(
            ui, &mut cave.underground_density_threshold, -1.0..=1.0, "underground_density_threshold",
            "Raw-density floor below which the noise carvers (cheese) go silent. Keeps near-surface voxels safe from accidental carving.",
        );
    });

    dirty
}

pub fn biomes_panel(ui: &mut Ui, cfg: &mut oxium::worldgen::config::BiomesConfig) -> bool {
    let mut dirty = false;
    ui.heading("Biomes");
    ui.collapsing("Weirdness noise", |ui| {
        dirty |= slider(
            ui, &mut cfg.weirdness_period, 50.0..=2000.0, "weirdness_period",
            "Wavelength of the weirdness noise. Adds variant biomes (ice spikes, sunflower plains analogues) within the same temperature/humidity region.",
        );
        dirty |= slider(
            ui, &mut cfg.weirdness_amplitude, 0.0..=2.0, "weirdness_amplitude",
            "Strength of weirdness. 0 = no variant biomes; higher = more rare biome variants.",
        );
    });
    dirty
}

pub fn surface_panel(ui: &mut Ui, cfg: &mut WorldgenConfig) -> bool {
    use oxium::voxel::block::Block;
    use oxium::worldgen::surface::{ConditionSource, RuleSource};

    ui.heading("Surface");
    ui.label("Stamp-and-decorate rules applied after density carving. Block leaves and numeric thresholds are editable inline; structural conditions show as labels.");

    let mut id_counter: u32 = 0;
    let mut changed = false;
    egui::ScrollArea::vertical()
        .id_salt("surface_rule_tree")
        .show(ui, |ui| {
            changed |= render_rule(ui, &mut cfg.surface, 0, &mut id_counter);
        });
    return changed;

    /// Render one node of the rule tree. Returns true if any
    /// editable leaf changed this frame.
    fn render_rule(ui: &mut Ui, rule: &mut RuleSource, depth: usize, id_counter: &mut u32) -> bool {
        let mut changed = false;
        match rule {
            RuleSource::Block(b) => {
                ui.horizontal(|ui| {
                    indent_label(ui, depth, "→");
                    *id_counter += 1;
                    let prev = *b;
                    egui::ComboBox::from_id_salt(format!("surf_blk_{id_counter}"))
                        .selected_text(format!("{b:?}"))
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            for opt in [
                                Block::Air,
                                Block::Stone,
                                Block::Dirt,
                                Block::Grass,
                                Block::Sand,
                                Block::Snow,
                                Block::Water,
                                Block::Lava,
                                Block::Wood,
                                Block::Leaves,
                            ] {
                                ui.selectable_value(b, opt, format!("{opt:?}"));
                            }
                        });
                    if *b != prev {
                        changed = true;
                    }
                });
            }
            RuleSource::Sequence(rules) => {
                indent_label(ui, depth, "Sequence:");
                for r in rules {
                    changed |= render_rule(ui, r, depth + 1, id_counter);
                }
            }
            RuleSource::If { condition, then } => {
                ui.horizontal(|ui| {
                    indent_label(ui, depth, "if");
                    changed |= render_condition(ui, condition, id_counter);
                    ui.monospace(":");
                });
                changed |= render_rule(ui, then, depth + 1, id_counter);
            }
        }
        changed
    }

    fn render_condition(ui: &mut Ui, cond: &mut ConditionSource, id_counter: &mut u32) -> bool {
        // Numeric variants get inline DragValue widgets so common
        // tuning (snow line, beach band, surface band thickness) is
        // a single drag away. Compound and pure-categorical
        // conditions render as monospace labels — their structure is
        // out of scope for inline editing.
        let mut changed = false;
        match cond {
            ConditionSource::YAbove(y) => {
                ui.monospace("Y ≥");
                *id_counter += 1;
                changed |= ui
                    .add(egui::DragValue::new(y).speed(1.0).range(-128..=320))
                    .changed();
            }
            ConditionSource::YBelow(y) => {
                ui.monospace("Y ≤");
                *id_counter += 1;
                changed |= ui
                    .add(egui::DragValue::new(y).speed(1.0).range(-128..=320))
                    .changed();
            }
            ConditionSource::UnderFloor(n) => {
                ui.monospace("depth ≤");
                *id_counter += 1;
                changed |= ui
                    .add(egui::DragValue::new(n).speed(1.0).range(0..=32))
                    .changed();
            }
            ConditionSource::WithinSurfaceBand(w) => {
                ui.monospace("|y − h_target| ≤");
                *id_counter += 1;
                changed |= ui
                    .add(egui::DragValue::new(w).speed(1.0).range(0..=64))
                    .changed();
            }
            ConditionSource::AbovePreliminarySurface(o) => {
                ui.monospace("y ≥ h_target +");
                *id_counter += 1;
                changed |= ui
                    .add(egui::DragValue::new(o).speed(1.0).range(-32..=32))
                    .changed();
            }
            ConditionSource::BeachBand {
                below_sea,
                above_sea,
            } => {
                ui.monospace("beach band [−");
                changed |= ui
                    .add(egui::DragValue::new(below_sea).speed(1.0).range(0..=16))
                    .changed();
                ui.monospace(", +");
                changed |= ui
                    .add(egui::DragValue::new(above_sea).speed(1.0).range(0..=16))
                    .changed();
                ui.monospace("]");
            }
            ConditionSource::SandTransitionRoll {
                temp_min,
                probability,
            } => {
                ui.monospace("sand roll (temp ≥");
                changed |= ui
                    .add(egui::DragValue::new(temp_min).speed(0.01).range(-1.0..=1.0))
                    .changed();
                ui.monospace(", p =");
                changed |= ui
                    .add(
                        egui::DragValue::new(probability)
                            .speed(0.01)
                            .range(0.0..=1.0),
                    )
                    .changed();
                ui.monospace(")");
            }
            ConditionSource::Always(v) => {
                ui.monospace("always");
                changed |= ui.checkbox(v, "").changed();
            }
            ConditionSource::IsCliff => {
                ui.monospace("is cliff");
            }
            ConditionSource::IsCold => {
                ui.monospace("is cold biome");
            }
            ConditionSource::OnFloor => {
                ui.monospace("on floor");
            }
            ConditionSource::Biome(biomes) => {
                ui.monospace(format!("biome ∈ {biomes:?}"));
            }
            ConditionSource::Not(inner) => {
                ui.monospace("not (");
                changed |= render_condition(ui, inner, id_counter);
                ui.monospace(")");
            }
            ConditionSource::All(parts) => {
                ui.monospace("all [");
                for (i, c) in parts.iter_mut().enumerate() {
                    if i > 0 {
                        ui.monospace(",");
                    }
                    changed |= render_condition(ui, c, id_counter);
                }
                ui.monospace("]");
            }
            ConditionSource::Any(parts) => {
                ui.monospace("any [");
                for (i, c) in parts.iter_mut().enumerate() {
                    if i > 0 {
                        ui.monospace(",");
                    }
                    changed |= render_condition(ui, c, id_counter);
                }
                ui.monospace("]");
            }
            ConditionSource::BelowWaterSurface { offset } => {
                ui.monospace("below water −");
                *id_counter += 1;
                changed |= ui
                    .add(egui::DragValue::new(offset).speed(1.0).range(0..=32))
                    .changed();
            }
        }
        changed
    }

    fn indent_label(ui: &mut Ui, depth: usize, leader: &str) {
        // Two spaces per depth level; egui's monospace gives stable
        // alignment column-to-column even with mixed glyph widths.
        ui.monospace(format!("{}{leader}", "  ".repeat(depth)));
    }
}

/// Inline density graph: plots the density curve across the Y range.
pub fn graph_panel(ui: &mut Ui, cfg: &oxium::worldgen::config::DensityConfig) {
    ui.heading("Density graph");
    let y_min = cfg.y_min as f32;
    let y_max = cfg.y_max as f32;
    let height = (y_max - y_min).max(1.0);
    ui.label(format!(
        "y [{:.0}, {:.0}]  gradient ±{:.2}  factor {:.2}",
        y_min, y_max, cfg.y_gradient_amplitude, cfg.factor
    ));
    let steps = 8usize;
    let mut line = String::new();
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let y = y_min + t * height;
        let grad = cfg.y_gradient_amplitude * (1.0 - 2.0 * t);
        let chars = ((grad + cfg.y_gradient_amplitude) / (2.0 * cfg.y_gradient_amplitude) * 10.0)
            .round()
            .clamp(0.0, 10.0) as usize;
        line.push_str(&format!("  y{:.0}: {}\n", y, "#".repeat(chars)));
    }
    ui.monospace(&line);
}

// Preset library lives in `crate::layout::preset_library_section` /
// `crate::preset` — it needs AppState access for its UI state which
// this thin per-panel function can't provide.
