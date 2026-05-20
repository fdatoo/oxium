//! Cross-section panel: a configurable cut plane through the world,
//! sampling a per-voxel scalar (density today; cave SDF / aquifer
//! follow). 64×64 px so each render is ~4k generator samples — slow
//! enough to debounce on edits but fast enough to feel interactive.

use egui::{ColorImage, TextureHandle, TextureOptions, Ui};
use oxium::worldgen::Generator;

const PLOT_PX: usize = 128;
const PLOT_BLOCKS_PER_PIXEL_INITIAL: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// Horizontal slice at constant world Y. Pixel-x = world X, pixel-y = world Z.
    Xz,
    /// Vertical slice at constant world Z. Pixel-x = world X, pixel-y = world Y (up = up).
    Xy,
    /// Vertical slice at constant world X. Pixel-x = world Z, pixel-y = world Y.
    Yz,
}

impl Orientation {
    fn label(self) -> &'static str {
        match self {
            Orientation::Xz => "XZ (horizontal)",
            Orientation::Xy => "XY (vertical, fixed Z)",
            Orientation::Yz => "YZ (vertical, fixed X)",
        }
    }

    fn slice_axis_label(self) -> &'static str {
        match self {
            Orientation::Xz => "slice Y",
            Orientation::Xy => "slice Z",
            Orientation::Yz => "slice X",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Signed density value, divergent gradient through 0. Solid > 0, air <= 0.
    Density,
    /// Resolved block at the voxel, coloured the same way the mesher
    /// would colour it. Reads through the existing density+surface
    /// pipeline so the slice matches what fill_chunk would write.
    Block,
}

impl Layer {
    fn label(self) -> &'static str {
        match self {
            Layer::Density => "Density (signed)",
            Layer::Block => "Block",
        }
    }
}

pub struct CrossSection {
    pub orientation: Orientation,
    pub layer: Layer,
    /// World value of the orientation's constant axis (Y for XZ, Z for XY, X for YZ).
    pub slice_axis: i32,
    /// World coords of the slice's centre, along the two view axes.
    pub center_a: f32,
    pub center_b: f32,
    pub blocks_per_pixel: f32,
    /// The pinned column seen on the most recent `show()` call.
    /// When the pin changes (user clicks a new column), we auto-snap
    /// the cross-section's centre + slice to that column so the
    /// freshly-pinned target lands in the middle of the slice.
    last_seen_pin: Option<(i32, i32, i32)>,
    texture: Option<TextureHandle>,
    last_key: Option<RenderKey>,
}

#[derive(PartialEq)]
struct RenderKey {
    orientation: Orientation,
    layer: Layer,
    slice_axis: i32,
    center_a: i32,
    center_b: i32,
    blocks_per_pixel: i32,
    config_revision: u64,
}

impl CrossSection {
    pub fn new() -> Self {
        Self {
            orientation: Orientation::Xy,
            layer: Layer::Density,
            slice_axis: 0,
            center_a: 0.0,
            center_b: 70.0,
            blocks_per_pixel: PLOT_BLOCKS_PER_PIXEL_INITIAL,
            last_seen_pin: None,
            texture: None,
            last_key: None,
        }
    }

    /// Snap centre + slice axis to a pinned column. Independent of
    /// the auto-snap on-pin-change so callers can also expose a
    /// manual "Center on pinned" button.
    pub fn focus_on(&mut self, wx: i32, h_target: i32, wz: i32) {
        match self.orientation {
            Orientation::Xz => {
                // Horizontal slice: centre on (wx, wz), slice at the surface Y.
                self.center_a = wx as f32;
                self.center_b = wz as f32;
                self.slice_axis = h_target;
            }
            Orientation::Xy => {
                // Vertical slice fixed-Z: centre on (wx, h_target), slice plane = wz.
                self.center_a = wx as f32;
                self.center_b = h_target as f32;
                self.slice_axis = wz;
            }
            Orientation::Yz => {
                // Vertical slice fixed-X: centre on (wz, h_target), slice plane = wx.
                self.center_a = wz as f32;
                self.center_b = h_target as f32;
                self.slice_axis = wx;
            }
        }
    }

    /// Map a pixel in the cross-section image to a world `(wx, wy, wz)`.
    fn pixel_to_world(&self, px: f32, py: f32) -> (i32, i32, i32) {
        let half = PLOT_PX as f32 * 0.5;
        let a = self.center_a + (px - half) * self.blocks_per_pixel;
        // py grows downward in image space; flip for "up = +" feel.
        let b = self.center_b - (py - half) * self.blocks_per_pixel;
        let (wx, wy, wz) = match self.orientation {
            Orientation::Xz => (a, self.slice_axis as f32, b), // XZ slice: b is world Z, slice = Y
            Orientation::Xy => (a, b, self.slice_axis as f32), // XY slice: a = X, b = Y, slice = Z
            Orientation::Yz => (self.slice_axis as f32, b, a), // YZ slice: a = Z, b = Y, slice = X
        };
        (wx.round() as i32, wy.round() as i32, wz.round() as i32)
    }

    fn regenerate(&mut self, generator: &Generator, ctx: &egui::Context, revision: u64) {
        let key = RenderKey {
            orientation: self.orientation,
            layer: self.layer,
            slice_axis: self.slice_axis,
            center_a: self.center_a as i32,
            center_b: self.center_b as i32,
            blocks_per_pixel: (self.blocks_per_pixel * 100.0) as i32,
            config_revision: revision,
        };
        if self.last_key.as_ref() == Some(&key) && self.texture.is_some() {
            return;
        }
        let mut pixels = vec![egui::Color32::TRANSPARENT; PLOT_PX * PLOT_PX];
        for py in 0..PLOT_PX {
            for px in 0..PLOT_PX {
                let (wx, wy, wz) = self.pixel_to_world(px as f32, py as f32);
                let rgba = sample_pixel(generator, self.layer, wx, wy, wz);
                pixels[py * PLOT_PX + px] = egui::Color32::from_rgba_premultiplied(
                    rgba[0], rgba[1], rgba[2], rgba[3],
                );
            }
        }
        let img = ColorImage {
            size: [PLOT_PX, PLOT_PX],
            pixels,
        };
        let tex = ctx.load_texture("viz_crosssection", img, TextureOptions::NEAREST);
        self.texture = Some(tex);
        self.last_key = Some(key);
    }

    /// Render UI; returns true if any control mutated. `pin` is the
    /// currently-pinned column (`(wx, h_target, wz)`), if any —
    /// when it differs from what we saw last frame the slice auto-
    /// snaps to it. Orientation changes also re-snap (since each
    /// orientation needs the pin mapped to different axes).
    pub fn show(
        &mut self,
        ui: &mut Ui,
        generator: &Generator,
        revision: u64,
        pin: Option<(i32, i32, i32)>,
    ) -> bool {
        // Auto-snap on pin change.
        if pin != self.last_seen_pin {
            if let Some((wx, h, wz)) = pin {
                self.focus_on(wx, h, wz);
            }
            self.last_seen_pin = pin;
        }

        let prev_orient = self.orientation;
        let mut dirty = false;
        ui.horizontal(|ui| {
            ui.label("Orientation:");
            egui::ComboBox::from_id_source("cross_orient_combo")
                .selected_text(self.orientation.label())
                .show_ui(ui, |ui| {
                    for o in [Orientation::Xz, Orientation::Xy, Orientation::Yz] {
                        if ui.selectable_label(self.orientation == o, o.label()).clicked() {
                            self.orientation = o;
                            dirty = true;
                        }
                    }
                });
            ui.separator();
            ui.label("Layer:");
            egui::ComboBox::from_id_source("cross_layer_combo")
                .selected_text(self.layer.label())
                .show_ui(ui, |ui| {
                    for l in [Layer::Density, Layer::Block] {
                        if ui.selectable_label(self.layer == l, l.label()).clicked() {
                            self.layer = l;
                            dirty = true;
                        }
                    }
                });
        });

        // If the orientation changed and we have a pin, re-snap to
        // it under the new axes (otherwise sliders look stale).
        if self.orientation != prev_orient {
            if let Some((wx, h, wz)) = pin {
                self.focus_on(wx, h, wz);
            }
        }

        // "Center on pinned" button — explicit re-snap for users who
        // manually drifted away with the sliders and want to reset.
        if let Some((wx, h, wz)) = pin {
            ui.horizontal(|ui| {
                if ui
                    .small_button(format!("⌖ Centre on pinned ({wx}, {h}, {wz})"))
                    .on_hover_text("Snap the slice's centre + slice axis back to the pinned column.")
                    .clicked()
                {
                    self.focus_on(wx, h, wz);
                    dirty = true;
                }
            });
        }
        dirty |= ui
            .add(egui::Slider::new(&mut self.slice_axis, -128..=256).text(self.orientation.slice_axis_label()))
            .on_hover_text("World coordinate of the cut plane along the orientation's fixed axis.")
            .changed();
        dirty |= ui
            .add(egui::Slider::new(&mut self.center_a, -512.0..=512.0).text("center a"))
            .on_hover_text("World coord at the centre of the slice along the image's horizontal axis.")
            .changed();
        dirty |= ui
            .add(egui::Slider::new(&mut self.center_b, -64.0..=256.0).text("center b"))
            .on_hover_text("World coord at the centre of the slice along the image's vertical axis.")
            .changed();
        dirty |= ui
            .add(egui::Slider::new(&mut self.blocks_per_pixel, 0.25..=4.0).text("blocks/px"))
            .on_hover_text("Zoom. Smaller = more detail per pixel; larger = wider area.")
            .changed();

        self.regenerate(generator, ui.ctx(), revision);
        if let Some(tex) = self.texture.as_ref() {
            let size = egui::vec2(PLOT_PX as f32 * 2.0, PLOT_PX as f32 * 2.0);
            ui.image((tex.id(), size));
        }
        ui.label(
            egui::RichText::new(format!(
                "slice {} ({:.0}, {:.0}) · {:.2} blocks/px · {}×{} px",
                self.orientation.slice_axis_label(),
                self.center_a, self.center_b, self.blocks_per_pixel, PLOT_PX, PLOT_PX,
            ))
            .small()
            .weak(),
        );
        dirty
    }
}

fn sample_pixel(generator: &Generator, layer: Layer, wx: i32, wy: i32, wz: i32) -> [u8; 4] {
    let bd = generator.evaluate_density_breakdown(wx, wy, wz);
    match layer {
        Layer::Density => density_color(bd.final_density),
        Layer::Block => block_color(bd.block),
    }
}

fn density_color(d: f32) -> [u8; 4] {
    // Divergent gradient through 0: blue (air) → black (zero) → red (deep solid).
    // Clamp magnitude into [0, 1] over ±8 — most surface-region densities
    // fall well within that window.
    let t = (d / 8.0).clamp(-1.0, 1.0);
    if t < 0.0 {
        let s = -t;
        [(20.0 * (1.0 - s)) as u8, (60.0 * (1.0 - s) + 60.0 * s) as u8, (60.0 * (1.0 - s) + 220.0 * s) as u8, 255]
    } else {
        let s = t;
        [(20.0 * (1.0 - s) + 220.0 * s) as u8, (60.0 * (1.0 - s) + 30.0 * s) as u8, (60.0 * (1.0 - s) + 30.0 * s) as u8, 255]
    }
}

fn block_color(b: oxium::voxel::block::Block) -> [u8; 4] {
    use oxium::voxel::block::Block;
    let rgb = match b {
        Block::Air => [10u8, 12, 24],
        Block::Stone => [140, 140, 140],
        Block::Dirt => [128, 82, 46],
        Block::Grass => [77, 166, 64],
        Block::Sand => [235, 217, 158],
        Block::Snow => [242, 242, 247],
        Block::Water => [60, 100, 200],
        Block::Lava => [255, 115, 20],
        _ => [80, 80, 80],
    };
    [rgb[0], rgb[1], rgb[2], 255]
}
