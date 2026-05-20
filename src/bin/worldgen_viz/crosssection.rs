//! Cross-section panel: a configurable cut plane through the world,
//! sampling a per-voxel scalar (density today; cave SDF / aquifer
//! follow). 64×64 px so each render is ~4k generator samples — slow
//! enough to debounce on edits but fast enough to feel interactive.

use crossbeam_channel::{unbounded, Receiver, Sender};
use egui::{Color32, ColorImage, TextureHandle, TextureOptions, Ui};
use oxium::worldgen::Generator;
use std::sync::Arc;

const PLOT_BLOCKS_PER_PIXEL_INITIAL: f32 = 1.0;
/// Horizontal slice (XZ) is square — both axes are XZ-plane axes,
/// equally interesting.
const SQUARE_PX: usize = 160;
/// Vertical slices (XY / YZ) get a 2:3 portrait so the world's
/// vertical extent (≈256 blocks of useful Y) doesn't get cropped by
/// the square aspect we used in v1. Right panel is much taller than
/// wide on a typical screen — exploit it.
const TALL_W_PX: usize = 128;
const TALL_H_PX: usize = 256;

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

    /// Internal image size for this orientation. Horizontal slices
    /// stay square; vertical slices go portrait so the world's
    /// vertical extent isn't cropped to the square aspect.
    fn dimensions(self) -> (usize, usize) {
        match self {
            Orientation::Xz => (SQUARE_PX, SQUARE_PX),
            Orientation::Xy | Orientation::Yz => (TALL_W_PX, TALL_H_PX),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Final signed density value (post-carver). Solid > 0, air <= 0.
    Density,
    /// Resolved block at the voxel, coloured the same way the mesher
    /// would colour it. Reads through the existing density+surface
    /// pipeline so the slice matches what fill_chunk would write.
    Block,
    /// Combined cave-contribution magnitude (max of cave_sdf + cheese
    /// + spaghetti). Bright = the carvers are removing material here.
    /// Pillar contribution is positive in the opposite direction
    /// and isn't included; see the dedicated `Pillar` layer for that.
    CaveSdf,
    /// Cheese noise contribution only (blobby caves). Bright = strong
    /// cheese carve at this voxel.
    Cheese,
    /// Spaghetti tube contribution only (thin worming tubes). Bright =
    /// strong spaghetti carve.
    Spaghetti,
    /// Pillar density-add-back. Bright = strong pillar add-back
    /// (resists carving inside cave volumes).
    Pillar,
}

impl Layer {
    fn label(self) -> &'static str {
        match self {
            Layer::Density => "Density (signed)",
            Layer::Block => "Block",
            Layer::CaveSdf => "Cave SDF (combined)",
            Layer::Cheese => "Cheese",
            Layer::Spaghetti => "Spaghetti",
            Layer::Pillar => "Pillar",
        }
    }

    pub const ALL: &'static [Layer] = &[
        Layer::Density,
        Layer::Block,
        Layer::CaveSdf,
        Layer::Cheese,
        Layer::Spaghetti,
        Layer::Pillar,
    ];
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
    /// Texture currently being displayed; stays mounted while a new
    /// render is in flight so the UI never blanks out.
    texture: Option<TextureHandle>,
    /// Key the displayed texture was rendered for. None on first frame.
    displayed_key: Option<RenderKey>,
    /// Key of the render in flight, if any. Used to dedup spawns and
    /// to discard stale results when the user changes a control mid-render.
    pending_key: Option<RenderKey>,
    /// Channel back from the rayon sampling task.
    tx: Sender<RenderResult>,
    rx: Receiver<RenderResult>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RenderKey {
    orientation: Orientation,
    layer: Layer,
    slice_axis: i32,
    center_a: i32,
    center_b: i32,
    blocks_per_pixel: i32,
    config_revision: u64,
}

/// One async sampling result: the pixel buffer + the key it was
/// rendered for (so stale results can be dropped on receipt).
struct RenderResult {
    key: RenderKey,
    width: usize,
    height: usize,
    pixels: Vec<Color32>,
}

impl CrossSection {
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        Self {
            orientation: Orientation::Xy,
            layer: Layer::Density,
            slice_axis: 0,
            center_a: 0.0,
            center_b: 70.0,
            blocks_per_pixel: PLOT_BLOCKS_PER_PIXEL_INITIAL,
            last_seen_pin: None,
            texture: None,
            displayed_key: None,
            pending_key: None,
            tx,
            rx,
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
        let (w, h) = self.orientation.dimensions();
        let a = self.center_a + (px - w as f32 * 0.5) * self.blocks_per_pixel;
        // py grows downward in image space; flip for "up = +" feel.
        let b = self.center_b - (py - h as f32 * 0.5) * self.blocks_per_pixel;
        let (wx, wy, wz) = match self.orientation {
            Orientation::Xz => (a, self.slice_axis as f32, b), // XZ slice: b is world Z, slice = Y
            Orientation::Xy => (a, b, self.slice_axis as f32), // XY slice: a = X, b = Y, slice = Z
            Orientation::Yz => (self.slice_axis as f32, b, a), // YZ slice: a = Z, b = Y, slice = X
        };
        (wx.round() as i32, wy.round() as i32, wz.round() as i32)
    }

    fn current_key(&self, revision: u64) -> RenderKey {
        RenderKey {
            orientation: self.orientation,
            layer: self.layer,
            slice_axis: self.slice_axis,
            center_a: self.center_a as i32,
            center_b: self.center_b as i32,
            blocks_per_pixel: (self.blocks_per_pixel * 100.0) as i32,
            config_revision: revision,
        }
    }

    /// Drain any results that landed on the channel. Stale results
    /// (whose key no longer matches the desired key) are discarded.
    fn drain_results(&mut self, ctx: &egui::Context, desired: &RenderKey) {
        while let Ok(result) = self.rx.try_recv() {
            if result.key != *desired {
                // Stale — user has moved on; drop.
                continue;
            }
            let img = ColorImage {
                size: [result.width, result.height],
                pixels: result.pixels,
            };
            self.texture = Some(ctx.load_texture("viz_crosssection", img, TextureOptions::NEAREST));
            self.displayed_key = Some(result.key);
            self.pending_key = None;
        }
    }

    /// Spawn a sampling task on the global rayon pool. Captures a
    /// snapshot of the centre / orientation / layer / etc. so the
    /// worker doesn't need any further state from us.
    fn spawn_render(&mut self, generator: Arc<Generator>, key: RenderKey) {
        self.pending_key = Some(key.clone());
        let (w, h) = self.orientation.dimensions();
        let center_a = self.center_a;
        let center_b = self.center_b;
        let bpp = self.blocks_per_pixel;
        let orientation = self.orientation;
        let layer = self.layer;
        let slice_axis = self.slice_axis;
        let tx = self.tx.clone();
        rayon::spawn(move || {
            let mut pixels = vec![Color32::TRANSPARENT; w * h];
            for py in 0..h {
                for px in 0..w {
                    let a = center_a + (px as f32 - w as f32 * 0.5) * bpp;
                    let b = center_b - (py as f32 - h as f32 * 0.5) * bpp;
                    let (wx, wy, wz) = match orientation {
                        Orientation::Xz => (a, slice_axis as f32, b),
                        Orientation::Xy => (a, b, slice_axis as f32),
                        Orientation::Yz => (slice_axis as f32, b, a),
                    };
                    let rgba = sample_pixel(
                        &generator,
                        layer,
                        wx.round() as i32,
                        wy.round() as i32,
                        wz.round() as i32,
                    );
                    pixels[py * w + px] =
                        Color32::from_rgba_premultiplied(rgba[0], rgba[1], rgba[2], rgba[3]);
                }
            }
            let _ = tx.send(RenderResult { key, width: w, height: h, pixels });
        });
    }

    /// Render UI; returns true if any control mutated. `pin` is the
    /// currently-pinned column (`(wx, h_target, wz)`), if any —
    /// when it differs from what we saw last frame the slice auto-
    /// snaps to it. Orientation changes also re-snap (since each
    /// orientation needs the pin mapped to different axes).
    pub fn show(
        &mut self,
        ui: &mut Ui,
        generator: &Arc<Generator>,
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
                    for &l in Layer::ALL {
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

        // Only two sliders left now that the pin drives the centre:
        // the slice depth (perpendicular to the cut plane) and the
        // zoom. Centre comes from the pin; if the user wants to look
        // somewhere else they re-pin in the 3D view or on the map.
        let (axis_lo, axis_hi) = match self.orientation {
            // Horizontal slice walks Y; cover the world's actual range.
            Orientation::Xz => (-64, 192),
            // Vertical slices walk Z or X — large blocks-of-world range.
            Orientation::Xy | Orientation::Yz => (-512, 512),
        };
        dirty |= ui
            .add(
                egui::Slider::new(&mut self.slice_axis, axis_lo..=axis_hi)
                    .text(self.orientation.slice_axis_label()),
            )
            .on_hover_text(
                "Depth: world coordinate of the cut plane perpendicular to the slice. \
                 Move this to walk the cross-section through the world.",
            )
            .changed();
        dirty |= ui
            .add(egui::Slider::new(&mut self.blocks_per_pixel, 0.25..=4.0).text("blocks/px"))
            .on_hover_text("Zoom. Smaller = more detail per pixel; larger = wider area.")
            .changed();

        // Async render pipeline. Compute the key the *current* UI
        // state wants; drain anything that just finished; if there's
        // nothing in flight that matches the desired key, spawn one
        // on the rayon pool. The old texture stays mounted so the
        // panel never blanks while a render is in progress.
        let desired = self.current_key(revision);
        self.drain_results(ui.ctx(), &desired);
        let needs_render = self.displayed_key.as_ref() != Some(&desired)
            && self.pending_key.as_ref() != Some(&desired);
        if needs_render {
            self.spawn_render(generator.clone(), desired.clone());
            // Request a repaint a few hundred ms out so the channel
            // gets drained even if the user stops poking the UI.
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(60));
        }
        // While a render is in flight, keep poking the runtime so
        // the eventual result drains promptly.
        if self.pending_key.is_some() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(60));
        }
        let (w_px, h_px) = self.orientation.dimensions();
        if let Some(tex) = self.texture.as_ref() {
            // Display at native panel width if it fits, else scale
            // down preserving aspect. 2x is the visual sweet spot —
            // each sampled pixel becomes a 2×2 block on screen,
            // matches the map widget's NEAREST upsampling.
            let panel_w = ui.available_width();
            let scale = (panel_w / w_px as f32).min(2.0);
            let size = egui::vec2(w_px as f32 * scale, h_px as f32 * scale);
            ui.image((tex.id(), size));
        } else {
            // First-render placeholder — empty rect so the panel
            // doesn't reflow as soon as the texture arrives.
            let panel_w = ui.available_width();
            let scale = (panel_w / w_px as f32).min(2.0);
            ui.allocate_exact_size(
                egui::vec2(w_px as f32 * scale, h_px as f32 * scale),
                egui::Sense::hover(),
            );
        }
        if self.pending_key.is_some() {
            ui.label(
                egui::RichText::new("⏳ rendering…")
                    .small()
                    .color(egui::Color32::from_rgb(230, 180, 80)),
            );
        }
        let centre_status = match pin {
            Some(_) => format!(
                "centred on pinned column · ({:.0}, {:.0})",
                self.center_a, self.center_b,
            ),
            None => "no pin — click a column in the 3D view or map to centre".to_string(),
        };
        ui.label(
            egui::RichText::new(format!(
                "{centre_status} · {:.2} blocks/px · {}×{} px",
                self.blocks_per_pixel, w_px, h_px,
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
        // For each carver, normalise into [0, 1] over a sensible
        // magnitude (~2.0 — most contributions sit there). Hot ramp
        // so cave material is easy to spot against solid rock.
        Layer::CaveSdf => {
            let combined = bd.cave_sdf.max(bd.cheese).max(bd.spaghetti);
            hot_color((combined / 2.0).clamp(0.0, 1.0))
        }
        Layer::Cheese => hot_color((bd.cheese / 2.0).clamp(0.0, 1.0)),
        Layer::Spaghetti => hot_color((bd.spaghetti / 2.0).clamp(0.0, 1.0)),
        Layer::Pillar => hot_color((bd.pillar / 2.0).clamp(0.0, 1.0)),
    }
}

/// Black → red → orange → white hot gradient. For carver-strength layers.
fn hot_color(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    if t < 0.001 {
        // Empty (no carver here) — keep the background dark so the
        // overall scene reads as "mostly solid with a few hot spots".
        return [12, 12, 18, 255];
    }
    if t < 0.4 {
        let s = t / 0.4;
        [(s * 200.0) as u8, 0, 0, 255]
    } else if t < 0.75 {
        let s = (t - 0.4) / 0.35;
        [200 + (s * 55.0) as u8, (s * 165.0) as u8, 0, 255]
    } else {
        let s = (t - 0.75) / 0.25;
        [255, 165 + (s * 90.0) as u8, (s * 255.0) as u8, 255]
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
