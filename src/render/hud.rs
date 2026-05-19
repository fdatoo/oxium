//! HUD vertex builder + per-frame batch.
//!
//! The HUD is a stack of textured 2D quads drawn over the world in
//! screen space: debug text in the top-left, a hotbar centred at the
//! bottom. Two textures are involved — the [`crate::render::font`]
//! atlas for glyphs and the [`crate::render::atlas`] block atlas for
//! hotbar slot icons — so the HUD frame is encoded as **two batches**,
//! one per texture, that the renderer draws back-to-back with a swap
//! of the active bind group.
//!
//! Coordinates are *pixels* with the origin at the top-left of the
//! framebuffer. The vertex shader projects pixels into NDC using the
//! framebuffer size passed in as a uniform — keeping the CPU code in
//! pixel space makes layout obvious ("hotbar slot 3 at x = 200") and
//! lets the same builder serve any window size.

use bytemuck::{Pod, Zeroable};

use crate::render::atlas::{ATLAS_PX, TILE_PX};
use crate::render::font::{ATLAS_H as FONT_ATLAS_H, ATLAS_W as FONT_ATLAS_W, CELL_W as FONT_CELL_W, GLYPH_H as FONT_GLYPH_H, GLYPH_W as FONT_GLYPH_W};

/// One HUD vertex: 16 bytes, two `vec4<f32>` slots.
///
/// | Offset | Size | Field     | Notes                              |
/// |-------:|-----:|-----------|------------------------------------|
/// |     0  |   8  | `pos_px`  | screen-space position, pixels      |
/// |     8  |   4  | `uv`      | atlas UV in [0, 1]; (-1,-1) ⇒ flat |
/// |    12  |   4  | `color`   | RGBA tint (Unorm8x4)               |
///
/// `uv = (-1, -1)` is the "no texture, fill flat" sentinel — used for
/// the solid coloured backgrounds and selection borders so the HUD
/// can mix textured and untextured quads in a single batch.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct HudVertex {
    pub pos_px: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
}

/// Sentinel UV for "this quad is a solid colour, skip the texture
/// sample". The shader checks `uv.x < 0` to detect it.
pub const UV_FLAT: [f32; 2] = [-1.0, -1.0];

/// One HUD draw batch: a vertex+index span that all sample the same
/// texture. The renderer issues one `set_bind_group(textures)` +
/// `draw` per [`HudBatch`].
#[derive(Default)]
pub struct HudBatch {
    pub vertices: Vec<HudVertex>,
    pub indices: Vec<u32>,
}

impl HudBatch {
    /// Append a screen-space quad covering pixel rect `[x..x+w, y..y+h]`
    /// with the given atlas UV rect and per-vertex colour tint. The
    /// rect order is top-left, top-right, bottom-right, bottom-left so
    /// CCW from the camera (no back-face cull is applied for the HUD).
    pub fn push_quad(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        color: [u8; 4],
    ) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            HudVertex { pos_px: [x, y], uv: [uv_min[0], uv_min[1]], color },
            HudVertex { pos_px: [x + w, y], uv: [uv_max[0], uv_min[1]], color },
            HudVertex { pos_px: [x + w, y + h], uv: [uv_max[0], uv_max[1]], color },
            HudVertex { pos_px: [x, y + h], uv: [uv_min[0], uv_max[1]], color },
        ]);
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// Append a solid (non-textured) coloured rectangle. Convenience
    /// wrapper around `push_quad` with the [`UV_FLAT`] sentinel.
    pub fn push_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [u8; 4]) {
        self.push_quad(x, y, w, h, UV_FLAT, UV_FLAT, color);
    }
}

/// HUD content for one frame: one batch per source texture.
///
/// `text` uses the font atlas (R8 → tinted by vertex colour); `icons`
/// uses the block atlas (RGBA, already coloured). Both batches share
/// the same vertex layout and pipeline — only the bind group differs.
#[derive(Default)]
pub struct HudFrame {
    pub text: HudBatch,
    pub icons: HudBatch,
}

impl HudFrame {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a string of [`crate::render::font`] glyphs anchored at
    /// pixel `(x, y)`, scaled `scale`× the source 5×7 cell, and tinted
    /// by `color`. Returns the X coordinate past the last glyph so
    /// callers can chain runs.
    pub fn push_text(&mut self, mut x: f32, y: f32, text: &str, scale: f32, color: [u8; 4]) -> f32 {
        let cell_w_px = FONT_CELL_W as f32 * scale;
        let glyph_w_px = FONT_GLYPH_W as f32 * scale;
        let glyph_h_px = FONT_GLYPH_H as f32 * scale;
        let atlas_w = FONT_ATLAS_W as f32;
        let atlas_h = FONT_ATLAS_H as f32;
        for ch in text.chars() {
            let slot = crate::render::font::slot_for(ch);
            let u0 = (slot * FONT_CELL_W) as f32 / atlas_w;
            let u1 = u0 + FONT_GLYPH_W as f32 / atlas_w;
            // The font atlas is one cell tall, so V always spans the
            // glyph height (= cell height in the v1 layout).
            let v0 = 0.0;
            let v1 = FONT_GLYPH_H as f32 / atlas_h;
            self.text.push_quad(
                x,
                y,
                glyph_w_px,
                glyph_h_px,
                [u0, v0],
                [u1, v1],
                color,
            );
            x += cell_w_px;
        }
        x
    }

    /// Append a hotbar icon quad: the top-face tile of `tile_index`
    /// from the block atlas, drawn into the pixel rect `(x, y, size)`.
    /// `tint` multiplies the texel so the same grayscale grass top
    /// tile reads green on the hotbar (matching the biome render).
    pub fn push_block_icon(&mut self, x: f32, y: f32, size: f32, tile_index: u8, tint: [u8; 4]) {
        let tile_uv = TILE_PX as f32 / ATLAS_PX as f32;
        let col = (tile_index % 4) as f32;
        let row = (tile_index / 4) as f32;
        let u0 = col * tile_uv;
        let v0 = row * tile_uv;
        let u1 = u0 + tile_uv;
        let v1 = v0 + tile_uv;
        self.icons.push_quad(x, y, size, size, [u0, v0], [u1, v1], tint);
    }
}

use crate::app::PerfSnapshot;
use crate::voxel::block::{Block, BlockRegistry};

/// Blocks shown on the hotbar, in slot order. Slot 9 (index 8) is
/// intentionally `None` to leave a familiar Minecraft-style trailing
/// empty slot — we have 9 visible cells but only 8 placeable blocks.
pub const HOTBAR_BLOCKS: [Option<Block>; 9] = [
    Some(Block::Stone),
    Some(Block::Dirt),
    Some(Block::Grass),
    Some(Block::Sand),
    Some(Block::Water),
    Some(Block::Wood),
    Some(Block::Leaves),
    Some(Block::Torch),
    None,
];

/// Build one frame of HUD content: the upper-left debug overlay
/// (FPS + XYZ) and the bottom-centre hotbar. `screen_px` is the
/// current framebuffer size so the hotbar can be centred without the
/// caller needing to know the layout details.
///
/// Pulling all the HUD layout into a single function keeps the call
/// site in the ECS render system tiny — it just hands us the numbers.
pub fn build_hud(
    screen_px: (u32, u32),
    fps: f32,
    eye: glam::Vec3,
    selected_slot: usize,
    registry: &BlockRegistry,
    perf: &PerfSnapshot,
) -> HudFrame {
    let mut frame = HudFrame::new();
    let (sw, sh) = (screen_px.0 as f32, screen_px.1 as f32);

    // ── Upper-left debug overlay ────────────────────────────────────
    // Scale 3 keeps the 5×7 glyphs at 15×21 px — readable at 1080p
    // without dominating the frame.
    let text_scale = 3.0;
    let pad = 12.0;
    let inner_pad = 6.0;
    let line_h = 28.0;
    let white = [255, 255, 255, 255];
    let cyan = [120, 220, 255, 255];

    let fps_str = format!("FPS: {:.0}", fps);
    let xyz_str = format!("XYZ: {:.1}  {:.1}  {:.1}", eye.x, eye.y, eye.z);
    // Perf line: live counters for the debug HUD. "LQ" = light
    // queue depth (chunks waiting for the relight pump); steady
    // non-zero means cascade isn't terminating. "CH" = chunk meshes
    // currently held by the renderer.
    let perf_str = format!(
        "LQ: {}  CH: {}  WMS: {:.1}",
        perf.light_queue, perf.chunks_rendered, perf.work_ms,
    );
    // A semi-transparent dark backdrop behind the three text lines so
    // the cyan/white glyphs stay readable against bright skies and
    // grass without us having to author per-character outlines. The
    // panel width tracks the widest string; the icons batch draws
    // first in the pass, so this lands beneath the text.
    let glyph_w = crate::render::font::CELL_W as f32 * text_scale;
    let widest = fps_str
        .chars()
        .count()
        .max(xyz_str.chars().count())
        .max(perf_str.chars().count());
    let panel_w = widest as f32 * glyph_w + inner_pad * 2.0;
    let panel_h = line_h * 3.0 + inner_pad * 2.0;
    frame.icons.push_rect(
        pad - inner_pad,
        pad - inner_pad,
        panel_w,
        panel_h,
        [0, 0, 0, 0xA0],
    );

    let yellow = [255, 220, 120, 255];
    frame.push_text(pad, pad, &fps_str, text_scale, white);
    frame.push_text(pad, pad + line_h, &xyz_str, text_scale, cyan);
    frame.push_text(pad, pad + line_h * 2.0, &perf_str, text_scale, yellow);

    // ── Bottom-centre hotbar ────────────────────────────────────────
    // 9 cells, 48 px each, 4 px gap. Centred horizontally; 16 px
    // above the bottom edge.
    let cell = 48.0;
    let gap = 4.0;
    let cells = HOTBAR_BLOCKS.len() as f32;
    let bar_w = cell * cells + gap * (cells - 1.0);
    let bar_x = (sw - bar_w) * 0.5;
    let bar_y = sh - cell - 16.0;
    let bg_alpha = 0x80;
    let border = 4.0;

    // Outer translucent panel — one big rectangle behind every slot
    // so the hotbar reads as a single UI element.
    frame.icons.push_rect(
        bar_x - border,
        bar_y - border,
        bar_w + border * 2.0,
        cell + border * 2.0,
        [0, 0, 0, bg_alpha],
    );

    for (i, slot) in HOTBAR_BLOCKS.iter().enumerate() {
        let x = bar_x + (cell + gap) * i as f32;
        // Selection highlight: bright outline behind the icon.
        if i == selected_slot {
            let pad = 3.0;
            frame.icons.push_rect(
                x - pad,
                bar_y - pad,
                cell + pad * 2.0,
                cell + pad * 2.0,
                [255, 255, 255, 255],
            );
            frame.icons.push_rect(x, bar_y, cell, cell, [40, 40, 40, 200]);
        }
        if let Some(block) = slot {
            let info = registry.info(*block);
            // Icon = the block's top tile (grass top, stone, sand…).
            if let Some(tile) = info.tile_top.or(info.tile_side) {
                // Tint comes from `top_color` if present (grass), else
                // `color` (everything else). Same logic the world
                // mesher uses, so the hotbar icon matches in-world.
                let tint = info.top_color.unwrap_or(info.color);
                let tint_u8 = [
                    (tint[0] * 255.0) as u8,
                    (tint[1] * 255.0) as u8,
                    (tint[2] * 255.0) as u8,
                    255,
                ];
                frame.push_block_icon(x, bar_y, cell, tile.index(), tint_u8);
            }
        }
    }

    frame
}
