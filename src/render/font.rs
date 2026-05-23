//! TrueType font renderer for the HUD overlay.
//!
//! Replaces the hand-baked 5×7 bitmap table with **PixelOperatorMono8**, a
//! pixel-art monospace font designed for 8 pt rendering. The font is embedded
//! at compile time via [`include_bytes!`] and rasterised once at startup by
//! [`fontdue`] into the same flat `Rgba8Unorm` atlas the HUD pipeline already
//! consumes. No changes to the GPU pipeline, shader, or vertex builder are
//! required — only this module and the handful of call sites that read the old
//! `const` metrics are updated.
//!
//! ## Why PixelOperatorMono8?
//!
//! The font is distributed under the MIT license (see
//! `assets/fonts/FONT_LICENSE.txt`), has a consistent 8-px advance width for
//! every glyph, and covers the full printable ASCII range — so the HUD can now
//! render any character rather than the curated 96-glyph table we maintained by
//! hand.
//!
//! ## Atlas layout
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  slot 0   │  slot 1   │  slot 2   │  …  │  slot 94  │          │
//! │  U+0020   │  U+0021   │  U+0022   │     │  U+007E   │          │
//! │  (space)  │    '!'    │    '"'    │     │    '~'    │          │
//! └──────────────────────────────────────────────────────────────────┘
//!   ←cell_w→   ←cell_w→
//! ```
//!
//! All 95 printable ASCII glyphs (U+0020 … U+007E) are packed into a single
//! horizontal row, one glyph per cell of width [`cell_w`]. Slot index is
//! `codepoint − 0x20`. [`slot_for`] maps any character to its slot, falling
//! back to the space glyph for code points outside the ASCII printable range.

use std::sync::OnceLock;

use fontdue::{Font, FontSettings};

/// Embedded PixelOperatorMono8 TrueType data. Licensed under MIT —
/// see `assets/fonts/FONT_LICENSE.txt`.
const FONT_DATA: &[u8] = include_bytes!("../../assets/fonts/PixelOperatorMono8.ttf");

/// Rasterisation size in pixels. PixelOperator8 is hinted for 8 pt; rendering
/// at exactly 8 px produces crisply aligned, zero-antialiased pixel edges. The
/// HUD vertex builder scales character quads independently via a float factor,
/// so this only affects the resolution of the atlas itself.
const PX_SIZE: f32 = 8.0;

/// First code point stored in the atlas: U+0020 (space). Slot index = cp − FIRST.
const FIRST: u32 = 0x0020;
/// Last code point stored in the atlas (inclusive): U+007E ('~').
const LAST: u32 = 0x007E;
/// Number of glyph slots — the 95 printable ASCII characters.
const SLOT_COUNT: u32 = LAST - FIRST + 1;

/// All rasterised metrics and atlas data, computed once and cached.
struct FontState {
    /// Flat RGBA8 atlas buffer: `atlas_w × atlas_h × 4` bytes.
    atlas: Vec<u8>,
    /// Horizontal advance width in pixels, uniform across all glyphs because
    /// PixelOperatorMono is a monospace family.
    cell_w: u32,
    /// Full line-box height: ascent + |descent|. Character quads are drawn at
    /// `glyph_h × scale` pixels tall.
    glyph_h: u32,
    /// Atlas pixel width: `SLOT_COUNT × cell_w`.
    atlas_w: u32,
    /// Atlas pixel height: equals `glyph_h`.
    atlas_h: u32,
}

static FONT: OnceLock<FontState> = OnceLock::new();

/// Initialise the [`OnceLock`] and return a reference to the cached state.
/// Called by every public accessor; the first call triggers rasterisation.
fn state() -> &'static FontState {
    FONT.get_or_init(build)
}

/// Build the [`FontState`] by rasterising every printable ASCII glyph with
/// fontdue and compositing the coverage bitmaps into the atlas.
fn build() -> FontState {
    let font = Font::from_bytes(FONT_DATA, FontSettings::default())
        .expect("PixelOperatorMono8.ttf embedded in the binary is valid; this is a build bug");

    // Derive cell dimensions from the font's line metrics at the chosen size.
    let line = font
        .horizontal_line_metrics(PX_SIZE)
        .expect("PixelOperatorMono8.ttf has horizontal line metrics");

    // fontdue reports descent as a negative value (below the baseline).
    let ascent = line.ascent.ceil() as i32;
    let descent = (-line.descent).ceil() as i32;
    let glyph_h = (ascent + descent) as u32;

    // Monospace fonts share a single advance width; measure against 'M'.
    let (ref_m, _) = font.rasterize('M', PX_SIZE);
    let cell_w = ref_m.advance_width.ceil() as u32;

    let atlas_w = SLOT_COUNT * cell_w;
    let atlas_h = glyph_h;
    let mut atlas = vec![0u8; (atlas_w * atlas_h * 4) as usize];

    for slot in 0..SLOT_COUNT {
        let ch = char::from_u32(FIRST + slot).unwrap_or(' ');
        let (m, bitmap) = font.rasterize(ch, PX_SIZE);
        let col0 = (slot * cell_w) as i32;

        // Place the glyph bitmap into the cell using baseline alignment.
        //
        // fontdue's coordinate system has y increasing upward; `ymin` is the
        // signed distance from the baseline to the *bottom* of the bitmap
        // (positive = above the baseline, negative = descender).
        //
        // The atlas uses screen-space y (increasing downward). The baseline
        // sits at y = ascent from the top of the cell, so:
        //
        //   cell_y_for_bitmap_row_0 = ascent − ymin − height
        //
        // This aligns cap-height letters flush with the top of the cell and
        // lets descenders (g, p, q, y …) spill into the descent region.
        let glyph_top = ascent - m.ymin - m.height as i32;

        for row in 0..m.height {
            let cell_y = glyph_top + row as i32;
            if cell_y < 0 || cell_y >= atlas_h as i32 {
                // Glyph extends outside the allocated cell — clip silently.
                continue;
            }
            for col in 0..m.width {
                let coverage = bitmap[row * m.width + col];
                if coverage == 0 {
                    continue;
                }
                // xmin: left bearing (usually 0 or 1 for a pixel font).
                let cell_x = col0 + m.xmin + col as i32;
                if cell_x < 0 || cell_x >= atlas_w as i32 {
                    continue;
                }
                let idx = (cell_y as u32 * atlas_w + cell_x as u32) as usize * 4;
                // Coverage is typically 0 or 255 for a pixel font at its design
                // size. Intermediate values arise at sub-pixel glyph edges and
                // give slightly smoother rendering when the quad is GPU-scaled.
                atlas[idx..idx + 4].copy_from_slice(&[255, 255, 255, coverage]);
            }
        }
    }

    FontState {
        atlas,
        cell_w,
        glyph_h,
        atlas_w,
        atlas_h,
    }
}

// ── Public accessors ─────────────────────────────────────────────────────────
//
// All callers go through these instead of the old `pub const` values because
// the dimensions are not known until fontdue has parsed the font at startup.

/// Horizontal pixel advance for any character. Uniform across all glyphs
/// because PixelOperatorMono is monospace.
pub fn cell_w() -> u32 {
    state().cell_w
}

/// Full line-box height in pixels: ascent + |descent|. Used to size the
/// character quad vertically in the HUD vertex builder.
pub fn glyph_h() -> u32 {
    state().glyph_h
}

/// Total pixel width of the font atlas texture.
pub fn atlas_w() -> u32 {
    state().atlas_w
}

/// Pixel height of the font atlas texture (equal to [`glyph_h`]).
pub fn atlas_h() -> u32 {
    state().atlas_h
}

/// Slot index within the atlas for character `c`.
///
/// Maps printable ASCII (U+0020 … U+007E) directly to its slot (`cp − 0x20`).
/// Any character outside that range — including all Unicode above U+007E —
/// silently maps to slot 0 (space), so a missing glyph renders as blank rather
/// than panicking.
pub fn slot_for(c: char) -> u32 {
    let cp = c as u32;
    if cp >= FIRST && cp <= LAST {
        cp - FIRST
    } else {
        0 // fall back to space (slot 0)
    }
}

/// Build the font atlas as a flat `Rgba8` byte buffer.
///
/// Lit pixels are encoded as `(255, 255, 255, coverage)` where `coverage` is
/// fontdue's per-pixel rasterisation output. For a pixel font at its design
/// size this is predominantly 0 or 255, matching the binary behaviour of the
/// old hand-baked table. Unlit pixels are `(0, 0, 0, 0)`.
///
/// The atlas is built exactly once and cached in a [`OnceLock`]; this call
/// clones only the ~24 KB result, which happens once at renderer startup.
pub fn build_font_atlas() -> Vec<u8> {
    state().atlas.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_slot_is_zero() {
        assert_eq!(slot_for(' '), 0);
    }

    #[test]
    fn printable_ascii_slots_are_contiguous() {
        // '!' is U+0021, slot 1; '~' is U+007E, slot 94.
        assert_eq!(slot_for('!'), 1);
        assert_eq!(slot_for('~'), 94);
        assert_eq!(slot_for('A'), b'A' as u32 - FIRST);
        assert_eq!(slot_for('z'), b'z' as u32 - FIRST);
    }

    #[test]
    fn out_of_range_chars_fall_back_to_space() {
        // '@' IS in ASCII printable range (0x40); '€' is not.
        assert_eq!(slot_for('@'), b'@' as u32 - FIRST);
        assert_eq!(slot_for('€'), 0); // falls back to space
        assert_eq!(slot_for('\n'), 0);
    }

    #[test]
    fn atlas_dimensions_are_consistent() {
        // Building the atlas must not panic and dimensions must be non-zero.
        let data = build_font_atlas();
        let w = atlas_w();
        let h = atlas_h();
        assert!(w > 0 && h > 0, "atlas must have positive dimensions");
        assert_eq!(
            data.len(),
            (w * h * 4) as usize,
            "atlas byte count must match w × h × 4"
        );
        assert_eq!(w, SLOT_COUNT * cell_w(), "atlas_w must equal SLOT_COUNT × cell_w");
        assert_eq!(h, glyph_h(), "atlas_h must equal glyph_h");
    }

    #[test]
    fn space_slot_is_blank() {
        // Every pixel in the space cell must have alpha = 0.
        let data = build_font_atlas();
        let cw = cell_w();
        let aw = atlas_w();
        let ah = glyph_h();
        for y in 0..ah {
            for x in 0..cw {
                let idx = (y * aw + x) as usize * 4;
                assert_eq!(data[idx + 3], 0, "space cell alpha at ({x},{y}) must be 0");
            }
        }
    }
}
