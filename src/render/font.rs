//! Tiny 5×7 bitmap font for the HUD overlay.
//!
//! Why a hand-baked bitmap font rather than a TrueType renderer? The
//! HUD only needs to show short debug strings ("FPS: 60.0", "XYZ:
//! -100.0  64.0  86.0") — pulling in `glyphon` / `ab_glyph` /
//! `cosmic-text` to render two lines of monospace digits is overkill.
//! A hand-drawn 5×7 font fits the pixel-art aesthetic of the textured
//! blocks, ships zero external assets, and the glyphs compile straight
//! into the binary as a tiny `const` table.
//!
//! Layout: each glyph is 5 columns × 7 rows of single-bit pixels,
//! encoded as 7 `u8` values (one per row). Bit 4 is the leftmost
//! pixel, bit 0 the rightmost. The font texture lays glyphs out
//! horizontally on a single 256×8 row, one glyph per 8-pixel cell
//! (the 3-pixel right margin gives the HUD vertex builder a clean
//! per-character advance and avoids texel bleed at quad edges).
//!
//! Only the characters the HUD actually prints are defined; missing
//! characters render as the blank "space" cell so a typo never panics
//! at runtime.

/// Pixel width of one glyph. The HUD vertex builder uses this to scale
/// per-character quads.
pub const GLYPH_W: u32 = 5;
/// Pixel height of one glyph. Cells in the atlas are slightly taller
/// (`CELL_H`) to give 1 row of padding above.
pub const GLYPH_H: u32 = 7;
/// Horizontal stride between glyph slots in the atlas.
pub const CELL_W: u32 = 8;
/// Vertical stride; only one row of glyphs is laid out today.
pub const CELL_H: u32 = 8;
/// Number of glyph slots in the atlas. Holds digits + the full
/// uppercase + lowercase alphabet + common punctuation with room to
/// spare. Bumped from 64 to 96 when the lowercase set was added — the
/// extra atlas width (256 → 768 texels) is still trivial.
pub const SLOT_COUNT: u32 = 96;
/// Atlas pixel width.
pub const ATLAS_W: u32 = SLOT_COUNT * CELL_W;
/// Atlas pixel height.
pub const ATLAS_H: u32 = CELL_H;

/// One glyph's 7-row bit pattern. Convention: row 0 = top of glyph,
/// bit 4 = leftmost pixel. A `1` bit is opaque white; a `0` bit is
/// transparent.
type Glyph = [u8; 7];

/// Per-character glyph table. The HUD looks up each printable char via
/// [`glyph_for`]; characters not in this list render as blanks.
///
/// The set covers digits, the full uppercase alphabet, and common
/// punctuation — enough for any HUD string we'd want to compose
/// without having to think about which letters happen to exist.
/// Add a new char by extending this list (the slot index is just its
/// position in the array, so order doesn't matter to callers).
const GLYPHS: &[(char, Glyph)] = &[
    ('0', [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110]),
    ('1', [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('2', [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111]),
    ('3', [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110]),
    ('4', [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010]),
    ('5', [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110]),
    ('6', [0b01110, 0b10001, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110]),
    ('7', [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000]),
    ('8', [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110]),
    ('9', [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b10001, 0b01110]),
    ('A', [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    ('B', [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110]),
    ('C', [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110]),
    ('D', [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110]),
    ('E', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111]),
    ('F', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000]),
    ('G', [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110]),
    ('H', [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    ('I', [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('J', [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100]),
    ('K', [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001]),
    ('L', [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111]),
    ('M', [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001]),
    ('N', [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001]),
    ('O', [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('P', [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000]),
    ('Q', [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101]),
    ('R', [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001]),
    ('S', [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110]),
    ('T', [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
    ('U', [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('V', [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
    ('W', [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001]),
    ('X', [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001]),
    ('Y', [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100]),
    ('Z', [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111]),
    (':', [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000]),
    ('.', [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100]),
    ('-', [0b00000, 0b00000, 0b00000, 0b01110, 0b00000, 0b00000, 0b00000]),
    (',', [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000]),
    // Lowercase alphabet. Tall letters (b, d, f, h, k, l, t) extend up
    // to row 0; descenders (g, j, p, q, y) are clipped to the baseline
    // because the cell has no extra row below it.
    ('a', [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111]),
    ('b', [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110]),
    ('c', [0b00000, 0b00000, 0b01110, 0b10001, 0b10000, 0b10001, 0b01110]),
    ('d', [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111]),
    ('e', [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110]),
    ('f', [0b00110, 0b01001, 0b01000, 0b11110, 0b01000, 0b01000, 0b01000]),
    ('g', [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110]),
    ('h', [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001]),
    ('i', [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('j', [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100]),
    ('k', [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010]),
    ('l', [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('m', [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101]),
    ('n', [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001]),
    ('o', [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('p', [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000]),
    ('q', [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b00001]),
    ('r', [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000]),
    ('s', [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110]),
    ('t', [0b01000, 0b01000, 0b11110, 0b01000, 0b01000, 0b01001, 0b00110]),
    ('u', [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('v', [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
    ('w', [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010]),
    ('x', [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001]),
    ('y', [0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110]),
    ('z', [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111]),
    // Punctuation needed by the chat console + pause overlay.
    ('/', [0b00001, 0b00010, 0b00010, 0b00100, 0b00100, 0b01000, 0b01000]),
    ('>', [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000]),
    ('<', [0b00001, 0b00010, 0b00100, 0b01000, 0b00100, 0b00010, 0b00001]),
    ('(', [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010]),
    (')', [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000]),
    ('!', [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100]),
    ('?', [0b01110, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100]),
    (' ', [0; 7]),
];

/// Return the bit pattern for `c`, or the `space` blank if `c` isn't
/// in the table. The linear scan is fine: `GLYPHS` is short and this
/// runs only at startup (the atlas is baked once). Test-only — the
/// production atlas builder walks `GLYPHS` directly without this
/// indirection.
#[cfg(test)]
fn glyph_for(c: char) -> Glyph {
    for (k, g) in GLYPHS {
        if *k == c {
            return *g;
        }
    }
    [0; 7]
}

/// Index a character maps to within the atlas's horizontal grid. The
/// HUD vertex builder uses this to compute per-glyph UV rects.
pub fn slot_for(c: char) -> u32 {
    for (i, (k, _)) in GLYPHS.iter().enumerate() {
        if *k == c {
            return i as u32;
        }
    }
    // Space falls through to its own slot; if "space" isn't in the
    // table either, return the last slot which is guaranteed blank
    // (we always end with `' '`).
    (GLYPHS.len() - 1) as u32
}

/// Build the font atlas as a flat `Rgba8` byte buffer matching the
/// block atlas's format. Lit pixels are `(255, 255, 255, 255)`,
/// unlit are `(0, 0, 0, 0)` — so the HUD shader's `tex * vertex_color`
/// works uniformly for both the font batch (alpha-masked tint) and
/// the block-icon batch (pre-coloured RGBA).
pub fn build_font_atlas() -> Vec<u8> {
    let mut atlas = vec![0u8; (ATLAS_W * ATLAS_H * 4) as usize];
    for (i, (_, glyph)) in GLYPHS.iter().enumerate() {
        let col0 = (i as u32) * CELL_W;
        for row in 0..GLYPH_H {
            let bits = glyph[row as usize];
            for col in 0..GLYPH_W {
                let mask = 1u8 << (GLYPH_W - 1 - col);
                if bits & mask != 0 {
                    let x = col0 + col;
                    let y = row;
                    let i = ((y * ATLAS_W + x) * 4) as usize;
                    atlas[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
                }
            }
        }
    }
    atlas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_slot_is_blank() {
        let atlas = build_font_atlas();
        let slot = slot_for(' ');
        let col0 = slot * CELL_W;
        // Every pixel in the space cell is fully transparent.
        for y in 0..GLYPH_H {
            for x in 0..GLYPH_W {
                let i = ((y * ATLAS_W + col0 + x) * 4) as usize;
                assert_eq!(atlas[i + 3], 0, "space cell alpha at ({x},{y}) not 0");
            }
        }
    }

    #[test]
    fn digit_zero_has_a_filled_top_row() {
        // '0' starts with 0b01110: pixels 1..=3 set, edges clear.
        let atlas = build_font_atlas();
        let slot = slot_for('0');
        let col0 = slot * CELL_W;
        let alpha = |x: u32| atlas[((col0 + x) * 4 + 3) as usize];
        assert_eq!(alpha(0), 0);
        assert_eq!(alpha(1), 255);
        assert_eq!(alpha(2), 255);
        assert_eq!(alpha(3), 255);
        assert_eq!(alpha(4), 0);
    }

    #[test]
    fn missing_char_falls_back_to_blank() {
        // '@' isn't in the table — glyph_for returns the blank pattern.
        let g = glyph_for('@');
        assert_eq!(g, [0; 7]);
    }
}
