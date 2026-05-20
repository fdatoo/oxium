//! Paint modes for the 3D viewport. PR 1 ships only Block; PR 3 adds
//! biome, height-delta, density, cave-distance, plate, slope.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintMode {
    Block,
}

impl Default for PaintMode {
    fn default() -> Self {
        PaintMode::Block
    }
}
