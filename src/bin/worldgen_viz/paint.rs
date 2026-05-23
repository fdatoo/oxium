//! Paint modes for the 3D viewport.
//!
//! Each mode picks a face colour from worldgen state: the block type
//! (default), the column's biome, the plate's hashed colour, height
//! delta, slope, etc. The mesher consumes `PaintContext` instead of
//! hard-coded block colours so toggling the mode just reruns the
//! paint pass.

use oxium::mesher::Face;
use oxium::voxel::block::Block;
use oxium::voxel::coords::CHUNK_DIM_U;
use oxium::worldgen::Biome;
use oxium::worldgen::Generator;
use oxium::worldgen::probe::PaintColumn;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaintMode {
    /// Block-type colour. Default.
    #[default]
    Block,
    /// Categorical hue keyed on column biome.
    Biome,
    /// Categorical hue keyed on the column's plate ID.
    PlateId,
    /// Divergent gradient on `(h_target - h_pre)` — red = valley carved,
    /// blue = ridge raised.
    HeightDelta,
    /// Hot gradient on `|∇h_pre|` magnitude.
    Slope,
}

impl PaintMode {
    pub const ALL: &'static [PaintMode] = &[
        PaintMode::Block,
        PaintMode::Biome,
        PaintMode::PlateId,
        PaintMode::HeightDelta,
        PaintMode::Slope,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PaintMode::Block => "Block",
            PaintMode::Biome => "Biome",
            PaintMode::PlateId => "Plate ID",
            PaintMode::HeightDelta => "Height Δ",
            PaintMode::Slope => "Slope",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            PaintMode::Block => {
                "Block-type colour (Stone grey, Grass green, Snow white, Lava orange)."
            }
            PaintMode::Biome => "Column biome — categorical hue per biome variant.",
            PaintMode::PlateId => "Tectonic plate ID — categorical hue. Reveals plate boundaries.",
            PaintMode::HeightDelta => {
                "Heightmap delta (h_target - h_pre). Red = valley carved by hydrology, blue = lifted by ridge noise."
            }
            PaintMode::Slope => {
                "Heightmap slope magnitude. Brighter = steeper. Useful for finding cliff-thresholds."
            }
        }
    }

    /// Returns true if rendering this mode needs per-column data
    /// (i.e. requires calling `Generator::paint_column`). `Block` mode
    /// doesn't — it just reads the voxel directly — so we skip the
    /// per-column pre-pass for it.
    pub fn needs_column_data(self) -> bool {
        !matches!(self, PaintMode::Block)
    }
}

/// Paint context for one chunk. Holds the active mode + a precomputed
/// 32×32 grid of per-column data (only populated when the mode reads
/// from it). Built once per chunk by `World::build_paint_context`.
pub struct PaintContext {
    pub mode: PaintMode,
    /// Chunk origin in world blocks (chunk_coord * CHUNK_DIM_U).
    pub origin_x: i32,
    pub origin_z: i32,
    /// 32×32 grid of paint columns, row-major by `lz * 32 + lx`.
    /// `None` if `mode.needs_column_data()` is false.
    columns: Option<Box<[PaintColumn]>>,
}

impl PaintContext {
    pub fn build(
        mode: PaintMode,
        generator: &Arc<Generator>,
        origin_x: i32,
        origin_z: i32,
    ) -> Self {
        let columns = if mode.needs_column_data() {
            let dim = CHUNK_DIM_U as i32;
            let mut buf = Vec::with_capacity((dim * dim) as usize);
            for lz in 0..dim {
                for lx in 0..dim {
                    buf.push(generator.paint_column(origin_x + lx, origin_z + lz));
                }
            }
            Some(buf.into_boxed_slice())
        } else {
            None
        };
        Self {
            mode,
            origin_x,
            origin_z,
            columns,
        }
    }

    /// Build a paint context that skips the per-column pre-pass. Only
    /// valid for modes where `needs_column_data()` is false (currently
    /// only `Block`). Used by tests that don't want to spin up a
    /// Generator just to mesh a synthetic chunk.
    #[allow(dead_code)]
    pub fn without_columns(mode: PaintMode, origin_x: i32, origin_z: i32) -> Self {
        debug_assert!(!mode.needs_column_data(), "{:?} requires column data", mode);
        Self {
            mode,
            origin_x,
            origin_z,
            columns: None,
        }
    }

    fn column(&self, wx: i32, wz: i32) -> Option<&PaintColumn> {
        let cols = self.columns.as_ref()?;
        let dim = CHUNK_DIM_U as i32;
        let lx = wx - self.origin_x;
        let lz = wz - self.origin_z;
        if lx < 0 || lz < 0 || lx >= dim || lz >= dim {
            return None;
        }
        cols.get((lz * dim + lx) as usize)
    }

    /// Pick a face colour. Always tinted by the face's direction for
    /// depth cue (top brighter, bottom darker — same as PR 1's
    /// face_tint).
    pub fn color_for(&self, wx: i32, wy: i32, wz: i32, face: Face, block: Block) -> [f32; 3] {
        let base = match self.mode {
            PaintMode::Block => block_color(block),
            PaintMode::Biome => self
                .column(wx, wz)
                .map(|c| biome_color(c.biome))
                .unwrap_or([0.4, 0.4, 0.4]),
            PaintMode::PlateId => self
                .column(wx, wz)
                .map(|c| plate_color(c.plate_id))
                .unwrap_or([0.4, 0.4, 0.4]),
            PaintMode::HeightDelta => self
                .column(wx, wz)
                .map(height_delta_color)
                .unwrap_or([0.4, 0.4, 0.4]),
            PaintMode::Slope => self
                .column(wx, wz)
                .map(slope_color)
                .unwrap_or([0.4, 0.4, 0.4]),
        };
        let _ = wy;
        let tint = face_tint(face);
        [base[0] * tint, base[1] * tint, base[2] * tint]
    }
}

/// Per-face brightness multiplier. Top faces full sun; bottom faces
/// deep shadow; sides graduated so cliffs read as cliffs. Matches the
/// values shipped in the PR 1 face-shading pass.
pub fn face_tint(face: Face) -> f32 {
    match face {
        Face::PosY => 1.00,
        Face::PosX | Face::NegZ => 0.82,
        Face::NegX | Face::PosZ => 0.66,
        Face::NegY => 0.40,
    }
}

fn block_color(b: Block) -> [f32; 3] {
    match b {
        Block::Stone => [0.55, 0.55, 0.55],
        Block::Dirt => [0.50, 0.32, 0.18],
        Block::Grass => [0.30, 0.65, 0.25],
        Block::Sand => [0.92, 0.85, 0.62],
        Block::Snow => [0.95, 0.95, 0.97],
        Block::Lava => [1.0, 0.45, 0.08],
        _ => [0.4, 0.4, 0.4],
    }
}

fn biome_color(b: Biome) -> [f32; 3] {
    match b {
        Biome::Tundra => [0.85, 0.92, 0.96],
        Biome::SnowyForest => [0.55, 0.78, 0.78],
        Biome::Plains => [0.66, 0.86, 0.52],
        Biome::Forest => [0.20, 0.55, 0.20],
        Biome::Desert => [0.95, 0.86, 0.50],
        Biome::Tropical => [0.20, 0.78, 0.40],
    }
}

fn plate_color(id: oxium::worldgen::plates::PlateId) -> [f32; 3] {
    // Stable per-plate hue from the (cell_x, cell_z) hash. We pull the
    // bottom 24 bits to drive a 3D RGB cube — gives ~16 M distinct
    // visible colours, plenty for the few hundred plates a world ever
    // touches.
    let h = oxium::worldgen::hash::mix(0xC0DE_FACE, &[id.cell_x, id.cell_z]);
    let r = ((h & 0xFF) as f32) / 255.0;
    let g = (((h >> 8) & 0xFF) as f32) / 255.0;
    let b = (((h >> 16) & 0xFF) as f32) / 255.0;
    // Mix toward grey so neighbouring plates remain readable rather
    // than searing.
    [0.5 + 0.5 * r, 0.5 + 0.5 * g, 0.5 + 0.5 * b]
}

fn height_delta_color(c: &PaintColumn) -> [f32; 3] {
    let delta = c.h_target as f32 - c.h_pre;
    // Map [-16, +16] blocks to a divergent palette: red carved, blue lifted.
    let t = (delta / 16.0).clamp(-1.0, 1.0);
    if t < 0.0 {
        let s = -t;
        [
            0.85 * s + 0.4 * (1.0 - s),
            0.25 * (1.0 - s) + 0.4 * (1.0 - s),
            0.25 * (1.0 - s) + 0.4 * (1.0 - s),
        ]
    } else {
        [
            0.4 * (1.0 - t),
            0.4 * (1.0 - t) + 0.2 * t,
            0.4 * (1.0 - t) + 0.85 * t,
        ]
    }
}

fn slope_color(c: &PaintColumn) -> [f32; 3] {
    // Slope is `|∇h_pre|` per block of XZ; cliffs are around 4+, flat
    // plains are < 0.1. Normalise into [0,1] over [0, 6].
    let t = (c.slope / 6.0).clamp(0.0, 1.0);
    // Hot: black → red → yellow → white.
    if t < 0.5 {
        let s = t / 0.5;
        [s, 0.0, 0.0]
    } else {
        let s = (t - 0.5) / 0.5;
        [1.0, s, s * 0.7]
    }
}
