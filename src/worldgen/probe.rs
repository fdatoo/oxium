//! Inspection types for the worldgen visualizer (PR 2).
//!
//! `ColumnProbe` is a snapshot of every value the pipeline computes
//! for one (wx, wz) column. `Stage` enumerates the per-column scalar
//! stages exposed as 2D overlays. `DensityBreakdown` is the
//! per-voxel density decomposition surfaced in the probe panel's
//! "sliding y" section.

use crate::voxel::block::Block;
use crate::worldgen::plates::PlateLookup;
use crate::worldgen::Biome;

/// Full pipeline trace for one (wx, wz) column.
#[derive(Debug, Clone)]
pub struct ColumnProbe {
    pub wx: i32,
    pub wz: i32,
    // Geometry
    pub plate: PlateLookup,
    pub continentalness: f32,
    pub h_pre: f32,
    pub valley_carve: f32,
    pub h_target: i32,
    pub is_cliff: bool,
    pub slope: f32,
    // Climate
    pub temperature: f32,
    pub humidity: f32,
    pub desertness: f32,
    pub weirdness: f32,
    pub biome: Biome,
    // Hydrology
    pub flow_accum: u32,
    pub lake_rim: Option<i32>,
    // Aquifer
    pub aquifer_y_top: i32,
    /// The fluid the aquifer cell holds (always `Block::Water` or
    /// `Block::Lava`). Distinct from `aquifer::Substance`, which is
    /// the per-voxel resolution result; this is the per-cell choice.
    pub aquifer_fluid: Block,
    // Cave systems whose bbox intersects this column's region
    pub cave_systems_count: usize,
}

/// Per-voxel density decomposition. Filled lazily as the user drags
/// the y-slider in the probe panel.
#[derive(Debug, Clone, Copy)]
pub struct DensityBreakdown {
    pub wx: i32,
    pub wy: i32,
    pub wz: i32,
    /// Bias from `(h_target - wy) / FALLOFF`.
    pub bias: f32,
    /// 3D base noise contribution at this voxel.
    pub base_3d: f32,
    /// Graph cave / entrance / wormhole combined SDF.
    pub cave_sdf: f32,
    /// Cheese carver contribution.
    pub cheese: f32,
    /// Spaghetti carver contribution.
    pub spaghetti: f32,
    /// Pillar contribution (adds back density inside carved volumes).
    pub pillar: f32,
    /// Composed density after all contributions (post-slide).
    pub final_density: f32,
    /// The block this voxel would resolve to. Computed by re-running
    /// the same selection logic `fill_chunk` uses.
    pub block: Block,
}

/// Per-column scalar stages the overlay map can render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Continentalness,
    PlateId,
    Temperature,
    Humidity,
    Desertness,
    Weirdness,
    HPre,
    ValleyCarve,
    HTarget,
    FlowAccum,
    BiomeId,
    AquiferY,
    AquiferSubstance,
}

impl Stage {
    pub const ALL: &'static [Stage] = &[
        Stage::Continentalness,
        Stage::PlateId,
        Stage::Temperature,
        Stage::Humidity,
        Stage::Desertness,
        Stage::Weirdness,
        Stage::HPre,
        Stage::ValleyCarve,
        Stage::HTarget,
        Stage::FlowAccum,
        Stage::BiomeId,
        Stage::AquiferY,
        Stage::AquiferSubstance,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Stage::Continentalness => "Continentalness",
            Stage::PlateId => "Plate ID",
            Stage::Temperature => "Temperature",
            Stage::Humidity => "Humidity",
            Stage::Desertness => "Desertness",
            Stage::Weirdness => "Weirdness",
            Stage::HPre => "h_pre",
            Stage::ValleyCarve => "Valley carve",
            Stage::HTarget => "h_target",
            Stage::FlowAccum => "Flow accumulation",
            Stage::BiomeId => "Biome",
            Stage::AquiferY => "Aquifer Y",
            Stage::AquiferSubstance => "Aquifer substance",
        }
    }

    /// Categorical stages need a discrete colormap (PlateId, BiomeId,
    /// AquiferSubstance); the rest are scalar.
    pub fn is_categorical(self) -> bool {
        matches!(self, Stage::PlateId | Stage::BiomeId | Stage::AquiferSubstance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::Generator;

    #[test]
    fn probe_column_is_deterministic() {
        let g = Generator::new(42);
        let a = g.probe_column(100, 200);
        let b = g.probe_column(100, 200);
        assert_eq!(a.h_target, b.h_target);
        assert_eq!(a.biome, b.biome);
        assert!((a.continentalness - b.continentalness).abs() < 1e-6);
        assert!((a.temperature - b.temperature).abs() < 1e-6);
    }

    #[test]
    fn probe_column_height_matches_column_data() {
        let g = Generator::new(42);
        let p = g.probe_column(100, 200);
        let c = g.column_data(100, 200);
        assert_eq!(p.h_target, c.height);
        assert_eq!(p.biome, c.biome);
        assert_eq!(p.is_cliff, c.is_cliff);
    }
}
