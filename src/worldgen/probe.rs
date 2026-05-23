//! Inspection types for the worldgen visualizer (PR 2).
//!
//! `ColumnProbe` is a snapshot of every value the pipeline computes
//! for one (wx, wz) column. `Stage` enumerates the per-column scalar
//! stages exposed as 2D overlays. `DensityBreakdown` is the
//! per-voxel density decomposition surfaced in the probe panel's
//! "sliding y" section.

use crate::voxel::block::Block;
use crate::worldgen::Biome;
use crate::worldgen::plates::{PlateId, PlateLookup};

/// Lean per-column snapshot for viz paint modes. Cheaper than
/// `ColumnProbe` — populated with just the fields the per-face paint
/// pass needs, so a 32×32 chunk's worth (1024 columns) can be
/// precomputed in a few ms.
#[derive(Debug, Clone, Copy)]
pub struct PaintColumn {
    pub biome: Biome,
    /// Primary plate (nearest by Voronoi). Used by the PlateId paint mode.
    pub plate_id: PlateId,
    /// Pre-carve heightmap value. Used by HeightDelta = h_target - h_pre.
    pub h_pre: f32,
    pub h_target: i32,
    /// `|∇h_pre|` proxy from the cliff-detection gradient. Used by Slope.
    pub slope: f32,
}

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
    pub river_water_y: Option<i32>,
    pub river_bed_y: Option<i32>,
    /// Unified water surface Y from the new plate-driven model.
    /// `Some(y)` = column is submerged; topmost Water at world Y == y.
    pub water_surface_y: Option<i32>,
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
    /// Graph cave / entrance combined SDF.
    pub cave_sdf: f32,
    /// Cheese carver contribution.
    pub cheese: f32,
    /// Terasology ambient carver contribution.
    pub tera: f32,
    /// Pillar contribution (adds back density inside carved volumes).
    pub pillar: f32,
    /// Composed density after all contributions (post-slide).
    pub final_density: f32,
    /// The block this voxel would resolve to. Computed by re-running
    /// the same selection logic `fill_chunk` uses.
    pub block: Block,
    /// Fluid planner explanation for generated Water/Lava at this voxel.
    pub fluid_reason: Option<crate::worldgen::fluid::FluidReason>,
    /// Name of the cave style if the probe point is inside a graph cave
    /// chamber or tunnel, else `None`.
    pub cave_style: Option<&'static str>,
    /// Depth band ("shallow"/"middle"/"deep") inferred from the cave
    /// system's bounding-box Y midpoint, if inside a cave system.
    pub cave_band: Option<&'static str>,
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
    RiverWaterSurface,
    RiverBed,
    LakeRim,
    WaterSurfaceY,
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
        Stage::RiverWaterSurface,
        Stage::RiverBed,
        Stage::LakeRim,
        Stage::WaterSurfaceY,
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
            Stage::RiverWaterSurface => "River water",
            Stage::RiverBed => "River bed",
            Stage::LakeRim => "Lake rim",
            Stage::WaterSurfaceY => "Water surface Y",
            Stage::BiomeId => "Biome",
            Stage::AquiferY => "Aquifer Y",
            Stage::AquiferSubstance => "Aquifer substance",
        }
    }

    /// Categorical stages need a discrete colormap (PlateId, BiomeId,
    /// AquiferSubstance); the rest are scalar.
    pub fn is_categorical(self) -> bool {
        matches!(
            self,
            Stage::PlateId | Stage::BiomeId | Stage::AquiferSubstance
        )
    }

    /// One-line human description of what this stage represents.
    /// Shown beneath the map's stage dropdown.
    pub fn description(self) -> &'static str {
        match self {
            Stage::Continentalness => {
                "Signed plate-Voronoi distance field. Positive inland, negative offshore — drives continent/ocean shape."
            }
            Stage::PlateId => {
                "Hashed plate ID (categorical). Each tectonic plate gets a stable hue."
            }
            Stage::Temperature => {
                "Raw temperature noise in [-1, 1]. Combines with humidity + continentalness to pick the biome."
            }
            Stage::Humidity => {
                "Raw humidity noise in [-1, 1]. Combines with temperature + continentalness to pick the biome."
            }
            Stage::Desertness => {
                "Desert-mask noise. Above the desert threshold the column flips to sand surface."
            }
            Stage::Weirdness => {
                "Weirdness noise. Selects rare biome variants (ice spikes / sunflower plains analogues)."
            }
            Stage::HPre => {
                "Pre-carve heightmap value (blocks). Plate base + ridges + warped FBM, before river carving."
            }
            Stage::ValleyCarve => {
                "Depth (blocks) that hydrology subtracts from h_pre to cut rivers. Brighter = deeper carve."
            }
            Stage::HTarget => {
                "Final terrain height (blocks) after the river carve. The actual top of the column."
            }
            Stage::FlowAccum => {
                "Hydrology flow accumulation (log-scaled). High values are trunk rivers; low values are headwaters."
            }
            Stage::RiverWaterSurface => {
                "Resolved generated river water-surface Y. Dry rapids and non-river cells are zero."
            }
            Stage::RiverBed => {
                "Resolved generated river bed Y below the water surface. Non-river cells are zero."
            }
            Stage::LakeRim => {
                "Filtered sink-fill lake rim Y. Tiny one-cell wet scratches are zero."
            }
            Stage::WaterSurfaceY => {
                "Unified water surface Y (plate-driven). Ocean/lake/river columns show their water level; dry land shows zero."
            }
            Stage::BiomeId => {
                "Discrete biome label (categorical): Tundra, SnowyForest, Plains, Forest, Desert, Tropical."
            }
            Stage::AquiferY => {
                "Legacy aquifer water-table Y retained for comparison; not used by default fluid generation."
            }
            Stage::AquiferSubstance => {
                "Legacy aquifer cell fluid: blue = Water, orange = Lava. Not used by default fluid generation."
            }
        }
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

    #[test]
    fn sample_stage_is_deterministic() {
        let g = Generator::new(42);
        for &stage in Stage::ALL {
            let a = g.sample_stage(stage, 200, 300);
            let b = g.sample_stage(stage, 200, 300);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "stage {:?} not byte-stable",
                stage
            );
        }
    }

    #[test]
    fn sample_stage_matches_probe_for_scalar_stages() {
        let g = Generator::new(42);
        let p = g.probe_column(200, 300);
        let cont = g.sample_stage(Stage::Continentalness, 200, 300);
        assert!((cont - p.continentalness).abs() < 1e-5);
        let h = g.sample_stage(Stage::HTarget, 200, 300);
        assert!((h - p.h_target as f32).abs() < 1e-5);
    }

    #[test]
    fn sample_stage_varies_across_coords() {
        // Defensive: if the dispatch is broken and always returns 0.0
        // for some stage, this catches it.
        let g = Generator::new(42);
        for &stage in Stage::ALL {
            let a = g.sample_stage(stage, 0, 0);
            let b = g.sample_stage(stage, 1000, 1000);
            // It is OK if a single stage happens to be equal at two
            // points (e.g., flat ocean continentalness); we only fail
            // if EVERY stage matches at both coords.
            if a != b {
                return;
            }
        }
        panic!("no stage varied across (0,0) vs (1000,1000) — dispatch broken");
    }

    #[test]
    fn density_breakdown_is_deterministic() {
        let g = Generator::new(42);
        let a = g.evaluate_density_breakdown(0, 70, 0);
        let b = g.evaluate_density_breakdown(0, 70, 0);
        assert_eq!(a.final_density.to_bits(), b.final_density.to_bits());
        assert_eq!(a.block, b.block);
    }

    #[test]
    fn density_high_y_is_air_or_water() {
        // y=200 is well above any reasonable surface — should be air or water.
        let g = Generator::new(42);
        let b = g.evaluate_density_breakdown(0, 200, 0);
        assert!(
            matches!(
                b.block,
                crate::voxel::block::Block::Air | crate::voxel::block::Block::Water
            ),
            "got {:?}",
            b.block
        );
    }

    #[test]
    fn density_low_y_is_solid() {
        // y=-100 (deep underground) should almost always be solid.
        let g = Generator::new(42);
        let b = g.evaluate_density_breakdown(0, -100, 0);
        assert!(
            b.final_density > 0.0,
            "expected positive density deep underground, got {}",
            b.final_density
        );
        assert!(
            !matches!(b.block, crate::voxel::block::Block::Air),
            "expected solid block deep underground, got {:?}",
            b.block
        );
    }
}
