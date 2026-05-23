//! Per-Stage sampler + colormap dispatch. Pure layer over
//! `Generator::sample_stage` — no rendering happens here; `MapView`
//! consumes this module's `render_pixel` to fill its texture.

use crate::viz_render::colormap;
use crate::worldgen::Generator;
use crate::worldgen::probe::Stage;

/// Sensible value range per stage for normalisation into `[0, 1]`.
/// Returning `None` means the stage is categorical and the colormap
/// keys on the raw value rather than normalising.
pub fn range(stage: Stage) -> Option<(f32, f32)> {
    match stage {
        Stage::Continentalness => Some((-1.0, 1.0)),
        Stage::Temperature => Some((-1.0, 1.0)),
        Stage::Humidity => Some((-1.0, 1.0)),
        Stage::Desertness => Some((-1.0, 1.0)),
        Stage::Weirdness => Some((-1.0, 1.0)),
        Stage::HPre => Some((40.0, 160.0)),
        Stage::ValleyCarve => Some((0.0, 16.0)),
        Stage::HTarget => Some((40.0, 160.0)),
        Stage::FlowAccum => Some((0.0, 4096.0)),
        Stage::RiverWaterSurface | Stage::RiverBed | Stage::LakeRim | Stage::WaterSurfaceY => {
            Some((40.0, 100.0))
        }
        Stage::AquiferY => Some((-64.0, 96.0)),
        Stage::PlateId | Stage::BiomeId | Stage::AquiferSubstance => None,
    }
}

/// Color one pixel given the raw sampler output for that stage.
pub fn pixel(stage: Stage, raw: f32) -> [u8; 4] {
    if let Some((lo, hi)) = range(stage) {
        let t = ((raw - lo) / (hi - lo)).clamp(0.0, 1.0);
        match stage {
            Stage::Continentalness => colormap::divergent(t),
            Stage::Temperature | Stage::Humidity | Stage::Desertness | Stage::Weirdness => {
                colormap::viridis(t)
            }
            Stage::HPre | Stage::HTarget => colormap::terrain_ramp(t),
            Stage::ValleyCarve => colormap::hot(t),
            Stage::RiverWaterSurface | Stage::RiverBed | Stage::LakeRim | Stage::WaterSurfaceY => {
                colormap::viridis(t)
            }
            Stage::FlowAccum => {
                // log-scale flow accumulation before colormap
                let t_log = (raw.max(1.0).ln() / 4096_f32.ln()).clamp(0.0, 1.0);
                colormap::viridis(t_log)
            }
            Stage::AquiferY => colormap::divergent(t),
            // Categorical stages are handled by the outer else branch;
            // these arms are unreachable here but required for exhaustiveness.
            Stage::PlateId | Stage::BiomeId | Stage::AquiferSubstance => {
                [0, 0, 0, 255]
            }
        }
    } else {
        match stage {
            Stage::PlateId | Stage::BiomeId => colormap::categorical(raw),
            Stage::AquiferSubstance => {
                colormap::binary(raw, [38, 99, 200, 255], [220, 110, 30, 255])
            }
            // WaterSurfaceY is now scalar; handled by the `if let Some` branch above.
            // This arm is unreachable but required for exhaustiveness.
            Stage::WaterSurfaceY => [0, 0, 0, 255],
            // Scalar stages are handled by the outer if branch;
            // these arms are unreachable here but required for exhaustiveness.
            _ => [0, 0, 0, 255],
        }
    }
}

/// Render one pixel by sampling the generator and colouring.
pub fn render_pixel(generator: &Generator, stage: Stage, wx: i32, wz: i32) -> [u8; 4] {
    let raw = generator.sample_stage(stage, wx, wz);
    pixel(stage, raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::Generator;

    #[test]
    fn render_pixel_is_deterministic() {
        let g = Generator::new(42);
        let a = render_pixel(&g, Stage::HTarget, 0, 0);
        let b = render_pixel(&g, Stage::HTarget, 0, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn categorical_stages_have_no_range() {
        assert!(range(Stage::PlateId).is_none());
        assert!(range(Stage::BiomeId).is_none());
        assert!(range(Stage::AquiferSubstance).is_none());
    }

    #[test]
    fn scalar_stages_have_range() {
        assert!(range(Stage::HTarget).is_some());
        assert!(range(Stage::Continentalness).is_some());
    }
}
