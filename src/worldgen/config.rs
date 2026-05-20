//! Hot-reloadable worldgen configuration.
//!
//! All tunable parameters live here, loaded at startup from
//! `assets/worldgen/default.ron` and atomically swapped on file
//! change via [`ConfigHolder`]. The graph topology (which density
//! functions exist, what marker wrappers apply) stays in Rust;
//! only values are file-driven.

use crate::worldgen::spline::CubicSpline;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level config. All worldgen-tunable values root here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldgenConfig {
    pub density: DensityConfig,
}

/// Density composition tuning (PR 2 introduces this section; PR 3+
/// expand it with biome / cave / aquifer subsections).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DensityConfig {
    /// World Y range used by the y_gradient. At `y_min`, gradient
    /// equals `+y_gradient_amplitude`; at `y_max`, equals the
    /// negative of that.
    pub y_min: i32,
    pub y_max: i32,
    pub y_gradient_amplitude: f32,

    /// Multiplier on the (depth + jagged) * factor term. MC uses 4.
    pub composition_scale: f32,

    /// Fraction by which above-surface (negative-depth) magnitudes
    /// are scaled before composition_scale. MC uses 0.25.
    pub above_surface_softening: f32,

    /// Constant `factor` for PR 2 (PR 3 makes this a spline of
    /// (continentalness, erosion, ridges)). Higher → sharper
    /// surface transition.
    pub factor: f32,

    /// Base 3D noise period in blocks.
    pub base_3d_period: f32,
    /// Base 3D noise amplitude.
    pub base_3d_amplitude: f32,
    /// Y-axis scale relative to XZ. 0.5 = vertical features 2×
    /// taller than wide (matches MC).
    pub base_3d_y_scale: f32,

    /// Top-slide: within `slide_top_blocks` of `y_max`, density is
    /// lerped toward `slide_top_target` (negative → forces air).
    pub slide_top_blocks: i32,
    pub slide_top_target: f32,

    /// Bottom-slide: within `slide_bottom_blocks` of `y_min`,
    /// density is lerped toward `slide_bottom_target` (positive →
    /// forces solid).
    pub slide_bottom_blocks: i32,
    pub slide_bottom_target: f32,

    /// Placeholder for the PR 3 offset spline. Today a Constant.
    pub offset_spline: CubicSpline,
}

impl WorldgenConfig {
    /// Load and parse from a RON file path.
    pub fn from_ron_file(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = ron::from_str(&text)?;
        Ok(cfg)
    }

    /// Load the bundled default config. Looks at
    /// `<crate-root>/assets/worldgen/default.ron`.
    pub fn bundled_default() -> anyhow::Result<Self> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("worldgen")
            .join("default.ron");
        Self::from_ron_file(&path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_default_loads_and_parses() {
        let cfg = WorldgenConfig::bundled_default().expect("default.ron must load");
        assert!(cfg.density.y_max > cfg.density.y_min);
        assert!(cfg.density.composition_scale > 0.0);
        assert!(cfg.density.factor > 0.0);
        assert!(cfg.density.above_surface_softening > 0.0);
        assert!(cfg.density.above_surface_softening <= 1.0);
    }

    #[test]
    fn ron_roundtrip_preserves_values() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let s = ron::to_string(&cfg).unwrap();
        let parsed: WorldgenConfig = ron::from_str(&s).unwrap();
        assert_eq!(parsed.density.y_min, cfg.density.y_min);
        assert_eq!(parsed.density.factor, cfg.density.factor);
    }
}
