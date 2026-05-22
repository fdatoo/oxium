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
    pub climate: ClimateConfig,
    pub biomes: BiomesConfig,
    pub surface: crate::worldgen::surface::RuleSource,
    pub aquifer: crate::worldgen::aquifer::AquiferConfig,
    pub cave: CaveConfig,
}

/// Noise channel descriptor. `first_octave` sets the lowest-
/// frequency octave's wavelength (`2^-first_octave` ≈ the wavelength
/// in blocks). `amplitudes` weights successive octaves; a zero entry
/// skips that octave entirely.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelParams {
    pub first_octave: i32,
    pub amplitudes: Vec<f32>,
}

impl ChannelParams {
    /// Effective number of octaves (count of nonzero amplitudes).
    pub fn octave_count(&self) -> usize {
        self.amplitudes.iter().filter(|&&a| a != 0.0).count()
    }

    /// Frequency of the first (lowest) octave, in cycles/block.
    pub fn first_frequency(&self) -> f64 {
        2.0_f64.powi(self.first_octave)
    }
}

/// Per-style parameter ranges for cave-system construction. Lives in
/// `CaveConfig` so RON hot reload can retune styles without recompile.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaveStyleTable {
    /// (min, max) chambers per system, per style.
    pub cathedral_chamber_count: (u32, u32),
    pub warren_chamber_count: (u32, u32),
    pub slot_chamber_count: (u32, u32),
    pub sump_chamber_count: (u32, u32),
    pub karst_chamber_count: (u32, u32),
    /// (min, max) chamber radii in XZ.
    pub cathedral_r_xz: (f32, f32),
    pub warren_r_xz: (f32, f32),
    pub slot_r_xz: (f32, f32),
    pub sump_r_xz: (f32, f32),
    pub karst_r_xz: (f32, f32),
    /// (min, max) chamber radii in Y.
    pub cathedral_r_y: (f32, f32),
    pub warren_r_y: (f32, f32),
    pub slot_r_y: (f32, f32),
    pub sump_r_y: (f32, f32),
    pub karst_r_y: (f32, f32),
    /// (min, max) tunnel radius.
    pub cathedral_tunnel_r: (f32, f32),
    pub warren_tunnel_r: (f32, f32),
    pub slot_tunnel_r: (f32, f32),
    pub sump_tunnel_r: (f32, f32),
    pub karst_tunnel_r: (f32, f32),
    /// Band-biased style weights `[Cathedral, Warren, Slot, Sump, Karst]`.
    /// Each must sum to 1.0.
    pub style_weights_shallow: [f32; 5],
    pub style_weights_middle: [f32; 5],
    pub style_weights_deep: [f32; 5],
}

/// Cave-carving tunables — applies to the noise carvers (cheese,
/// pillars), not the graph cave systems.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaveConfig {
    // Cheese: signed-density carver, see `cheese_contribution`.
    pub cheese: ChannelParams,
    /// XZ multiplier on world coords when sampling the cheese noise.
    /// MC uses 1.0 (sample at the channel's natural frequency).
    pub cheese_xz_scale: f32,
    /// Y multiplier on world coords when sampling the cheese noise.
    /// Decoupled from xz so the cheese rooms can be anisotropic — MC
    /// uses 0.6666 (Y advances slower → features taller in Y →
    /// vertical caverns rather than spherical pockets).
    pub cheese_y_scale: f32,
    /// Constant added to the cheese noise sample. Positive = solid
    /// bias; lowering it lets more voxels carve.
    pub cheese_offset: f32,
    /// Surface-suppression term, evaluated as
    /// `clamp(supp_offset + supp_slope * raw_density, supp_min,
    /// supp_max)`. At the surface (raw_density ≈ 0) the full
    /// suppression applies, pushing cheese toward solid; deep
    /// underground it dies off and cheese is free to carve.
    pub cheese_suppression_offset: f32,
    pub cheese_suppression_slope: f32,
    pub cheese_suppression_min: f32,
    pub cheese_suppression_max: f32,

    /// MC-parity `cave_layer` noise. Sampled at `(xz=1, y=8)` so
    /// the noise advances 8× faster in Y than XZ → forms thin
    /// horizontal "layers" of cave-rich vs cave-poor strata. The
    /// term `cave_layer_intensity * cave_layer²` is ADDED to the
    /// cheese sum before the carve check: where `|layer|≈0` (cave-
    /// rich band) cheese can carve; where `|layer|` is large the
    /// layer² term swamps the cheese signal and the band stays
    /// solid. Without this term cheese carves uniformly at
    /// ~50% of deep voxels, producing chaotic swiss cheese.
    pub cave_layer: ChannelParams,
    pub cave_layer_xz_scale: f32,
    pub cave_layer_y_scale: f32,
    /// Multiplier on `cave_layer²` added to cheese. MC uses 4.0.
    /// Higher → fewer / sparser cave-rich bands; lower → uniform
    /// carving everywhere.
    pub cave_layer_intensity: f32,

    // Pillars: positive density that gets max()'d at the end so
    // they refill carved voxels (stone columns inside open caves).
    pub pillar: ChannelParams,
    pub pillar_rareness: ChannelParams,
    pub pillar_thickness: ChannelParams,
    pub pillar_xz_scale: f32,
    pub pillar_y_scale: f32,
    pub pillar_cutoff: f32,
    pub pillar_intensity: f32,

    /// Raw-density threshold below which the noise carvers (cheese)
    /// are silent. The graph cave system's entrances still carve
    /// below this, providing the deliberate surface openings. Above
    /// it (deeper underground), all carvers operate.
    pub underground_density_threshold: f32,

    // ── Terasology depth-driven ambient ──────────────────────────────
    /// 4-octave FBM-Simplex channels for the two-noise intersection
    /// that defines the meandering tubes of the ambient cave layer.
    pub tera_a: ChannelParams,
    pub tera_b: ChannelParams,
    /// Noise wavelength in blocks. Default 200.
    pub tera_wave: f32,
    /// Surface-band suppression magnitude — shift applied to noise B
    /// near the heightmap to push the cave region off-axis. Default 0.17.
    pub tera_supp: f32,
    /// Block depth over which the suppression fades to zero. Default 123.
    pub tera_supp_depth: f32,
    /// Base radius of the cave region in noise space (at depth 0). Default 0.073.
    pub tera_thresh_base: f32,
    /// Depth-divisor: threshold += depth / this. Default 2229.
    pub tera_thresh_depth: f32,
    /// Y-anisotropy: multiplier on wy when sampling tera noise. Higher
    /// values force tube iso-surfaces to bend horizontal. Default 3.56.
    pub tera_y_factor: f32,

    pub style_table: CaveStyleTable,
    /// Per-chamber depth-driven radius multiplier.
    /// `mult(cy) = 1.0 + depth_scale * max(0, (40 - cy) / 80)`.
    pub depth_scale: f32,
    /// Share of cave systems rolled into the Deep band.
    /// 0.0 = uniform thirds; 1.0 = heavily deep.
    pub deep_band_bias: f32,
    /// Max cave systems per region; sweep-chosen 3.
    pub systems_per_region_max: u32,
    /// Per-chamber radius jitter multiplier range. (0.7, 1.3) → ×0.7..×1.3.
    pub chamber_radius_jitter: (f32, f32),
    /// Probability that two systems in adjacent bands of the same region
    /// are linked by a vertical connector tunnel.
    pub vertical_connector_prob: f32,
    /// Vertical-connector tunnel radius.
    pub vertical_connector_r: f32,
    /// Probability that a cave system has a cross-region trunk to a
    /// neighbour-region system's chamber 0.
    pub trunk_prob: f32,
    /// Cross-region trunk radius.
    pub trunk_r: f32,
    /// Smooth-min radius for cave layer joins. 0.0 = strict min().
    /// Default 1.2 merges nearly-touching pockets (within ~1.2 in SDF units).
    pub smin_k: f32,
}

/// PR 4 biome lookup config. The 6 existing biomes (Tundra,
/// SnowyForest, Plains, Forest, Desert, Tropical) are selected by
/// hyperbox claims in the 6D climate space — see
/// `crate::worldgen::climate::ParameterPoint`. The `weirdness`
/// noise is a new 2D Fbm sampled per column.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BiomesConfig {
    /// Each entry claims one biome on the 6 climate axes.
    pub entries: Vec<crate::worldgen::climate::ParameterPoint>,
    /// Period (blocks) of the weirdness Fbm noise. Adds variant
    /// biomes (ice spikes / sunflower plains analogues) inside
    /// the same temperature/humidity/continentalness regions.
    pub weirdness_period: f32,
    pub weirdness_amplitude: f32,
}

/// Climate-driven spline pipeline tuning (PR 3).
///
/// Three 2D inputs feed nested cubic Hermite splines:
///
/// * `continentalness` — plate Voronoi signed distance field
///   blended via `plate_t`. Positive inland, negative offshore.
/// * `terrain_shape` — low-frequency 2D noise (~2000-block period)
///   plus per-plate `roughness_bias`. Low values produce mountains.
/// * `ridges_pv` — peaks-and-valleys triangle fold on a
///   higher-frequency ridge noise. Drives jaggedness.
///
/// Each spline is nested: outer keyed on continentalness, inner
/// (the knot's value) keyed on terrain_shape or ridges_pv. The
/// innermost result is a scalar in roughly `[-1.5, 1.5]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClimateConfig {
    pub offset_spline: NestedSpline,
    pub factor_spline: NestedSpline,
    pub jaggedness_spline: NestedSpline,
    pub plate_roughness_bias_range: (f32, f32),
    pub terrain_shape_period: f32,
    pub terrain_shape_amplitude: f32,
    pub ridges_period: f32,
    pub ridges_amplitude: f32,
}

/// A nested spline: each knot's value is itself a spline. The
/// `evaluate` method takes three inputs `(c, s, r)` and walks the
/// nesting — outer on `c`, mid on `s`, inner on `r`. Constants
/// short-circuit (so any depth can be a leaf).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NestedSpline {
    Constant(f32),
    Multipoint(Vec<NestedKnot>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NestedKnot {
    pub loc: f32,
    pub val: NestedSpline,
    pub slope: f32,
}

impl NestedSpline {
    pub fn evaluate(&self, c: f32, s: f32, r: f32) -> f32 {
        self.evaluate_inner(&[c, s, r], 0)
    }

    fn evaluate_inner(&self, inputs: &[f32], depth: usize) -> f32 {
        match self {
            NestedSpline::Constant(v) => *v,
            NestedSpline::Multipoint(knots) => {
                assert!(!knots.is_empty(), "nested spline must have ≥1 knot");
                let input = inputs.get(depth).copied().unwrap_or(0.0);
                if input <= knots[0].loc {
                    let base = knots[0].val.evaluate_inner(inputs, depth + 1);
                    return base + knots[0].slope * (input - knots[0].loc);
                }
                let last = knots.last().unwrap();
                if input >= last.loc {
                    let base = last.val.evaluate_inner(inputs, depth + 1);
                    return base + last.slope * (input - last.loc);
                }
                let mut i = 0;
                while i + 1 < knots.len() && knots[i + 1].loc < input {
                    i += 1;
                }
                let k1 = &knots[i];
                let k2 = &knots[i + 1];
                let v1 = k1.val.evaluate_inner(inputs, depth + 1);
                let v2 = k2.val.evaluate_inner(inputs, depth + 1);
                let dx = k2.loc - k1.loc;
                let t = (input - k1.loc) / dx;
                let a = k1.slope * dx - (v2 - v1);
                let b = -k2.slope * dx + (v2 - v1);
                let lerp_y = v1 + t * (v2 - v1);
                let lerp_ab = a + t * (b - a);
                lerp_y + t * (1.0 - t) * lerp_ab
            }
        }
    }
}

impl ClimateConfig {
    pub fn offset_spline_at(&self, c: f32, s: f32, r: f32) -> f32 {
        self.offset_spline.evaluate(c, s, r)
    }
    pub fn factor_spline_at(&self, c: f32, s: f32, r: f32) -> f32 {
        self.factor_spline.evaluate(c, s, r)
    }
    pub fn jaggedness_spline_at(&self, c: f32, s: f32, r: f32) -> f32 {
        self.jaggedness_spline.evaluate(c, s, r)
    }
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

use arc_swap::ArcSwap;
use std::sync::Arc;

/// Thread-safe holder for the current worldgen config. Readers use
/// `holder.load()` to get a snapshot `Arc<WorldgenConfig>`; the file
/// watcher swaps in a new value via `holder.swap(new)` without
/// blocking readers.
#[derive(Clone)]
pub struct ConfigHolder(Arc<ArcSwap<WorldgenConfig>>);

impl ConfigHolder {
    pub fn new(initial: WorldgenConfig) -> Self {
        Self(Arc::new(ArcSwap::new(Arc::new(initial))))
    }

    /// Cheap atomic read of the current config. Returns an
    /// `Arc<WorldgenConfig>` snapshot — held references stay
    /// valid even if the holder is swapped concurrently.
    pub fn load(&self) -> Arc<WorldgenConfig> {
        self.0.load_full()
    }

    /// Atomically replace the held config. Existing snapshots
    /// returned by `load()` remain valid.
    pub fn swap(&self, new: WorldgenConfig) {
        self.0.store(Arc::new(new));
    }
}

use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};
use std::path::PathBuf;
use std::time::Duration;

/// Spawn a file watcher on `path`. On any change, re-parse the RON
/// file and (if valid) atomically swap the new config into
/// `holder`. Parse errors are logged at `error` level; the previous
/// config stays in effect.
///
/// Returns the debouncer — caller must keep it alive for the
/// watcher to keep running. Dropping it shuts the watcher down.
pub fn spawn_watcher(
    path: PathBuf,
    holder: ConfigHolder,
) -> anyhow::Result<Debouncer<notify_debouncer_mini::notify::RecommendedWatcher>> {
    let watch_path = path.clone();
    let mut debouncer = new_debouncer(
        Duration::from_millis(300),
        move |res: DebounceEventResult| match res {
            Ok(_events) => match WorldgenConfig::from_ron_file(&watch_path) {
                Ok(cfg) => {
                    log::info!("worldgen config reloaded from {:?}", watch_path);
                    holder.swap(cfg);
                }
                Err(e) => {
                    log::error!(
                        "worldgen config reload failed ({:?}): {} — keeping previous",
                        watch_path,
                        e
                    );
                }
            },
            Err(e) => log::error!("watcher error: {:?}", e),
        },
    )?;
    debouncer.watcher().watch(
        &path,
        notify_debouncer_mini::notify::RecursiveMode::NonRecursive,
    )?;
    Ok(debouncer)
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

    #[test]
    fn holder_swap_visible_to_subsequent_load() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let holder = ConfigHolder::new(cfg);
        let initial_factor = holder.load().density.factor;
        let mut new_cfg = (*holder.load()).clone();
        new_cfg.density.factor = 99.0;
        holder.swap(new_cfg);
        assert!((initial_factor - 4.0).abs() < 1e-5);
        assert!((holder.load().density.factor - 99.0).abs() < 1e-5);
    }

    #[test]
    fn holder_load_returns_independent_snapshot() {
        let cfg = WorldgenConfig::bundled_default().unwrap();
        let holder = ConfigHolder::new(cfg);
        let snapshot = holder.load();
        let mut new_cfg = (*snapshot).clone();
        new_cfg.density.factor = 42.0;
        holder.swap(new_cfg);
        // The previously-held snapshot must NOT see the new value.
        assert!((snapshot.density.factor - 4.0).abs() < 1e-5);
    }

    #[test]
    fn watcher_picks_up_file_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_config.ron");

        // Write initial config with factor=4.0.
        let initial = WorldgenConfig::bundled_default().unwrap();
        let initial_text =
            ron::ser::to_string_pretty(&initial, ron::ser::PrettyConfig::default()).unwrap();
        std::fs::write(&path, &initial_text).unwrap();

        let cfg = WorldgenConfig::from_ron_file(&path).unwrap();
        let holder = ConfigHolder::new(cfg);
        let _debouncer = spawn_watcher(path.clone(), holder.clone()).unwrap();

        // Modify the file: change factor to 7.0.
        let modified = initial_text.replace("factor: 4.0", "factor: 7.0");
        // Sleep briefly so the file mtime ticks past initial write.
        std::thread::sleep(Duration::from_millis(100));
        std::fs::write(&path, modified).unwrap();

        // Poll up to 2 seconds for the swap to occur.
        for _ in 0..20 {
            std::thread::sleep(Duration::from_millis(100));
            if (holder.load().density.factor - 7.0).abs() < 1e-3 {
                return;
            }
        }
        panic!(
            "watcher did not pick up file change; factor still {}",
            holder.load().density.factor
        );
    }
}
