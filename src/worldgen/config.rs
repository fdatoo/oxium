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

use notify_debouncer_mini::{new_debouncer, DebounceEventResult, Debouncer};
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
        let initial_text = ron::ser::to_string_pretty(
            &initial,
            ron::ser::PrettyConfig::default(),
        ).unwrap();
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
