//! Cave-system personality types, depth-band placement, and hash-domain
//! constants.
//!
//! Every per-system roll is namespaced by a `SALT_*` constant so two
//! different rolls at the same coordinates cannot produce correlated output.
//! See the `// ── Hash domain separators ──` block below.
use crate::worldgen::hash::mix_unit;
use crate::worldgen::region::RegionCoord;
use crate::worldgen::tuning::*;

// ── Hash domain separators ────────────────────────────────────────────────────
// Each constant uniquely namespaces one per-region or per-system hash roll so
// two different rolls at the same coordinates cannot produce correlated output.
// Values are arbitrary but must be globally unique within the caves submodule.

/// Namespaces the depth-band (Shallow / Middle / Deep) roll for a cave system.
pub(super) const SALT_DEPTH_BAND: i32 = 100;
/// Namespaces the style roll (Cathedral / Warren / Slot / Sump / Karst).
pub(super) const SALT_CAVE_STYLE: i32 = 7000;
/// Namespaces the total-system-count roll for the region.
pub(super) const SALT_SYSTEM_COUNT: i32 = 1;
/// Namespaces the lava-vs-water pool kind roll for a chamber.
pub(super) const SALT_POOL_LAVA: i32 = 99_001;
/// Namespaces the vertical-connector probability roll between two systems.
pub(super) const SALT_VERTICAL_CONNECTOR: i32 = 9500;
/// Namespaces the cross-region trunk probability roll. (See `build_trunks`.)
pub(super) const SALT_TRUNK_PROB: i32 = 9000;
/// Namespaces the trunk midpoint lateral-offset roll (determines which side of
/// the straight line the arc bows toward).
pub(super) const SALT_TRUNK_MID_OFFSET: i32 = 9001;
/// Namespaces the bounding-box X-origin roll within the region.
pub(super) const SALT_BB_ORIGIN_X: i32 = 10;
/// Namespaces the bounding-box Z-origin roll within the region.
pub(super) const SALT_BB_ORIGIN_Z: i32 = 11;
/// Namespaces the chamber-count roll (how many chambers this system has).
pub(super) const SALT_CHAMBER_COUNT: i32 = 20;
/// Namespaces the extra-loop-count roll (how many non-MST tunnel edges to add).
pub(super) const SALT_EXTRA_LOOPS: i32 = 50;

/// Distinct cave-system personalities, rolled per system from the
/// region cell id and the system's depth band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaveStyle {
    /// Few large chambers, wide tunnels. Deep-band-biased.
    Cathedral,
    /// Many small chambers, narrow tunnels. Shallow-band-biased.
    Warren,
    /// XZ-stretched chambers, narrow vertical sheets. Mid-band-biased.
    Slot,
    /// Low-clustered chambers (flooded look). Deep-band-biased.
    Sump,
    /// Default — medium chambers, medium tunnels.
    Karst,
}

/// Depth band a system belongs to. Drives bounding-box Y placement
/// and entrance-roll probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthBand {
    Shallow,
    Middle,
    Deep,
}

impl DepthBand {
    pub(super) fn pick(seed: u64, system_id: i32, region: RegionCoord) -> Self {
        // Three-way roll: 35% Shallow, 35% Middle, 30% Deep.
        let r = mix_unit(seed, &[region.x, region.z, system_id, SALT_DEPTH_BAND]);
        if r < 0.35 {
            DepthBand::Shallow
        } else if r < 0.70 {
            DepthBand::Middle
        } else {
            DepthBand::Deep
        }
    }

    /// `[y_min, y_max]` for chamber placement (inclusive).
    pub(super) fn range(self) -> (i32, i32) {
        match self {
            DepthBand::Shallow => CAVE_BAND_SHALLOW,
            DepthBand::Middle => CAVE_BAND_MIDDLE,
            DepthBand::Deep => CAVE_BAND_DEEP,
        }
    }

    pub(super) fn entrance_prob(self) -> f32 {
        match self {
            DepthBand::Shallow => ENTRANCE_PROB_SHALLOW,
            DepthBand::Middle => ENTRANCE_PROB_MIDDLE,
            DepthBand::Deep => ENTRANCE_PROB_DEEP,
        }
    }
}

/// Roll a `CaveStyle` deterministically from `(seed, region_coord,
/// system_idx, band)`. Band-weighted via `CaveStyleTable`.
pub fn pick_style(
    seed: u64,
    coord: RegionCoord,
    system_idx: i32,
    band: DepthBand,
    cfg: &crate::worldgen::config::CaveConfig,
) -> CaveStyle {
    let u = mix_unit(seed, &[coord.x, coord.z, system_idx, SALT_CAVE_STYLE]);
    let weights = match band {
        DepthBand::Shallow => &cfg.style_table.style_weights_shallow,
        DepthBand::Middle => &cfg.style_table.style_weights_middle,
        DepthBand::Deep => &cfg.style_table.style_weights_deep,
    };
    let mut acc = 0.0;
    let styles = [
        CaveStyle::Cathedral,
        CaveStyle::Warren,
        CaveStyle::Slot,
        CaveStyle::Sump,
        CaveStyle::Karst,
    ];
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if u <= acc {
            return styles[i];
        }
    }
    CaveStyle::Karst
}
