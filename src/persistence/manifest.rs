//! Per-save metadata: seed, worldgen version, creation timestamp.
//!
//! Lives alongside the `regions/` directory inside each save folder
//! as a small TOML file. Replaces the hard-coded `seed = 42` in
//! `app.rs` — each new save gets a freshly-rolled seed at creation
//! time so different worlds look different.
//!
//! Legacy saves (created before this file existed) get a synthetic
//! manifest written on first open with `seed = 42` and
//! `worldgen_version = 1`, preserving the world they already know.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// On-disk filename inside the save directory.
pub const MANIFEST_FILENAME: &str = "world.toml";

/// Current worldgen pipeline version. v1 = pre-overhaul; v2 = the
/// plate-driven pipeline from the worldgen-overhaul spec.
pub const CURRENT_VERSION: u32 = 2;

/// Legacy worldgen version assumed for saves that pre-date the
/// manifest. Used when writing a synthetic manifest for an existing
/// save so the player's already-explored terrain stays in the old
/// generator's frame.
pub const LEGACY_VERSION: u32 = 1;

/// Legacy seed assumed for saves that pre-date the manifest. Matches
/// the value `app.rs` used to hard-code.
pub const LEGACY_SEED: u64 = 42;

/// Whole-world metadata.
///
/// `seed` is stored as a 16-character lowercase hex string on disk
/// because TOML integers are limited to i64 and we want the full u64
/// range. Custom serde adapter `seed_hex` handles the conversion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldManifest {
    /// World seed. Everything downstream — plate layout, climate,
    /// caves — is deterministic in `(seed, world_coord)`.
    #[serde(with = "seed_hex")]
    pub seed: u64,
    /// Worldgen pipeline version this save was created with. v2 ships
    /// with the overhaul spec; v1 is legacy.
    pub worldgen_version: u32,
    /// Unix timestamp (seconds) at world creation. Informational only.
    pub created_at: i64,
}

/// Custom serde adapter: serialize `u64` as a 16-char lowercase hex
/// string so the full range fits in TOML (which caps integers at i64).
/// Round-trips through `format!("{:016x}", x)` + `u64::from_str_radix`.
mod seed_hex {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(seed: &u64, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&format!("{:016x}", seed))
    }

    pub fn deserialize<'de, D>(d: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(d)?;
        u64::from_str_radix(&s, 16).map_err(serde::de::Error::custom)
    }
}

impl WorldManifest {
    /// Build a manifest for a brand-new world. Seed is rolled
    /// randomly; version is the current pipeline version.
    pub fn fresh() -> Self {
        Self {
            seed: rand::random::<u64>(),
            worldgen_version: CURRENT_VERSION,
            created_at: now_unix(),
        }
    }

    /// Build the synthetic manifest used when opening a legacy save
    /// that has no `world.toml`. Preserves the v1 seed so chunks
    /// streamed in after the upgrade keep the same generator they
    /// would have had before.
    pub fn legacy_fallback() -> Self {
        Self {
            seed: LEGACY_SEED,
            worldgen_version: LEGACY_VERSION,
            created_at: now_unix(),
        }
    }
}

/// Path of the manifest inside `saves_dir`.
pub fn manifest_path(saves_dir: &Path) -> PathBuf {
    saves_dir.join(MANIFEST_FILENAME)
}

/// Load the manifest from `saves_dir`. If no manifest exists yet,
/// **does not** write one — `load_or_init` does that. The caller
/// chooses which to use depending on whether they want to create a
/// fresh world or fall back to legacy.
pub fn load(saves_dir: &Path) -> Result<Option<WorldManifest>> {
    let p = manifest_path(saves_dir);
    if !p.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&p)
        .with_context(|| format!("reading world manifest at {}", p.display()))?;
    let m: WorldManifest = toml::from_str(&text)
        .with_context(|| format!("parsing world manifest at {}", p.display()))?;
    Ok(Some(m))
}

/// Write `m` to `saves_dir/world.toml`. Creates the directory if
/// missing.
pub fn save(saves_dir: &Path, m: &WorldManifest) -> Result<()> {
    std::fs::create_dir_all(saves_dir)
        .with_context(|| format!("creating save dir {}", saves_dir.display()))?;
    let p = manifest_path(saves_dir);
    let text = toml::to_string_pretty(m).context("serialising world manifest")?;
    std::fs::write(&p, text)
        .with_context(|| format!("writing world manifest to {}", p.display()))?;
    Ok(())
}

/// Load the manifest if present; otherwise pick:
///
/// * If `saves_dir/regions/` already exists, this is a legacy save —
///   write the legacy-fallback manifest (`seed = 42`,
///   `worldgen_version = 1`).
/// * Otherwise it's a brand-new world — roll a random seed at the
///   current version.
///
/// Either way, the resulting manifest is persisted before being
/// returned so subsequent loads find it.
pub fn load_or_init(saves_dir: &Path) -> Result<WorldManifest> {
    if let Some(m) = load(saves_dir)? {
        return Ok(m);
    }
    let regions_dir = saves_dir.join("regions");
    let m = if regions_dir.exists() {
        log::info!(
            "save at {} predates world.toml — writing synthetic legacy manifest \
             (seed=42, worldgen_version=1)",
            saves_dir.display()
        );
        WorldManifest::legacy_fallback()
    } else {
        WorldManifest::fresh()
    };
    save(saves_dir, &m)?;
    Ok(m)
}

/// Seconds since the Unix epoch.
fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trip_through_disk() {
        let dir = tempdir().unwrap();
        let m = WorldManifest {
            seed: 0xDEAD_BEEF_F00D_CAFE,
            worldgen_version: CURRENT_VERSION,
            created_at: 1_000_000,
        };
        save(dir.path(), &m).unwrap();
        let loaded = load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded, m);
    }

    #[test]
    fn load_missing_returns_none() {
        let dir = tempdir().unwrap();
        assert!(load(dir.path()).unwrap().is_none());
    }

    #[test]
    fn load_or_init_creates_fresh_when_empty() {
        let dir = tempdir().unwrap();
        let m = load_or_init(dir.path()).unwrap();
        assert_eq!(m.worldgen_version, CURRENT_VERSION);
        // And it was persisted.
        assert!(manifest_path(dir.path()).exists());
        // Reload returns the same value.
        let m2 = load_or_init(dir.path()).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn load_or_init_treats_existing_regions_as_legacy() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("regions")).unwrap();
        let m = load_or_init(dir.path()).unwrap();
        assert_eq!(m.seed, LEGACY_SEED);
        assert_eq!(m.worldgen_version, LEGACY_VERSION);
    }

    #[test]
    fn fresh_uses_current_version() {
        let m = WorldManifest::fresh();
        assert_eq!(m.worldgen_version, CURRENT_VERSION);
    }

    #[test]
    fn fresh_seeds_differ_in_a_small_batch() {
        // Two fresh manifests should almost always have different
        // seeds — `rand::random` is well-distributed. A coincidental
        // collision in 64 bits is vanishingly unlikely.
        let a = WorldManifest::fresh();
        let b = WorldManifest::fresh();
        assert_ne!(a.seed, b.seed);
    }
}
