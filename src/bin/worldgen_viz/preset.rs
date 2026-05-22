//! Preset library: named WorldgenConfig snapshots saved as RON files
//! under `assets/worldgen/presets/`. Each preset is a single
//! `<name>.ron` file holding the full config; an optional
//! `<name>.notes.md` sibling carries freeform commentary the user
//! can attach to a tuning state (what worked, what to try next).
//!
//! The default config (`assets/worldgen/default.ron`) sits one
//! directory up and is treated as the immutable factory baseline —
//! always loadable, never overwritten by the viz.

use oxium::worldgen::config::WorldgenConfig;
use std::path::{Path, PathBuf};

const ASSETS_SUBDIR: &str = "assets/worldgen/presets";

/// One preset on disk.
#[derive(Debug, Clone)]
pub struct PresetEntry {
    pub name: String,
    pub config_path: PathBuf,
    pub notes_path: PathBuf,
}

pub fn presets_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ASSETS_SUBDIR)
}

/// Scan `presets_root()` for `*.ron` files. Notes paths are computed
/// (they may not exist yet — caller checks before reading).
pub fn list_presets() -> Vec<PresetEntry> {
    let mut out = Vec::new();
    let root = presets_root();
    let read = match std::fs::read_dir(&root) {
        Ok(r) => r,
        Err(_) => return out,
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ron") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let notes_path = root.join(format!("{stem}.notes.md"));
        out.push(PresetEntry {
            name: stem,
            config_path: path,
            notes_path,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Load a preset's WorldgenConfig from disk.
pub fn load(entry: &PresetEntry) -> anyhow::Result<WorldgenConfig> {
    let text = std::fs::read_to_string(&entry.config_path)?;
    let cfg: WorldgenConfig = ron::from_str(&text)?;
    Ok(cfg)
}

/// Load notes for a preset (empty string if no notes file).
pub fn load_notes(entry: &PresetEntry) -> String {
    std::fs::read_to_string(&entry.notes_path).unwrap_or_default()
}

/// Save (or overwrite) a preset's config under the given name.
/// Returns the resulting entry.
pub fn save(name: &str, cfg: &WorldgenConfig) -> anyhow::Result<PresetEntry> {
    let sanitized = sanitize_name(name);
    if sanitized.is_empty() {
        anyhow::bail!("preset name resolves to empty after sanitising");
    }
    let root = presets_root();
    std::fs::create_dir_all(&root)?;
    let config_path = root.join(format!("{sanitized}.ron"));
    let notes_path = root.join(format!("{sanitized}.notes.md"));
    let serialized = ron::ser::to_string_pretty(cfg, ron::ser::PrettyConfig::default())?;
    std::fs::write(&config_path, serialized)?;
    Ok(PresetEntry {
        name: sanitized,
        config_path,
        notes_path,
    })
}

/// Save (or overwrite) notes for a preset.
pub fn save_notes(entry: &PresetEntry, notes: &str) -> anyhow::Result<()> {
    if notes.is_empty() {
        // Remove the file rather than leaving an empty notes file behind.
        let _ = std::fs::remove_file(&entry.notes_path);
        return Ok(());
    }
    std::fs::write(&entry.notes_path, notes)?;
    Ok(())
}

/// Delete a preset (config + notes).
pub fn delete(entry: &PresetEntry) -> anyhow::Result<()> {
    let _ = std::fs::remove_file(&entry.notes_path);
    std::fs::remove_file(&entry.config_path)?;
    Ok(())
}

/// UI state for the preset-library widget. Holds the most recent
/// scan of disk + the currently-active preset name + the in-edit
/// notes buffer + the "save current as" name input. Kept in AppState
/// so the widget is stateless beyond what's drawn each frame.
pub struct PresetUi {
    pub entries: Vec<PresetEntry>,
    pub new_name: String,
    pub active_name: Option<String>,
    /// Notes for the active preset, edited in-place and flushed to
    /// disk on save / when the active preset changes.
    pub notes_buffer: String,
    /// Notes buffer's content when last loaded from disk; used to
    /// decide whether the buffer is dirty.
    pub notes_pristine: String,
}

impl PresetUi {
    pub fn new() -> Self {
        Self {
            entries: list_presets(),
            new_name: String::new(),
            active_name: None,
            notes_buffer: String::new(),
            notes_pristine: String::new(),
        }
    }

    pub fn refresh_entries(&mut self) {
        self.entries = list_presets();
    }

    /// Mark `name` as the active preset and load its notes. Pass
    /// `None` to clear.
    pub fn activate(&mut self, name: Option<String>) {
        self.active_name = name;
        let notes = match &self.active_name {
            Some(n) => self
                .entries
                .iter()
                .find(|e| e.name == *n)
                .map(load_notes)
                .unwrap_or_default(),
            None => String::new(),
        };
        self.notes_buffer = notes.clone();
        self.notes_pristine = notes;
    }

    pub fn notes_dirty(&self) -> bool {
        self.notes_buffer != self.notes_pristine
    }

    pub fn mark_notes_saved(&mut self) {
        self.notes_pristine = self.notes_buffer.clone();
    }
}

/// Collapse a user-supplied preset name to a filesystem-safe form.
/// Only ASCII alphanumerics and `_` survive — every other character
/// becomes a `-`, runs of separators collapse to a single `-`, and
/// leading/trailing `-` is trimmed. This blocks directory traversal
/// (`../`) and similar by construction: even after sanitising,
/// what's left can only be appended to `presets_root()` as a single
/// filename component.
fn sanitize_name(raw: &str) -> String {
    let mut s = String::with_capacity(raw.len());
    let mut last_was_sep = true;
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            s.push(c.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            s.push('-');
            last_was_sep = true;
        }
    }
    s.trim_end_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_unsafe_chars() {
        assert_eq!(sanitize_name("Alpine v2"), "alpine-v2");
        assert_eq!(sanitize_name("../etc/passwd"), "etc-passwd");
        assert_eq!(sanitize_name("ok_name"), "ok_name");
        assert_eq!(sanitize_name("with.dot"), "with-dot");
        assert_eq!(sanitize_name("a..b"), "a-b");
        assert_eq!(sanitize_name("-alpine-"), "alpine");
    }

    #[test]
    fn sanitize_empty_on_garbage() {
        assert_eq!(sanitize_name("!!!"), "");
        assert_eq!(sanitize_name(""), "");
    }
}
