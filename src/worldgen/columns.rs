//! Per-column terrain summary: [`ColumnData`].
//!
//! A [`ColumnData`] is computed once per `(wx, wz)` column at the start of
//! `fill_chunk` and reused by block selection, tree placement, and light
//! surface computation. Keeping it as a small struct avoids re-running
//! expensive noise samples for each use.
//!
//! See `docs/book/content/part-4-chunk-fill/4.1-column-data.mdx` and
//! `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`.

use crate::worldgen::Biome;

/// Per-column biome + geometry summary used by both `fill_chunk` and
/// `add_trees` so block selection and tree placement stay in sync.
///
/// Computed once per column by `Generator::column_data_with`; read-only
/// everywhere else. Cheap to copy (`Copy` impl) — 32 bytes.
#[derive(Debug, Clone, Copy)]
pub struct ColumnData {
    /// Surface height in world Y, post-carve, clamped.
    pub height: i32,
    /// Pre-carve surface height (`h_pre` from the heightmap pass).
    /// Used by the terasology ambient carver's surface suppression:
    /// the carver doesn't punch through into the top few blocks of
    /// the terrain even when the height has been valley-carved lower.
    pub h_pre: f32,
    /// True if the column's `h_pre` slope exceeds `CLIFF_SLOPE_THRESH`
    /// AND its elevation is at/above `CLIFF_MIN_HEIGHT`. Cliff
    /// columns expose stone faces directly, skipping the dirt/grass cap.
    pub is_cliff: bool,
    /// Jitter-perturbed `desertness` noise value in roughly `[0, 1]`.
    /// Used by the sand/grass transition band: inside the band on the
    /// grass side of the desert boundary, the surface block is rolled
    /// stochastically against this value. Higher → more sand bleed
    /// into the neighboring biome's surface.
    pub desertness: f32,
    /// Discrete biome label derived from temperature, humidity, and
    /// the desert mask, with threshold perturbation applied.
    pub biome: Biome,
    /// Unified water-surface Y: `Some(y)` means this column is submerged
    /// and the topmost Water voxel sits at world Y == y. Priority:
    /// river > lake > ocean > `None`. River priority is patched in by
    /// `fill_chunk` after the river grid is built; other callers receive
    /// only the lake/ocean classification computed here.
    pub water_surface_y: Option<i32>,
}
