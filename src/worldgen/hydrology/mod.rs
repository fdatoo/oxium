//! Hydrology: D8 flow accumulation at fine (8 m) and macro (64 m)
//! resolution, sink-fill into lakes, trunk-river injection, river
//! segment extraction, valley carving.
//!
//! ### Algorithm summary
//!
//! A region's hydrology is built on a 5 × 5-region window of fine
//! cells (2-region halo on each side, total 320 × 320 cells at 8 m
//! per cell). Operating on the full window means flow that crosses
//! into the region from outside is computed correctly — and two
//! adjacent regions that share a halo see the same flow direction
//! at the shared border (with the caveat that closed basins bigger
//! than the window terrace at region boundaries; the macro pass
//! catches the big ones).
//!
//! 1. **Sample h_pre** at every fine cell in the window.
//! 2. **Sink fill** (priority-queue Planchon-Darboux) — every cell
//!    ends up with a non-strictly-decreasing path to the window
//!    boundary. Cells whose filled height exceeds their natural
//!    height are flagged as lake water; the filled value is the
//!    lake's rim elevation.
//! 3. **Macro trunk injection** — for each macro cell flagged as a
//!    trunk river by the macro pass, inject `macro_acc * 64` units
//!    of starting accumulation at the fine cell at the macro cell's
//!    center. Trunk drainage from outside the visible fine window
//!    appears as already-fat rivers entering the region.
//! 4. **D8 flow direction** on the filled heightmap → each cell
//!    points to its single steepest-downhill neighbour. Lake-rim
//!    cells get an outflow direction (over the rim toward the
//!    lowest neighbour outside the lake).
//! 5. **Flow accumulation** — topological sort cells by filled
//!    height descending; each cell donates its own area plus any
//!    injected trunk units to its downstream neighbour. O(N).
//! 6. **River cells** = cells where accumulation ≥ `RIVER_THRESH`.
//!    Width = `clamp(sqrt(acc) * SCALE, MIN, MAX)`.
//! 7. **Extract region-interior data** into the cache (boxed
//!    slices, halo discarded).
//! 8. **Build river segment list** by walking each river cell to
//!    its downstream neighbour — the cell-center to next-cell-center
//!    polyline plus a domain-warped meander offset.
//!
//! ### Macro pass
//!
//! Same algorithm at coarser resolution (64 m cells, 1-macro-region
//! halo). Operates on 128 × 128 cells per macro region (window 3 ×
//! 128 = 384 cells); sees a 24 km drainage horizon — enough for
//! continental-scale trunk rivers and inland-basin lakes.
//!
//! See `docs/book/content/part-3-region-build/3.4-hydrology.mdx`,
//! `docs/book/content/part-3-region-build/3.5-rivers-lakes.mdx`, and
//! `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`.

pub mod fine_pass;
pub mod grid;
pub mod lakes;
pub mod macro_pass;
pub mod rivers;
pub mod valley;

// Re-export the public API so callers use `hydrology::build_fine_hydro`
// rather than `hydrology::fine_pass::build_fine_hydro`.
pub use fine_pass::{NeighbourEdges, build_fine_hydro, gather_neighbour_edges};
pub use grid::{DIR_NONE, DIR_OFFSETS};
pub use lakes::lake_rim_at;
pub use macro_pass::build_macro_region;
pub(crate) use valley::{for_each_segment, perpendicular_distance};
pub use valley::{valley_carve, valley_grid};

#[cfg(test)]
mod tests;
