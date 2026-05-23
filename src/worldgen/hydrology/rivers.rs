//! River segment classification and helpers.
//!
//! This file is intentionally sparse in part 1/2 of the hydrology-split
//! refactor. In part 2/2, `build_segments_from_fine` will be extracted
//! from `fine_pass::build_fine_hydro` and placed here.
//!
//! The segment-building loop classifies each river cell into one of three
//! kinds based on terrain drop to the downstream neighbour:
//!
//! - `Waterfall` — drop ≥ 12 blocks
//! - `Rapid` — drop ≥ 5 blocks
//! - `Channel` — all other flowing cells
//!
//! See `docs/book/content/part-3-region-build/3.5-rivers-lakes.mdx`.
