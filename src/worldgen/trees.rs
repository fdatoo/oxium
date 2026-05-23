//! Tree placement: per-cell deterministic rolls, `TreeKind::{Oak,
//! Palm}`, palm-shape stamping.
//!
//! Tree placement is **cell-based**: the world is divided into
//! `TREE_CELL_SIZE × TREE_CELL_SIZE` (8×8) columns cells. Each cell
//! independently rolls a biome-weighted probability (`TREE_RATE_*`)
//! to decide whether a tree should appear there, and if so, jitters
//! the trunk position deterministically within a `TREE_MARGIN` border
//! band. Cells in adjacent chunks are also evaluated so cross-chunk
//! trees (trunk in a neighbour, leaves in this chunk) are placed
//! correctly.
//!
//! ### Implementation location
//!
//! The entry point `tree_in_cell(seed, cell_x, cell_z, generator)` and
//! the leaf-stamping logic for both `Oak` (round sphere canopy) and
//! `Palm` (spreading-frond crown) live in `src/worldgen/mod.rs` in the
//! `fill_chunk` post-pass. This file contains only the module-level
//! documentation and the `TreeKind` enum (introduced in PR 5).
//!
//! ### Design notes
//!
//! - Determinism: `hash::mix(seed, &[cell_x, cell_z, salt])` drives all
//!   per-cell rolls so the same seed always produces the same forest.
//! - Cross-chunk correctness: `fill_chunk` expands the scan radius to
//!   `TREE_MARGIN` extra cells on each side so fronds and canopy
//!   overhangs that originate outside the chunk boundary are still drawn.
//! - Biome blending: tree density interpolates across biome boundaries
//!   over `TREE_BLEND_WIDTH` blocks so forest/plains edges aren't a
//!   hard pixel step.
//!
//! See `docs/book/content/part-4-chunk-fill/4.9-trees.mdx` for the
//! visual design rationale.
