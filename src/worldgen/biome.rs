//! Discrete biome labels and per-biome tree-placement policy.
//!
//! ### Key concepts
//!
//! - [`Biome`]: the six discrete labels assigned per column after climate
//!   evaluation. The set is deliberately small — every variant has a
//!   visually distinct surface block or tree density, so the
//!   biome boundary reads in a screenshot.
//! - [`TreeKind`]: the two canopy shapes currently implemented (Oak
//!   round-sphere; Palm spreading-frond). Stamped by the tree placement
//!   pass in `trees.rs`.
//!
//! ### Biome → tree policy
//!
//! Two private methods on [`Biome`] drive the tree pass:
//! `tree_rate_percentile` returns the probability (0–100) that a
//! `TREE_CELL_SIZE × TREE_CELL_SIZE` patch hosts a tree, and
//! `tree_kind` picks the canopy shape. Both live here rather than in
//! `trees.rs` to keep the biome enum self-contained: every piece of
//! "what does this biome look like?" logic is in one place.
//!
//! See `docs/book/content/part-4-chunk-fill/4.7-biomes.mdx` and
//! `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`.

use crate::worldgen::tuning::{TREE_RATE_FOREST, TREE_RATE_PLAINS, TREE_RATE_TROPICAL};

/// Discrete biome label assigned to each column. The set is small on
/// purpose — every variant has a distinct visual signature (different
/// surface block or noticeably different tree density), so the
/// difference between biomes reads from a screenshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Biome {
    /// Cold column. Snow on the surface; no trees grow here.
    Tundra,
    /// Cold *and* humid. Same Snow surface as Tundra but trees do
    /// grow (taiga / boreal forest analogue).
    SnowyForest,
    /// Temperate, dry. Grass surface, very sparse trees — open
    /// rolling fields.
    Plains,
    /// Temperate, humid. Grass surface, dense tree cover.
    Forest,
    /// Hot, dry. Sand surface, no trees.
    Desert,
    /// Hot, humid (new in PR 5). Grass surface, denser tree cover
    /// than Forest — placeholder for the future palm/jungle pass.
    /// Palm-shape trees are stamped here via [`TreeKind::Palm`].
    Tropical,
}

impl Biome {
    /// Probability (0..100) that a `TREE_CELL_SIZE × TREE_CELL_SIZE`
    /// patch in this biome rolls a tree. Returns `None` for biomes
    /// that never host trees (`Tundra`, `Desert`).
    ///
    /// Higher → denser forest cover. Lower → more open terrain.
    pub(crate) fn tree_rate_percentile(self) -> Option<u32> {
        match self {
            Biome::Tundra | Biome::Desert => None,
            Biome::Plains => Some(TREE_RATE_PLAINS),
            Biome::Forest | Biome::SnowyForest => Some(TREE_RATE_FOREST),
            Biome::Tropical => Some(TREE_RATE_TROPICAL),
        }
    }

    /// Which tree canopy shape to stamp in this biome's cells.
    /// Oak for temperate / boreal columns; Palm for tropical columns.
    pub(crate) fn tree_kind(self) -> TreeKind {
        match self {
            Biome::Tropical => TreeKind::Palm,
            _ => TreeKind::Oak,
        }
    }
}

/// Tree canopy shape selector.
///
/// PR 5 introduces [`Palm`] for `Tropical`; follow-up PRs may add
/// jungle / pine variants. The stamping logic for each variant lives
/// in `src/worldgen/mod.rs::stamp_tree`.
///
/// [`Palm`]: TreeKind::Palm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TreeKind {
    /// Round-sphere canopy, 5 blocks tall. Used by all temperate /
    /// boreal biomes.
    Oak,
    /// Spreading-frond crown, 7 blocks tall. Used by `Tropical`.
    Palm,
}
