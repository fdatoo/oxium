//! Graph-based cave systems plus ambient noise carvers.
//!
//! Each 512×512-block fine region deterministically rolls 0–3 cave systems.
//! A system is a small graph of ellipsoidal chambers connected by spline
//! tunnels (minimum spanning tree + 1–2 loop edges), placed inside a
//! per-system bounding box at one of three depth bands:
//!
//! - **Shallow** (`CAVE_BAND_SHALLOW`): near surface, high entrance rate.
//! - **Middle** (`CAVE_BAND_MIDDLE`): mid-depth, moderate entrance rate.
//! - **Deep** (`CAVE_BAND_DEEP`): very deep, rare entrances.
//!
//! Each system is assigned a style ([`CaveStyle`]) that parametrises
//! chamber count, radii, and tunnel widths. Adjacent systems in different
//! bands may be linked by vertical connectors.
//!
//! ### Carving at chunk fill time
//!
//! Carving is deferred to chunk fill. For each system whose bounding box
//! intersects the chunk, signed SDF functions (`cave_sdf`, `trunks_sdf`,
//! `entrance_sdf`) return a positive intensity wherever a voxel lies inside
//! a chamber or tunnel. The fill loop subtracts these from the base density
//! via `smin` (smooth-min), so cave walls have soft, chamfered edges.
//!
//! The `CAVE_SURFACE_BUFFER` guard preserves the grass cap by refusing to
//! apply the graph SDFs within `CAVE_SURFACE_BUFFER` blocks of `h_pre`.
//! Entrance features (sinkholes, cliff mouths, skylights) bypass this guard
//! via a separate `entrance_sdf` gate so intentional cave openings can still
//! breach the surface.
//!
//! ### Ambient noise carvers
//!
//! Alongside the graph systems, two MC-derived ambient carvers fire on every
//! underground voxel:
//!
//! - **Cheese** ([`cheese_contribution`]): threshold-sampled 3D FBM produces
//!   Swiss-cheese-like isolated pockets. A `cave_layer²` stratification term
//!   concentrates carving at specific depth bands.
//! - **Terasology ambient** ([`terasology_ambient`]): two independent FBM
//!   channels are intersected; carving occurs where both are near zero —
//!   geometrically a disk in 2D noise space. Disk radius grows with depth.
//!
//! Both layers are sampled via a `CarverEvaluator` corner-lattice trilerp
//! (same 9³ pattern as `density_graph::CellEvaluator`) to avoid per-voxel
//! FBM cost.
//!
//! See `docs/superpowers/specs/2026-05-21-cave-system-overhaul-design.md`,
//! `docs/book/content/part-3-region-build/3.6-cave-systems.mdx`, and
//! `docs/book/content/part-4-chunk-fill/4.3-composing-caves.mdx`.

pub mod connectors;
pub mod noise_carvers;
pub mod pools;
pub mod sdf;
pub mod style;
pub mod system;

// Re-export the public API surface so external callers use `caves::*`.
pub use connectors::{build_trunks, build_vertical_connectors};
pub use noise_carvers::{
    CarverEvaluator, NoiseCarvers, cheese_contribution, pillar_contribution, terasology_ambient,
};
pub use sdf::{
    any_system_y_in_range, cave_air, cave_sdf, entrance_air, entrance_sdf, smin, trunks_sdf,
};
pub use style::{CaveStyle, DepthBand, pick_style};
pub use system::build_systems_for_region;

#[cfg(test)]
mod tests;
