//! Player physics: gravity, jumping, and AABB-vs-voxel collision.
//!
//! The single hot function is the *axis-by-axis swept AABB*. Sweeping each
//! velocity axis independently — rather than resolving all three at once —
//! avoids the classic corner-snag bug where a player coming at a wall on a
//! diagonal trajectory gets stuck on an inside edge.

pub mod sweep;
