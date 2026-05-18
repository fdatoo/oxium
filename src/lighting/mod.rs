//! Per-voxel light propagation (sky + emissive block sources).
//!
//! Two BFS flood-fills, run on a worker thread:
//!
//! - **Sky light** drops 15 from the world ceiling and falls off by 1 per
//!   non-opaque step (water costs 3).
//! - **Block light** spreads outward from blocks with `info.emission > 0`.
//!
//! Both are *recompute-on-dirty*: an edit reseeds and re-runs the BFS instead
//! of doing an incremental update. The recompute approach trades a few extra
//! milliseconds of worker time for ~10× less code complexity — see the design
//! spec's "Why recompute over incremental" table.
