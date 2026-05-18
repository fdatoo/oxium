//! Game-entity ECS (player, camera, sun, cursor). Chunks live in `voxel::World`.
//!
//! We use `hecs` — a tiny archetypal ECS — and a hand-written per-frame
//! schedule (no macros, no automatic parallelism). The chunk grid is
//! intentionally *not* an ECS resource: voxel data is too cache-hot and
//! co-accessed by jobs running off the main thread, so it lives in its own
//! [`crate::voxel::World`] container.
