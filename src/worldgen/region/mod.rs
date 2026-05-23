//! Region cache: lazy, deterministic, two-level memoisation.
//!
//! The new worldgen wants per-region precomputation (flow accumulation,
//! cave system rolls) that's too expensive to redo per chunk. This
//! module owns the two LRU caches — one for fine regions (512-block
//! granularity) and one for macro regions (8192-block granularity for
//! the trunk-river pass).
//!
//! The cache is **pure in `(seed, coord)`**: the same key always
//! produces the same value. Evicted entries rebuild on access; rebuilds
//! are byte-identical to the original. No serialisation across
//! runs — caches start cold on every program start.
//!
//! ### Submodule layout
//!
//! | Submodule  | Contents                                                   |
//! |------------|------------------------------------------------------------|
//! | `coords`   | `RegionCoord`, `MacroRegionCoord`                          |
//! | `data`     | `FineRegion`, `MacroRegion`, river + cave payload types    |
//! | `cache`    | `BuildCache` + `InFlight` concurrency core (Mutex+Condvar) |
//! | `bitset`   | `bitset_get` / `bitset_set` bit-packed boolean helpers     |
//!
//! ### BuildCache concurrency model
//!
//! `BuildCache<K, V>` is a **dual-Mutex + Condvar** design. The outer
//! `Mutex<BuildCacheInner>` guards the LRU map and a `HashMap` of
//! in-flight builds. The inner `Mutex<InFlightState>` + `Condvar` guards
//! the result of a single in-progress build.
//!
//! The protocol for a miss:
//! 1. Lock the outer mutex; check the LRU (hit → return immediately).
//! 2. Check `in_flight`: if another worker is already building this key,
//!    grab a clone of the `Arc<InFlight>` slot, **drop the outer lock**,
//!    then wait on the `Condvar` inside the slot.
//! 3. If no in-flight entry exists, insert one and drop the outer lock,
//!    then run the build closure outside any lock.
//! 4. On build completion, re-acquire the outer lock to install the
//!    result into the LRU and remove the in-flight entry; then signal
//!    all waiters through the slot's `Condvar`.
//!
//! The `loop` in `get_or_build` is necessary: if the builder panics,
//! `BuildClaim`'s `Drop` impl marks the slot as `aborted` and wakes all
//! waiters. A waiter re-enters the loop from the top to either claim
//! ownership of a fresh build (if no other thread got there first) or
//! wait on the next builder.
//!
//! In PR 1 the cache types exist but the build functions return
//! default-filled placeholders. PRs 2–4 fill them in: PR 2 populates
//! the heightmap samples, PR 3 the river network, PR 4 the cave systems.
//!
//! See `docs/book/content/part-5-engineering/5.2-region-cache.mdx` and
//! `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`.

pub mod bitset;
pub mod cache;
pub mod coords;
pub mod data;

#[cfg(test)]
mod tests;

// ── Re-exports — every previously-public item stays accessible at
//    `worldgen::region::X` so callers (hydrology.rs, caves.rs, mod.rs,
//    tests) require no import changes.

// coords
pub use coords::{MacroRegionCoord, RegionCoord};

// data
pub use data::{
    build_fine_region_placeholder, build_macro_region_placeholder, Chamber, CavePool, CaveSystem,
    Entrance, EntranceKind, FineRegion, MacroRegion, RiverSegment, RiverSegmentKind, Tunnel,
};

// cache
pub use cache::{
    fresh_fine_cache, fresh_macro_cache, get_fine, get_macro, peek_fine, BuildCache, FineCache,
    MacroCache,
};

// bitset
pub use bitset::{bitset_get, bitset_set};
