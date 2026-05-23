//! Concurrent LRU cache for region data: `BuildCache<K, V>`, the
//! `InFlight` one-builder-per-key concurrency primitive, and the
//! convenience wrappers `get_fine` / `get_macro` / `peek_fine`.
//!
//! ### Concurrency model
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
//! The `loop` in [`get_or_build`] is necessary: if the builder panics,
//! `BuildClaim`'s `Drop` impl marks the slot as `aborted` and wakes all
//! waiters. A waiter re-enters the loop from the top to either claim
//! ownership of a fresh build (if no other thread got there first) or
//! wait on the next builder.

use crate::worldgen::tuning::{FINE_CACHE_CAP, MACRO_CACHE_CAP};
use lru::LruCache;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex};
use super::coords::{MacroRegionCoord, RegionCoord};
use super::data::{FineRegion, MacroRegion};

// ── Public type aliases ───────────────────────────────────────────────

/// Shared LRU cache of fine regions. `Arc`-wrapped so all concurrent chunk
/// generation threads share a single cache instance with atomic eviction.
pub type FineCache = Arc<BuildCache<RegionCoord, FineRegion>>;
/// Shared LRU cache of macro regions.
pub type MacroCache = Arc<BuildCache<MacroRegionCoord, MacroRegion>>;

// ── Cache data structure ──────────────────────────────────────────────

/// Concurrent LRU cache with one-builder-per-key semantics.
///
/// On a miss, exactly one thread builds the value; all others wait on
/// a `Condvar`. See the module-level concurrency section for the full
/// protocol.
pub struct BuildCache<K, V> {
    inner: Mutex<BuildCacheInner<K, V>>,
}

struct BuildCacheInner<K, V> {
    lru: LruCache<K, Arc<V>>,
    in_flight: HashMap<K, Arc<InFlight<V>>>,
}

struct InFlight<V> {
    state: Mutex<InFlightState<V>>,
    ready: Condvar,
}

struct InFlightState<V> {
    result: Option<Arc<V>>,
    aborted: bool,
}

struct BuildClaim<'a, K, V>
where
    K: Copy + Eq + Hash,
{
    cache: &'a BuildCache<K, V>,
    key: K,
    slot: Arc<InFlight<V>>,
    active: bool,
}

impl<K, V> BuildCache<K, V>
where
    K: Copy + Eq + Hash,
{
    fn new(cap: NonZeroUsize) -> Self {
        Self {
            inner: Mutex::new(BuildCacheInner {
                lru: LruCache::new(cap),
                in_flight: HashMap::new(),
            }),
        }
    }

    /// Look up `key` **without building on miss** and without promoting
    /// the entry's LRU position. Returns `None` if the entry is not cached.
    pub(super) fn peek(&self, key: &K) -> Option<Arc<V>> {
        self.inner
            .lock()
            .expect("region cache mutex poisoned")
            .lru
            .peek(key)
            .cloned()
    }
}

impl<K, V> Drop for BuildClaim<'_, K, V>
where
    K: Copy + Eq + Hash,
{
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        self.cache
            .inner
            .lock()
            .expect("region cache mutex poisoned")
            .in_flight
            .remove(&self.key);

        let mut state = self
            .slot
            .state
            .lock()
            .expect("region build state mutex poisoned");
        state.aborted = true;
        self.slot.ready.notify_all();
    }
}

// ── Cache constructors ────────────────────────────────────────────────

/// Build a fresh, capped fine LRU.
pub fn fresh_fine_cache() -> FineCache {
    Arc::new(BuildCache::new(
        NonZeroUsize::new(FINE_CACHE_CAP).expect("FINE_CACHE_CAP must be > 0"),
    ))
}

/// Build a fresh, capped macro LRU.
pub fn fresh_macro_cache() -> MacroCache {
    Arc::new(BuildCache::new(
        NonZeroUsize::new(MACRO_CACHE_CAP).expect("MACRO_CACHE_CAP must be > 0"),
    ))
}

// ── Public access functions ───────────────────────────────────────────

/// Look up `coord` in `cache`; build it on miss. The build runs outside
/// the lock, but only one worker builds a given cold key. Other workers
/// wait for that result instead of rebuilding the same region.
pub fn get_fine<F>(cache: &FineCache, coord: RegionCoord, build: F) -> Arc<FineRegion>
where
    F: FnOnce() -> FineRegion,
{
    get_or_build(cache, coord, build)
}

/// Look up `coord` in the fine cache **without building on miss**.
/// Returns `None` if the entry is not cached. Unlike `get_fine`, this
/// does NOT promote the entry's LRU position (uses `LruCache::peek`),
/// so repeated peeks from the hydrology stitcher don't reshape the
/// eviction order.
pub fn peek_fine(cache: &FineCache, coord: RegionCoord) -> Option<Arc<FineRegion>> {
    cache.peek(&coord)
}

pub fn get_macro<F>(cache: &MacroCache, coord: MacroRegionCoord, build: F) -> Arc<MacroRegion>
where
    F: FnOnce() -> MacroRegion,
{
    get_or_build(cache, coord, build)
}

// ── Core get-or-build protocol ────────────────────────────────────────

/// Core cache lookup + one-builder-per-key build protocol.
///
/// Returns the cached value if present, otherwise runs `build` once and
/// caches the result. If another thread is already building this key, the
/// caller waits on a `Condvar` inside the in-flight slot until that build
/// completes (or aborts).
///
/// **Why the `loop`:** if the builder panics, `BuildClaim`'s `Drop` impl
/// sets `aborted = true` and wakes all waiters. A waiter that wakes to an
/// aborted slot continues to the top of the loop to either claim ownership
/// of a fresh rebuild (if no other thread has done so) or wait for the next
/// builder to succeed. Without the loop, an aborted build would leave all
/// waiters stuck with no value.
fn get_or_build<K, V, F>(cache: &Arc<BuildCache<K, V>>, key: K, build: F) -> Arc<V>
where
    K: Copy + Eq + Hash,
    F: FnOnce() -> V,
{
    let mut build = Some(build);
    loop {
        let slot = {
            let mut inner = cache.inner.lock().expect("region cache mutex poisoned");
            if let Some(value) = inner.lru.get(&key) {
                return value.clone();
            }
            if let Some(slot) = inner.in_flight.get(&key) {
                Some(slot.clone())
            } else {
                let slot = Arc::new(InFlight {
                    state: Mutex::new(InFlightState {
                        result: None,
                        aborted: false,
                    }),
                    ready: Condvar::new(),
                });
                inner.in_flight.insert(key, slot.clone());
                None
            }
        };

        if let Some(slot) = slot {
            let mut state = slot
                .state
                .lock()
                .expect("region build state mutex poisoned");
            while state.result.is_none() && !state.aborted {
                state = slot
                    .ready
                    .wait(state)
                    .expect("region build state mutex poisoned");
            }
            if let Some(value) = &state.result {
                return value.clone();
            }
            continue;
        }

        let slot = {
            let inner = cache.inner.lock().expect("region cache mutex poisoned");
            inner
                .in_flight
                .get(&key)
                .expect("new in-flight region build must be registered")
                .clone()
        };
        let mut claim = BuildClaim {
            cache,
            key,
            slot: slot.clone(),
            active: true,
        };
        let value = Arc::new(build.take().expect("region build closure consumed")());
        {
            let mut inner = cache.inner.lock().expect("region cache mutex poisoned");
            inner.lru.put(key, value.clone());
            inner.in_flight.remove(&key);
        }
        {
            let mut state = slot
                .state
                .lock()
                .expect("region build state mutex poisoned");
            state.result = Some(value.clone());
        }
        claim.active = false;
        slot.ready.notify_all();
        return value;
    }
}
