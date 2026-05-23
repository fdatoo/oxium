//! Region cache: lazy, deterministic, two-level memoization.
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
//! In PR 1 the cache types exist but the build functions return
//! default-filled placeholders. PRs 2–4 fill them in: PR 2 populates
//! the heightmap samples, PR 3 the river network, PR 4 the cave
//! systems.

use crate::worldgen::tuning::*;
use lru::LruCache;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex};

// ── Region coordinate keys ────────────────────────────────────────────

/// Coordinate of a fine region (512 × 512 blocks). `(world_x /
/// FINE_REGION_SIZE).floor()` etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionCoord {
    pub x: i32,
    pub z: i32,
}

impl RegionCoord {
    /// The fine region containing `(wx, wz)`.
    pub fn containing(wx: i32, wz: i32) -> Self {
        Self {
            x: wx.div_euclid(FINE_REGION_SIZE),
            z: wz.div_euclid(FINE_REGION_SIZE),
        }
    }

    /// World coordinate of the region's south-west corner.
    pub fn origin(self) -> (i32, i32) {
        (self.x * FINE_REGION_SIZE, self.z * FINE_REGION_SIZE)
    }
}

/// Coordinate of a macro region (8192 × 8192 blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroRegionCoord {
    pub x: i32,
    pub z: i32,
}

impl MacroRegionCoord {
    pub fn containing(wx: i32, wz: i32) -> Self {
        Self {
            x: wx.div_euclid(MACRO_REGION_SIZE),
            z: wz.div_euclid(MACRO_REGION_SIZE),
        }
    }

    pub fn origin(self) -> (i32, i32) {
        (self.x * MACRO_REGION_SIZE, self.z * MACRO_REGION_SIZE)
    }
}

// ── Region data ───────────────────────────────────────────────────────

/// What the fine cache stores for one 512×512 region.
///
/// In PR 1 these fields are empty placeholders; PRs 2–4 populate
/// them with real data. Kept inside `Box<[…]>` rather than `Vec<…>` so
/// the size is fixed at compile time and we can rely on cheap
/// `Clone`-free sharing through `Arc<FineRegion>`.
#[derive(Debug)]
pub struct FineRegion {
    pub coord: RegionCoord,
    /// Pre-river heightmap samples on the region's fine grid, one
    /// sample per `FINE_CELL` × `FINE_CELL` block. Populated by PR 2.
    /// Length = `FINE_CELLS_PER_REGION * FINE_CELLS_PER_REGION`.
    pub h_pre: Box<[i16]>,
    /// D8 downstream direction at each fine cell. 0..=7 are compass
    /// directions; 8 = no downhill neighbour (sink). Populated by
    /// PR 3.
    pub flow_dir: Box<[u8]>,
    /// Upstream flow accumulation. Populated by PR 3.
    pub flow_acc: Box<[u32]>,
    /// Bit set: true if the fine cell is a river. Populated by PR 3.
    pub is_river: Box<[u8]>,
    /// Bit set: true if the fine cell is a lake interior. Populated by
    /// PR 3.
    pub is_lake: Box<[u8]>,
    /// River width per fine cell (0.0 if not a river). Populated by
    /// PR 3.
    pub width: Box<[f32]>,
    /// Lake rim elevation per fine cell (only meaningful where
    /// `is_lake` is true). Populated by PR 3.
    pub lake_rim: Box<[i16]>,
    /// River segments derived from the fine flow field. Populated by
    /// PR 3.
    pub segments: Vec<RiverSegment>,
    /// Cave systems whose primary anchor lives in this region. Their
    /// bounding boxes can spill into neighbours; chunk fill consults
    /// the 3×3 region neighborhood. Populated by PR 4.
    pub cave_systems: Vec<CaveSystem>,
}

impl FineRegion {
    /// An empty `FineRegion` with all data buffers zero-filled and
    /// the coord set. Callers populate the fields via the hydrology
    /// and (PR 4) caves builders.
    pub fn empty(coord: RegionCoord) -> Self {
        let n = (FINE_CELLS_PER_REGION * FINE_CELLS_PER_REGION) as usize;
        let bitset_bytes = n.div_ceil(8);
        Self {
            coord,
            h_pre: vec![0i16; n].into_boxed_slice(),
            flow_dir: vec![8u8; n].into_boxed_slice(),
            flow_acc: vec![0u32; n].into_boxed_slice(),
            is_river: vec![0u8; bitset_bytes].into_boxed_slice(),
            is_lake: vec![0u8; bitset_bytes].into_boxed_slice(),
            width: vec![0.0f32; n].into_boxed_slice(),
            lake_rim: vec![0i16; n].into_boxed_slice(),
            segments: Vec::new(),
            cave_systems: Vec::new(),
        }
    }

    /// Linear index of the fine cell at integer offsets `(ix, iz)`
    /// within the region. Caller must ensure `0 <= ix, iz <
    /// FINE_CELLS_PER_REGION`.
    #[inline]
    pub fn cell_index(ix: i32, iz: i32) -> usize {
        debug_assert!(ix >= 0 && ix < FINE_CELLS_PER_REGION);
        debug_assert!(iz >= 0 && iz < FINE_CELLS_PER_REGION);
        (iz * FINE_CELLS_PER_REGION + ix) as usize
    }
}

/// What the macro cache stores. Populated by PR 3.
#[derive(Debug)]
pub struct MacroRegion {
    pub coord: MacroRegionCoord,
    pub flow_dir: Box<[u8]>,
    pub flow_acc: Box<[u32]>,
    pub is_trunk: Box<[u8]>,
    pub is_lake: Box<[u8]>,
    pub lake_rim: Box<[i16]>,
}

impl MacroRegion {
    fn empty(coord: MacroRegionCoord) -> Self {
        let n = (MACRO_CELLS_PER_REGION * MACRO_CELLS_PER_REGION) as usize;
        let bitset_bytes = n.div_ceil(8);
        Self {
            coord,
            flow_dir: vec![8u8; n].into_boxed_slice(),
            flow_acc: vec![0u32; n].into_boxed_slice(),
            is_trunk: vec![0u8; bitset_bytes].into_boxed_slice(),
            is_lake: vec![0u8; bitset_bytes].into_boxed_slice(),
            lake_rim: vec![0i16; n].into_boxed_slice(),
        }
    }

    #[inline]
    pub fn cell_index(ix: i32, iz: i32) -> usize {
        debug_assert!(ix >= 0 && ix < MACRO_CELLS_PER_REGION);
        debug_assert!(iz >= 0 && iz < MACRO_CELLS_PER_REGION);
        (iz * MACRO_CELLS_PER_REGION + ix) as usize
    }
}

// ── River + cave system payloads ──────────────────────────────────────

/// One linear segment of a river — runs from cell center `(from)` to
/// cell center `(to)` at the cell's width and depth. PR 3 populates
/// per-region segment lists; the kd-tree spatial index is built lazily
/// on first `valley_carve` query.
#[derive(Debug, Clone, Copy)]
pub struct RiverSegment {
    /// Centerline endpoints in world coordinates.
    pub from: (i32, i32),
    pub to: (i32, i32),
    pub width: f32,
    /// True for an ocean-mouth segment (flared by `MOUTH_FLARE_MULT`).
    pub mouth: bool,
}

/// Pre-built cave system. Stored in the region cache; carving happens
/// at chunk fill time.
#[derive(Debug, Clone)]
pub struct CaveSystem {
    /// World-space axis-aligned bounding box, inclusive.
    pub bb_min: glam::IVec3,
    pub bb_max: glam::IVec3,
    pub chambers: Vec<Chamber>,
    pub tunnels: Vec<Tunnel>,
    pub entrances: Vec<Entrance>,
    /// Style rolled once per system, drives chamber/tunnel parameters.
    pub style: crate::worldgen::caves::CaveStyle,
    /// Optional cross-region trunk to a neighbour-region cave system.
    /// Populated in PR3.3 by `build_trunks`.
    pub trunk: Option<Tunnel>,
    /// In-region vertical connectors between this system and adjacent-band
    /// systems in the same region (Shallow↔Middle, Middle↔Deep).
    /// Populated in PR3.2 by `build_vertical_connectors`.
    pub vertical_connectors: Vec<Tunnel>,
}

/// One chamber — an ellipsoid of air. Radii independent per axis.
#[derive(Debug, Clone, Copy)]
pub struct Chamber {
    pub center: glam::Vec3,
    pub radii: glam::Vec3,
}

/// One tunnel — a Catmull-Rom spline through 2–4 control points.
/// The capsule along this spline is carved out of the world.
#[derive(Debug, Clone)]
pub struct Tunnel {
    pub control_points: Vec<glam::Vec3>,
    pub radius: f32,
}

/// One surface entrance feature attached to a chamber.
#[derive(Debug, Clone, Copy)]
pub struct Entrance {
    pub chamber_idx: u32,
    pub kind: EntranceKind,
    /// Anchor point at the surface in world coordinates.
    pub surface: glam::IVec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntranceKind {
    Sinkhole,
    CliffMouth,
    Skylight,
}

// ── Bitset helpers ────────────────────────────────────────────────────

/// Read bit `i` from a packed bytestring.
#[inline]
pub fn bitset_get(bytes: &[u8], i: usize) -> bool {
    (bytes[i >> 3] >> (i & 7)) & 1 != 0
}

/// Set bit `i` in a packed bytestring.
#[inline]
pub fn bitset_set(bytes: &mut [u8], i: usize, v: bool) {
    let mask = 1u8 << (i & 7);
    if v {
        bytes[i >> 3] |= mask;
    } else {
        bytes[i >> 3] &= !mask;
    }
}

// ── Cache infrastructure ──────────────────────────────────────────────

/// LRU cache of fine regions. Shared so concurrent chunk jobs can hit it.
pub type FineCache = Arc<BuildCache<RegionCoord, FineRegion>>;
pub type MacroCache = Arc<BuildCache<MacroRegionCoord, MacroRegion>>;

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

    fn peek(&self, key: &K) -> Option<Arc<V>> {
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

/// Convenience: build a fine region for `coord` filled with PR 1
/// placeholders. PRs 2–4 replace this with real population code.
pub fn build_fine_region_placeholder(coord: RegionCoord) -> FineRegion {
    FineRegion::empty(coord)
}

pub fn build_macro_region_placeholder(coord: MacroRegionCoord) -> MacroRegion {
    MacroRegion::empty(coord)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_coord_containing_is_floor_divide() {
        assert_eq!(RegionCoord::containing(0, 0), RegionCoord { x: 0, z: 0 });
        assert_eq!(
            RegionCoord::containing(511, 511),
            RegionCoord { x: 0, z: 0 }
        );
        assert_eq!(
            RegionCoord::containing(512, 512),
            RegionCoord { x: 1, z: 1 }
        );
        assert_eq!(
            RegionCoord::containing(-1, -1),
            RegionCoord { x: -1, z: -1 }
        );
        assert_eq!(
            RegionCoord::containing(-FINE_REGION_SIZE, -FINE_REGION_SIZE),
            RegionCoord { x: -1, z: -1 }
        );
    }

    #[test]
    fn macro_region_coord_containing_is_floor_divide() {
        assert_eq!(
            MacroRegionCoord::containing(0, 0),
            MacroRegionCoord { x: 0, z: 0 }
        );
        assert_eq!(
            MacroRegionCoord::containing(MACRO_REGION_SIZE - 1, 0),
            MacroRegionCoord { x: 0, z: 0 }
        );
        assert_eq!(
            MacroRegionCoord::containing(MACRO_REGION_SIZE, 0),
            MacroRegionCoord { x: 1, z: 0 }
        );
    }

    #[test]
    fn lru_evicts_oldest() {
        let cache = fresh_fine_cache();
        // Insert FINE_CACHE_CAP + 1 entries; the first should be gone.
        for i in 0..=(FINE_CACHE_CAP as i32) {
            let c = RegionCoord { x: i, z: 0 };
            let _ = get_fine(&cache, c, || build_fine_region_placeholder(c));
        }
        let was_evicted = cache.peek(&RegionCoord { x: 0, z: 0 }).is_none();
        assert!(
            was_evicted,
            "expected the first-inserted entry to be evicted after CAP+1 inserts"
        );
        // And the most recent one should still be there.
        assert!(
            cache
                .peek(&RegionCoord {
                    x: FINE_CACHE_CAP as i32,
                    z: 0
                })
                .is_some()
        );
    }

    #[test]
    fn cache_returns_same_arc_on_repeat_lookup() {
        let cache = fresh_fine_cache();
        let coord = RegionCoord { x: 4, z: 5 };
        let a = get_fine(&cache, coord, || build_fine_region_placeholder(coord));
        let b = get_fine(&cache, coord, || build_fine_region_placeholder(coord));
        assert!(Arc::ptr_eq(&a, &b));

        // peek_fine returns the same Arc without building.
        let c = peek_fine(&cache, coord).expect("peek must find cached");
        assert!(Arc::ptr_eq(&a, &c));

        // peek_fine on cold key returns None and doesn't insert.
        let cold = RegionCoord { x: 999, z: 999 };
        assert!(peek_fine(&cache, cold).is_none());
        assert!(cache.peek(&cold).is_none(), "peek must not insert");
    }

    #[test]
    fn concurrent_lookup_builds_key_once() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;
        use std::time::Duration;

        let cache = fresh_fine_cache();
        let coord = RegionCoord { x: 7, z: 8 };
        let starts = Arc::new(Barrier::new(8));
        let builds = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache = cache.clone();
                let starts = starts.clone();
                let builds = builds.clone();
                thread::spawn(move || {
                    starts.wait();
                    get_fine(&cache, coord, || {
                        builds.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(25));
                        build_fine_region_placeholder(coord)
                    })
                })
            })
            .collect();

        let first = handles
            .into_iter()
            .map(|h| h.join().expect("worker must not panic"))
            .reduce(|a, b| {
                assert!(Arc::ptr_eq(&a, &b));
                a
            })
            .expect("at least one handle");

        assert!(Arc::ptr_eq(
            &first,
            &peek_fine(&cache, coord).expect("region must be cached")
        ));
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn bitset_get_and_set_round_trip() {
        let mut bytes = vec![0u8; 4];
        bitset_set(&mut bytes, 0, true);
        bitset_set(&mut bytes, 7, true);
        bitset_set(&mut bytes, 9, true);
        bitset_set(&mut bytes, 31, true);
        assert!(bitset_get(&bytes, 0));
        assert!(bitset_get(&bytes, 7));
        assert!(!bitset_get(&bytes, 8));
        assert!(bitset_get(&bytes, 9));
        assert!(bitset_get(&bytes, 31));
        bitset_set(&mut bytes, 7, false);
        assert!(!bitset_get(&bytes, 7));
    }

    #[test]
    fn placeholder_region_has_correct_buffer_sizes() {
        let r = FineRegion::empty(RegionCoord { x: 0, z: 0 });
        let n = (FINE_CELLS_PER_REGION * FINE_CELLS_PER_REGION) as usize;
        assert_eq!(r.h_pre.len(), n);
        assert_eq!(r.flow_dir.len(), n);
        assert_eq!(r.flow_acc.len(), n);
        assert_eq!(r.width.len(), n);
    }
}
