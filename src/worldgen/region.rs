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
//! The `loop` in [`get_or_build`] is necessary: if the builder panics,
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

use crate::worldgen::fluid::FluidBodyKind;
use crate::worldgen::tuning::*;
use lru::LruCache;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex};

// ── Region coordinate keys ────────────────────────────────────────────

/// Grid coordinate of a fine region (512 × 512 blocks).
///
/// `x` and `z` equal `floor(world_coord / FINE_REGION_SIZE)`. Negative
/// values are valid — the world grid extends in all directions.
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

/// Grid coordinate of a macro region (8192 × 8192 blocks).
///
/// The macro grid covers the same infinite plane as fine regions but at
/// 16× coarser granularity. One macro region covers 16 × 16 fine regions.
/// Used by the trunk-river pass to provide long-distance drainage context.
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
    /// How many blocks the lake bed has been carved below the natural
    /// terrain height. Guaranteed ≥ `MIN_LAKE_BED_DROP` where
    /// `is_lake` is true; zero elsewhere. Populated by PR 3.
    pub lake_bed_depth: Box<[i16]>,
    /// River segments derived from the fine flow field. Populated by
    /// PR 3.
    pub segments: Vec<RiverSegment>,
    /// Cave systems whose primary anchor lives in this region. Their
    /// bounding boxes can spill into neighbours; chunk fill consults
    /// the 3×3 region neighborhood. Populated by PR 4.
    pub cave_systems: Vec<CaveSystem>,
    /// Cave pools derived from qualifying chambers in this region.
    /// Populated alongside `cave_systems` in PR 5.
    pub cave_pools: Vec<CavePool>,
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
            lake_bed_depth: vec![0i16; n].into_boxed_slice(),
            segments: Vec::new(),
            cave_systems: Vec::new(),
            cave_pools: Vec::new(),
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

/// Pre-computed coarse hydrology for one 8192×8192-block macro region.
///
/// Stores the D8 flow field and accumulation at 64 m/cell resolution.
/// Cells with `flow_acc >= MACRO_RIVER_THRESH` are flagged as trunk rivers
/// in `is_trunk` and their accumulation is injected into the fine grid
/// when building overlapping fine regions, so intercontinental rivers stay
/// fat even when they first appear in a fine region window.
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
    /// Voxel Y of the static generated river surface.
    pub water_y: i32,
    /// Voxel Y of the carved bed below the water surface.
    pub bed_y: i32,
    /// Surface-water classification for this segment.
    pub kind: RiverSegmentKind,
    /// True for an ocean-mouth segment (flared by `MOUTH_FLARE_MULT`).
    pub mouth: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiverSegmentKind {
    Channel,
    Rapid,
    Waterfall,
}

/// A static fluid pool inside a cave chamber.
///
/// Derived from qualifying `Chamber` ellipsoids during region build and
/// stored per-region. At chunk fill time the fluid planner reads all pools
/// whose bounding ellipsoid intersects the chunk and stamps the fluid into
/// Air voxels between `bed_y` and `surface_y`. Large chambers deep in the
/// lava band may roll as lava pools; shallower ones are always water.
#[derive(Debug, Clone)]
pub struct CavePool {
    /// World-space center of the originating ellipsoid chamber.
    pub center: glam::Vec3,
    /// Semi-axis lengths (x, y, z) of the chamber.
    pub radii: glam::Vec3,
    /// Y of the static fluid surface (air above, fluid below).
    pub surface_y: i32,
    /// Y of the lowest solid voxel below the fluid column.
    pub bed_y: i32,
    /// Water or lava.
    pub kind: FluidBodyKind,
}

/// A fully-resolved graph-based cave system.
///
/// Stored in the fine region cache (immutable behind `Arc`). Carving
/// happens at chunk fill time: the SDF functions in `caves.rs` query
/// `chambers`, `tunnels`, `entrances`, and `vertical_connectors` to
/// decide which voxels are air. The `bb_min`/`bb_max` bounding box lets
/// `fill_chunk` cull the list to only the systems that overlap the chunk
/// before entering the per-voxel inner loop.
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

/// One ellipsoidal chamber — the primary air volume in a cave system.
///
/// A voxel at position `p` is inside the chamber when
/// `(p - center)^2 / radii^2 <= 1` (normalised squared distance ≤ 1).
/// Radii are independent per axis so chambers can be wide (Cathedral,
/// Sump) or tall (Slot).
#[derive(Debug, Clone, Copy)]
pub struct Chamber {
    pub center: glam::Vec3,
    pub radii: glam::Vec3,
}

/// A tunnel corridor connecting two chambers.
///
/// Represented as a polyline of 2–4 control points. The SDF carver
/// approximates the smooth Catmull-Rom path as a sequence of straight
/// capsule segments (`control_points[i] → control_points[i+1]`); any
/// voxel within `radius` of the nearest point on any segment is carved.
#[derive(Debug, Clone)]
pub struct Tunnel {
    pub control_points: Vec<glam::Vec3>,
    pub radius: f32,
}

/// A surface entrance feature carved above a chamber to connect it to the
/// open world.
///
/// There are three kinds (see [`EntranceKind`]): `Sinkhole` (vertical shaft
/// from chamber top to surface), `CliffMouth` (horizontal tunnel to a
/// cliff face), and `Skylight` (narrow vertical shaft). The `entrance_sdf`
/// function uses `surface` as the anchor for the carved geometry.
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

/// Shared LRU cache of fine regions. `Arc`-wrapped so all concurrent chunk
/// generation threads share a single cache instance with atomic eviction.
pub type FineCache = Arc<BuildCache<RegionCoord, FineRegion>>;
/// Shared LRU cache of macro regions.
pub type MacroCache = Arc<BuildCache<MacroRegionCoord, MacroRegion>>;

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

/// Convenience: build a fine region for `coord` filled with PR 1
/// placeholders. PRs 2–4 replace this with real population code.
pub fn build_fine_region_placeholder(coord: RegionCoord) -> FineRegion {
    FineRegion::empty(coord)
}

pub fn build_macro_region_placeholder(coord: MacroRegionCoord) -> MacroRegion {
    MacroRegion::empty(coord)
}

#[cfg(test)]
#[path = "region_tests.rs"]
mod tests;
