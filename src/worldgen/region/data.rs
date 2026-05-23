//! Region data structs: `FineRegion`, `MacroRegion`, and the cave / river
//! payload types that live inside them.
//!
//! All structs in this file are populated incrementally across PRs 2–4.
//! PR 1 leaves the data buffers zero-filled via `FineRegion::empty` /
//! `MacroRegion::empty`; later PRs replace the placeholder builders with
//! real population code.

use super::coords::{MacroRegionCoord, RegionCoord};
use crate::worldgen::fluid::FluidBodyKind;
use crate::worldgen::tuning::{
    FINE_CELLS_PER_REGION, MACRO_CELLS_PER_REGION, MAX_RIVER_WIDTH, MIN_RIVER_WIDTH,
};

// ── RiverWidth ────────────────────────────────────────────────────────

/// A river-width value in voxels, clamped to `[MIN_RIVER_WIDTH,
/// MAX_RIVER_WIDTH]` at construction time.
///
/// Wrapping the bare `f32` makes the clamp invariant visible in type
/// signatures: any `RiverWidth` the caller receives is guaranteed to be
/// in the valid range, eliminating ad-hoc clamping at read sites.
///
/// Use `.0` to extract the inner `f32` for arithmetic. Smaller values
/// produce narrower rivers; larger values widen them. The effective range
/// is controlled by [`MIN_RIVER_WIDTH`] and [`MAX_RIVER_WIDTH`] in
/// `tuning.rs`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RiverWidth(pub f32);

impl RiverWidth {
    /// Wrap `v` and clamp it to `[MIN_RIVER_WIDTH, MAX_RIVER_WIDTH]`.
    /// This is the canonical constructor — prefer it over the tuple-struct
    /// literal to ensure the invariant is enforced at every entry point.
    #[inline]
    pub fn new(v: f32) -> Self {
        Self(v.clamp(MIN_RIVER_WIDTH, MAX_RIVER_WIDTH))
    }

    /// Zero-width placeholder used to initialise the per-cell width buffer
    /// before river tagging. Not a valid river width (it is below
    /// `MIN_RIVER_WIDTH`), but no code reads these cells as rivers because
    /// `is_river` is false for them.
    #[inline]
    pub const fn zero() -> Self {
        Self(0.0)
    }
}

impl From<f32> for RiverWidth {
    /// Clamp and wrap `v` into a `RiverWidth`.
    #[inline]
    fn from(v: f32) -> Self {
        Self::new(v)
    }
}

// ── SystemBoundingBox ─────────────────────────────────────────────────

/// Axis-aligned bounding box for a cave system, stored as world-space
/// inclusive min / max corners.
///
/// Extracted from the raw `(IVec3, IVec3)` tuples that previously appeared
/// wherever cave systems were rolled, stored, and queried, so that the
/// bounding-box invariants and helper methods live in one place.
///
/// `min` and `max` are both **inclusive** — a point exactly on an edge
/// is inside the box. This matches the convention used by the SDF culling
/// loop and the chunk-overlap test in `pipeline.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemBoundingBox {
    /// Inclusive world-space minimum corner.
    pub min: glam::IVec3,
    /// Inclusive world-space maximum corner.
    pub max: glam::IVec3,
}

impl SystemBoundingBox {
    /// True when world point `(wx, wy, wz)` is inside this box (inclusive
    /// on both ends). Used as a fast early-exit before evaluating the full
    /// chamber / tunnel SDF.
    #[inline]
    pub fn contains_point(&self, wx: i32, wy: i32, wz: i32) -> bool {
        wx >= self.min.x
            && wx <= self.max.x
            && wy >= self.min.y
            && wy <= self.max.y
            && wz >= self.min.z
            && wz <= self.max.z
    }

    /// True when this box overlaps the axis-aligned box `[box_min, box_max)`.
    /// Used in `pipeline.rs` to pre-filter cave systems before the per-voxel
    /// inner loop.
    #[inline]
    pub fn overlaps_box(&self, box_min: glam::IVec3, box_max: glam::IVec3) -> bool {
        self.max.x >= box_min.x
            && self.min.x <= box_max.x
            && self.max.y >= box_min.y
            && self.min.y <= box_max.y
            && self.max.z >= box_min.z
            && self.min.z <= box_max.z
    }
}

// ── Fine region ───────────────────────────────────────────────────────

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
    /// River width per fine cell. `RiverWidth::zero()` if the cell is not
    /// a river; otherwise guaranteed to be in `[MIN_RIVER_WIDTH,
    /// MAX_RIVER_WIDTH]`. Populated by PR 3.
    pub width: Box<[RiverWidth]>,
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
            width: vec![RiverWidth::zero(); n].into_boxed_slice(),
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
        debug_assert!((0..FINE_CELLS_PER_REGION).contains(&ix));
        debug_assert!((0..FINE_CELLS_PER_REGION).contains(&iz));
        (iz * FINE_CELLS_PER_REGION + ix) as usize
    }
}

// ── Macro region ──────────────────────────────────────────────────────

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
    pub(super) fn empty(coord: MacroRegionCoord) -> Self {
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
        debug_assert!((0..MACRO_CELLS_PER_REGION).contains(&ix));
        debug_assert!((0..MACRO_CELLS_PER_REGION).contains(&iz));
        (iz * MACRO_CELLS_PER_REGION + ix) as usize
    }
}

// ── River payload ─────────────────────────────────────────────────────

/// One linear segment of a river — runs from cell center `(from)` to
/// cell center `(to)` at the cell's width and depth. PR 3 populates
/// per-region segment lists; the kd-tree spatial index is built lazily
/// on first `valley_carve` query.
#[derive(Debug, Clone, Copy)]
pub struct RiverSegment {
    /// Centerline endpoints in world coordinates.
    pub from: (i32, i32),
    pub to: (i32, i32),
    pub width: RiverWidth,
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

// ── Cave payload ──────────────────────────────────────────────────────

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
/// decide which voxels are air. The `bbox` bounding box lets `fill_chunk`
/// cull the list to only the systems that overlap the chunk before
/// entering the per-voxel inner loop.
#[derive(Debug, Clone)]
pub struct CaveSystem {
    /// World-space axis-aligned bounding box (inclusive on both ends).
    /// Used for fast chunk-overlap culling before the per-voxel SDF loop.
    pub bbox: SystemBoundingBox,
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

impl CaveSystem {
    /// True when world point `(wx, wy, wz)` is inside this system's
    /// axis-aligned bounding box. Delegates to [`SystemBoundingBox::contains_point`].
    #[inline]
    pub fn contains_point(&self, wx: i32, wy: i32, wz: i32) -> bool {
        self.bbox.contains_point(wx, wy, wz)
    }

    /// True when this system's bounding box overlaps `[box_min, box_max)`.
    /// Delegates to [`SystemBoundingBox::overlaps_box`].
    #[inline]
    pub fn overlaps_box(&self, box_min: glam::IVec3, box_max: glam::IVec3) -> bool {
        self.bbox.overlaps_box(box_min, box_max)
    }
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

// ── Placeholder builders ──────────────────────────────────────────────

/// Convenience: build a fine region for `coord` filled with PR 1
/// placeholders. PRs 2–4 replace this with real population code.
pub fn build_fine_region_placeholder(coord: RegionCoord) -> FineRegion {
    FineRegion::empty(coord)
}

pub fn build_macro_region_placeholder(coord: MacroRegionCoord) -> MacroRegion {
    MacroRegion::empty(coord)
}
