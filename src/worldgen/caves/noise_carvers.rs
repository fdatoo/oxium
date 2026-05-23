//! MC-derived ambient noise carvers: cheese, pillars, and Terasology ambient.
//!
//! These run on every underground voxel alongside the graph-based cave
//! systems. They use a `CarverEvaluator` corner-lattice trilerp (9³ corners,
//! 4-block spacing) to avoid per-voxel FBM cost.
use crate::worldgen::config::CaveConfig;
use crate::worldgen::density::cell_evaluator::CornerLatticeEvaluator;
use crate::worldgen::noise_channel::build_channel;
use crate::worldgen::tuning::*;
use glam::IVec3;
use noise::{Fbm, NoiseFn, Simplex};

/// All MC-derived noise channels needed for the cheese and pillar
/// carvers. Built once per Generator.
pub struct NoiseCarvers {
    // Cheese.
    pub cheese: Fbm<Simplex>,
    /// `cave_layer` — the regional gating noise added to cheese as
    /// `intensity * layer²`. See [`CaveConfig::cave_layer`].
    pub cave_layer: Fbm<Simplex>,
    // Pillars.
    pub pillar: Fbm<Simplex>,
    pub pillar_rareness: Fbm<Simplex>,
    pub pillar_thickness: Fbm<Simplex>,
    // Terasology ambient.
    pub tera_a: Fbm<Simplex>,
    pub tera_b: Fbm<Simplex>,
}

impl NoiseCarvers {
    /// Build all channels from their config descriptors. Each
    /// channel uses a different seed-salt so they're uncorrelated.
    pub fn new(seed: u64, cfg: &CaveConfig) -> Self {
        Self {
            cheese: build_channel(&cfg.cheese, seed, 1001),
            cave_layer: build_channel(&cfg.cave_layer, seed, 1011),
            pillar: build_channel(&cfg.pillar, seed, 1006),
            pillar_rareness: build_channel(&cfg.pillar_rareness, seed, 1007),
            pillar_thickness: build_channel(&cfg.pillar_thickness, seed, 1008),
            tera_a: build_channel(&cfg.tera_a, seed, 2000),
            tera_b: build_channel(&cfg.tera_b, seed, 2001),
        }
    }
}

/// MC-style cheese cave contribution.
///
/// Where the signed FBM noise is negative enough (below `cheese_offset`
/// as a threshold), the cheese term goes negative, carving a hole. The
/// "cheese" metaphor: if you sample a random 3D FBM and threshold it at
/// zero, you get a Swiss-cheese-like collection of blobs where the field
/// dips below the threshold, each blob being an isolated pocket of air.
///
/// The `cave_layer² × intensity` term adds horizontal stratification:
/// `cave_layer` is a low-frequency noise that controls which horizontal
/// strata are rich in caves. Where `cave_layer ≈ 0`, the `layer²` term
/// is near-zero so the raw cheese signal dominates and carves freely.
/// Where `|cave_layer|` is large, the `layer²` term is strongly positive,
/// pushing the total toward solid and suppressing caves in that stratum.
/// This produces the characteristic Minecraft "cave layer" banding —
/// caves that cluster at specific depths rather than distributing evenly.
///
/// `term1` (raw cheese signal) + `term2` (surface suppression, using
/// `raw_density` as a proxy for depth near the surface) + `layerized`.
/// The caller composes via `smin(density, cheese, k)` so the whole
/// signed value participates in the soft-blend.
///
/// ```text
///   term1      = clamp(cheese_offset + cheese_noise, -1, 1)
///   term2      = clamp(supp_offset + supp_slope × raw_density,
///                      supp_min, supp_max)
///   layerized  = cave_layer_intensity × layer²
///   result     = term1 + term2 + layerized
/// ```
pub fn cheese_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    raw_density: f32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    let xz_scale = cfg.cheese_xz_scale as f64;
    let y_scale = cfg.cheese_y_scale as f64;
    let cheese = carvers.cheese.get([
        wx as f64 * xz_scale,
        wy as f64 * y_scale,
        wz as f64 * xz_scale,
    ]) as f32;
    let term1 = (cfg.cheese_offset + cheese).clamp(-1.0, 1.0);
    let supp = (cfg.cheese_suppression_offset + cfg.cheese_suppression_slope * raw_density)
        .clamp(cfg.cheese_suppression_min, cfg.cheese_suppression_max);

    // MC-parity `layerizedCaverns`: add `intensity * layer²` so
    // cheese carving is regionally gated by horizontal strata.
    // Without this the cheese clamp at -1 makes ~50% of deep
    // voxels carve, producing uniform swiss cheese.
    let layer = carvers.cave_layer.get([
        wx as f64 * cfg.cave_layer_xz_scale as f64,
        wy as f64 * cfg.cave_layer_y_scale as f64,
        wz as f64 * cfg.cave_layer_xz_scale as f64,
    ]) as f32;
    let layerized = cfg.cave_layer_intensity * layer * layer;

    term1 + supp + layerized
}

/// Per-voxel pillar contribution. Returns a non-negative value in
/// `[0, pillar_intensity]` that is ADDED to density (not subtracted)
/// in `fill_chunk`'s composition. Composition order matters:
/// pillars apply AFTER all cave subtractions so they can refill
/// previously-carved voxels — the MC "columns inside open caves"
/// look.
///
/// MC formula (from `data/.../caves/pillars.json`):
///
/// ```text
///   pillar_raw  = 2 * noise(pillar, xz_scale, y_scale)
///   pillar_rare = -1 - noise(pillar_rareness)
///   thickness   = (0.55 + 0.55 * noise(pillar_thickness))^3
///   pillars     = (pillar_raw + pillar_rare) * thickness
///   range_choice: if pillars >= cutoff → pillars, else 0
/// ```
pub fn pillar_contribution(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
) -> f32 {
    let p_x = wx as f64 * cfg.pillar_xz_scale as f64;
    let p_y = wy as f64 * cfg.pillar_y_scale as f64;
    let p_z = wz as f64 * cfg.pillar_xz_scale as f64;
    let pillar_raw = 2.0 * carvers.pillar.get([p_x, p_y, p_z]) as f32;
    let pillar_rare = -1.0
        - carvers
            .pillar_rareness
            .get([wx as f64, wy as f64, wz as f64]) as f32;
    let thickness_noise = carvers
        .pillar_thickness
        .get([wx as f64, wy as f64, wz as f64]) as f32;
    let thickness = (PILLAR_THICKNESS_BASE + PILLAR_THICKNESS_BASE * thickness_noise).powi(3);
    let raw = (pillar_raw + pillar_rare) * thickness;
    if raw < cfg.pillar_cutoff {
        return 0.0;
    }
    let depth = (raw - cfg.pillar_cutoff).clamp(0.0, 1.0);
    cfg.pillar_intensity * depth
}

/// Terasology-style depth-driven 2-noise disk carver.
///
/// Two independently seeded 3D FBM noise channels (`tera_a`, `tera_b`) are
/// each evaluated at the same scaled position. Geometrically, the pair
/// `(n0, n1)` defines a point in 2D noise space; carving occurs where that
/// point falls inside a disk of radius `freq_depth` centred near the
/// origin. Because noise values cluster near zero, the disk selects a
/// thin connected manifold — visually a set of nearly-horizontal tubes
/// threading through the rock, mimicking the stratigraphy-following
/// caves found in real karst.
///
/// Two depth-driven offsets modulate the disk:
/// - `freq_reduction` shifts the disk center off-axis near the surface,
///   suppressing carving there (`tera_supp` controls the suppression
///   depth). This replaces the blunt `CAVE_SURFACE_BUFFER` for tera-caves.
/// - `freq_depth` grows with depth so caves become more frequent
///   underground. The growth rate is `tera_thresh_depth`.
///
/// The Y axis is sampled at `tera_y_factor × freq` to squash the noise
/// vertically, keeping the tubes lean-horizontal.
///
/// Returns signed density: negative = carve, positive = solid. Output is
/// scaled by `* 5.0` to align its magnitude with the cheese carver for
/// downstream `smin` composition. Typical range: `[-5.4, +6.7]`.
///
/// Reference: `org.terasology.caves.CaveFacetProvider`.
pub fn terasology_ambient(
    wx: i32,
    wy: i32,
    wz: i32,
    carvers: &NoiseCarvers,
    cfg: &CaveConfig,
    surface_y: f32,
) -> f32 {
    let depth = (surface_y - wy as f32).max(0.0);
    let freq_reduction = (cfg.tera_supp - depth / cfg.tera_supp_depth).max(0.0);
    let freq_depth = cfg.tera_thresh_base + depth / cfg.tera_thresh_depth;
    let freq = 1.0 / cfg.tera_wave;
    let wy_scaled = wy as f32 * cfg.tera_y_factor;
    let n0 = carvers.tera_a.get([
        (wx as f32 * freq) as f64,
        (wy_scaled * freq) as f64,
        (wz as f32 * freq) as f64,
    ]) as f32;
    let n1 = carvers.tera_b.get([
        (wx as f32 * freq) as f64,
        (wy_scaled * freq) as f64,
        (wz as f32 * freq) as f64,
    ]) as f32
        + freq_reduction;
    ((n0 * n0 + n1 * n1).sqrt() - freq_depth) * TERA_OUTPUT_SCALE
    // scale: align magnitude with cheese carver for downstream smin composition
}

// ── Carver evaluator (corner-lattice trilerp) ────────────────────────
//
// The per-voxel `cheese_contribution` and `pillar_contribution` each
// issue several FBM samples per voxel. With 32³ voxels per chunk and
// 2-octave FBM, that dominates the chunk-fill cost.
//
// `CarverEvaluator` mirrors `density_graph::CellEvaluator`: sample
// each underlying noise on a 9³ corner lattice (4-block spacing,
// 729 corners per chunk), then trilerp per voxel. The per-voxel
// formulas (clamps, `raw_density`-dependent suppression, gates) run
// unchanged on the lerped values, so cave shapes track the original
// formulas to within FBM's local smoothness — visually equivalent at
// 4-block resolution, the same precedent as the base density.

pub(crate) const CARVER_CELL_COUNT: usize =
    (crate::voxel::coords::CHUNK_DIM_U as usize) / CARVER_CELL_SIZE as usize; // 8
const CARVER_CORNER_COUNT: usize = CARVER_CELL_COUNT + 1; // 9
const CARVER_CORNER_CUBE: usize = CARVER_CORNER_COUNT * CARVER_CORNER_COUNT * CARVER_CORNER_COUNT; // 729

/// One corner's worth of pre-sampled noise. Keeping the channels
/// AoS means each voxel touches 8 contiguous corner structs instead
/// of striding separate `Vec<f32>` arenas — better cache behavior
/// in the per-voxel inner loop.
///
/// `pub(crate)` because it appears as the `Corner` associated type of the
/// [`CornerLatticeEvaluator`] impl for [`CarverEvaluator`], and the trait
/// is `pub(crate)` — the compiler requires the associated type to be at
/// least as visible as the trait.
#[derive(Default, Clone, Copy)]
pub(crate) struct CarverCorner {
    cheese: f32,
    cave_layer: f32,
    pillar: f32,
    pillar_rare: f32,
    pillar_thick: f32,
    tera_a: f32,
    tera_b: f32,
}

/// Pre-sampled noise lattice for the carver layers. Built once per
/// chunk fill; `*_at` accessors trilerp from the 8 surrounding
/// corners and apply the original per-voxel formula.
pub struct CarverEvaluator {
    corners: Box<[CarverCorner]>,
    origin: IVec3,
}

impl CarverEvaluator {
    pub fn new(carvers: &NoiseCarvers, cfg: &CaveConfig, chunk_origin: IVec3) -> Self {
        let mut corners = vec![CarverCorner::default(); CARVER_CORNER_CUBE].into_boxed_slice();
        for cz in 0..CARVER_CORNER_COUNT {
            for cy in 0..CARVER_CORNER_COUNT {
                for cx in 0..CARVER_CORNER_COUNT {
                    let wx = chunk_origin.x + (cx as i32) * CARVER_CELL_SIZE;
                    let wy = chunk_origin.y + (cy as i32) * CARVER_CELL_SIZE;
                    let wz = chunk_origin.z + (cz as i32) * CARVER_CELL_SIZE;
                    let idx = corner_index(cx, cy, cz);

                    let cheese = carvers.cheese.get([
                        wx as f64 * cfg.cheese_xz_scale as f64,
                        wy as f64 * cfg.cheese_y_scale as f64,
                        wz as f64 * cfg.cheese_xz_scale as f64,
                    ]) as f32;
                    let cave_layer = carvers.cave_layer.get([
                        wx as f64 * cfg.cave_layer_xz_scale as f64,
                        wy as f64 * cfg.cave_layer_y_scale as f64,
                        wz as f64 * cfg.cave_layer_xz_scale as f64,
                    ]) as f32;

                    let pillar = carvers.pillar.get([
                        wx as f64 * cfg.pillar_xz_scale as f64,
                        wy as f64 * cfg.pillar_y_scale as f64,
                        wz as f64 * cfg.pillar_xz_scale as f64,
                    ]) as f32;
                    let pillar_rare = carvers
                        .pillar_rareness
                        .get([wx as f64, wy as f64, wz as f64])
                        as f32;
                    let pillar_thick = carvers
                        .pillar_thickness
                        .get([wx as f64, wy as f64, wz as f64])
                        as f32;

                    let tera_freq = 1.0 / cfg.tera_wave;
                    let tera_wy = wy as f32 * cfg.tera_y_factor;
                    let tera_a = carvers.tera_a.get([
                        (wx as f32 * tera_freq) as f64,
                        (tera_wy * tera_freq) as f64,
                        (wz as f32 * tera_freq) as f64,
                    ]) as f32;
                    let tera_b = carvers.tera_b.get([
                        (wx as f32 * tera_freq) as f64,
                        (tera_wy * tera_freq) as f64,
                        (wz as f32 * tera_freq) as f64,
                    ]) as f32;

                    corners[idx] = CarverCorner {
                        cheese,
                        cave_layer,
                        pillar,
                        pillar_rare,
                        pillar_thick,
                        tera_a,
                        tera_b,
                    };
                }
            }
        }
        Self {
            corners,
            origin: chunk_origin,
        }
    }

    #[inline]
    fn lerp_coords(&self, wx: i32, wy: i32, wz: i32) -> LerpCoords {
        let lx = wx - self.origin.x;
        let ly = wy - self.origin.y;
        let lz = wz - self.origin.z;
        let cx = ((lx / CARVER_CELL_SIZE) as usize).min(CARVER_CELL_COUNT - 1);
        let cy = ((ly / CARVER_CELL_SIZE) as usize).min(CARVER_CELL_COUNT - 1);
        let cz = ((lz / CARVER_CELL_SIZE) as usize).min(CARVER_CELL_COUNT - 1);
        let tx = (lx - cx as i32 * CARVER_CELL_SIZE) as f32 / CARVER_CELL_SIZE as f32;
        let ty = (ly - cy as i32 * CARVER_CELL_SIZE) as f32 / CARVER_CELL_SIZE as f32;
        let tz = (lz - cz as i32 * CARVER_CELL_SIZE) as f32 / CARVER_CELL_SIZE as f32;
        LerpCoords {
            cx,
            cy,
            cz,
            tx,
            ty,
            tz,
        }
    }

    /// Trilinear interp of one channel — `get` picks the channel from a
    /// `CarverCorner`. Delegates to [`CornerLatticeEvaluator::trilerp_at`]
    /// so this evaluator uses the same Y→X→Z lerp order as [`CellEvaluator`].
    #[inline]
    fn trilerp<F: Fn(&CarverCorner) -> f32>(&self, c: &LerpCoords, get: F) -> f32 {
        self.trilerp_at(c.cx, c.cy, c.cz, c.tx, c.ty, c.tz, get)
    }

    /// Same shape as `cheese_contribution`: `term1 + supp + layerized`.
    /// Trilerps the noise channels; the clamps and `raw_density`-driven
    /// suppression term run unchanged per voxel.
    pub fn cheese_at(&self, wx: i32, wy: i32, wz: i32, raw_density: f32, cfg: &CaveConfig) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let cheese = self.trilerp(&lc, |c| c.cheese);
        let cave_layer = self.trilerp(&lc, |c| c.cave_layer);
        let term1 = (cfg.cheese_offset + cheese).clamp(-1.0, 1.0);
        let supp = (cfg.cheese_suppression_offset + cfg.cheese_suppression_slope * raw_density)
            .clamp(cfg.cheese_suppression_min, cfg.cheese_suppression_max);
        let layerized = cfg.cave_layer_intensity * cave_layer * cave_layer;
        term1 + supp + layerized
    }

    /// Same shape as `pillar_contribution`. The cutoff gate runs per voxel.
    pub fn pillar_at(&self, wx: i32, wy: i32, wz: i32, cfg: &CaveConfig) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let pillar = self.trilerp(&lc, |c| c.pillar);
        let pillar_rare_n = self.trilerp(&lc, |c| c.pillar_rare);
        let thickness_noise = self.trilerp(&lc, |c| c.pillar_thick);
        let pillar_raw = 2.0 * pillar;
        let pillar_rare = -1.0 - pillar_rare_n;
        let thickness = (PILLAR_THICKNESS_BASE + PILLAR_THICKNESS_BASE * thickness_noise).powi(3);
        let raw = (pillar_raw + pillar_rare) * thickness;
        if raw < cfg.pillar_cutoff {
            return 0.0;
        }
        let depth = (raw - cfg.pillar_cutoff).clamp(0.0, 1.0);
        cfg.pillar_intensity * depth
    }

    /// Trilerp the pre-sampled tera_a and tera_b noise values, then run the
    /// same `terasology_ambient` arithmetic on the lerped result. Exact at
    /// corners by construction.
    pub fn terasology_ambient_at(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        cfg: &CaveConfig,
        surface_y: f32,
    ) -> f32 {
        let lc = self.lerp_coords(wx, wy, wz);
        let n0_raw = self.trilerp(&lc, |c| c.tera_a);
        let n1_raw = self.trilerp(&lc, |c| c.tera_b);
        let depth = (surface_y - wy as f32).max(0.0);
        let freq_reduction = (cfg.tera_supp - depth / cfg.tera_supp_depth).max(0.0);
        let freq_depth = cfg.tera_thresh_base + depth / cfg.tera_thresh_depth;
        let n1 = n1_raw + freq_reduction;
        ((n0_raw * n0_raw + n1 * n1).sqrt() - freq_depth) * TERA_OUTPUT_SCALE
    }
}

impl CornerLatticeEvaluator for CarverEvaluator {
    type Corner = CarverCorner;

    /// Return the pre-sampled multi-channel corner at lattice position
    /// `(cx, cy, cz)`. Storage uses [`corner_index`] (same X/Y/Z order as
    /// the build loop) so lerp coordinates computed in [`lerp_coords`] match
    /// exactly.
    #[inline]
    fn corner(&self, cx: usize, cy: usize, cz: usize) -> &CarverCorner {
        &self.corners[corner_index(cx, cy, cz)]
    }
}

#[derive(Clone, Copy)]
struct LerpCoords {
    cx: usize,
    cy: usize,
    cz: usize,
    tx: f32,
    ty: f32,
    tz: f32,
}

#[inline]
pub(crate) fn corner_index(cx: usize, cy: usize, cz: usize) -> usize {
    cx + cy * CARVER_CORNER_COUNT + cz * CARVER_CORNER_COUNT * CARVER_CORNER_COUNT
}
