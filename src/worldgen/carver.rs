//! MC-style procedural cave carver.
//!
//! Each chunk has a ~15% chance of seeding 0-3 tunnels. Each tunnel
//! walks a random curve through the world (`MAX_DISTANCE` ≈ 112
//! unit steps), dropping ellipsoid spheres at each step. Sphere
//! radius is a sine profile over the tunnel lifetime (small at the
//! ends, fat in the middle, modulated by a per-tunnel `thickness`).
//! Tunnels branch once at their midpoint, spawning two perpendicular
//! sub-walks that inherit half the parent's thickness. Tunnels
//! exit early if they wander too far from the origin chunk.
//!
//! All randomness routes through `hash::mix`, so the same `(seed,
//! chunk_coord)` always produces the same set of tunnels — caves
//! are byte-reproducible.
//!
//! At chunk fill time, the caller rasterises the tunnel sphere list
//! into a per-chunk boolean mask (see [`rasterize_into_mask`]) and
//! tests `mask[voxel]` to decide if the voxel is carved. This keeps
//! the per-voxel test at O(1) regardless of how many tunnels
//! converge on the chunk.
//!
//! References:
//! - MC `net/minecraft/world/level/levelgen/carver/CaveWorldCarver.java`
//! - MC `net/minecraft/world/level/levelgen/carver/WorldCarver.java`
//! - MC `data/minecraft/worldgen/configured_carver/cave.json`
//! - Oxium spec: `docs/superpowers/specs/2026-05-21-cave-system-overhaul-design.md`

use crate::voxel::coords::{CHUNK_DIM, ChunkCoord};
use crate::worldgen::hash;
use glam::{IVec3, Vec3};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

// ── Tuning ───────────────────────────────────────────────────────

/// Probability that a chunk seeds carvers at all. MC: 0.15.
const CHUNK_PROBABILITY: f32 = 0.15;

/// Bound on the triple-nested random for cave count per chunk.
/// MC: 15. Combined with the triple-nesting this skews the
/// distribution toward 0-2 caves per firing chunk.
const COUNT_BOUND: u32 = 15;

/// Y range for carver origins. Walks can drift ~15 blocks beyond.
const Y_MIN: i32 = -80;
const Y_MAX: i32 = 80;

/// Max walk distance per tunnel (unit-length steps). MC: 112.
const MAX_DISTANCE: i32 = 112;

/// Reachability cap. A tunnel exits early if it walks further than
/// `thickness + 2 + REACH_BUFFER` from its origin chunk's center.
/// MC: 16 (matches MC chunk width). Ours is 32 since chunks are
/// twice as wide — letting tunnels reach roughly the same physical
/// distance.
const REACH_BUFFER: f32 = 32.0;

/// Max branch depth. MC implicitly stops at 1 because branches get
/// thickness ≤ 1.0 which fails the `thickness > 1` branch gate.
/// We cap explicitly as a defensive limit.
const MAX_BRANCH_DEPTH: u8 = 2;

// ── Geometry ─────────────────────────────────────────────────────

/// One sphere along a tunnel path.
#[derive(Debug, Clone, Copy)]
pub struct CarverSphere {
    pub center: Vec3,
    pub h_radius: f32,
    pub v_radius: f32,
    /// Normalized Y cutoff: voxels with `yd / v_radius <= floor_level`
    /// are NOT carved. MC: U[-1.0, -0.4] — gives the cave a flat-ish
    /// floor instead of a hemispherical pit at the bottom.
    pub floor_level: f32,
}

/// One tunnel = a chain of spheres + an AABB for quick rejection.
#[derive(Debug, Clone)]
pub struct CarverTunnel {
    pub spheres: Vec<CarverSphere>,
    pub aabb_min: IVec3,
    pub aabb_max: IVec3,
}

impl CarverTunnel {
    fn new() -> Self {
        Self {
            spheres: Vec::new(),
            aabb_min: IVec3::splat(i32::MAX),
            aabb_max: IVec3::splat(i32::MIN),
        }
    }

    fn push_sphere(&mut self, sphere: CarverSphere) {
        let r_h = (sphere.h_radius.ceil() as i32) + 1;
        let r_v = (sphere.v_radius.ceil() as i32) + 1;
        let cx = sphere.center.x.round() as i32;
        let cy = sphere.center.y.round() as i32;
        let cz = sphere.center.z.round() as i32;
        let smin = IVec3::new(cx - r_h, cy - r_v, cz - r_h);
        let smax = IVec3::new(cx + r_h, cy + r_v, cz + r_h);
        self.aabb_min = self.aabb_min.min(smin);
        self.aabb_max = self.aabb_max.max(smax);
        self.spheres.push(sphere);
    }

    /// True if this tunnel's AABB intersects the inclusive chunk AABB.
    pub fn intersects_chunk(&self, chunk_min: IVec3, chunk_max: IVec3) -> bool {
        if self.spheres.is_empty() {
            return false;
        }
        self.aabb_min.x <= chunk_max.x
            && self.aabb_max.x >= chunk_min.x
            && self.aabb_min.y <= chunk_max.y
            && self.aabb_max.y >= chunk_min.y
            && self.aabb_min.z <= chunk_max.z
            && self.aabb_max.z >= chunk_min.z
    }
}

/// True iff the voxel center sits inside the sphere's ellipsoid AND
/// above the sphere's floor cutoff.
pub fn sphere_carves(s: &CarverSphere, wx: i32, wy: i32, wz: i32) -> bool {
    let xd = (wx as f32 + 0.5 - s.center.x) / s.h_radius;
    let yd = (wy as f32 + 0.5 - s.center.y) / s.v_radius;
    let zd = (wz as f32 + 0.5 - s.center.z) / s.h_radius;
    if yd <= s.floor_level {
        return false;
    }
    xd * xd + yd * yd + zd * zd < 1.0
}

// ── Deterministic RNG ────────────────────────────────────────────

/// Stream RNG routed through `hash::mix`. Each `next_*` call advances
/// a counter so the same `(seed, chunk, salt)` reproduces the same
/// stream byte-for-byte.
struct CarverRng {
    seed: u64,
    salt: [i32; 4],
    counter: u32,
}

impl CarverRng {
    fn new(seed: u64, chunk: ChunkCoord, sub_seed: u32) -> Self {
        Self {
            seed,
            salt: [chunk.0.x, chunk.0.y, chunk.0.z, sub_seed as i32],
            counter: 0,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let r = hash::mix_u32(
            self.seed,
            &[
                self.salt[0],
                self.salt[1],
                self.salt[2],
                self.salt[3],
                self.counter as i32,
            ],
        );
        self.counter = self.counter.wrapping_add(1);
        r
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    fn next_range(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        self.next_u32() % max
    }
}

// ── Builder ──────────────────────────────────────────────────────

/// Build all carver tunnels originating in this chunk. Deterministic
/// on `(seed, chunk)`. Empty when the per-chunk probability roll
/// fails — most chunks return no tunnels.
pub fn build_tunnels_for_chunk(seed: u64, chunk: ChunkCoord) -> Vec<CarverTunnel> {
    let mut rng = CarverRng::new(seed, chunk, 0);

    if rng.next_f32() > CHUNK_PROBABILITY {
        return Vec::new();
    }

    // MC's triple-nested random: heavily skews toward 0-2 caves.
    let n0 = rng.next_range(COUNT_BOUND);
    let n1 = rng.next_range(n0 + 1);
    let cave_count = rng.next_range(n1 + 1) as i32;

    let mut tunnels = Vec::new();
    let chunk_origin = chunk.0 * CHUNK_DIM;
    let chunk_center = Vec3::new(
        chunk_origin.x as f32 + CHUNK_DIM as f32 * 0.5,
        chunk_origin.y as f32 + CHUNK_DIM as f32 * 0.5,
        chunk_origin.z as f32 + CHUNK_DIM as f32 * 0.5,
    );

    for cave_idx in 0..cave_count {
        // Each cave gets its own RNG sub-stream so adding more caves
        // doesn't perturb earlier caves' randomness.
        let cave_seed = rng.next_u32();
        let mut cave_rng = CarverRng::new(seed, chunk, cave_seed);

        let ox = chunk_origin.x + cave_rng.next_range(CHUNK_DIM as u32) as i32;
        let oz = chunk_origin.z + cave_rng.next_range(CHUNK_DIM as u32) as i32;
        let oy = Y_MIN + cave_rng.next_range((Y_MAX - Y_MIN) as u32) as i32;
        let pos = Vec3::new(ox as f32 + 0.5, oy as f32 + 0.5, oz as f32 + 0.5);

        let h_rot = cave_rng.next_f32() * TAU;
        let v_rot = (cave_rng.next_f32() - 0.5) / 4.0;

        // MC thickness formula: U[1, 3) + 10% chance to scale by U[1, 4).
        let mut thickness = cave_rng.next_f32() * 2.0 + cave_rng.next_f32();
        if cave_rng.next_range(10) == 0 {
            thickness *= cave_rng.next_f32() * cave_rng.next_f32() * 3.0 + 1.0;
        }

        let y_scale = 0.1 + cave_rng.next_f32() * 0.8;
        let floor_level = -1.0 + cave_rng.next_f32() * 0.6;

        let dist = MAX_DISTANCE - cave_rng.next_range(MAX_DISTANCE as u32 / 4) as i32;
        let split = (cave_rng.next_range((dist / 2).max(1) as u32) as i32) + dist / 4;

        walk(
            &mut cave_rng,
            &mut tunnels,
            chunk_center,
            pos,
            h_rot,
            v_rot,
            thickness,
            y_scale,
            floor_level,
            0,
            dist,
            Some(split),
            0,
            cave_idx as u32,
        );
    }

    tunnels
}

/// One walk produces (potentially) one `CarverTunnel` of spheres plus
/// any recursive branch tunnels.
#[allow(clippy::too_many_arguments)]
fn walk(
    rng: &mut CarverRng,
    tunnels: &mut Vec<CarverTunnel>,
    chunk_center: Vec3,
    mut pos: Vec3,
    mut h_rot: f32,
    mut v_rot: f32,
    thickness: f32,
    y_scale: f32,
    floor_level: f32,
    start_step: i32,
    dist: i32,
    split: Option<i32>,
    branch_depth: u8,
    branch_id: u32,
) {
    if branch_depth > MAX_BRANCH_DEPTH {
        return;
    }

    let is_steep = rng.next_range(6) == 0;
    let vert_decay: f32 = if is_steep { 0.92 } else { 0.7 };

    let mut x_rota: f32 = 0.0;
    let mut y_rota: f32 = 0.0;
    let mut tunnel = CarverTunnel::new();

    for step in start_step..dist {
        // Advance one unit in the current direction.
        let cos_v = v_rot.cos();
        pos.x += h_rot.cos() * cos_v;
        pos.y += v_rot.sin();
        pos.z += h_rot.sin() * cos_v;

        // Update rotations with damped random walk (MC formula).
        v_rot *= vert_decay;
        v_rot += x_rota * 0.1;
        h_rot += y_rota * 0.1;
        x_rota *= 0.9;
        y_rota *= 0.75;
        x_rota += (rng.next_f32() - rng.next_f32()) * rng.next_f32() * 2.0;
        y_rota += (rng.next_f32() - rng.next_f32()) * rng.next_f32() * 4.0;

        // Branch at the split point — only the parent (with the
        // original `split`) branches; child walks have `split = None`.
        if let Some(sp) = split {
            if step == sp && thickness > 1.0 && branch_depth < MAX_BRANCH_DEPTH {
                let branch_id_a = branch_id.wrapping_mul(7) ^ step as u32;
                let branch_id_b = branch_id_a.wrapping_add(1);
                let new_thickness_a = rng.next_f32() * 0.5 + 0.5;
                let new_thickness_b = rng.next_f32() * 0.5 + 0.5;
                walk(
                    rng,
                    tunnels,
                    chunk_center,
                    pos,
                    h_rot - FRAC_PI_2,
                    v_rot / 3.0,
                    new_thickness_a,
                    y_scale,
                    floor_level,
                    step,
                    dist,
                    None,
                    branch_depth + 1,
                    branch_id_a,
                );
                walk(
                    rng,
                    tunnels,
                    chunk_center,
                    pos,
                    h_rot + FRAC_PI_2,
                    v_rot / 3.0,
                    new_thickness_b,
                    y_scale,
                    floor_level,
                    step,
                    dist,
                    None,
                    branch_depth + 1,
                    branch_id_b,
                );
                // Parent stops after branching (matches MC: return).
                break;
            }
        }

        // MC: 1-in-4 chance to skip this step entirely. Adds spatial
        // jitter and reduces sphere count.
        if rng.next_range(4) == 0 {
            continue;
        }

        // Reachability: if we've wandered too far from the origin
        // chunk's center to plausibly return, abort.
        let dx = pos.x - chunk_center.x;
        let dz = pos.z - chunk_center.z;
        let remaining = (dist - step) as f32;
        let rr = thickness + 2.0 + REACH_BUFFER;
        if dx * dx + dz * dz - remaining * remaining > rr * rr {
            break;
        }

        let progress = step as f32 / dist.max(1) as f32;
        let h_radius = 1.5 + (PI * progress).sin() * thickness;
        let v_radius = h_radius * y_scale;

        tunnel.push_sphere(CarverSphere {
            center: pos,
            h_radius,
            v_radius,
            floor_level,
        });
    }

    if !tunnel.spheres.is_empty() {
        tunnels.push(tunnel);
    }
}

// ── Rasterisation into per-chunk mask ────────────────────────────

/// Rasterise a tunnel's spheres into a per-chunk boolean mask.
///
/// `mask` is indexed as `local_x + CHUNK_DIM * local_y + CHUNK_DIM^2 * local_z`
/// (matching [`LocalPos::to_index`]). `chunk_origin` is the world-space
/// position of the chunk's `(0,0,0)` corner. Voxels carved by any
/// sphere are set to `true`; voxels outside the chunk are skipped.
pub fn rasterize_into_mask(tunnel: &CarverTunnel, chunk_origin: IVec3, mask: &mut [bool]) {
    let chunk_max = chunk_origin + IVec3::splat(CHUNK_DIM - 1);
    if !tunnel.intersects_chunk(chunk_origin, chunk_max) {
        return;
    }
    for sphere in &tunnel.spheres {
        let r_h = sphere.h_radius.ceil() as i32 + 1;
        let r_v = sphere.v_radius.ceil() as i32 + 1;
        let cx = sphere.center.x.round() as i32;
        let cy = sphere.center.y.round() as i32;
        let cz = sphere.center.z.round() as i32;
        let xmin = (cx - r_h).max(chunk_origin.x);
        let xmax = (cx + r_h).min(chunk_max.x);
        let ymin = (cy - r_v).max(chunk_origin.y);
        let ymax = (cy + r_v).min(chunk_max.y);
        let zmin = (cz - r_h).max(chunk_origin.z);
        let zmax = (cz + r_h).min(chunk_max.z);
        if xmin > xmax || ymin > ymax || zmin > zmax {
            continue;
        }
        for wz in zmin..=zmax {
            for wy in ymin..=ymax {
                for wx in xmin..=xmax {
                    if !sphere_carves(sphere, wx, wy, wz) {
                        continue;
                    }
                    let lx = (wx - chunk_origin.x) as usize;
                    let ly = (wy - chunk_origin.y) as usize;
                    let lz = (wz - chunk_origin.z) as usize;
                    let idx = lx
                        + (CHUNK_DIM as usize) * ly
                        + (CHUNK_DIM as usize) * (CHUNK_DIM as usize) * lz;
                    mask[idx] = true;
                }
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_is_deterministic() {
        let a = build_tunnels_for_chunk(42, ChunkCoord(IVec3::new(3, 0, -1)));
        let b = build_tunnels_for_chunk(42, ChunkCoord(IVec3::new(3, 0, -1)));
        assert_eq!(a.len(), b.len());
        for (ta, tb) in a.iter().zip(b.iter()) {
            assert_eq!(ta.spheres.len(), tb.spheres.len());
            assert_eq!(ta.aabb_min, tb.aabb_min);
            assert_eq!(ta.aabb_max, tb.aabb_max);
        }
    }

    #[test]
    fn most_chunks_have_no_carvers() {
        // ~15% of chunks should have any tunnels; verify across a
        // broad scan that the rate is in the right ballpark.
        let mut hits = 0;
        let n = 400;
        for cx in 0..20 {
            for cz in 0..20 {
                let t = build_tunnels_for_chunk(42, ChunkCoord(IVec3::new(cx, 0, cz)));
                if !t.is_empty() {
                    hits += 1;
                }
            }
        }
        let rate = hits as f32 / n as f32;
        assert!(
            (0.08..0.25).contains(&rate),
            "carver chunk-firing rate {rate:.3} outside expected ~0.15 band"
        );
    }

    #[test]
    fn some_chunks_produce_tunnels_with_many_spheres() {
        // Across the same scan, at least one firing chunk should
        // produce a tunnel with substantial length — proves the walk
        // actually walks instead of bailing immediately.
        let mut max_spheres = 0;
        for cx in 0..20 {
            for cz in 0..20 {
                for t in build_tunnels_for_chunk(42, ChunkCoord(IVec3::new(cx, 0, cz))) {
                    max_spheres = max_spheres.max(t.spheres.len());
                }
            }
        }
        assert!(
            max_spheres >= 30,
            "longest tunnel has {max_spheres} spheres — walker exiting too early?"
        );
    }

    #[test]
    fn sphere_carves_unit_sphere_correctly() {
        let s = CarverSphere {
            center: Vec3::new(0.5, 0.5, 0.5),
            h_radius: 5.0,
            v_radius: 5.0,
            floor_level: -1.0,
        };
        // Origin is inside.
        assert!(sphere_carves(&s, 0, 0, 0));
        // 4 blocks away in xz: inside (4/5 = 0.8, dist²=0.64<1).
        assert!(sphere_carves(&s, 4, 0, 0));
        // 5 blocks away: at boundary (1.0 >= 1, not inside per strict <).
        assert!(!sphere_carves(&s, 5, 0, 0));
    }

    #[test]
    fn sphere_floor_level_cuts_bottom() {
        let s = CarverSphere {
            center: Vec3::new(0.5, 10.5, 0.5),
            h_radius: 4.0,
            v_radius: 4.0,
            floor_level: -0.5,
        };
        // Directly below center by 3 (yd = -3/4 = -0.75, below floor).
        assert!(!sphere_carves(&s, 0, 7, 0));
        // Below center by 1 (yd = -0.25, above floor): carved.
        assert!(sphere_carves(&s, 0, 9, 0));
    }

    #[test]
    fn rasterize_into_mask_marks_correct_voxels() {
        // Build a synthetic single-sphere tunnel and rasterise.
        let mut tunnel = CarverTunnel::new();
        tunnel.push_sphere(CarverSphere {
            center: Vec3::new(16.5, 16.5, 16.5),
            h_radius: 3.0,
            v_radius: 3.0,
            floor_level: -1.0,
        });
        let mut mask = vec![false; (CHUNK_DIM as usize).pow(3)];
        let chunk_origin = IVec3::new(0, 0, 0);
        rasterize_into_mask(&tunnel, chunk_origin, &mut mask);
        let count = mask.iter().filter(|b| **b).count();
        // A radius-3 sphere has volume 4/3 π r³ ≈ 113 voxels.
        assert!(
            (80..160).contains(&count),
            "sphere voxel count {count} outside expected ~113 range"
        );
    }
}
