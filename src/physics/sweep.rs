//! Axis-by-axis swept AABB collision against the voxel world.
//!
//! "Swept" means we advance the AABB along the velocity vector and clip
//! the motion against the first solid voxel it overlaps. "Axis-by-axis"
//! means we resolve one axis at a time (Y first, then X, then Z) — the
//! classic remedy for the "diagonal corner snag" bug that simultaneous
//! resolution suffers from.
//!
//! Why Y first? So a falling player lands on the floor *before* lateral
//! motion is applied — otherwise a diagonal jump can clip the corner of
//! a ledge and lose its footing.
//!
//! The DDA-style step uses a maximum sub-step of 0.5 blocks so a fast
//! arrow-key dash never tunnels through a thin wall. With `dt` clamped to
//! 100 ms and max walk speed 5 m/s, the natural per-frame displacement is
//! already well under 0.5; the cap is a defensive belt-and-braces.

use crate::voxel::coords::BlockPos;
use crate::voxel::world::World;
use glam::{IVec3, Vec3};

/// World gravity in blocks per second². Pulls −Y at ~Earth-ish strength.
/// (Real-world 9.81 feels floaty in a voxel game; 28 m/s² is the Minecraft
/// reference value and gives satisfying snappy jumps with `jump_v ≈ 8.4`.)
pub const GRAVITY: f32 = -28.0;

/// Epsilon used to (a) decide that residual motion is too small to bother
/// stepping and (b) shrink the upper edge of the AABB-to-voxel test so a
/// flat floor at integer y doesn't read as a 1-block-tall hit.
const EPS: f32 = 1e-3;

/// Output of [`sweep_player`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CollisionResult {
    /// Final feet position after sweeping.
    pub pos: Vec3,
    /// Velocity adjusted (axes that hit get zeroed).
    pub vel: Vec3,
    /// True when the downward sweep made contact.
    pub grounded: bool,
}

/// Resolve `pos`/`vel` for one tick against `world`'s solid voxels.
///
/// `half` is the half-extents of the player AABB (e.g. `(0.3, 0.9, 0.3)`
/// for a 0.6 × 1.8 × 0.6 m humanoid). `pos` is the *feet* position; the
/// AABB extends from `(pos.x − half.x, pos.y, pos.z − half.z)` to
/// `(pos.x + half.x, pos.y + 2·half.y, pos.z + half.z)`.
pub fn sweep_player(world: &World, pos: Vec3, half: Vec3, vel: Vec3, dt: f32) -> CollisionResult {
    let mut p = pos;
    let mut v = vel;
    let mut grounded = false;

    // Order: Y first (so gravity lands before lateral motion), then X, Z.
    let (new_y, hit_y_down) = sweep_axis(world, p, half, 1, v.y * dt);
    p.y = new_y;
    if hit_y_down && v.y <= 0.0 {
        grounded = true;
    }
    if hit_y_down || (v.y > 0.0 && new_y < pos.y + v.y * dt - EPS) {
        v.y = 0.0;
    }

    let (new_x, hit_x) = sweep_axis(world, p, half, 0, v.x * dt);
    p.x = new_x;
    if hit_x {
        v.x = 0.0;
    }

    let (new_z, hit_z) = sweep_axis(world, p, half, 2, v.z * dt);
    p.z = new_z;
    if hit_z {
        v.z = 0.0;
    }

    CollisionResult {
        pos: p,
        vel: v,
        grounded,
    }
}

/// Move `pos[axis]` by `delta`, stopping just before the first solid
/// block hit. Returns `(new_value_for_pos[axis], hit)`. `pos` is read as
/// *feet*; the AABB extends Y upward from there.
fn sweep_axis(world: &World, pos: Vec3, half: Vec3, axis: usize, delta: f32) -> (f32, bool) {
    if delta.abs() < EPS {
        return (pos[axis], false);
    }
    let sign = delta.signum();
    // Cap each sub-step at 0.5 block so the AABB never jumps past a thin
    // wall between samples.
    let mut remaining = delta.abs();
    let mut p = pos;
    let mut hit = false;
    while remaining > EPS {
        let step = remaining.min(0.5);
        let mut next = p;
        next[axis] += sign * step;
        if player_aabb_blocked(world, next, half) {
            hit = true;
            break;
        }
        p = next;
        remaining -= step;
    }
    (p[axis], hit)
}

/// True when the player AABB at the given feet position overlaps any
/// solid voxel in `world`. An unloaded chunk is conservatively treated as
/// solid so the player doesn't fall through pending terrain.
fn player_aabb_blocked(world: &World, feet: Vec3, half: Vec3) -> bool {
    // AABB in world coords: feet → feet + (0, 2·half.y, 0), expanded by half on X/Z.
    let min = Vec3::new(feet.x - half.x, feet.y, feet.z - half.z);
    let max = Vec3::new(feet.x + half.x, feet.y + 2.0 * half.y, feet.z + half.z);
    // Inclusive-floor on min, exclusive-floor on max (minus EPS) so a
    // player flush against the top of a block doesn't read the block
    // above as overlapping.
    let x0 = min.x.floor() as i32;
    let x1 = (max.x - EPS).floor() as i32;
    let y0 = min.y.floor() as i32;
    let y1 = (max.y - EPS).floor() as i32;
    let z0 = min.z.floor() as i32;
    let z1 = (max.z - EPS).floor() as i32;

    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                match world.get_block(BlockPos(IVec3::new(x, y, z))) {
                    Some(b) if world.registry.info(b).solid => return true,
                    Some(_) => {}        // non-solid (air, water) — walk through
                    None => return true, // unloaded chunk — solid by default
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::block::Block;
    use crate::voxel::chunk::{DenseChunk, PalettedChunk};
    use crate::voxel::coords::{ChunkCoord, LocalPos};
    use crate::voxel::world::{ChunkSlot, World};
    use glam::UVec3;

    /// World with a 1-block-thick stone floor at y=0 in chunk (0, 0, 0)
    /// and all the surrounding chunks pre-air-filled so the player can
    /// move freely above it.
    fn world_with_floor() -> World {
        let mut w = World::new(0);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let coord = ChunkCoord(IVec3::new(dx, dy, dz));
                    let mut dense = DenseChunk::empty();
                    if dx == 0 && dy == 0 && dz == 0 {
                        for z in 0..32 {
                            for x in 0..32 {
                                dense.set(LocalPos(UVec3::new(x, 0, z)), Block::Stone);
                            }
                        }
                    }
                    let paletted = PalettedChunk::compress(&dense);
                    w.insert(coord, paletted);
                }
            }
        }
        w
    }

    #[test]
    fn falls_onto_floor() {
        let w = world_with_floor();
        let res = sweep_player(
            &w,
            Vec3::new(5.0, 10.0, 5.0),
            Vec3::new(0.3, 0.9, 0.3),
            Vec3::new(0.0, -100.0, 0.0),
            1.0,
        );
        assert!(res.grounded, "should be grounded after falling onto floor");
        assert!(
            res.pos.y >= 1.0 - EPS,
            "feet should rest just above y=1, got {}",
            res.pos.y
        );
    }

    #[test]
    fn walks_into_wall_stops() {
        let mut w = world_with_floor();
        // Add a single wall block at (2, 1, 5).
        if let Some(ChunkSlot::Stored { data, .. }) = w.chunks.get_mut(&ChunkCoord(IVec3::ZERO)) {
            let mut dense = data.decompress();
            dense.set(LocalPos(UVec3::new(2, 1, 5)), Block::Stone);
            *data = std::sync::Arc::new(PalettedChunk::compress(&dense));
        }
        // Starting at x=0.5, walking +X at 10 m/s for 1s should be blocked
        // by the wall whose -X face is at x=2.
        let res = sweep_player(
            &w,
            Vec3::new(0.5, 1.0, 5.0),
            Vec3::new(0.3, 0.9, 0.3),
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
        );
        assert!(
            res.pos.x < 2.0 - 0.3 + 0.05,
            "should stop before the wall; got x={}",
            res.pos.x
        );
        // Velocity on the blocked axis should be zeroed.
        assert!(res.vel.x.abs() < EPS, "x velocity should be zeroed");
    }

    #[test]
    fn wall_slide_keeps_z_motion() {
        let mut w = world_with_floor();
        // Wall on +X side blocking lateral X, but free along Z.
        if let Some(ChunkSlot::Stored { data, .. }) = w.chunks.get_mut(&ChunkCoord(IVec3::ZERO)) {
            let mut dense = data.decompress();
            for y in 1..3 {
                dense.set(LocalPos(UVec3::new(2, y, 5)), Block::Stone);
            }
            *data = std::sync::Arc::new(PalettedChunk::compress(&dense));
        }
        let res = sweep_player(
            &w,
            Vec3::new(0.5, 1.0, 5.0),
            Vec3::new(0.3, 0.9, 0.3),
            Vec3::new(5.0, 0.0, 3.0),
            1.0,
        );
        assert!(res.pos.x < 2.0, "x should hit the wall: got {}", res.pos.x);
        assert!(
            res.pos.z > 5.0,
            "z should keep sliding past start: got {}",
            res.pos.z
        );
    }
}
