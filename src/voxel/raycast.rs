//! Amanatides–Woo voxel DDA raycast.
//!
//! Walks the integer voxel grid one cell at a time along a ray, stepping
//! on whichever of the three axes will cross a cell boundary next. This is
//! both fast (a handful of `f32` ops per step) and exact (every voxel the
//! ray pierces is visited in order).
//!
//! Output also reports which *face* of the hit block the ray entered
//! through — needed by the interaction system to decide where a "place
//! block" action puts the new voxel.

use crate::mesher::Face;
use crate::voxel::block::Block;
use crate::voxel::coords::BlockPos;
use crate::voxel::world::World;
use glam::{IVec3, Vec3};

/// What the raycast hit.
#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    /// The integer-coordinate block the ray collided with.
    pub block: BlockPos,
    /// The face of `block` that the ray entered through. Used by the place
    /// action to compute the cell adjacent to the hit face.
    pub face: Face,
    /// Distance from `origin` to the hit point along the ray.
    pub distance: f32,
}

/// Cast a ray through the loaded world. Returns the first non-air,
/// non-water block hit, or `None` if no hit occurs within `max_dist`.
///
/// Water is treated as a *non-collidable* medium — the cursor passes
/// through to whatever solid is on the far side.
///
/// An unloaded chunk along the ray's path stops the cast and returns
/// `None`: better than silently treating it as an air column.
pub fn raycast(world: &World, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RaycastHit> {
    let dir = dir.normalize_or_zero();
    if dir.length_squared() < 1e-6 {
        return None;
    }

    // Current voxel coords (floor of the float origin).
    let mut block = IVec3::new(
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );
    // Step direction (±1, 0) on each axis.
    let step = IVec3::new(
        dir.x.signum() as i32,
        dir.y.signum() as i32,
        dir.z.signum() as i32,
    );

    // `next_boundary` returns the world-coord of the next cell boundary
    // the ray will cross on the given axis.
    let next_boundary = |coord: f32, step: i32| -> f32 {
        if step > 0 {
            coord.floor() + 1.0
        } else {
            coord.floor()
        }
    };
    // `t_max[axis]` = ray-parameter `t` at which the next boundary on that
    // axis will be crossed. Infinity if `dir[axis] == 0`.
    let mut t_max = Vec3::new(
        if dir.x != 0.0 {
            (next_boundary(origin.x, step.x) - origin.x) / dir.x
        } else {
            f32::INFINITY
        },
        if dir.y != 0.0 {
            (next_boundary(origin.y, step.y) - origin.y) / dir.y
        } else {
            f32::INFINITY
        },
        if dir.z != 0.0 {
            (next_boundary(origin.z, step.z) - origin.z) / dir.z
        } else {
            f32::INFINITY
        },
    );
    // How much `t` advances per full cell on each axis.
    let t_delta = Vec3::new(
        if dir.x != 0.0 { (1.0 / dir.x).abs() } else { f32::INFINITY },
        if dir.y != 0.0 { (1.0 / dir.y).abs() } else { f32::INFINITY },
        if dir.z != 0.0 { (1.0 / dir.z).abs() } else { f32::INFINITY },
    );

    // Inside-block start: hit immediately.
    if let Some(b) = world.get_block(BlockPos(block)) {
        if b != Block::Air && b != Block::Water {
            return Some(RaycastHit {
                block: BlockPos(block),
                face: Face::PosY,
                distance: 0.0,
            });
        }
    }

    let mut last_face = Face::PosY;
    let mut traveled;
    loop {
        // Step on the axis with the smallest t_max — that's the next
        // boundary the ray crosses.
        if t_max.x < t_max.y && t_max.x < t_max.z {
            traveled = t_max.x;
            block.x += step.x;
            // We entered the new cell from the *opposite* face of the step.
            last_face = if step.x > 0 { Face::NegX } else { Face::PosX };
            t_max.x += t_delta.x;
        } else if t_max.y < t_max.z {
            traveled = t_max.y;
            block.y += step.y;
            last_face = if step.y > 0 { Face::NegY } else { Face::PosY };
            t_max.y += t_delta.y;
        } else {
            traveled = t_max.z;
            block.z += step.z;
            last_face = if step.z > 0 { Face::NegZ } else { Face::PosZ };
            t_max.z += t_delta.z;
        }
        if traveled > max_dist {
            return None;
        }
        let pos = BlockPos(block);
        match world.get_block(pos) {
            Some(b) if b != Block::Air && b != Block::Water => {
                return Some(RaycastHit {
                    block: pos,
                    face: last_face,
                    distance: traveled,
                });
            }
            Some(_) => continue,
            None => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::{DenseChunk, PalettedChunk};
    use crate::voxel::coords::{ChunkCoord, LocalPos};
    use glam::UVec3;

    fn world_with_block(at: (u32, u32, u32)) -> World {
        let mut w = World::new(0);
        let mut d = DenseChunk::empty();
        d.set(LocalPos(UVec3::new(at.0, at.1, at.2)), Block::Stone);
        w.insert(ChunkCoord(IVec3::ZERO), PalettedChunk::compress(&d));
        w
    }

    #[test]
    fn hits_block_along_x() {
        let w = world_with_block((5, 0, 0));
        let hit = raycast(
            &w,
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.0, 0.0, 0.0),
            10.0,
        )
        .unwrap();
        assert_eq!(hit.block.0, IVec3::new(5, 0, 0));
        assert_eq!(hit.face as u8, Face::NegX as u8);
    }

    #[test]
    fn misses_when_nothing_in_path() {
        let w = world_with_block((5, 0, 0));
        let hit = raycast(
            &w,
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
            10.0,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn max_distance_respected() {
        let w = world_with_block((10, 0, 0));
        let hit = raycast(
            &w,
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.0, 0.0, 0.0),
            5.0,
        );
        assert!(hit.is_none());
    }
}
