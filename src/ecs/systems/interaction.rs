//! Block-interaction system: raycast from the eye, place/break on click.
//!
//! Once per frame the system:
//!
//! 1. Casts a ray from the camera eye along the look direction.
//! 2. Writes the result into the player's `CursorTarget` so the renderer
//!    can paint the wireframe highlight.
//! 3. If the player edge-pressed left-mouse (`break_`), sets the targeted
//!    block to `Air`.
//! 4. If they edge-pressed right-mouse (`place`), sets the cell *adjacent
//!    to the targeted face* to the currently-selected block — unless
//!    placing there would overlap the player AABB.
//!
//! Returns the list of chunks that became dirty (relight + remesh) so the
//! app schedule can spawn follow-up jobs.

use crate::ecs::GameEcs;
use crate::ecs::components::{Aabb, Camera, CursorTarget, PlayerInput, Position, Selected};
use crate::render::hud::HOTBAR_BLOCKS;
use crate::voxel::block::Block;
use crate::voxel::coords::{BlockPos, ChunkCoord};
use crate::voxel::raycast::raycast;
use crate::voxel::world::World;
use glam::{IVec3, Vec3};

/// Maximum interaction reach, in blocks.
const REACH: f32 = 6.0;

/// Run interaction. Returns chunks whose meshes need recomputing.
pub fn interaction(ecs: &mut GameEcs, world: &mut World) -> Vec<ChunkCoord> {
    let mut q = ecs
        .world
        .query_one::<(
            &Position,
            &Camera,
            &Aabb,
            &mut PlayerInput,
            &mut Selected,
            &mut CursorTarget,
        )>(ecs.player)
        .unwrap();
    let (pos, cam, aabb, input, sel, target) = q.get().unwrap();

    // Build the look direction from yaw/pitch (matches the camera basis
    // used by movement + view-proj).
    let (sy, cy) = cam.yaw.sin_cos();
    let (sp, cp) = cam.pitch.sin_cos();
    let eye = pos.0 + cam.eye_offset;
    let dir = Vec3::new(cy * cp, sp, sy * cp).normalize();
    let hit = raycast(world, eye, dir, REACH);
    target.hit = hit.map(|h| (h.block, h.face));

    let mut dirty = Vec::new();
    if let Some(h) = hit {
        if input.pick_block {
            if let Some(block) = world.get_block(h.block)
                && HOTBAR_BLOCKS.contains(&Some(block))
            {
                sel.0 = block;
            }
        } else if input.break_ {
            dirty = world.set_block(h.block, Block::Air);
        } else if input.place {
            // Place the new block adjacent to the face the ray entered
            // through — that's where the player wants the cube to land.
            let off = h.face.normal();
            let target_pos = BlockPos(h.block.0 + IVec3::new(off[0], off[1], off[2]));
            if !overlaps_player(target_pos, pos.0, aabb.half) {
                dirty = world.set_block(target_pos, sel.0);
            }
        }
    }
    // Consume edge triggers so a held click doesn't repeat.
    input.break_ = false;
    input.place = false;
    input.pick_block = false;
    dirty
}

/// True when a 1×1×1 block at `block` overlaps the player AABB. Used to
/// stop the place action from suffocating the player inside a new block.
fn overlaps_player(block: BlockPos, feet: Vec3, half: Vec3) -> bool {
    let bmin = Vec3::new(block.0.x as f32, block.0.y as f32, block.0.z as f32);
    let bmax = bmin + Vec3::ONE;
    let pmin = Vec3::new(feet.x - half.x, feet.y, feet.z - half.z);
    let pmax = Vec3::new(feet.x + half.x, feet.y + 2.0 * half.y, feet.z + half.z);
    pmin.x < bmax.x
        && pmax.x > bmin.x
        && pmin.y < bmax.y
        && pmax.y > bmin.y
        && pmin.z < bmax.z
        && pmax.z > bmin.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::{PlayerInput, Selected};
    use crate::voxel::chunk::{DenseChunk, PalettedChunk};
    use crate::voxel::coords::{ChunkCoord, LocalPos};
    use glam::{IVec3, UVec3};

    #[test]
    fn pick_block_selects_looked_at_hotbar_block() {
        let mut ecs = GameEcs::new(Vec3::new(0.5, 0.0, 0.5));
        {
            let mut q = ecs
                .world
                .query_one::<(&mut PlayerInput, &mut Selected)>(ecs.player)
                .unwrap();
            let (input, selected) = q.get().unwrap();
            input.pick_block = true;
            selected.0 = Block::Stone;
        }

        let mut dense = DenseChunk::empty();
        dense.set(LocalPos(UVec3::new(3, 1, 0)), Block::Grass);
        let mut world = World::new(42);
        world.insert(ChunkCoord(IVec3::ZERO), PalettedChunk::compress(&dense));

        let dirty = interaction(&mut ecs, &mut world);
        assert!(dirty.is_empty());
        let mut q = ecs
            .world
            .query_one::<(&PlayerInput, &Selected)>(ecs.player)
            .unwrap();
        let (input, selected) = q.get().unwrap();
        assert!(!input.pick_block);
        assert_eq!(selected.0, Block::Grass);
    }
}
