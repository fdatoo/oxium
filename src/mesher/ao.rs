//! Baked vertex ambient occlusion.
//!
//! "Ambient occlusion" approximates the soft shadowing you get in corners
//! where light has trouble reaching: a corner with three occluding
//! neighbours is darker than one with none. We *bake* this at mesh time
//! rather than computing it per-pixel because:
//!
//! - Voxel worlds change geometry slowly; per-vertex is cheaper than
//!   per-pixel screen-space AO.
//! - Storing a 2-bit `ao` value per vertex (0..=3) packs into the same
//!   16-byte `Vertex` we already have.
//!
//! The classic Minecraft AO formula uses three samples around each face
//! corner: the two "sides" sharing the corner along the face's plane plus
//! the diagonal "corner" block. If both sides are solid, the corner is
//! fully occluded (= 0). Otherwise the AO value is `3 − occupied_count`.
//!
//! Corner index convention (must match `greedy::emit`'s corner layout):
//!
//!   index 0 → (-u, -v)
//!   index 1 → (+u, -v)
//!   index 2 → (+u, +v)
//!   index 3 → (-u, +v)
//!
//! Per face, the `(u_axis, v_axis)` directions are the same ones the
//! greedy mesher uses to unmap `(slice, u, v)` back into world space.
//! Keeping that mapping consistent here is what makes adjacent quads agree
//! on the AO value at a shared vertex — without it, neighbouring grass
//! tops sample *different* occluders for the same world-space corner, and
//! the resulting AO discontinuity reads as visible block-sized shadow
//! tiles that don't flow into each other.

use crate::mesher::Face;
use crate::voxel::block::Block;

/// Compute the four-corner AO for a face on the cube at the origin.
///
/// `query(dx, dy, dz)` returns the block at the offset, or `None` when the
/// caller can't supply that voxel (out-of-chunk and the neighbour chunk
/// isn't loaded). Out-of-bounds samples are treated as *unoccluded* —
/// erring on the side of brighter rather than darker at the world edge.
///
/// Returns `[ao_c0, ao_c1, ao_c2, ao_c3]` in the corner order described in
/// the module doc, which matches the mesher's vertex emission order.
pub fn corner_ao_at<F>(face: Face, query: F) -> [u8; 4]
where
    F: Fn(i32, i32, i32) -> Option<Block>,
{
    // `normal` is the unit step from the cube into the air side of the
    // face; `u_unit` / `v_unit` are unit steps in the face's in-plane
    // axes, in the same orientation the greedy mesher uses.
    let (normal, u_unit, v_unit) = face_axes(face);

    let is_solid = |off: (i32, i32, i32)| -> bool {
        query(off.0, off.1, off.2)
            .map(|b| b != Block::Air)
            .unwrap_or(false)
    };

    let mut out = [3u8; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        // Corner i sits at (u_sign · u_unit + v_sign · v_unit) from the
        // cube origin (plus the face's normal, on the air side). The two
        // "side" occluders are one step out along each axis; the "diag"
        // occluder is one step along both.
        let u_sign = if i == 0 || i == 3 { -1 } else { 1 };
        let v_sign = if i == 0 || i == 1 { -1 } else { 1 };
        let side1 = add(normal, scale(u_unit, u_sign));
        let side2 = add(normal, scale(v_unit, v_sign));
        let diag = add(side1, scale(v_unit, v_sign));
        *slot = ao_value(is_solid(side1), is_solid(side2), is_solid(diag));
    }
    out
}

/// Per-face `(normal, u_unit, v_unit)` triple, matching the axis mapping
/// used by `greedy::mask::greedy_one_face`. Changing one without the other
/// would re-introduce the corner-mismatch artefact this function exists
/// to prevent.
fn face_axes(face: Face) -> (Off, Off, Off) {
    match face {
        Face::PosY => ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
        Face::NegY => ((0, -1, 0), (0, 0, 1), (1, 0, 0)),
        Face::PosX => ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
        Face::NegX => ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
        Face::PosZ => ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
        Face::NegZ => ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
    }
}

type Off = (i32, i32, i32);
fn add(a: Off, b: Off) -> Off {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}
fn scale(a: Off, k: i32) -> Off {
    (a.0 * k, a.1 * k, a.2 * k)
}

/// Minecraft-style AO function. The "both sides solid" shortcut is
/// important — without it a corner with side1=side2=true but corner=false
/// would map to `3 − 2 = 1` instead of the visually correct 0.
fn ao_value(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        return 0;
    }
    let occ = side1 as u8 + side2 as u8 + corner as u8;
    3u8.saturating_sub(occ)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::block::Block;

    #[test]
    fn no_occluders_returns_three() {
        let ao = corner_ao_at(Face::PosY, |_x, _y, _z| Some(Block::Air));
        assert_eq!(ao, [3, 3, 3, 3]);
    }

    #[test]
    fn two_sides_solid_returns_zero() {
        // PosY corner 0 is at (-u, -v) = (-x, -z). Both side occluders
        // sit one step along the air side (y=+1) in -x and -z.
        let solids: [(i32, i32, i32); 2] = [(-1, 1, 0), (0, 1, -1)];
        let ao = corner_ao_at(Face::PosY, |x, y, z| {
            if solids.contains(&(x, y, z)) {
                Some(Block::Stone)
            } else {
                Some(Block::Air)
            }
        });
        assert_eq!(
            ao[0], 0,
            "corner 0 should be fully occluded (two sides solid)"
        );
    }

    #[test]
    fn one_side_solid_returns_two() {
        // PosY corner 0 with only one of its two side occluders solid.
        let ao = corner_ao_at(Face::PosY, |x, y, z| {
            if (x, y, z) == (0, 1, -1) {
                Some(Block::Stone)
            } else {
                Some(Block::Air)
            }
        });
        assert_eq!(ao[0], 2, "one solid neighbour -> 2");
    }

    /// Regression guard: adjacent cells on the same face must agree on the
    /// AO value at a shared vertex. Two PosY-top cells at (0,0,0) and
    /// (1,0,0) share the edge along x=1. With a single stone occluder
    /// floating to one side of that shared edge, the AO computed for
    /// cell A's corner-at-(1,1,0) must equal the AO computed for cell B's
    /// corner-at-(1,1,0). Otherwise the mesher emits two adjacent quads
    /// whose interpolated AO discontinuously snaps at the shared edge,
    /// producing the "shadows don't flow into next tile" artefact.
    #[test]
    fn shared_vertex_ao_agrees_across_adjacent_cells() {
        // Occluder sits one block above and one block in -z from the
        // shared vertex at world (1, 1, 0). This block contributes as a
        // "side" occluder for both cells' corners that touch (1, 1, 0).
        let occluder = (1, 1, -1);
        let q = |x: i32, y: i32, z: i32| -> Option<Block> {
            if (x, y, z) == occluder {
                Some(Block::Stone)
            } else {
                Some(Block::Air)
            }
        };

        // Cell A at (0, 0, 0). Its corner 1 is at world (1, 1, 0).
        let ao_a = corner_ao_at(Face::PosY, q);
        // Cell B at (1, 0, 0). Its corner 0 is at world (1, 1, 0).
        let ao_b = corner_ao_at(Face::PosY, |dx, dy, dz| q(1 + dx, dy, dz));

        assert_eq!(
            ao_a[1], ao_b[0],
            "AO at shared vertex must agree between adjacent cells"
        );
    }

    /// Same agreement check on a vertical face, since the bug also
    /// affected `Face::PosX` / `NegX` / `PosZ` / `NegZ`.
    #[test]
    fn shared_vertex_ao_agrees_on_vertical_face() {
        // PosX face, u_axis=z, v_axis=y. Two cells stacked along +y at
        // (0, 0, 0) and (0, 1, 0) share the edge at y=1. Cell A's
        // corner 3 (=(−u,+v)=(z=0,y=1)) and Cell B's corner 0
        // (=(−u,−v)=(z=0,y=1)) both sit at world (1, 1, 0).
        let occluder = (1, 1, -1);
        let q = |x: i32, y: i32, z: i32| -> Option<Block> {
            if (x, y, z) == occluder {
                Some(Block::Stone)
            } else {
                Some(Block::Air)
            }
        };
        let ao_a = corner_ao_at(Face::PosX, q);
        let ao_b = corner_ao_at(Face::PosX, |dx, dy, dz| q(dx, 1 + dy, dz));
        assert_eq!(
            ao_a[3], ao_b[0],
            "PosX shared vertex AO must agree between vertically-adjacent cells"
        );
    }
}
