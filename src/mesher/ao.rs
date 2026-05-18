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

use crate::mesher::Face;
use crate::voxel::block::Block;

/// Compute the four-corner AO for a face on the cube at the origin.
///
/// `query(dx, dy, dz)` returns the block at the offset, or `None` when the
/// caller can't supply that voxel (out-of-chunk and the neighbour chunk
/// isn't loaded). Out-of-bounds samples are treated as *unoccluded* —
/// erring on the side of brighter rather than darker at the world edge.
///
/// Returns `[ao_c0, ao_c1, ao_c2, ao_c3]` matching the corner order used by
/// the mesher's quad emitter.
pub fn corner_ao_at<F>(face: Face, query: F) -> [u8; 4]
where
    F: Fn(i32, i32, i32) -> Option<Block>,
{
    // `n_off` is the offset to the face's outward neighbour (where AO is
    // measured). `plane` is the 4 in-plane neighbour offsets in winding
    // order — these define the (side1, side2, corner) sampling triple for
    // each corner of the face.
    let (n_off, plane) = match face {
        Face::PosX => ((1, 0, 0), [(0, 1, 0), (0, 0, 1), (0, -1, 0), (0, 0, -1)]),
        Face::NegX => ((-1, 0, 0), [(0, 1, 0), (0, 0, -1), (0, -1, 0), (0, 0, 1)]),
        Face::PosY => ((0, 1, 0), [(-1, 0, 0), (0, 0, 1), (1, 0, 0), (0, 0, -1)]),
        Face::NegY => ((0, -1, 0), [(1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, 0, -1)]),
        Face::PosZ => ((0, 0, 1), [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]),
        Face::NegZ => ((0, 0, -1), [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0)]),
    };

    // Closure: is the block at `(n_off + off)` solid (i.e. anything but Air)?
    let is_solid = |off: (i32, i32, i32)| -> bool {
        let p = (n_off.0 + off.0, n_off.1 + off.1, n_off.2 + off.2);
        query(p.0, p.1, p.2)
            .map(|b| b != Block::Air)
            .unwrap_or(false)
    };

    // The four edges (side1+side2 pairs) wrapping around the face's plane.
    // Each adjacent pair shares one corner of the quad.
    let plane_solid: [bool; 4] = [
        is_solid(plane[0]),
        is_solid(plane[1]),
        is_solid(plane[2]),
        is_solid(plane[3]),
    ];
    let corner_solid = |a: (i32, i32, i32), b: (i32, i32, i32)| -> bool {
        is_solid((a.0 + b.0, a.1 + b.1, a.2 + b.2))
    };

    // Corner i uses plane[pairs[i].0] + plane[pairs[i].1] + their diagonal.
    let pairs = [(3, 0), (0, 1), (1, 2), (2, 3)];
    let mut out = [3u8; 4];
    for (i, (a, b)) in pairs.iter().enumerate() {
        let s1 = plane_solid[*a];
        let s2 = plane_solid[*b];
        let c = corner_solid(plane[*a], plane[*b]);
        out[i] = ao_value(s1, s2, c);
    }
    out
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
        // For the PosY face at the origin, corner 0 samples plane[3] + plane[0]
        // after the +Y normal offset:
        //   plane[3] = (0, 0, -1)  → side1 offset (0, 1, -1)
        //   plane[0] = (-1, 0, 0)  → side2 offset (-1, 1, 0)
        let solids: [(i32, i32, i32); 2] = [(0, 1, -1), (-1, 1, 0)];
        let ao = corner_ao_at(Face::PosY, |x, y, z| {
            if solids.iter().any(|s| *s == (x, y, z)) {
                Some(Block::Stone)
            } else {
                Some(Block::Air)
            }
        });
        assert_eq!(ao[0], 0, "corner 0 should be fully occluded (two sides solid)");
    }

    #[test]
    fn one_side_solid_returns_two() {
        // PosY, only one of the two sides for corner 0 is solid.
        let ao = corner_ao_at(Face::PosY, |x, y, z| {
            if (x, y, z) == (0, 1, -1) {
                Some(Block::Stone)
            } else {
                Some(Block::Air)
            }
        });
        assert_eq!(ao[0], 2, "one solid neighbour -> 2");
    }
}
