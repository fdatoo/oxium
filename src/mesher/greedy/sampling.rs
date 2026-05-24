//! Boundary-aware block and light sampling for greedy meshing.

use crate::mesher::Face;
use crate::voxel::block::Block;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::LocalPos;
use glam::UVec3;

use super::D;

/// Read-only view over a chunk plus its six face neighbours.
pub(super) struct Sampler<'a> {
    chunk: &'a DenseChunk,
    neighbors: &'a [Option<&'a DenseChunk>; 6],
}

impl<'a> Sampler<'a> {
    pub(super) fn new(chunk: &'a DenseChunk, neighbors: &'a [Option<&'a DenseChunk>; 6]) -> Self {
        Self { chunk, neighbors }
    }

    /// Read a block at `(x, y, z)`, crossing into a face neighbour when exactly
    /// one axis is out of range.
    ///
    /// Multi-axis corner samples return `None`; AO treats that as unoccluded.
    /// Missing face neighbours also return `None`; visibility code decides
    /// whether that should be conservative visible geometry or suppressed water.
    pub(super) fn block_at(&self, x: i32, y: i32, z: i32) -> Option<Block> {
        let dim = D as i32;
        let in_range = x >= 0 && y >= 0 && z >= 0 && x < dim && y < dim && z < dim;
        if in_range {
            return Some(
                self.chunk
                    .get(LocalPos(UVec3::new(x as u32, y as u32, z as u32))),
            );
        }

        let out_x = (x < 0) as i32 + (x >= dim) as i32;
        let out_y = (y < 0) as i32 + (y >= dim) as i32;
        let out_z = (z < 0) as i32 + (z >= dim) as i32;
        if out_x + out_y + out_z >= 2 {
            return None;
        }

        let (face, lx, ly, lz) = if x < 0 {
            (Face::NegX, (dim - 1) as u32, y as u32, z as u32)
        } else if x >= dim {
            (Face::PosX, 0u32, y as u32, z as u32)
        } else if y < 0 {
            (Face::NegY, x as u32, (dim - 1) as u32, z as u32)
        } else if y >= dim {
            (Face::PosY, x as u32, 0u32, z as u32)
        } else if z < 0 {
            (Face::NegZ, x as u32, y as u32, (dim - 1) as u32)
        } else {
            (Face::PosZ, x as u32, y as u32, 0u32)
        };
        let n = self.neighbors[face as usize]?;
        Some(n.get(LocalPos(UVec3::new(lx, ly, lz))))
    }

    /// Read packed `(sky << 4) | block` light at a face-adjacent sample point.
    ///
    /// In-range samples read the source chunk. One-axis-out samples prefer the
    /// real neighbour if it has non-zero light; otherwise they clamp to this
    /// chunk's boundary to avoid baking black seams while neighbours stream in.
    pub(super) fn light_at(&self, x: i32, y: i32, z: i32) -> u8 {
        let dim = D as i32;
        let in_range = x >= 0 && y >= 0 && z >= 0 && x < dim && y < dim && z < dim;
        if in_range {
            return pack_light(self.chunk, x as u32, y as u32, z as u32);
        }

        let own_x = x.clamp(0, dim - 1) as u32;
        let own_y = y.clamp(0, dim - 1) as u32;
        let own_z = z.clamp(0, dim - 1) as u32;
        let own_light = pack_light(self.chunk, own_x, own_y, own_z);

        let out_x = (x < 0) as i32 + (x >= dim) as i32;
        let out_y = (y < 0) as i32 + (y >= dim) as i32;
        let out_z = (z < 0) as i32 + (z >= dim) as i32;
        if out_x + out_y + out_z >= 2 {
            return own_light;
        }

        let (face_idx, lx, ly, lz) = if x < 0 {
            (Face::NegX as usize, (dim - 1) as u32, y as u32, z as u32)
        } else if x >= dim {
            (Face::PosX as usize, 0u32, y as u32, z as u32)
        } else if y < 0 {
            (Face::NegY as usize, x as u32, (dim - 1) as u32, z as u32)
        } else if y >= dim {
            (Face::PosY as usize, x as u32, 0u32, z as u32)
        } else if z < 0 {
            (Face::NegZ as usize, x as u32, y as u32, (dim - 1) as u32)
        } else {
            (Face::PosZ as usize, x as u32, y as u32, 0u32)
        };
        if let Some(n) = self.neighbors[face_idx] {
            let neighbor_light = pack_light(n, lx, ly, lz);
            if neighbor_light != 0 {
                return neighbor_light;
            }
        }
        own_light
    }
}

fn pack_light(chunk: &DenseChunk, lx: u32, ly: u32, lz: u32) -> u8 {
    let idx = LocalPos(UVec3::new(lx, ly, lz)).to_index();
    let brightness = crate::voxel::chunk::rgb_brightness(chunk.block_rgb[idx]);
    (chunk.sky_light[idx] & 0x0F) << 4 | (brightness & 0x0F)
}
