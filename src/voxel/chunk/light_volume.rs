//! GPU light-volume blob construction.

use crate::mesher::Face;
use crate::voxel::block::Block;
use crate::voxel::coords::LocalPos;
use glam::UVec3;

use super::{DenseChunk, Neighbors, unpack_rgb};

/// Build the 33^3 Rgba8Unorm blob for a chunk's GPU light volume.
///
/// Each axis spans 0..=32; index 32 reads from the +X/+Y/+Z neighbor's index 0
/// so trilinear sampling at the chunk's far face sees valid neighbour values.
pub fn build_light_volume_blob(
    dense: &DenseChunk,
    neighbors: &Neighbors,
) -> Box<[u8; 33 * 33 * 33 * 4]> {
    let mut buf = vec![0u8; 33 * 33 * 33 * 4].into_boxed_slice();
    let out: &mut [u8; 33 * 33 * 33 * 4] = buf.as_mut().try_into().expect("size mismatch");
    let scale = |v: u8| ((v as u32 * 255 + 7) / 15) as u8;
    for z in 0..33 {
        for y in 0..33 {
            for x in 0..33 {
                let (r, g, b, a) = sample_for_blob(dense, neighbors, x, y, z);
                let idx = (z * 33 * 33 + y * 33 + x) * 4;
                out[idx] = scale(r);
                out[idx + 1] = scale(g);
                out[idx + 2] = scale(b);
                out[idx + 3] = scale(a);
            }
        }
    }
    buf.try_into().expect("size mismatch")
}

fn sample_for_blob(
    dense: &DenseChunk,
    neighbors: &Neighbors,
    x: usize,
    y: usize,
    z: usize,
) -> (u8, u8, u8, u8) {
    let (chunk_src, lx, ly, lz) = resolve_cell(dense, neighbors, x, y, z);
    let idx = LocalPos(UVec3::new(lx as u32, ly as u32, lz as u32)).to_index();
    let block = chunk_src.blocks[idx];
    let (mut r, mut g, mut b) = unpack_rgb(chunk_src.block_rgb[idx]);
    let mut a = chunk_src.sky_light[idx] & 0x0F;

    // Halo fill for opaque cells. The BFS stores 0 in opaque cells, but the
    // shader's trilinear sample can pull them into face corners; copying the
    // max axial neighbour light prevents false dark corners.
    if block != Block::Air {
        for (dx, dy, dz) in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            let nz = z as isize + dz;
            if !(-1..=32).contains(&nx) || !(-1..=32).contains(&ny) || !(-1..=32).contains(&nz) {
                continue;
            }
            let Some((ns, nlx, nly, nlz)) = resolve_cell_signed(dense, neighbors, nx, ny, nz)
            else {
                continue;
            };
            let nidx = LocalPos(UVec3::new(nlx as u32, nly as u32, nlz as u32)).to_index();
            let (nr, ng, nb) = unpack_rgb(ns.block_rgb[nidx]);
            let na = ns.sky_light[nidx] & 0x0F;
            r = r.max(nr);
            g = g.max(ng);
            b = b.max(nb);
            a = a.max(na);
        }
    }
    (r, g, b, a)
}

fn resolve_cell_signed<'a>(
    dense: &'a DenseChunk,
    neighbors: &'a Neighbors,
    x: isize,
    y: isize,
    z: isize,
) -> Option<(&'a DenseChunk, usize, usize, usize)> {
    let negative_axes = (x < 0) as u8 + (y < 0) as u8 + (z < 0) as u8;
    if negative_axes > 1 {
        return None;
    }
    if x < 0 {
        return Some(match neighbors.chunks[Face::NegX as usize] {
            Some(n) => (n, 31, y.clamp(0, 31) as usize, z.clamp(0, 31) as usize),
            None => (dense, 0, y.clamp(0, 31) as usize, z.clamp(0, 31) as usize),
        });
    }
    if y < 0 {
        return Some(match neighbors.chunks[Face::NegY as usize] {
            Some(n) => (n, x.clamp(0, 31) as usize, 31, z.clamp(0, 31) as usize),
            None => (dense, x.clamp(0, 31) as usize, 0, z.clamp(0, 31) as usize),
        });
    }
    if z < 0 {
        return Some(match neighbors.chunks[Face::NegZ as usize] {
            Some(n) => (n, x.clamp(0, 31) as usize, y.clamp(0, 31) as usize, 31),
            None => (dense, x.clamp(0, 31) as usize, y.clamp(0, 31) as usize, 0),
        });
    }
    Some(resolve_cell(
        dense, neighbors, x as usize, y as usize, z as usize,
    ))
}

fn resolve_cell<'a>(
    dense: &'a DenseChunk,
    neighbors: &'a Neighbors,
    x: usize,
    y: usize,
    z: usize,
) -> (&'a DenseChunk, usize, usize, usize) {
    if x == 32 {
        match neighbors.chunks[Face::PosX as usize] {
            Some(n) => (n, 0, y.min(31), z.min(31)),
            None => (dense, 31, y.min(31), z.min(31)),
        }
    } else if y == 32 {
        match neighbors.chunks[Face::PosY as usize] {
            Some(n) => (n, x.min(31), 0, z.min(31)),
            None => (dense, x.min(31), 31, z.min(31)),
        }
    } else if z == 32 {
        match neighbors.chunks[Face::PosZ as usize] {
            Some(n) => (n, x.min(31), y.min(31), 0),
            None => (dense, x.min(31), y.min(31), 31),
        }
    } else {
        (dense, x, y, z)
    }
}
