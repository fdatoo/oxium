//! Level-of-detail: voxel-grid downsampling + a simpler greedy mesher.
//!
//! Rendering 32³ chunks all the way to the horizon is infeasible — at
//! render distance 16, that's ~17k chunks × ~hundreds of greedy quads
//! each. We solve it by collapsing each 2³ (L1) or 4³ (L2) voxel group
//! to a single representative block, then meshing the downsampled chunk
//! with the same per-face logic but per-cell scaled.
//!
//! Visual result: distant terrain looks slightly chunkier, but the
//! silhouette and colour are preserved. LOD is picked per chunk by
//! camera distance — see `render::Renderer::render`.

use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{LocalPos, CHUNK_DIM_U};
use glam::UVec3;
use std::collections::HashMap;

/// A downsampled chunk. `dim` is `CHUNK_DIM_U / factor`. Stored as a flat
/// row-major array (x fastest), like the rest of the engine.
pub struct LodChunk {
    pub dim: u32,
    pub blocks: Vec<Block>,
    /// Packed `(sky << 4) | block` light byte, averaged over the
    /// downsampled group's cells.
    pub light: Vec<u8>,
}

/// Build an `LodChunk` from a `DenseChunk` by collapsing each `factor³`
/// voxel group into one block. `factor` must be 2 (L1) or 4 (L2).
///
/// **Surface-aware vote.** Each `(dx, dz)` column inside the group
/// contributes its *topmost non-air block within the cell*. That
/// contribution is classified as either a **surface** vote (the block
/// directly above it is air — i.e. it's actually visible from outside)
/// or a **bulk** vote (it has a solid neighbour above, so this cell sits
/// underneath the terrain in that column).
///
/// Surface votes win over bulk votes for the cell's representative block.
/// This stops the "stone roof" tiling — where one column's bulk-stone
/// vote was outvoting the grass surface from neighbouring columns — and
/// preserves the visible top across the LOD0/LOD1/LOD2 boundary.
///
/// Light is averaged across the whole group — the eye can't distinguish
/// per-cell light at LOD distances anyway.
pub fn downsample(src: &DenseChunk, factor: u32) -> LodChunk {
    assert!(factor == 2 || factor == 4, "factor must be 2 or 4");
    let dim = CHUNK_DIM_U / factor;
    let len = (dim * dim * dim) as usize;
    let mut blocks = vec![Block::Air; len];
    let mut light = vec![0u8; len];

    for z in 0..dim {
        for y in 0..dim {
            for x in 0..dim {
                let mut surface_counts: HashMap<Block, u32> = HashMap::new();
                let mut bulk_counts: HashMap<Block, u32> = HashMap::new();
                let mut sky_sum: u32 = 0;
                let mut blk_sum: u32 = 0;

                // Walk each column inside the cell top-to-bottom.
                for dz in 0..factor {
                    for dx in 0..factor {
                        let mut found_top = false;
                        for dy_top in 0..factor {
                            let dy = factor - 1 - dy_top;
                            let yl = y * factor + dy;
                            let lp = LocalPos(UVec3::new(
                                x * factor + dx,
                                yl,
                                z * factor + dz,
                            ));
                            let b = src.blocks[lp.to_index()];
                            sky_sum += src.sky_light[lp.to_index()] as u32;
                            blk_sum += src.block_light[lp.to_index()] as u32;
                            if !found_top && b != Block::Air {
                                found_top = true;
                                // Determine surface vs bulk by looking one
                                // block higher. Outside the chunk we treat
                                // the above-block as air (so a column whose
                                // surface coincides with the chunk's top
                                // counts as a surface, not as bulk).
                                let above = if yl + 1 < CHUNK_DIM_U {
                                    src.blocks[LocalPos(UVec3::new(
                                        x * factor + dx,
                                        yl + 1,
                                        z * factor + dz,
                                    ))
                                    .to_index()]
                                } else {
                                    Block::Air
                                };
                                if above == Block::Air {
                                    *surface_counts.entry(b).or_insert(0) += 1;
                                } else {
                                    *bulk_counts.entry(b).or_insert(0) += 1;
                                }
                            }
                        }
                    }
                }

                // Surface votes always win when any exist; bulk votes
                // only matter for cells entirely below the visible
                // terrain.
                let chosen = surface_counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .or_else(|| bulk_counts.into_iter().max_by_key(|(_, c)| *c))
                    .map(|(b, _)| b)
                    .unwrap_or(Block::Air);
                let n = factor * factor * factor;
                let sky = (sky_sum / n) as u8;
                let blk = (blk_sum / n) as u8;
                let idx = (x + y * dim + z * dim * dim) as usize;
                blocks[idx] = chosen;
                light[idx] = (sky.min(15) << 4) | blk.min(15);
            }
        }
    }
    LodChunk { dim, blocks, light }
}

/// Mesh an `LodChunk` using a naive face emitter scaled by `factor`.
///
/// We don't bother with greedy merging at LOD levels — the gains are
/// smaller (the cells are already coarser) and the simpler emitter
/// makes each downsampled cube a single quad per visible face. Also,
/// AO is skipped (it'd compound the downsampling artefacts).
pub fn mesh_lod(
    lod: &LodChunk,
    factor: u32,
    reg: &BlockRegistry,
) -> crate::mesher::ChunkMesh {
    use crate::mesher::{ChunkMesh, Face, Vertex};
    let mut mesh = ChunkMesh::empty();
    let dim = lod.dim as i32;

    // Inside the LOD chunk we read real cells. *Outside* we treat the
    // neighbour as **opaque** instead of air — this is the
    // load-bearing trick of mesh_lod. The job doesn't have neighbour
    // chunks to consult, and the old "out-of-bounds = air" rule was
    // emitting spurious +Y/+X/+Z faces at every chunk boundary, showing
    // up as stone-coloured "shelves" across the rendered landscape.
    // Calling out-of-bounds opaque hides those boundary faces; the
    // neighbouring LOD chunk's own geometry covers the void from its
    // side. At LOD distance the player can't see the tiny single-cell
    // gaps this leaves where adjacent topographies differ.
    let block_at = |x: i32, y: i32, z: i32| -> Block {
        if x < 0 || y < 0 || z < 0 || x >= dim || y >= dim || z >= dim {
            return Block::Stone;
        }
        lod.blocks[(x + y * dim + z * dim * dim) as usize]
    };
    let light_at = |x: i32, y: i32, z: i32| -> u8 {
        if x < 0 || y < 0 || z < 0 || x >= dim || y >= dim || z >= dim {
            return 0xFF;
        }
        lod.light[(x + y * dim + z * dim * dim) as usize]
    };

    for z in 0..dim {
        for y in 0..dim {
            for x in 0..dim {
                let block = block_at(x, y, z);
                if block == Block::Air {
                    continue;
                }
                for face in Face::all() {
                    let [dx, dy, dz] = face.normal();
                    let nb = block_at(x + dx, y + dy, z + dz);
                    if reg.info(nb).opaque {
                        continue;
                    }
                    let light = light_at(x + dx, y + dy, z + dz);
                    let info = reg.info(block);
                    let color = match face {
                        Face::PosY if info.top_color.is_some() => info.top_color.unwrap(),
                        _ => info.color,
                    };
                    let color_u8 = [
                        (color[0] * 255.0) as u8,
                        (color[1] * 255.0) as u8,
                        (color[2] * 255.0) as u8,
                        (color[3] * 255.0) as u8,
                    ];

                    // Each LOD cell spans `factor` voxels — expand the
                    // local coords to keep the LOD mesh in the same world
                    // space as the full-res chunk.
                    let f = factor as u8;
                    let x0 = (x as u8) * f;
                    let y0 = (y as u8) * f;
                    let z0 = (z as u8) * f;
                    let normal_pos = matches!(face, Face::PosX | Face::PosY | Face::PosZ);
                    let s_off = if normal_pos { f } else { 0 };

                    let corners: [[u8; 3]; 4] = match face {
                        Face::PosX | Face::NegX => [
                            [x0 + s_off, y0, z0],
                            [x0 + s_off, y0 + f, z0],
                            [x0 + s_off, y0 + f, z0 + f],
                            [x0 + s_off, y0, z0 + f],
                        ],
                        Face::PosY | Face::NegY => [
                            [x0, y0 + s_off, z0],
                            [x0 + f, y0 + s_off, z0],
                            [x0 + f, y0 + s_off, z0 + f],
                            [x0, y0 + s_off, z0 + f],
                        ],
                        Face::PosZ | Face::NegZ => [
                            [x0, y0, z0 + s_off],
                            [x0 + f, y0, z0 + s_off],
                            [x0 + f, y0 + f, z0 + s_off],
                            [x0, y0 + f, z0 + s_off],
                        ],
                    };
                    // Negative-normal faces need their corner order
                    // reversed to remain CCW from outside (same trick
                    // greedy.rs uses).
                    let order: [usize; 4] = if normal_pos {
                        [0, 3, 2, 1]
                    } else {
                        [0, 1, 2, 3]
                    };
                    let base = mesh.vertices.len() as u32;
                    for i in 0..4 {
                        mesh.vertices.push(Vertex {
                            pos: corners[order[i]],
                            ao: 3, // no AO at LOD levels
                            color: color_u8,
                            normal_face: face as u8,
                            light,
                            _pad: [0; 2],
                        });
                    }
                    mesh.indices
                        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::DenseChunk;

    #[test]
    fn downsample_solid_yields_solid() {
        let d = DenseChunk::new_filled(Block::Stone);
        let lod = downsample(&d, 2);
        assert_eq!(lod.dim, 16);
        assert!(lod.blocks.iter().all(|&b| b == Block::Stone));
    }

    #[test]
    fn downsample_air_yields_air() {
        let d = DenseChunk::empty();
        let lod = downsample(&d, 4);
        assert_eq!(lod.dim, 8);
        assert!(lod.blocks.iter().all(|&b| b == Block::Air));
    }
}
