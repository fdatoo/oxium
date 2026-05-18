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

/// Build an `LodChunk` from a `DenseChunk`. `factor` must be 2 (L1) or
/// 4 (L2).
///
/// The downsample is built in two passes per LOD column:
///
/// 1. **Column scan.** For each of the `factor²` source columns inside
///    this LOD column we find the source-y of the *topmost non-air
///    block*. The LOD column's "surface y_lod" is the highest cell that
///    any source column reaches. The "surface block" is voted from the
///    source columns whose surface lands in (or above) the LOD-column's
///    surface cell.
///
/// 2. **Vertical fill.** Every LOD cell at `y_lod ≤ surface_y_lod` is
///    set: the surface cell gets the surface block; cells below get a
///    bulk block (the topmost solid block in the lowest source column).
///    Cells *above* `surface_y_lod` are Air.
///
/// Why the fill: the naive "vote per cell independently" approach
/// produced floating cube artefacts when adjacent LOD columns had their
/// terrain in different `y_lod` slots — the sky showed *through* the
/// stepped gap between them. Filling each LOD column from the bottom up
/// to its surface y_lod removes those gaps; adjacent LOD columns now
/// hide each others' vertical seams because they're back-to-back solid.
///
/// Light is averaged across the whole group.
pub fn downsample(src: &DenseChunk, factor: u32) -> LodChunk {
    assert!(factor == 2 || factor == 4, "factor must be 2 or 4");
    let dim = CHUNK_DIM_U / factor;
    let len = (dim * dim * dim) as usize;
    let mut blocks = vec![Block::Air; len];
    let mut light = vec![0u8; len];

    let chunk_dim = CHUNK_DIM_U;
    for z in 0..dim {
        for x in 0..dim {
            // (1) Column scan: per-source-column surface info.
            let mut max_surface_y: Option<u32> = None;
            let mut surface_counts: HashMap<Block, u32> = HashMap::new();
            let mut bulk_counts: HashMap<Block, u32> = HashMap::new();

            for dz in 0..factor {
                for dx in 0..factor {
                    let sx = x * factor + dx;
                    let sz = z * factor + dz;
                    // Find the highest non-air block in this source column.
                    let mut top_y: Option<u32> = None;
                    let mut top_b = Block::Air;
                    for sy in (0..chunk_dim).rev() {
                        let b = src.blocks
                            [LocalPos(UVec3::new(sx, sy, sz)).to_index()];
                        if b != Block::Air {
                            top_y = Some(sy);
                            top_b = b;
                            break;
                        }
                    }
                    let Some(top_y) = top_y else { continue };

                    // Update the LOD column's max surface y_lod.
                    let top_y_lod = top_y / factor;
                    if max_surface_y.map(|m| top_y_lod > m).unwrap_or(true) {
                        max_surface_y = Some(top_y_lod);
                    }
                    // Surface vote: the topmost block of this column.
                    *surface_counts.entry(top_b).or_insert(0) += 1;
                    // Bulk vote: a block well below the surface (skip
                    // the dirt-layer; sample the deepest source row of
                    // this column to get the underlying material).
                    let bulk_b = src.blocks
                        [LocalPos(UVec3::new(sx, 0, sz)).to_index()];
                    if bulk_b != Block::Air {
                        *bulk_counts.entry(bulk_b).or_insert(0) += 1;
                    }
                }
            }

            // Light: averaged across the whole LOD column footprint.
            // (Cheap and good enough for "sky vs cave" distinction.)
            let mut sky_sum: u32 = 0;
            let mut blk_sum: u32 = 0;
            for sy in 0..chunk_dim {
                for dz in 0..factor {
                    for dx in 0..factor {
                        let idx = LocalPos(UVec3::new(
                            x * factor + dx,
                            sy,
                            z * factor + dz,
                        ))
                        .to_index();
                        sky_sum += src.sky_light[idx] as u32;
                        blk_sum += src.block_light[idx] as u32;
                    }
                }
            }
            let n = chunk_dim * factor * factor;
            let avg_sky = (sky_sum / n) as u8;
            let avg_blk = (blk_sum / n) as u8;
            let light_byte = (avg_sky.min(15) << 4) | avg_blk.min(15);

            // (2) Vertical fill of this LOD column.
            let Some(surface_y_lod) = max_surface_y else {
                continue;
            };
            let surface_block = surface_counts
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(b, _)| b)
                .unwrap_or(Block::Stone);
            let bulk_block = bulk_counts
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(b, _)| b)
                .unwrap_or(Block::Stone);

            for y_lod in 0..=surface_y_lod {
                let idx = (x + y_lod * dim + z * dim * dim) as usize;
                blocks[idx] = if y_lod == surface_y_lod {
                    surface_block
                } else {
                    bulk_block
                };
                light[idx] = light_byte;
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

    // Inside the LOD chunk we read real cells. *Outside* we conservatively
    // return Air so boundary faces are emitted. We don't currently have
    // neighbour LOD chunks plumbed through, so the choice is between:
    //
    //   * "out = Stone": hide boundary faces. Looks clean for same-height
    //     adjacent LOD chunks but leaves *sky-visible holes* wherever
    //     adjacent LOD chunks fill to different heights (which is the
    //     common case with varied terrain).
    //
    //   * "out = Air": emit boundary faces. Always covers the height
    //     differences correctly; the cost is a faint duplicate face at
    //     same-height boundaries (z-fight resolved by last-drawn) plus
    //     some +Y "shelf" faces at the permanent load-range top.
    //
    // Air wins: holes are far worse than minor z-fighting on a flat
    // boundary at LOD distance.
    let block_at = |x: i32, y: i32, z: i32| -> Block {
        if x < 0 || y < 0 || z < 0 || x >= dim || y >= dim || z >= dim {
            return Block::Air;
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
