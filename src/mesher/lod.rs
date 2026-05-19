//! Level-of-detail: voxel-grid downsampling + heightmap-style meshing.
//!
//! Rendering 32³ chunks to the horizon is infeasible — at render
//! distance 16 that's ~17 k chunks × hundreds of greedy quads each.
//! We solve it by downsampling each chunk into a 2D heightmap (one
//! "tallest non-air column" per `factor × factor` source-column patch),
//! plus a surface block + bulk block for that patch.
//!
//! The mesher then emits, per LOD column:
//!
//! - One **top quad** at the column's surface height, coloured with the
//!   surface block (grass, dirt, water, …).
//! - One **side quad** on each cardinal direction where the neighbour
//!   LOD column is shorter, dropping from this column's top to the
//!   neighbour's top. The skirt is coloured with the bulk block (stone,
//!   dirt) — exactly what you'd see if you walked round a real terrain
//!   cliff at LOD distance.
//!
//! No more floating-cube scatter: the heightmap is a continuous surface
//! by construction, and the skirts close the height steps between
//! adjacent LOD columns. Per-chunk boundary steps still need adjacent
//! LOD chunk data to render perfectly — they emit a conservative skirt
//! to world y=0 of the chunk when the neighbour is out-of-bounds, which
//! the adjacent chunk's own surface paints over.

use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{LocalPos, CHUNK_DIM_U};
use glam::UVec3;
use std::collections::HashMap;

/// A downsampled chunk, stored as a 2-D heightmap rather than a 3-D
/// cube grid. `dim = CHUNK_DIM_U / factor` along both X and Z. For each
/// `(x, z)` LOD column we keep:
///
/// - `surface_y` — the highest source-y of any non-air block in the
///   `factor × factor` source-column patch, or `u32::MAX` for "this
///   LOD column has no terrain" (entirely air across the chunk).
/// - `surface_block` — the block to paint the column's top with.
/// - `bulk_block` — the block to paint the column's side walls with.
/// - `light` — packed `(sky << 4) | block` light, averaged over the
///   patch.
///
/// The mesher reads these per column to emit one top quad and up to
/// four side quads (skirts) per column.
pub struct LodChunk {
    pub dim: u32,
    pub surface_y: Vec<u32>,
    pub surface_block: Vec<Block>,
    pub bulk_block: Vec<Block>,
    pub light: Vec<u8>,
}

/// Sentinel for "this LOD column is entirely air, don't render it".
const NO_COLUMN: u32 = u32::MAX;

/// Downsample a `DenseChunk` into the heightmap-style `LodChunk`.
/// `factor` must be 2 (L1) or 4 (L2).
///
/// For each `(x, z)` LOD column we walk the `factor²` source columns
/// and:
///
/// - find the highest non-air block across all of them → `surface_y`
/// - vote the surface block (the highest block from each source column)
/// - vote the bulk block (the deepest non-air block from each column)
/// - average the sky + block-light bytes across the full source-column
///   patch
///
/// The result is *just* per-column metadata — no 3-D cube grid. The
/// mesher reads these per column to emit one quad per top + one quad
/// per side that needs a skirt.
pub fn downsample(src: &DenseChunk, factor: u32) -> LodChunk {
    assert!(factor == 2 || factor == 4, "factor must be 2 or 4");
    let dim = CHUNK_DIM_U / factor;
    let len = (dim * dim) as usize;
    let mut surface_y = vec![NO_COLUMN; len];
    let mut surface_block = vec![Block::Air; len];
    let mut bulk_block = vec![Block::Stone; len];
    let mut light = vec![0u8; len];

    let chunk_dim = CHUNK_DIM_U;
    for z in 0..dim {
        for x in 0..dim {
            let mut max_top_y: Option<u32> = None;
            let mut surface_counts: HashMap<Block, u32> = HashMap::new();
            let mut bulk_counts: HashMap<Block, u32> = HashMap::new();
            // Light for the entire LOD column is taken from the *tallest*
            // source column's "air above surface" cell. Averaging across
            // source columns sounds reasonable but dramatically darkens
            // LOD columns that mix one tall surface with many buried
            // columns (the buried ones contribute zero), so even a
            // brightly-lit hill peak ends up rendered with sky_light=1.
            // Using the tallest column's light keeps the surface bright.
            let mut light_sky: u8 = 0;
            let mut light_blk: u8 = 0;

            for dz in 0..factor {
                for dx in 0..factor {
                    let sx = x * factor + dx;
                    let sz = z * factor + dz;

                    let mut top: Option<(u32, Block)> = None;
                    let mut deepest_solid: Option<Block> = None;
                    for sy in 0..chunk_dim {
                        let idx = LocalPos(UVec3::new(sx, sy, sz)).to_index();
                        let b = src.blocks[idx];
                        if b != Block::Air {
                            if deepest_solid.is_none() {
                                deepest_solid = Some(b);
                            }
                            top = Some((sy, b));
                        }
                    }
                    let Some((top_y, top_b)) = top else { continue };

                    let new_max = max_top_y.map(|m| top_y > m).unwrap_or(true);
                    if new_max {
                        max_top_y = Some(top_y);
                        // Sample this column's light at the cell above
                        // its surface. Out-of-chunk (terrain continues
                        // into the chunk above) → keep the previous
                        // light value rather than dropping to zero,
                        // since the chunk above will paint its own top.
                        let above_y = top_y + 1;
                        if above_y < chunk_dim {
                            let idx = LocalPos(UVec3::new(sx, above_y, sz)).to_index();
                            light_sky = src.sky_light[idx];
                            light_blk = src.block_light[idx];
                        } else {
                            // Buried-continuing column: assume full sky
                            // light because the surface above will mask
                            // this one anyway, and we'd rather a
                            // brief mis-illumination during streaming
                            // than a near-black ground.
                            light_sky = 15;
                            light_blk = 0;
                        }
                    }
                    *surface_counts.entry(top_b).or_insert(0) += 1;
                    if let Some(b) = deepest_solid {
                        *bulk_counts.entry(b).or_insert(0) += 1;
                    }
                }
            }

            let light_byte = (light_sky.min(15) << 4) | light_blk.min(15);

            let idx = (x + z * dim) as usize;
            light[idx] = light_byte;
            if let Some(top_y) = max_top_y {
                surface_y[idx] = top_y;
                surface_block[idx] = surface_counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(b, _)| b)
                    .unwrap_or(Block::Stone);
                bulk_block[idx] = bulk_counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(b, _)| b)
                    .unwrap_or(Block::Stone);
            }
        }
    }
    LodChunk {
        dim,
        surface_y,
        surface_block,
        bulk_block,
        light,
    }
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
    use crate::mesher::{ChunkMesh, Face};
    let mut mesh = ChunkMesh::empty();
    let dim = lod.dim as i32;

    // Helpers for reading the per-column metadata. Out-of-bounds is
    // treated as "no column": skirts that face an out-of-bounds
    // neighbour drop all the way to y=0, and the adjacent LOD chunk's
    // own surface paints over the void from its side.
    let height = |x: i32, z: i32| -> Option<u32> {
        if x < 0 || z < 0 || x >= dim || z >= dim {
            return None;
        }
        let v = lod.surface_y[(x + z * dim) as usize];
        if v == NO_COLUMN { None } else { Some(v) }
    };
    let column_at = |x: i32, z: i32| -> Option<(Block, Block, u8)> {
        if x < 0 || z < 0 || x >= dim || z >= dim {
            return None;
        }
        let idx = (x + z * dim) as usize;
        if lod.surface_y[idx] == NO_COLUMN {
            return None;
        }
        Some((lod.surface_block[idx], lod.bulk_block[idx], lod.light[idx]))
    };

    let f = factor as u8;
    for z in 0..dim {
        for x in 0..dim {
            let Some((surface, bulk, light)) = column_at(x, z) else {
                continue;
            };
            // The "top" sits *above* the topmost solid block — matching
            // where the LOD0 +Y face of that block would render.
            let top_y_u32 = height(x, z).unwrap() + 1;
            let y_top = top_y_u32.min(u8::MAX as u32) as u8;
            let x0 = (x as u32 * factor) as u8;
            let z0 = (z as u32 * factor) as u8;
            let info = reg.info(surface);
            let top_color = match info.top_color {
                Some(c) => c,
                None => info.color,
            };
            let top_tile = info
                .tile_for_face(Face::PosY)
                .map(|t| t.index())
                .unwrap_or(crate::mesher::UNTEXTURED_TILE);
            // Top quad (PosY) — surface block colour. UV spans
            // `(0,0)..(f,f)` in tile units so each LOD-merged block
            // patch gets one tile repetition per source block.
            emit_quad(
                &mut mesh,
                Face::PosY,
                [
                    [x0, y_top, z0],
                    [x0, y_top, z0 + f],
                    [x0 + f, y_top, z0 + f],
                    [x0 + f, y_top, z0],
                ],
                [(0, 0), (0, f), (f, f), (f, 0)],
                top_tile,
                pack_color(top_color),
                light,
            );

            // Bottom cap (NegY) at the chunk floor. Without this the
            // LOD chunk has only a top surface + side skirts, so a
            // player looking up at it from deep underground (or down
            // at it through a hole in the world) sees right through
            // — the LOD top quad reads as a floating slab. The cap
            // closes the volume into a solid extrusion of the
            // heightfield: a single NegY quad per column, at the
            // chunk's local y=0, coloured + textured with the bulk
            // block so the underside matches the side skirts.
            let bulk_color = pack_color(reg.info(bulk).color);
            let bulk_tile = reg
                .info(bulk)
                .tile_for_face(Face::PosX)
                .map(|t| t.index())
                .unwrap_or(crate::mesher::UNTEXTURED_TILE);
            let bulk_bottom_tile = reg
                .info(bulk)
                .tile_for_face(Face::NegY)
                .map(|t| t.index())
                .unwrap_or(crate::mesher::UNTEXTURED_TILE);
            emit_quad(
                &mut mesh,
                Face::NegY,
                // CCW from below (looking up in +Y direction). Same
                // winding `naive::face_corners` uses for NegY so the
                // back-face cull keeps the cap visible from outside.
                [
                    [x0, 0, z0],
                    [x0 + f, 0, z0],
                    [x0 + f, 0, z0 + f],
                    [x0, 0, z0 + f],
                ],
                // NegY UV mapping mirrors the greedy mesher's
                // PosY/NegY table — `(0,0)/(f,0)/(f,f)/(0,f)` aligns
                // texture U with world X and V with world Z.
                [(0, 0), (f, 0), (f, f), (0, f)],
                bulk_bottom_tile,
                bulk_color,
                light,
            );
            for &(dx, dz, face) in &[
                (1i32, 0i32, Face::PosX),
                (-1, 0, Face::NegX),
                (0, 1, Face::PosZ),
                (0, -1, Face::NegZ),
            ] {
                let neighbour_top = height(x + dx, z + dz)
                    .map(|h| h + 1)
                    .unwrap_or(0);
                if neighbour_top >= top_y_u32 {
                    continue;
                }
                let y_bot = neighbour_top.min(u8::MAX as u32) as u8;
                // Vertical span of this skirt in block (= tile) units.
                let dy = y_top.saturating_sub(y_bot);
                // CCW from outside in the face's normal direction. The
                // matching UV order is (0,0)/(0,dy)/(f,dy)/(f,0) — one
                // tile per source block both horizontally and vertically.
                let corners = match face {
                    Face::PosX => [
                        [x0 + f, y_bot, z0],
                        [x0 + f, y_top, z0],
                        [x0 + f, y_top, z0 + f],
                        [x0 + f, y_bot, z0 + f],
                    ],
                    Face::NegX => [
                        [x0, y_bot, z0 + f],
                        [x0, y_top, z0 + f],
                        [x0, y_top, z0],
                        [x0, y_bot, z0],
                    ],
                    Face::PosZ => [
                        [x0 + f, y_bot, z0 + f],
                        [x0 + f, y_top, z0 + f],
                        [x0, y_top, z0 + f],
                        [x0, y_bot, z0 + f],
                    ],
                    Face::NegZ => [
                        [x0, y_bot, z0],
                        [x0, y_top, z0],
                        [x0 + f, y_top, z0],
                        [x0 + f, y_bot, z0],
                    ],
                    _ => unreachable!(),
                };
                // Skirt corners arrive in `[bottom-near, top-near,
                // top-far, bottom-far]` order (CCW from outside). The
                // matching UV must put high-Y corners at V=0 (top of
                // tile) and low-Y corners at V=dy (bottom of tile),
                // mirroring greedy's per-face flip — otherwise the
                // grass strip on a `grass_block_side` skirt lands at
                // the BOTTOM of the cliff face instead of the top.
                let uvs: [(u8, u8); 4] = [(0, dy), (0, 0), (f, 0), (f, dy)];
                emit_quad(&mut mesh, face, corners, uvs, bulk_tile, bulk_color, light);
            }
        }
    }
    mesh
}

/// Pack an `[f32; 4]` colour into the engine's vertex `[u8; 4]` format.
fn pack_color(c: [f32; 4]) -> [u8; 4] {
    [
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
        (c[3] * 255.0) as u8,
    ]
}

/// Push a single 4-vertex / 6-index quad into `mesh`. Corners are in
/// CCW order from outside the face's normal direction; we use the same
/// `0..1..2 / 0..2..3` triangle split as the greedy mesher. `corner_uv`
/// maps each corner to a per-vertex tile-unit UV (parallel to `corners`).
fn emit_quad(
    mesh: &mut crate::mesher::ChunkMesh,
    face: crate::mesher::Face,
    corners: [[u8; 3]; 4],
    corner_uv: [(u8, u8); 4],
    tile_index: u8,
    color: [u8; 4],
    light: u8,
) {
    use crate::mesher::Vertex;
    let base = mesh.vertices.len() as u32;
    for (i, c) in corners.into_iter().enumerate() {
        let (u_tile, v_tile) = corner_uv[i];
        mesh.vertices.push(Vertex {
            pos: c,
            ao: 3,
            color,
            normal_face: face as u8,
            light,
            _pad: [0; 2],
            tile_index,
            u_tile,
            v_tile,
            _pad2: 0,
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::DenseChunk;

    #[test]
    fn downsample_solid_yields_max_height() {
        // A fully-solid chunk: every LOD column's surface is at the
        // chunk's top (y=31). surface_block == bulk_block == Stone.
        let d = DenseChunk::new_filled(Block::Stone);
        let lod = downsample(&d, 2);
        assert_eq!(lod.dim, 16);
        for (i, &y) in lod.surface_y.iter().enumerate() {
            assert_eq!(y, 31, "col {i} expected top at y=31, got {y}");
            assert_eq!(lod.surface_block[i], Block::Stone);
            assert_eq!(lod.bulk_block[i], Block::Stone);
        }
    }

    #[test]
    fn downsample_air_yields_no_columns() {
        // An entirely-air chunk: every column reports NO_COLUMN, so the
        // mesher emits nothing.
        let d = DenseChunk::empty();
        let lod = downsample(&d, 4);
        assert_eq!(lod.dim, 8);
        assert!(lod.surface_y.iter().all(|&y| y == NO_COLUMN));
    }

    #[test]
    fn mesh_lod_solid_chunk_emits_top_skirts_and_bottom() {
        // Solid chunk at LOD2: 8×8 LOD columns. Per column:
        //   - 1 top quad (PosY)
        //   - 1 bottom cap (NegY) — closes the column's volume from
        //     below so a player underground doesn't see through it
        //   - 0..2 skirt quads (one per cardinal neighbour that's
        //     shorter; out-of-bounds is treated as "no column")
        // 64 columns × 2 closing quads = 128 (top + bottom).
        // Edge skirts: 4 corner × 2 + 24 edge × 1 = 32.
        // Total: 160 quads = 640 verts.
        let d = DenseChunk::new_filled(Block::Stone);
        let r = BlockRegistry::new();
        let lod = downsample(&d, 4);
        let mesh = mesh_lod(&lod, 4, &r);

        let expected_quads = 64 + 64 + 32;
        assert_eq!(mesh.vertices.len(), expected_quads * 4);
        assert_eq!(mesh.indices.len(), expected_quads * 6);
    }
}
