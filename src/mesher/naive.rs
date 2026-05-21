//! Naive culled mesher — one quad per *visible* block face.
//!
//! "Visible" means the neighbour in that face's direction is non-opaque.
//! At chunk boundaries we don't yet have access to the adjacent chunk's
//! contents, so we conservatively *do* emit those faces; M3 fixes this by
//! plumbing neighbour chunks into the mesher.
//!
//! The greedy mesher in `greedy.rs` (M4) replaces this on the hot path, but
//! the naive variant stays because:
//!
//! 1. It is the obvious-but-correct reference; greedy tests compare against it.
//! 2. It's useful as a fallback during debugging.

use crate::mesher::{ChunkMesh, Face, Vertex, UNTEXTURED_TILE};
use crate::voxel::block::{Block, BlockRegistry};
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{LocalPos, CHUNK_DIM_U};
use glam::UVec3;

/// Mesh a single chunk **without** neighbour information. Chunk-boundary
/// faces are always emitted (correct in isolation, but produces extra
/// overdraw between two adjacent chunks until M3 wires neighbours through).
pub fn mesh_chunk_no_neighbors(chunk: &DenseChunk, reg: &BlockRegistry) -> ChunkMesh {
    let mut mesh = ChunkMesh::empty();

    // Iteration order is z outer → y → x inner so the inner-most index is
    // the one stored fastest in memory (matches `LocalPos::to_index`'s layout).
    for z in 0..CHUNK_DIM_U {
        for y in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let p = LocalPos(UVec3::new(x, y, z));
                let block = chunk.get(p);
                if block == Block::Air {
                    continue;
                }

                for face in Face::all() {
                    if face_visible(chunk, reg, x as i32, y as i32, z as i32, face) {
                        emit_quad(&mut mesh, x as u8, y as u8, z as u8, face, block, reg);
                    }
                }
            }
        }
    }
    mesh.has_water = chunk.blocks.iter().any(|b| *b == Block::Water);
    mesh
}

/// Is the face of the block at `(x, y, z)` looking in `face` direction
/// visible — i.e. does the neighbour in that direction not block the view?
fn face_visible(
    chunk: &DenseChunk,
    reg: &BlockRegistry,
    x: i32,
    y: i32,
    z: i32,
    face: Face,
) -> bool {
    let [dx, dy, dz] = face.normal();
    let (nx, ny, nz) = (x + dx, y + dy, z + dz);
    let dim = CHUNK_DIM_U as i32;

    // Out-of-chunk neighbour: M3 supplies the real neighbour chunk; here we
    // assume the face is visible (worst case: a wasted quad, never an
    // incorrect missing quad).
    if nx < 0 || ny < 0 || nz < 0 || nx >= dim || ny >= dim || nz >= dim {
        return true;
    }

    let neighbor = chunk.get(LocalPos(UVec3::new(nx as u32, ny as u32, nz as u32)));
    !reg.info(neighbor).opaque
}

/// Push a single quad (two triangles, four vertices) into `mesh`.
fn emit_quad(
    mesh: &mut ChunkMesh,
    x: u8,
    y: u8,
    z: u8,
    face: Face,
    block: Block,
    reg: &BlockRegistry,
) {
    let corners = face_corners(x, y, z, face);
    let info = reg.info(block);
    // Grass-style top tint: if the block declares a separate top colour,
    // use it for the +Y face only.
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
    let tile_index = info
        .tile_for_face(face)
        .map(|t| t.index())
        .unwrap_or(UNTEXTURED_TILE);
    // 1×1 quad UV per face, matching `face_corners`' winding so the
    // texture reads right-side up on every side face. See the equivalent
    // comment in `greedy.rs::emit_greedy_quad` for the full rationale.
    // For naive: `face_corners` puts the two top-Y corners at indices 1
    // and 2, and the two bottom-Y corners at 0 and 3; the horizontal
    // (UV.x) side then follows the face's "near/far" corner ordering.
    let corner_tile_uv: [(u8, u8); 4] = match face {
        // Top + bottom: UV from the two horizontal axes; PosY's
        // (-x, -z) → (+x, -z) → (+x, +z) → (-x, +z) corner sweep gives
        // (0,0)/(0,1)/(1,1)/(1,0).
        Face::PosY => [(0, 0), (0, 1), (1, 1), (1, 0)],
        Face::NegY => [(0, 0), (1, 0), (1, 1), (0, 1)],
        // PosX / NegZ: corners[0,3] = bottom, [1,2] = top; UV.x grows
        // toward higher horizontal world coord.
        Face::PosX | Face::NegZ => [(0, 1), (0, 0), (1, 0), (1, 1)],
        // NegX / PosZ: same vertical pattern but the corners' near/far
        // ordering is mirrored, so UV.x flips.
        Face::NegX | Face::PosZ => [(1, 1), (1, 0), (0, 0), (0, 1)],
    };

    let base = mesh.vertices.len() as u32;
    for (i, c) in corners.into_iter().enumerate() {
        let (u_tile, v_tile) = corner_tile_uv[i];
        mesh.vertices.push(Vertex {
            pos: c,
            // No real AO yet — uniform "fully unoccluded" (3); M4 bakes proper AO.
            ao: 3,
            color: color_u8,
            normal_face: face as u8,
            // No real lighting yet — fully lit on both channels; M5 fills this in.
            light: 0xFF,
            _pad: [0; 2],
            tile_index,
            u_tile,
            v_tile,
            _pad2: 0,
        });
    }
    // Two triangles forming the quad, in counter-clockwise winding so the
    // pipeline's `front_face = Ccw` + back-face cull keep them visible.
    mesh.indices.extend_from_slice(&[
        base,
        base + 1,
        base + 2,
        base,
        base + 2,
        base + 3,
    ]);
}

/// Counter-clockwise corner positions for the cube face at `(x,y,z)`,
/// expressed in 0..=32 local coordinates (the cube spans
/// `(x,y,z)..(x+1,y+1,z+1)`).
///
/// "Counter-clockwise from outside" is the convention required by our
/// `front_face = Ccw` + back-face cull pipeline state. It means: standing
/// on the *outward* side of the face and tracing the corners in the order
/// returned, you should turn left at each step.
///
/// Validating one face is a one-line cross-product check: for any
/// `(v0, v1, v2)` from the returned list, `(v1-v0) × (v2-v0)` must equal
/// the face's outward normal. The PosY/NegY entries used to be wound the
/// other way and got silently back-face culled — fixed below.
fn face_corners(x: u8, y: u8, z: u8, face: Face) -> [[u8; 3]; 4] {
    let x1 = x + 1;
    let y1 = y + 1;
    let z1 = z + 1;
    match face {
        Face::PosX => [[x1, y, z], [x1, y1, z], [x1, y1, z1], [x1, y, z1]],
        Face::NegX => [[x, y, z1], [x, y1, z1], [x, y1, z], [x, y, z]],
        Face::PosY => [[x, y1, z], [x, y1, z1], [x1, y1, z1], [x1, y1, z]],
        Face::NegY => [[x, y, z], [x1, y, z], [x1, y, z1], [x, y, z1]],
        Face::PosZ => [[x1, y, z1], [x1, y1, z1], [x, y1, z1], [x, y, z1]],
        Face::NegZ => [[x, y, z], [x, y1, z], [x1, y1, z], [x1, y, z]],
    }
}

/// Mesh a chunk *with* optional access to its six neighbours.
///
/// `neighbors` is ordered by [`Face`] discriminant (PosX, NegX, PosY, NegY,
/// PosZ, NegZ). A `Some(&DenseChunk)` neighbour fixes the face culling at
/// the chunk boundary (so two adjacent solid chunks don't both emit their
/// shared face). A `None` neighbour means "we don't know what's there"; we
/// conservatively emit the boundary face and let the next mesh job (when
/// the neighbour generates) fix it up.
pub fn mesh_chunk_with_neighbors(
    chunk: &DenseChunk,
    neighbors: &[Option<&DenseChunk>; 6],
    reg: &BlockRegistry,
) -> ChunkMesh {
    let mut mesh = ChunkMesh::empty();
    for z in 0..CHUNK_DIM_U {
        for y in 0..CHUNK_DIM_U {
            for x in 0..CHUNK_DIM_U {
                let p = LocalPos(UVec3::new(x, y, z));
                let block = chunk.get(p);
                if block == Block::Air {
                    continue;
                }
                for face in Face::all() {
                    if face_visible_with_neighbors(
                        chunk,
                        neighbors,
                        reg,
                        x as i32,
                        y as i32,
                        z as i32,
                        face,
                    ) {
                        emit_quad(&mut mesh, x as u8, y as u8, z as u8, face, block, reg);
                    }
                }
            }
        }
    }
    mesh.has_water = chunk.blocks.iter().any(|b| *b == Block::Water);
    mesh
}

/// Variant of [`face_visible`] that consults the supplied neighbour chunk
/// when the face is at the chunk boundary.
fn face_visible_with_neighbors(
    chunk: &DenseChunk,
    neighbors: &[Option<&DenseChunk>; 6],
    reg: &BlockRegistry,
    x: i32,
    y: i32,
    z: i32,
    face: Face,
) -> bool {
    let [dx, dy, dz] = face.normal();
    let (nx, ny, nz) = (x + dx, y + dy, z + dz);
    let dim = CHUNK_DIM_U as i32;

    // Same-chunk case: simple in-bounds opacity test.
    if nx >= 0 && ny >= 0 && nz >= 0 && nx < dim && ny < dim && nz < dim {
        let neighbor = chunk.get(LocalPos(UVec3::new(nx as u32, ny as u32, nz as u32)));
        return !reg.info(neighbor).opaque;
    }

    // Boundary case: consult the neighbour chunk on the face's side.
    // When the neighbour isn't loaded we conservatively emit (return
    // true) — hiding it would leave large sky-visible holes at chunk
    // boundaries within the loaded set. The cost is some visible
    // boundary faces at the *permanent* edge of the loaded vertical
    // range (e.g. chunk-bottom NegY faces), accepted as a v0 limit.
    let neighbor_chunk = match neighbors[face as usize] {
        Some(c) => c,
        None => return true,
    };
    // Wrap the out-of-bounds coordinate around to the neighbour's local
    // space. `+ dim) % dim` handles both `-1 → dim-1` and `dim → 0`.
    let (lx, ly, lz) = (
        ((nx + dim) % dim) as u32,
        ((ny + dim) % dim) as u32,
        ((nz + dim) % dim) as u32,
    );
    let neighbor = neighbor_chunk.get(LocalPos(UVec3::new(lx, ly, lz)));
    !reg.info(neighbor).opaque
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::DenseChunk;
    use crate::voxel::coords::CHUNK_DIM_U;

    #[test]
    fn empty_chunk_produces_empty_mesh() {
        let c = DenseChunk::empty();
        let r = BlockRegistry::new();
        let mesh = mesh_chunk_no_neighbors(&c, &r);
        assert!(mesh.vertices.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn single_block_emits_six_quads() {
        let mut c = DenseChunk::empty();
        c.set(LocalPos(UVec3::new(5, 5, 5)), Block::Stone);
        let r = BlockRegistry::new();
        let mesh = mesh_chunk_no_neighbors(&c, &r);
        assert_eq!(mesh.vertices.len(), 24, "6 faces × 4 verts");
        assert_eq!(mesh.indices.len(), 36, "6 faces × 6 indices");
    }

    #[test]
    fn full_chunk_emits_only_outer_faces() {
        let c = DenseChunk::new_filled(Block::Stone);
        let r = BlockRegistry::new();
        let mesh = mesh_chunk_no_neighbors(&c, &r);
        // Interior faces are all culled. Every chunk-boundary face is emitted
        // (we don't have neighbour info), so we expect 6 × 32 × 32 quads.
        let expected_quads = 6 * CHUNK_DIM_U as usize * CHUNK_DIM_U as usize;
        assert_eq!(mesh.vertices.len(), expected_quads * 4);
        assert_eq!(mesh.indices.len(), expected_quads * 6);
    }

    #[test]
    fn with_solid_neighbor_no_boundary_face() {
        // Single stone block at (0,0,0). With a fully-solid neighbour on
        // the -X side the boundary face there is hidden. Other unloaded
        // neighbours conservatively emit (treated as Air for boundary
        // visibility) — see `face_visible_with_neighbors` for the
        // rationale.
        let mut center = DenseChunk::empty();
        center.set(LocalPos(UVec3::new(0, 0, 0)), Block::Stone);
        let neighbor = DenseChunk::new_filled(Block::Stone);
        let neighbors: [Option<&DenseChunk>; 6] = [
            None,            // PosX (in-chunk Air)
            Some(&neighbor), // NegX (solid → hidden)
            None,
            None,
            None,
            None,
        ];
        let r = BlockRegistry::new();
        let mesh = mesh_chunk_with_neighbors(&center, &neighbors, &r);
        // 6 total minus 1 hidden NegX face = 5 quads = 20 verts.
        assert_eq!(mesh.vertices.len(), 20);
    }
}
