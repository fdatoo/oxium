//! Vertex and index emission for greedy rectangles.

use crate::mesher::{ChunkMesh, Face, UNTEXTURED_TILE, Vertex};
use crate::voxel::block::{Block, BlockRegistry};

use super::mask::Cell;

/// Emit a single greedy-merged quad of size `w x h`.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_greedy_quad(
    mesh: &mut ChunkMesh,
    face: Face,
    slice: u8,
    ui: u8,
    vi: u8,
    w: u8,
    h: u8,
    n_axis: u8,
    u_axis: u8,
    v_axis: u8,
    cell: Cell,
    reg: &BlockRegistry,
) {
    let block = Block::from_repr(cell.block).unwrap_or(Block::Stone);
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
    let tile_index = info
        .tile_for_face(face)
        .map(|t| t.index())
        .unwrap_or(UNTEXTURED_TILE);

    let normal_pos = matches!(face, Face::PosX | Face::PosY | Face::PosZ);
    let s = if normal_pos { slice + 1 } else { slice };

    let corner_pos_uv: [(u8, u8); 4] = [(0, 0), (w, 0), (w, h), (0, h)];
    let mut positions = [[0u8; 3]; 4];
    for (i, (u, v)) in corner_pos_uv.iter().enumerate() {
        let mut p = [0u8; 3];
        p[n_axis as usize] = s;
        p[u_axis as usize] = ui + *u;
        p[v_axis as usize] = vi + *v;
        positions[i] = p;
    }

    // Texture UVs in tile units. Side faces remap U/V so image top follows
    // world +Y; top/bottom faces can use the in-plane axes directly.
    let corner_tex_uv: [(u8, u8); 4] = match face {
        Face::PosY | Face::NegY => [(0, 0), (w, 0), (w, h), (0, h)],
        Face::PosX | Face::NegZ => [(0, h), (w, h), (w, 0), (0, 0)],
        Face::NegX | Face::PosZ => [(0, w), (0, 0), (h, 0), (h, w)],
    };

    // This order gives all six faces outward winding for the axis mappings in
    // `mask::axis_map`; `tests::every_face_winds_outward` guards it.
    let order: [usize; 4] = [0, 3, 2, 1];

    let base = mesh.vertices.len() as u32;
    let ao = cell.ao;
    let light = cell.light;
    for i in 0..4 {
        let (u_tile, v_tile) = corner_tex_uv[order[i]];
        mesh.vertices.push(Vertex {
            pos: positions[order[i]],
            ao: ao[order[i]],
            color: color_u8,
            normal_face: face as u8,
            light,
            _pad: [0; 2],
            tile_index,
            u_tile,
            v_tile,
            _pad2: 0,
        });
    }

    let flip =
        ao[order[0]] as u32 + ao[order[2]] as u32 > ao[order[1]] as u32 + ao[order[3]] as u32;
    if flip {
        mesh.indices
            .extend_from_slice(&[base + 1, base + 2, base + 3, base + 1, base + 3, base]);
    } else {
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}
