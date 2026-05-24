use super::*;
use crate::voxel::block::BlockRegistry;
use crate::voxel::coords::LocalPos;
use glam::UVec3;

#[test]
fn empty_chunk_produces_no_quads() {
    let c = DenseChunk::empty();
    let r = BlockRegistry::new();
    let n: [Option<&DenseChunk>; 6] = [None; 6];
    assert_eq!(mesh_greedy(&c, &n, &r).vertices.len(), 0);
}

#[test]
fn single_block_produces_six_quads() {
    let mut c = DenseChunk::empty();
    c.set(LocalPos(UVec3::new(10, 10, 10)), Block::Stone);
    let r = BlockRegistry::new();
    let n: [Option<&DenseChunk>; 6] = [None; 6];
    let mesh = mesh_greedy(&c, &n, &r);
    assert_eq!(mesh.vertices.len(), 24, "6 faces x 4 verts");
}

#[test]
fn solid_chunk_produces_five_merged_quads() {
    let c = DenseChunk::new_filled(Block::Stone);
    let r = BlockRegistry::new();
    let n: [Option<&DenseChunk>; 6] = [None; 6];
    let mesh = mesh_greedy(&c, &n, &r);
    assert_eq!(mesh.vertices.len(), 20, "expected 5 merged 32x32 quads");
    assert_eq!(mesh.indices.len(), 30);
}

#[test]
fn border_face_uses_own_light_when_neighbor_is_unlit() {
    let mut c = DenseChunk::empty();
    let pos = LocalPos(UVec3::new(31, 5, 5));
    c.set(pos, Block::Stone);
    c.sky_light[pos.to_index()] = 12;

    let unlit_neighbor = DenseChunk::empty();
    let r = BlockRegistry::new();
    let mut n: [Option<&DenseChunk>; 6] = [None; 6];
    n[Face::PosX as usize] = Some(&unlit_neighbor);

    let mesh = mesh_greedy(&c, &n, &r);
    let pos_x_lights: Vec<u8> = mesh
        .vertices
        .iter()
        .filter(|v| v.normal_face == Face::PosX as u8)
        .map(|v| v.light)
        .collect();
    assert_eq!(pos_x_lights, vec![0xC0; 4]);
}

#[test]
fn missing_above_neighbor_does_not_force_sky_on_top_boundary_face() {
    let mut c = DenseChunk::empty();
    let pos = LocalPos(UVec3::new(5, 31, 5));
    c.set(pos, Block::Stone);

    let r = BlockRegistry::new();
    let n: [Option<&DenseChunk>; 6] = [None; 6];
    let mesh = mesh_greedy(&c, &n, &r);
    let pos_y_lights: Vec<u8> = mesh
        .vertices
        .iter()
        .filter(|v| v.normal_face == Face::PosY as u8)
        .map(|v| v.light)
        .collect();
    assert_eq!(pos_y_lights, vec![0; 4]);
}

#[test]
fn missing_neighbors_do_not_emit_water_chunk_sheets() {
    let c = DenseChunk::new_filled(Block::Water);
    let r = BlockRegistry::new();
    let n: [Option<&DenseChunk>; 6] = [None; 6];
    let mesh = mesh_greedy(&c, &n, &r);
    assert_eq!(mesh.vertices.len(), 0);
    assert_eq!(mesh.indices.len(), 0);
}

#[test]
fn every_face_winds_outward() {
    let mut c = DenseChunk::empty();
    c.set(LocalPos(UVec3::new(10, 10, 10)), Block::Stone);
    let r = BlockRegistry::new();
    let n: [Option<&DenseChunk>; 6] = [None; 6];
    let mesh = mesh_greedy(&c, &n, &r);
    assert_eq!(mesh.vertices.len(), 24, "6 faces x 4 verts");

    for tri in mesh.indices.chunks_exact(3) {
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];
        let face = match v0.normal_face {
            0 => Face::PosX,
            1 => Face::NegX,
            2 => Face::PosY,
            3 => Face::NegY,
            4 => Face::PosZ,
            5 => Face::NegZ,
            _ => panic!("bad face index"),
        };
        let expected = face.normal();
        let a = [
            v1.pos[0] as i32 - v0.pos[0] as i32,
            v1.pos[1] as i32 - v0.pos[1] as i32,
            v1.pos[2] as i32 - v0.pos[2] as i32,
        ];
        let b = [
            v2.pos[0] as i32 - v0.pos[0] as i32,
            v2.pos[1] as i32 - v0.pos[1] as i32,
            v2.pos[2] as i32 - v0.pos[2] as i32,
        ];
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let dot = cross[0] * expected[0] + cross[1] * expected[1] + cross[2] * expected[2];
        assert!(
            dot > 0,
            "{:?} triangle wound wrong: cross={:?} expected normal={:?}",
            face,
            cross,
            expected
        );
    }
}
