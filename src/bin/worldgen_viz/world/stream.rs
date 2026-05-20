//! Camera-follow chunk enumeration.

use glam::{IVec3, Vec3};
use oxium::voxel::coords::{ChunkCoord, CHUNK_DIM_U};

/// Radius (in chunks) around the camera. Spec defaults: 8 XZ, 4 Y.
#[derive(Debug, Clone, Copy)]
pub struct StreamRadius {
    pub xz: i32,
    pub y: i32,
}

impl StreamRadius {
    pub const DEFAULT: Self = Self { xz: 8, y: 4 };
}

/// Which chunk does the given world position sit in?
pub fn camera_chunk(pos: Vec3) -> ChunkCoord {
    let dim = CHUNK_DIM_U as f32;
    ChunkCoord(IVec3::new(
        (pos.x / dim).floor() as i32,
        (pos.y / dim).floor() as i32,
        (pos.z / dim).floor() as i32,
    ))
}

/// All chunk coords within `radius` of the chunk containing `pos`, sorted by
/// Chebyshev distance ascending (closest first). The center chunk is at
/// index 0 of the returned vec.
pub fn chunks_in_radius(pos: Vec3, radius: StreamRadius) -> Vec<ChunkCoord> {
    let center = camera_chunk(pos);
    let mut out = Vec::with_capacity(((2 * radius.xz + 1).pow(2) * (2 * radius.y + 1)) as usize);
    for dy in -radius.y..=radius.y {
        for dz in -radius.xz..=radius.xz {
            for dx in -radius.xz..=radius.xz {
                out.push(ChunkCoord(center.0 + IVec3::new(dx, dy, dz)));
            }
        }
    }
    out.sort_by_key(|c| {
        let d = c.0 - center.0;
        d.x.abs().max(d.z.abs()).max(d.y.abs())
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_chunk_at_origin_is_zero() {
        assert_eq!(camera_chunk(Vec3::new(0.0, 0.0, 0.0)).0, IVec3::ZERO);
    }

    #[test]
    fn camera_chunk_floors_negative() {
        // (-1, -1, -1) lies in chunk (-1, -1, -1), not (0,0,0).
        assert_eq!(
            camera_chunk(Vec3::new(-1.0, -1.0, -1.0)).0,
            IVec3::new(-1, -1, -1)
        );
    }

    #[test]
    fn chunks_in_radius_size_matches_3d_box() {
        let r = StreamRadius { xz: 2, y: 1 };
        let v = chunks_in_radius(Vec3::ZERO, r);
        let expected = (2 * r.xz + 1).pow(2) * (2 * r.y + 1);
        assert_eq!(v.len() as i32, expected);
    }

    #[test]
    fn chunks_in_radius_center_first() {
        let v = chunks_in_radius(Vec3::ZERO, StreamRadius::DEFAULT);
        assert_eq!(v[0].0, IVec3::ZERO);
    }

    #[test]
    fn chunks_in_radius_sorted_by_chebyshev() {
        let v = chunks_in_radius(Vec3::ZERO, StreamRadius { xz: 1, y: 0 });
        // Center, then 8 neighbours all at Chebyshev distance 1.
        assert_eq!(v.len(), 9);
        let chebyshev = |c: &ChunkCoord| c.0.x.abs().max(c.0.y.abs()).max(c.0.z.abs());
        for w in v.windows(2) {
            assert!(chebyshev(&w[0]) <= chebyshev(&w[1]));
        }
    }
}
