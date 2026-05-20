//! Camera-follow chunk enumeration.

use glam::{IVec3, Vec3};
use oxium::voxel::coords::{ChunkCoord, CHUNK_DIM_U};

/// Streaming radius. XZ tracks the camera; Y is a fixed world band
/// (chunks `-y..=y`, NOT camera-relative). The Y decoupling matters
/// because flying high above the world would otherwise leave the
/// terrain entirely outside the load radius — `scene.clear()` on a
/// config-edit wipe would empty the screen and never refill until
/// the camera came back down.
///
/// Cut from the spec's (xz=8, y=4)=2601 chunks down to (xz=3, y=4)=441
/// to make edit-time regens feel sub-second while still covering the
/// full world Y range (-128..=128). Tune via `--radius-xz` /
/// `--radius-y` for a wider visible window.
#[derive(Debug, Clone, Copy)]
pub struct StreamRadius {
    pub xz: i32,
    pub y: i32,
}

impl StreamRadius {
    pub const DEFAULT: Self = Self { xz: 3, y: 4 };
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

/// All chunk coords to stream around `pos`. XZ is camera-relative
/// (chunks within `radius.xz` of the camera's chunk); Y is a fixed
/// world band (`cy ∈ -radius.y..=radius.y`, ignoring camera Y).
/// Sorted by Chebyshev distance from the camera-XZ + Y=0 anchor
/// (closest first), so the center XZ column lands at index 0.
pub fn chunks_in_radius(pos: Vec3, radius: StreamRadius) -> Vec<ChunkCoord> {
    let cam = camera_chunk(pos);
    let anchor = IVec3::new(cam.0.x, 0, cam.0.z);
    let mut out = Vec::with_capacity(((2 * radius.xz + 1).pow(2) * (2 * radius.y + 1)) as usize);
    for cy in -radius.y..=radius.y {
        for dz in -radius.xz..=radius.xz {
            for dx in -radius.xz..=radius.xz {
                out.push(ChunkCoord(IVec3::new(cam.0.x + dx, cy, cam.0.z + dz)));
            }
        }
    }
    out.sort_by_key(|c| {
        let d = c.0 - anchor;
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

    #[test]
    fn camera_y_does_not_shift_returned_chunks_y() {
        // The regression this guards: with the camera high in the
        // sky, the streamed Y range used to slide upward with it,
        // leaving the terrain band outside the load radius. New
        // behaviour: Y range is always world-anchored.
        let low = chunks_in_radius(Vec3::new(0.0, 0.0, 0.0), StreamRadius::DEFAULT);
        let high = chunks_in_radius(Vec3::new(0.0, 400.0, 0.0), StreamRadius::DEFAULT);
        let low_ys: std::collections::BTreeSet<i32> = low.iter().map(|c| c.0.y).collect();
        let high_ys: std::collections::BTreeSet<i32> = high.iter().map(|c| c.0.y).collect();
        assert_eq!(low_ys, high_ys, "Y range must be camera-independent");
    }

    #[test]
    fn camera_xz_does_shift_returned_chunks_xz() {
        // The intended behaviour the previous test does NOT regress:
        // XZ should still track the camera.
        let a = chunks_in_radius(Vec3::new(0.0, 0.0, 0.0), StreamRadius::DEFAULT);
        let b = chunks_in_radius(Vec3::new(500.0, 0.0, 0.0), StreamRadius::DEFAULT);
        let a_xs: std::collections::BTreeSet<i32> = a.iter().map(|c| c.0.x).collect();
        let b_xs: std::collections::BTreeSet<i32> = b.iter().map(|c| c.0.x).collect();
        assert_ne!(a_xs, b_xs, "XZ must track the camera");
    }
}
