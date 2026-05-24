//! Kruskal MST tunnel graph builder.
//!
//! [`connect_chambers_mst`] connects all chambers in a cave system with the
//! minimum spanning tree of their pairwise Euclidean distances, then adds a
//! small number of extra (non-MST) edges to introduce cycles. Without cycles,
//! every room has exactly one entrance/exit, which feels unnatural to navigate.
//!
//! ## MST algorithm
//!
//! Kruskal's algorithm with path-compressed union-find: O(E log E + α(V))
//! amortised. Cave systems have at most a few dozen chambers, so this is
//! effectively O(n² log n) in the worst case, which is fast enough.
//!
//! ## Tunnel shape
//!
//! Each MST edge becomes a 4-point Catmull-Rom-friendly control polyline:
//! `[pa, p1, p2, pb]` where `p1` and `p2` are the 1/3 and 2/3 lerp points
//! displaced by random perpendicular offsets. The perpendicular frame is:
//!
//! ```text
//!   axis  = normalize(pb - pa)
//!   perp1 = normalize(axis × world_up)     // horizontal perpendicular
//!   perp2 = normalize(axis × perp1)        // vertical perpendicular
//! ```
//!
//! `world_up = Vec3::Y`. This gives a well-defined perpendicular frame for any
//! non-vertical tunnel (and a fallback to `Vec3::X`/`Vec3::Z` for edge cases).
//! The offsets are scaled by `radius * TUNNEL_WARP_AMP`, giving meander
//! amplitude proportional to tunnel width.
//!
//! See `docs/book/content/part-3-region-build/3.5-caves.mdx`.

use super::ctx::{CaveCtx, StyleParams};
use super::style::SALT_EXTRA_LOOPS;
use crate::worldgen::hash::{mix_range, mix_u32};
use crate::worldgen::region::{Chamber, Tunnel, TunnelRadius};
use crate::worldgen::tuning::*;
use glam::Vec3;

/// Build the tunnel graph connecting all `chambers` for one cave system.
///
/// Returns a `Vec<Tunnel>` (one per MST edge plus up to `MST_EXTRA_LOOPS`
/// additional short edges). Empty if `chambers.len() < 2`.
pub(super) fn connect_chambers_mst(
    ctx: CaveCtx,
    chambers: &[Chamber],
    sp: &StyleParams,
) -> Vec<Tunnel> {
    let CaveCtx {
        seed,
        coord,
        system_idx,
    } = ctx;
    let mut tunnels: Vec<Tunnel> = Vec::new();
    if chambers.len() < 2 {
        return tunnels;
    }

    // ── Kruskal's MST ────────────────────────────────────────────────────

    let mut edges: Vec<(f32, usize, usize)> = Vec::new();
    for i in 0..chambers.len() {
        for j in (i + 1)..chambers.len() {
            let d = (chambers[i].center - chambers[j].center).length();
            edges.push((d, i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Path-compressed union-find: parent[x] == x means x is a root.
    let mut parent: Vec<usize> = (0..chambers.len()).collect();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    let mut mst_edges: Vec<(usize, usize)> = Vec::new();
    for &(_, a, b) in &edges {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent[ra] = rb;
            mst_edges.push((a, b));
        }
    }

    // ── Extra loop edges ─────────────────────────────────────────────────
    //
    // A pure MST gives a tree (no cycles), which means every chamber has
    // exactly one way in/out. Adding a few short non-MST edges re-introduces
    // cycles so exploration feels natural.
    let extra_loop_count = MST_EXTRA_LOOPS.0
        + (mix_u32(seed, &[coord.x, coord.z, system_idx, SALT_EXTRA_LOOPS])
            % (MST_EXTRA_LOOPS.1 - MST_EXTRA_LOOPS.0 + 1));
    let mut added_extras = 0u32;
    for &(_, a, b) in &edges {
        if added_extras >= extra_loop_count {
            break;
        }
        if mst_edges
            .iter()
            .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
        {
            continue;
        }
        mst_edges.push((a, b));
        added_extras += 1;
    }

    // ── Build tunnels from edges ─────────────────────────────────────────

    for (idx, &(a, b)) in mst_edges.iter().enumerate() {
        let radius = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 60, idx as i32],
            sp.tunnel_r.0,
            sp.tunnel_r.1,
        );
        let pa = chambers[a].center;
        let pb = chambers[b].center;

        // Catmull-Rom-friendly control polyline: pa, p1, p2, pb.
        // p1 and p2 are 1/3 and 2/3 lerps displaced by perpendicular offsets
        // derived from the tunnel axis. This breaks the straight line into a
        // meander; the 4-point polyline is later sampled as N capsule segments.
        let axis = (pb - pa).normalize_or_zero();
        // Two orthogonal perpendicular axes for the warp.
        // axis × world_up gives a horizontal perp; axis × perp1 gives vertical.
        let world_up = Vec3::Y;
        let perp1 = axis.cross(world_up).try_normalize().unwrap_or(Vec3::X);
        let perp2 = axis.cross(perp1).try_normalize().unwrap_or(Vec3::Z);

        let amp = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 61, idx as i32],
            radius * 1.0,
            radius * 1.5,
        ) * TUNNEL_WARP_AMP;
        let off1_u = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 62, idx as i32],
            -amp,
            amp,
        );
        let off1_v = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 63, idx as i32],
            -amp,
            amp,
        );
        let off2_u = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 64, idx as i32],
            -amp,
            amp,
        );
        let off2_v = mix_range(
            seed,
            &[coord.x, coord.z, system_idx, 65, idx as i32],
            -amp,
            amp,
        );

        let p1 = pa.lerp(pb, 0.33) + perp1 * off1_u + perp2 * off1_v;
        let p2 = pa.lerp(pb, 0.66) + perp1 * off2_u + perp2 * off2_v;
        tunnels.push(Tunnel {
            control_points: vec![pa, p1, p2, pb],
            radius: TunnelRadius(radius),
        });
    }
    tunnels
}
