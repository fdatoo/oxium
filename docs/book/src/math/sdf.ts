// parity: src/worldgen/caves.rs
// Ellipsoid + capsule SDFs adapted to 2D for the chapter's visualization.
// The engine uses 3D versions of the same formulas inside cave_sdf.

/**
 * 2D ellipsoid SDF. Returns intensity ∈ [0, 1] inside, 0 outside.
 * Inside the boundary, the value is `1 - normalized_distance²` clamped to ≥0.
 *
 * Matches the chamber formula in src/worldgen/caves.rs::cave_sdf:
 *   ratio = (dx/rx)² + (dy/ry)² + (dz/rz)²
 *   sdf   = (1.0 - ratio).max(0.0) * CAVE_SDF_INTENSITY
 * (adapted to 2D, CAVE_SDF_INTENSITY factored out so intensity ∈ [0, 1])
 */
export function ellipsoid2D(
  px: number, py: number,
  cx: number, cy: number,
  rx: number, ry: number,
): number {
  const dx = (px - cx) / rx;
  const dy = (py - cy) / ry;
  return Math.max(0, 1 - (dx * dx + dy * dy));
}

/**
 * 2D capsule SDF: project the query point onto the line segment AB,
 * compute distance to the projection, and return `1 - d/radius`
 * (clamped to ≥0).
 *
 * Matches the tunnel formula in src/worldgen/caves.rs::cave_sdf:
 *   t_param = clamp(dot(p-a, ab) / len_sq, 0, 1)
 *   closest = a + ab * t_param
 *   sdf     = (1.0 - dist/radius).max(0.0) * CAVE_SDF_INTENSITY
 */
export function capsule2D(
  px: number, py: number,
  ax: number, ay: number, bx: number, by: number,
  radius: number,
): number {
  const abx = bx - ax, aby = by - ay;
  const apx = px - ax, apy = py - ay;
  const denom = abx * abx + aby * aby || 1e-9;
  let t = (apx * abx + apy * aby) / denom;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * abx, cy = ay + t * aby;
  const d = Math.hypot(px - cx, py - cy);
  return Math.max(0, 1 - d / radius);
}

/** Union via max: most-inside-of-anything wins. */
export function unionSdf(values: number[]): number {
  return values.reduce((a, b) => Math.max(a, b), 0);
}

/** Intersection via min: least-inside-of-anything wins. */
export function intersectionSdf(values: number[]): number {
  return values.reduce((a, b) => Math.min(a, b), Infinity);
}
