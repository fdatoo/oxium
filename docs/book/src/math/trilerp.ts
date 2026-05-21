// parity: src/worldgen/density_graph.rs (CellEvaluator::evaluate trilerp inside the cell)

/**
 * Trilinear interpolation. Given the 8 corner values of a unit cube,
 * compute the value at a point (tx, ty, tz) where each t ∈ [0, 1].
 *
 * Corner order: c[xi + 2*yi + 4*zi] for xi, yi, zi ∈ {0, 1}.
 * So index 0 is (0,0,0); index 1 is (1,0,0); index 2 is (0,1,0); etc.
 */
export function trilerp(
  corners: [number, number, number, number, number, number, number, number],
  tx: number, ty: number, tz: number,
): number {
  // Lerp along x at each of 4 (yi, zi) pairs.
  const c00 = corners[0] * (1 - tx) + corners[1] * tx;
  const c10 = corners[2] * (1 - tx) + corners[3] * tx;
  const c01 = corners[4] * (1 - tx) + corners[5] * tx;
  const c11 = corners[6] * (1 - tx) + corners[7] * tx;
  // Lerp along y at each of 2 zi.
  const c0 = c00 * (1 - ty) + c10 * ty;
  const c1 = c01 * (1 - ty) + c11 * ty;
  // Lerp along z.
  return c0 * (1 - tz) + c1 * tz;
}

/** 2D bilinear interpolation — used by the chapter's 2D widget visualization. */
export function bilerp(
  c00: number, c10: number, c01: number, c11: number,
  tx: number, ty: number,
): number {
  const a = c00 * (1 - tx) + c10 * tx;
  const b = c01 * (1 - tx) + c11 * tx;
  return a * (1 - ty) + b * ty;
}
