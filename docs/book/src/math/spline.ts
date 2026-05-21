// parity: src/worldgen/spline.rs (CubicSpline::evaluate)
//
// The Rust code's `CubicSpline::Multipoint` holds a Vec<Knot>; this
// TS port works directly on a Knot[] for the same evaluation. Outside
// the knot range, the result extrapolates linearly using the endpoint
// slope — same as the Rust.

export type Knot = { loc: number; val: number; slope: number };

/**
 * Evaluate a cubic Hermite spline at `input`. Mirrors
 * `src/worldgen/spline.rs::CubicSpline::Multipoint::evaluate`.
 *
 * Knots must be sorted ascending by `loc`. Outside the knot range,
 * extrapolates linearly using the endpoint slope.
 */
export function evaluateSpline(knots: Knot[], input: number): number {
  if (knots.length === 0) throw new Error('empty spline');
  if (knots.length === 1) return knots[0].val;
  if (input <= knots[0].loc) {
    const k = knots[0];
    return k.val + k.slope * (input - k.loc);
  }
  const last = knots[knots.length - 1];
  if (input >= last.loc) {
    return last.val + last.slope * (input - last.loc);
  }
  // Linear scan for the segment [k1, k2] containing input.
  // Matches Rust: while i + 1 < knots.len() && knots[i+1].loc < input { i++ }
  let i = 0;
  while (i + 1 < knots.length && knots[i + 1].loc < input) {
    i++;
  }
  const k1 = knots[i];
  const k2 = knots[i + 1];
  const dx = k2.loc - k1.loc;
  const t = (input - k1.loc) / dx;
  const a =  k1.slope * dx - (k2.val - k1.val);
  const b = -k2.slope * dx + (k2.val - k1.val);
  // lerp(t, k1.val, k2.val) + t*(1-t)*lerp(t, a, b)
  const lerpY = k1.val + t * (k2.val - k1.val);
  const lerpAb = a + t * (b - a);
  return lerpY + t * (1.0 - t) * lerpAb;
}
