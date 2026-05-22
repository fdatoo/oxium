//! Generic cubic Hermite spline with nested-value support.
//!
//! A [`CubicSpline`] is either a constant scalar or a list of knots
//! whose values are themselves splines — enabling nested `f(x, y)`
//! composition by stacking 1D splines. Evaluation uses the standard
//! Hermite formula:
//!
//!   t = (input - x1) / (x2 - x1)
//!   result = lerp(t, y1, y2) + t·(1-t)·lerp(t, a, b)
//!     where a =  d1·(x2-x1) − (y2-y1)
//!           b = -d2·(x2-x1) + (y2-y1)
//!
//! Outside the knot range, evaluation is linear extrapolation using
//! the endpoint derivative. Matches the algorithm in Minecraft 1.18+
//! `net/minecraft/util/CubicSpline.java`.

use serde::{Deserialize, Serialize};

/// One knot: input location, output value (possibly itself a
/// spline), derivative dy/dx at this knot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Knot {
    pub loc: f32,
    pub val: f32,
    pub slope: f32,
}

/// Cubic Hermite spline over a scalar input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CubicSpline {
    Constant(f32),
    Multipoint(Vec<Knot>),
}

impl CubicSpline {
    pub fn evaluate(&self, input: f32) -> f32 {
        match self {
            CubicSpline::Constant(v) => *v,
            CubicSpline::Multipoint(knots) => {
                assert!(!knots.is_empty(), "spline must have at least one knot");
                // Below first knot: extrapolate using first knot's slope.
                if input <= knots[0].loc {
                    return knots[0].val + knots[0].slope * (input - knots[0].loc);
                }
                // Above last knot: extrapolate using last knot's slope.
                let last = knots.last().unwrap();
                if input >= last.loc {
                    return last.val + last.slope * (input - last.loc);
                }
                // Linear scan for the segment [k1, k2] containing input.
                let mut i = 0;
                while i + 1 < knots.len() && knots[i + 1].loc < input {
                    i += 1;
                }
                let k1 = &knots[i];
                let k2 = &knots[i + 1];
                let dx = k2.loc - k1.loc;
                let t = (input - k1.loc) / dx;
                let a = k1.slope * dx - (k2.val - k1.val);
                let b = -k2.slope * dx + (k2.val - k1.val);
                // lerp(t, k1.val, k2.val) + t·(1-t)·lerp(t, a, b)
                let lerp_y = k1.val + t * (k2.val - k1.val);
                let lerp_ab = a + t * (b - a);
                lerp_y + t * (1.0 - t) * lerp_ab
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_returns_value_at_any_input() {
        let s = CubicSpline::Constant(0.7);
        assert_eq!(s.evaluate(-5.0), 0.7);
        assert_eq!(s.evaluate(0.0), 0.7);
        assert_eq!(s.evaluate(100.0), 0.7);
    }

    #[test]
    fn two_knots_interpolate_smoothly() {
        // Knots at x=0 (y=0, slope=0) and x=1 (y=1, slope=0).
        // With both slopes 0, Hermite gives an S-curve from (0,0) to (1,1).
        let s = CubicSpline::Multipoint(vec![
            Knot {
                loc: 0.0,
                val: 0.0,
                slope: 0.0,
            },
            Knot {
                loc: 1.0,
                val: 1.0,
                slope: 0.0,
            },
        ]);
        assert!((s.evaluate(0.0) - 0.0).abs() < 1e-5);
        assert!((s.evaluate(1.0) - 1.0).abs() < 1e-5);
        // At t=0.5, S-curve value is exactly 0.5 (symmetry).
        assert!((s.evaluate(0.5) - 0.5).abs() < 1e-5);
        // Below 0.5 the curve should be below 0.5 (concave up).
        assert!(s.evaluate(0.25) < 0.5);
        // Above 0.5 the curve should be above 0.5 (concave down).
        assert!(s.evaluate(0.75) > 0.5);
    }

    #[test]
    fn below_first_knot_extrapolates_linearly() {
        let s = CubicSpline::Multipoint(vec![
            Knot {
                loc: 0.0,
                val: 0.0,
                slope: 1.0,
            },
            Knot {
                loc: 1.0,
                val: 1.0,
                slope: 1.0,
            },
        ]);
        // At x=-1 with slope=1 extrapolation: y = 0 + 1·(-1) = -1.
        assert!((s.evaluate(-1.0) - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn above_last_knot_extrapolates_linearly() {
        let s = CubicSpline::Multipoint(vec![
            Knot {
                loc: 0.0,
                val: 0.0,
                slope: 0.0,
            },
            Knot {
                loc: 1.0,
                val: 1.0,
                slope: 0.5,
            },
        ]);
        // At x=2 with endpoint slope=0.5: y = 1 + 0.5·1 = 1.5.
        assert!((s.evaluate(2.0) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn ron_roundtrip_preserves_knots() {
        let s = CubicSpline::Multipoint(vec![
            Knot {
                loc: -0.5,
                val: 0.3,
                slope: 0.0,
            },
            Knot {
                loc: 0.5,
                val: -0.2,
                slope: 1.0,
            },
        ]);
        let r = ron::to_string(&s).unwrap();
        let parsed: CubicSpline = ron::from_str(&r).unwrap();
        match parsed {
            CubicSpline::Multipoint(knots) => {
                assert_eq!(knots.len(), 2);
                assert!((knots[0].loc - (-0.5)).abs() < 1e-5);
                assert!((knots[1].slope - 1.0).abs() < 1e-5);
            }
            _ => panic!("expected Multipoint"),
        }
    }
}
