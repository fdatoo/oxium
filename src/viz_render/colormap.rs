//! Named gradient palettes for stage overlays. Each maps a normalised
//! f32 in `[0, 1]` to an RGBA byte tuple.

/// Categorical → hue cycle around HSV. For PlateId / BiomeId etc.
pub fn categorical(value_01: f32) -> [u8; 4] {
    let hue = (value_01.fract() * 360.0).abs();
    hsv_to_rgba(hue, 0.65, 0.85)
}

/// Cool blue → green → warm tan. For h_pre / h_target.
pub fn terrain_ramp(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t / 0.5;
        (lerp(40.0, 80.0, s), lerp(80.0, 180.0, s), lerp(140.0, 100.0, s))
    } else {
        let s = (t - 0.5) / 0.5;
        (lerp(80.0, 220.0, s), lerp(180.0, 200.0, s), lerp(100.0, 140.0, s))
    };
    [r as u8, g as u8, b as u8, 255]
}

/// Approximate viridis (perceptually uniform). For temperature / humidity.
pub fn viridis(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    // 5-stop linear gradient: dark purple → blue → teal → green → yellow.
    let stops: [(f32, [f32; 3]); 5] = [
        (0.00, [ 68.0,   1.0,  84.0]),
        (0.25, [ 59.0,  82.0, 139.0]),
        (0.50, [ 33.0, 144.0, 140.0]),
        (0.75, [ 94.0, 201.0,  98.0]),
        (1.00, [253.0, 231.0,  37.0]),
    ];
    // find the segment containing t
    let mut lo = &stops[0];
    let mut hi = &stops[1];
    for i in 0..stops.len() - 1 {
        if t >= stops[i].0 && t <= stops[i + 1].0 {
            lo = &stops[i];
            hi = &stops[i + 1];
            break;
        }
    }
    let s = if (hi.0 - lo.0).abs() < 1e-6 { 0.0 } else { (t - lo.0) / (hi.0 - lo.0) };
    let r = [
        lerp(lo.1[0], hi.1[0], s),
        lerp(lo.1[1], hi.1[1], s),
        lerp(lo.1[2], hi.1[2], s),
    ];
    [r[0] as u8, r[1] as u8, r[2] as u8, 255]
}

/// Divergent: red below 0.5, blue above. For continentalness etc.
/// At t=0.5 (midpoint), both R and B channels are near zero (dark neutral).
pub fn divergent(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        // s goes from 1 (at t=0, full red) → 0 (at t=0.5, dark)
        let s = (0.5 - t) * 2.0;
        [(220.0 * s) as u8, 0, 0, 255]
    } else {
        // s goes from 0 (at t=0.5, dark) → 1 (at t=1, full blue)
        let s = (t - 0.5) * 2.0;
        [0, 0, (220.0 * s) as u8, 255]
    }
}

/// Hot (intensity) — black → red → yellow → white. For valley_carve depth.
pub fn hot(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.33 {
        let s = t / 0.33;
        (255.0 * s, 0.0, 0.0)
    } else if t < 0.66 {
        let s = (t - 0.33) / 0.33;
        (255.0, 255.0 * s, 0.0)
    } else {
        let s = (t - 0.66) / 0.34;
        (255.0, 255.0, 255.0 * s)
    };
    [r as u8, g as u8, b as u8, 255]
}

/// Binary (0 → A, 1 → B). For AquiferSubstance.
pub fn binary(t: f32, a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
    if t < 0.5 { a } else { b }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

fn hsv_to_rgba(h: f32, s: f32, v: f32) -> [u8; 4] {
    let c = v * s;
    let h_p = h / 60.0;
    let x = c * (1.0 - (h_p.rem_euclid(2.0) - 1.0).abs());
    let (r, g, b) = match h_p as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
        255,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_endpoints_match_stops() {
        assert_eq!(viridis(0.0), [68, 1, 84, 255]);
        assert_eq!(viridis(1.0), [253, 231, 37, 255]);
    }

    #[test]
    fn divergent_midpoint_is_neutral() {
        let mid = divergent(0.5);
        // Both red and blue components should be near zero at the midpoint.
        assert!(mid[0] < 50 && mid[2] < 50);
    }

    #[test]
    fn terrain_ramp_low_is_water_blue_ish() {
        let low = terrain_ramp(0.05);
        assert!(low[2] > low[0]); // more blue than red at the low end
    }
}
