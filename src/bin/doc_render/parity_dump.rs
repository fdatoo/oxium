//! Parity dump: writes a JSON file containing reference outputs of
//! the Rust math functions that have TS ports under
//! `docs/book/src/math/`. The TS parity-test page imports this JSON
//! and asserts the TS implementations match.
//!
//! New functions are added here as their TS ports land. Each entry
//! is a `(input, output)` pair using a deterministic input set.

use noise::{Fbm, MultiFractal, NoiseFn, Simplex};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<(), String> {
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let val = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--output" => output = Some(PathBuf::from(val)),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 2;
    }
    let output = output.ok_or("--output required")?;

    let mut entries: Vec<String> = Vec::new();

    // -- fbm --
    // Inputs span the unit square at several scales / seeds /
    // (octaves, persistence) pairs so a TS port that gets octave
    // composition wrong will show.
    let cases: &[(u32, usize, f64, f64, f64)] = &[
        // (seed, octaves, persistence, x, y)
        (42, 4, 0.5, 0.0, 0.0),
        (42, 4, 0.5, 0.25, 0.0),
        (42, 4, 0.5, 0.5, 0.5),
        (42, 1, 0.5, 0.5, 0.5),
        (42, 6, 0.6, 0.123, 0.789),
        (1337, 4, 0.5, 0.0, 0.0),
    ];
    for &(seed, oct, pers, x, y) in cases {
        let fbm = Fbm::<Simplex>::new(seed)
            .set_octaves(oct)
            .set_frequency(1.0)
            .set_persistence(pers);
        let v = fbm.get([x, y]);
        entries.push(format!(
            "    {{\"fn\":\"fbm\",\"args\":{{\"seed\":{seed},\"octaves\":{oct},\"persistence\":{pers},\"x\":{x},\"y\":{y}}},\"out\":{v}}}",
        ));
    }

    // -- spline --
    {
        use oxium::worldgen::spline::{CubicSpline, Knot};
        #[allow(clippy::type_complexity)]
        let spline_cases: &[(&str, &[(f32, f32, f32)], f32)] = &[
            // (name, knots [(loc, val, slope), ...], input)
            ("ramp", &[(0.0, 0.0, 1.0), (1.0, 1.0, 1.0)], 0.5),
            ("plateau", &[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)], 0.5),
            ("plateau", &[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)], 0.25),
            (
                "dip",
                &[(0.0, 1.0, -2.0), (0.5, 0.0, 0.0), (1.0, 1.0, 2.0)],
                0.5,
            ),
            (
                "dip",
                &[(0.0, 1.0, -2.0), (0.5, 0.0, 0.0), (1.0, 1.0, 2.0)],
                0.25,
            ),
        ];
        for (name, knots_raw, input) in spline_cases {
            let s = CubicSpline::Multipoint(
                knots_raw
                    .iter()
                    .map(|&(loc, val, slope)| Knot { loc, val, slope })
                    .collect(),
            );
            let v = s.evaluate(*input);
            let knots_json: Vec<String> = knots_raw
                .iter()
                .map(|(loc, val, slope)| format!("[{loc},{val},{slope}]"))
                .collect();
            entries.push(format!(
                "    {{\"fn\":\"spline\",\"args\":{{\"name\":\"{name}\",\"knots\":[{}],\"input\":{input}}},\"out\":{v}}}",
                knots_json.join(",")
            ));
        }
    }

    // -- ellipsoid_sdf --
    // The chapter's TS port uses a 2D version of the formula the engine
    // applies in 3D inside cave_sdf. This block asserts the TS port
    // matches the same formula evaluated in Rust on identical inputs.
    {
        let cases: &[(f32, f32, f32, f32, f32, f32)] = &[
            // (px, py, cx, cy, rx, ry)
            (0.0, 0.0, 0.0, 0.0, 1.0, 1.0), // dead center → 1.0
            (1.0, 0.0, 0.0, 0.0, 1.0, 1.0), // on boundary → 0.0
            (0.5, 0.0, 0.0, 0.0, 1.0, 1.0), // halfway radially → 0.75
            (2.0, 0.0, 0.0, 0.0, 1.0, 1.0), // outside → 0.0
            (0.0, 0.5, 0.0, 0.0, 1.0, 2.0), // off-y in elongated → 1 - 0.0625
        ];
        for &(px, py, cx, cy, rx, ry) in cases {
            let dx = (px - cx) / rx;
            let dy = (py - cy) / ry;
            let v = (1.0_f32 - (dx * dx + dy * dy)).max(0.0);
            entries.push(format!(
                "    {{\"fn\":\"ellipsoid2D\",\"args\":{{\"px\":{px},\"py\":{py},\"cx\":{cx},\"cy\":{cy},\"rx\":{rx},\"ry\":{ry}}},\"out\":{v}}}",
            ));
        }
    }

    // -- trilerp --
    // Pure math — assert TS port matches Rust exactly. Corner order:
    // c[xi + 2*yi + 4*zi].
    {
        let cases: &[([f32; 8], f32, f32, f32)] = &[
            ([0.0; 8], 0.5, 0.5, 0.5),                                 // all zero → 0
            ([1.0; 8], 0.0, 0.0, 0.0),                                 // all one  → 1
            ([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 0.5, 0.0, 0.0), // gradient in x → 0.5
            ([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 0.0, 0.0, 0.5), // gradient in z → 0.5
            ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5, 0.5, 0.5), // single corner → 0.125
        ];
        for &(corners, tx, ty, tz) in cases {
            let c = corners;
            let c00 = c[0] * (1.0 - tx) + c[1] * tx;
            let c10 = c[2] * (1.0 - tx) + c[3] * tx;
            let c01 = c[4] * (1.0 - tx) + c[5] * tx;
            let c11 = c[6] * (1.0 - tx) + c[7] * tx;
            let c0 = c00 * (1.0 - ty) + c10 * ty;
            let c1 = c01 * (1.0 - ty) + c11 * ty;
            let v = c0 * (1.0 - tz) + c1 * tz;
            let corners_json: Vec<String> = c.iter().map(|x| format!("{x}")).collect();
            entries.push(format!(
                "    {{\"fn\":\"trilerp\",\"args\":{{\"corners\":[{}],\"tx\":{tx},\"ty\":{ty},\"tz\":{tz}}},\"out\":{v}}}",
                corners_json.join(",")
            ));
        }
    }

    // Write as JSON array. We hand-build the JSON so we don't pull
    // in serde_json just for this (the schema is simple and stable).
    let json = format!("[\n{}\n]\n", entries.join(",\n"));
    let mut f = File::create(&output).map_err(|e| format!("create {:?}: {e}", output))?;
    f.write_all(json.as_bytes())
        .map_err(|e| format!("write {:?}: {e}", output))?;
    Ok(())
}
