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
        let val = args.get(i + 1).ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--output" => output = Some(PathBuf::from(val)),
            other      => return Err(format!("unknown flag: {other}")),
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

    // Write as JSON array. We hand-build the JSON so we don't pull
    // in serde_json just for this (the schema is simple and stable).
    let json = format!("[\n{}\n]\n", entries.join(",\n"));
    let mut f = File::create(&output).map_err(|e| format!("create {:?}: {e}", output))?;
    f.write_all(json.as_bytes()).map_err(|e| format!("write {:?}: {e}", output))?;
    Ok(())
}
