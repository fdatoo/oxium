//! `oxium-probe inspect` subcommand implementation.
//!
//! Handles three query modes:
//! - `--column WX,WZ` — [`ColumnData`] for a surface column as JSON.
//! - `--at WX,WY,WZ` — [`DensityBreakdown`] for one voxel as JSON.
//! - `--find <kind>` — nearest terrain features as JSON via [`features::find`].
//!
//! All output goes to stdout as a single JSON object. Errors are written to
//! stderr and the process exits with a non-zero status.

use anyhow::{bail, Context};
use oxium::worldgen::{Generator, features};

use crate::cli::{InspectArgs, parse_xz, parse_xyz};

/// Entry point called from `main.rs` for the `inspect` subcommand.
pub fn run(seed: u64, args: InspectArgs) -> anyhow::Result<()> {
    // Exactly one mode must be supplied.
    let mode_count = args.column.is_some() as u8
        + args.at.is_some() as u8
        + args.find.is_some() as u8;
    if mode_count == 0 {
        bail!("one of --column, --at, or --find is required");
    }
    if mode_count > 1 {
        bail!("--column, --at, and --find are mutually exclusive");
    }

    let g = Generator::new(seed);

    if let Some(col_str) = args.column {
        run_column(&g, seed, &col_str)
    } else if let Some(at_str) = args.at {
        run_at(&g, seed, &at_str)
    } else {
        let kind_str = args.find.unwrap();
        run_find(&g, seed, &kind_str, &args.origin, args.max_radius, args.count)
    }
}

// ── --column ──────────────────────────────────────────────────────────────────

fn run_column(g: &Generator, seed: u64, s: &str) -> anyhow::Result<()> {
    let (wx, wz) = parse_xz(s).context("--column: expected WX,WZ")?;
    let col = g.column_data(wx, wz);

    // Build output manually so field order is stable and readable.
    let out = serde_json::json!({
        "schema": 1,
        "query": "column",
        "seed": seed,
        "wx": wx,
        "wz": wz,
        "height": col.height,
        "biome": format!("{:?}", col.biome),
        "is_cliff": col.is_cliff,
        "desertness": col.desertness,
        "h_pre": col.h_pre,
        "water_surface_y": col.water_surface_y,
    });

    print_json(&out)
}

// ── --at ──────────────────────────────────────────────────────────────────────

fn run_at(g: &Generator, seed: u64, s: &str) -> anyhow::Result<()> {
    let (wx, wy, wz) = parse_xyz(s).context("--at: expected WX,WY,WZ")?;
    let bd = g.evaluate_density_breakdown(wx, wy, wz);

    let out = serde_json::json!({
        "schema": 1,
        "query": "voxel",
        "seed": seed,
        "wx": wx,
        "wy": wy,
        "wz": wz,
        "bias": bd.bias,
        "base_3d": bd.base_3d,
        "cave_sdf": bd.cave_sdf,
        "cheese": bd.cheese,
        "tera": bd.tera,
        "pillar": bd.pillar,
        "final_density": bd.final_density,
        "block": format!("{:?}", bd.block),
        "cave_style": bd.cave_style,
        "cave_band": bd.cave_band,
    });

    print_json(&out)
}

// ── --find ────────────────────────────────────────────────────────────────────

fn run_find(
    g: &Generator,
    seed: u64,
    kind_str: &str,
    origin_str: &str,
    max_radius: i32,
    count: usize,
) -> anyhow::Result<()> {
    let kind: features::FeatureKind = kind_str.parse()?;
    let (ox, oz) = parse_xz(origin_str).context("--origin: expected WX,WZ")?;

    let hits = features::find(g, ox, oz, kind, max_radius, count)?;

    // Serialize hits — serde_json knows how because FeatureHit derives Serialize.
    let hits_json: Vec<serde_json::Value> = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "kind": format!("{:?}", h.kind),
                "pos": { "x": h.pos[0], "y": h.pos[1], "z": h.pos[2] },
                "region": { "x": h.region[0], "z": h.region[1] },
                "distance_blocks": h.distance_blocks,
            })
        })
        .collect();

    let out = serde_json::json!({
        "schema": 1,
        "query": "find",
        "seed": seed,
        "feature": kind_str,
        "origin": { "x": ox, "z": oz },
        "max_radius": max_radius,
        "count_requested": count,
        "count_found": hits_json.len(),
        "hits": hits_json,
    });

    print_json(&out)
}

// ── Shared output ─────────────────────────────────────────────────────────────

fn print_json(v: &serde_json::Value) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}
