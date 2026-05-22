//! Snapshot tool that turns the engine's worldgen pipeline into PNGs
//! for the documentation book. Pure on `(seed, center, zoom, stage)`.
//!
//! Usage:
//!   doc_render snapshot --seed <u64> --center <wx>,<wz> --zoom <N> \
//!                       --stage <stage> --width <px> --output <path>
//!   doc_render parity   --output <path>
//!   doc_render --help

use std::process::ExitCode;

fn print_help() {
    eprintln!(
        "doc_render — generate engine-authoritative images for the docs book

Usage:
  doc_render snapshot --seed <u64> --center <wx,wz> --zoom <N> \\
                      --stage <stage> --width <px> --output <path>
  doc_render parity --output <path>
  doc_render --help

Stages (snapshot):
  continentalness, plate-id, temperature, humidity, desertness,
  weirdness, h-pre, valley-carve, h-target, flow-accum, biome-id,
  aquifer-y, aquifer-substance
"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }
    let sub = args[0].as_str();
    let rest = &args[1..];
    let result: Result<(), String> = match sub {
        "snapshot" => snapshot::run(rest),
        "parity" => parity_dump::run(rest),
        other => Err(format!("unknown subcommand: {other}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

mod parity_dump;
mod snapshot;
