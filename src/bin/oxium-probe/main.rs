//! `oxium-probe` — LLM testing harness for the Oxium voxel engine.
//!
//! Provides two subcommands:
//!
//! - **`inspect`** — pure worldgen queries, no renderer, fast. Returns JSON on
//!   stdout. Used for "what biome is at X,Z?", "find me a lava lake", etc.
//! - **`capture`** (future) — boots the renderer with a hidden window, teleports
//!   the camera, takes screenshots with sidecar JSON metadata and perf CSV.
//!
//! # Examples
//!
//! ```sh
//! # Column data at (100, -200) for seed 42
//! cargo run --release --bin oxium-probe -- inspect --column 100,-200
//!
//! # Nearest three cave entrances within 4096 blocks of the origin
//! cargo run --release --bin oxium-probe -- --seed 42 inspect \
//!     --find cave --max-radius 4096 --count 3
//!
//! # Density breakdown at a specific voxel
//! cargo run --release --bin oxium-probe -- inspect --at 100,64,-200
//! ```
//!
//! See `docs/superpowers/plans/let-s-plan-a-proper-stateful-sloth.md` for the
//! full harness design spec.

use clap::Parser;

mod capture;
mod cli;
mod inspect;
mod sidecar;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let cli = cli::Cli::parse();

    match cli.cmd {
        cli::Cmd::Inspect(args) => inspect::run(cli.seed, args),
        cli::Cmd::Capture(args) => capture::run(cli.seed, args),
    }
}
