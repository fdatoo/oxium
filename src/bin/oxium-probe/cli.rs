//! Clap CLI definitions for `oxium-probe`.

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "oxium-probe",
    version,
    about = "LLM testing harness for Oxium — inspect world state and capture screenshots"
)]
pub struct Cli {
    /// World seed. Defaults to the same seed used by visual regression baselines.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Query world state as JSON without booting the renderer (fast).
    ///
    /// Exactly one of --column, --at, or --find must be supplied.
    Inspect(InspectArgs),

    /// Boot the renderer, warm up the world stream, and capture a screenshot.
    ///
    /// Always produces a sidecar `<out>.json` file with camera state, biome,
    /// and performance counters alongside the PNG.
    Capture(CaptureArgs),
}

#[derive(Args)]
pub struct InspectArgs {
    /// Evaluate column data at world (WX,WZ). e.g. --column 100,-200
    #[arg(long, value_name = "WX,WZ")]
    pub column: Option<String>,

    /// Evaluate density breakdown at world (WX,WY,WZ). e.g. --at 100,64,-200
    #[arg(long, value_name = "WX,WY,WZ")]
    pub at: Option<String>,

    /// Find the nearest terrain feature. Recognised values:
    /// water, lava, cave, river, forest, tropical, desert, tundra, plains,
    /// snowy_forest
    #[arg(long, value_name = "KIND")]
    pub find: Option<String>,

    /// Origin (WX,WZ or WX,WY,WZ) for --find. Defaults to 0,0.
    #[arg(long, value_name = "WX,WZ", default_value = "0,0")]
    pub origin: String,

    /// Maximum search radius in blocks for --find.
    #[arg(long, default_value_t = 8192)]
    pub max_radius: i32,

    /// Number of results to return for --find.
    #[arg(long, default_value_t = 1)]
    pub count: usize,
}

#[derive(Args)]
pub struct CaptureArgs {
    /// Output path for the PNG. A sidecar `<out>.json` is always written next to it.
    #[arg(long, value_name = "PATH.png")]
    pub out: Option<std::path::PathBuf>,

    /// Spawn position (WX,WY,WZ). Overrides --find position if both are given.
    #[arg(long, value_name = "X,Y,Z")]
    pub spawn: Option<String>,

    /// Camera look direction: "yaw_deg,pitch_deg".
    /// Positive pitch = looking up; yaw 0 = facing +X.
    #[arg(long, value_name = "YAW,PITCH")]
    pub look: Option<String>,

    /// Aim the camera at a world position (WX,WY,WZ). Overrides --look.
    #[arg(long, value_name = "WX,WY,WZ")]
    pub look_at: Option<String>,

    /// When combined with --find, aim the camera at the found feature.
    /// Overrides both --look and --look-at.
    #[arg(long)]
    pub look_at_feature: bool,

    /// Time of day (0=midnight, 0.5=noon, 0.75=sunset).
    #[arg(long)]
    pub time: Option<f32>,

    /// Find the nearest terrain feature and use it as the spawn. Kinds:
    /// water, lava, cave, river, forest, tropical, desert, tundra, plains, snowy_forest
    #[arg(long, value_name = "KIND")]
    pub find: Option<String>,

    /// Origin (WX,WZ) for the feature search. Defaults to 0,0.
    #[arg(long, value_name = "WX,WZ", default_value = "0,0")]
    pub origin: String,

    /// Maximum search radius in blocks for --find.
    #[arg(long, default_value_t = 8192)]
    pub max_radius: i32,

    /// Window size for the hidden renderer window (e.g. 1280x720).
    #[arg(long)]
    pub window_size: Option<String>,

    /// Number of frames to render before the quiesce check begins.
    #[arg(long, default_value_t = 60)]
    pub warmup_frames: u32,

    /// Number of screenshots to capture. When > 1, --out must be a directory
    /// path. Screenshots are written as `0001.png`, `0002.png`, … and a
    /// `metrics.csv` is written in the same directory. Default: 1 (single
    /// shot).
    #[arg(long, default_value_t = 1)]
    pub frames: u32,

    /// Minimum wall-clock delay between consecutive captures in burst mode
    /// (--frames > 1). 0 = capture on every rendered frame. Default: 0.
    #[arg(long, default_value_t = 0)]
    pub interval_ms: u64,
}

// ── Coordinate parsers ─────────────────────────────────────────────────────

/// Parse `"WX,WZ"` into `(i32, i32)`.
pub fn parse_xz(s: &str) -> anyhow::Result<(i32, i32)> {
    let parts: Vec<&str> = s.split(',').collect();
    anyhow::ensure!(parts.len() == 2, "expected WX,WZ but got {:?}", s);
    Ok((parts[0].trim().parse()?, parts[1].trim().parse()?))
}

/// Parse `"WX,WY,WZ"` or `"WX,WZ"` (treating Y as 0) into `(i32, i32, i32)`.
pub fn parse_xyz(s: &str) -> anyhow::Result<(i32, i32, i32)> {
    let parts: Vec<&str> = s.split(',').collect();
    match parts.len() {
        2 => Ok((parts[0].trim().parse()?, 0, parts[1].trim().parse()?)),
        3 => Ok((
            parts[0].trim().parse()?,
            parts[1].trim().parse()?,
            parts[2].trim().parse()?,
        )),
        _ => anyhow::bail!("expected WX,WY,WZ or WX,WZ but got {:?}", s),
    }
}
