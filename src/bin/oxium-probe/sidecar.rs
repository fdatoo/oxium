//! Sidecar JSON metadata written alongside each captured screenshot.
//!
//! Every `<name>.png` produced by `oxium-probe capture` gets a companion
//! `<name>.png.json` with this struct serialised as pretty-printed JSON.
//! The schema lets an LLM answer questions like "what biome is the camera in?"
//! or "why is this frame slow?" without parsing the PNG itself.

use serde::Serialize;

/// Per-screenshot metadata written as `<path>.json`.
#[derive(Debug, Serialize)]
pub struct CaptureMeta {
    /// Schema version. Bump when fields are removed or semantics change.
    pub schema: u32,
    /// World seed used for this capture.
    pub seed: u64,
    /// Camera state at capture time.
    pub camera: CameraInfo,
    /// Surface column data underneath the camera position.
    pub ground: GroundInfo,
    /// Time of day when the frame was captured (0=midnight, 0.5=noon).
    pub time_of_day: f32,
    /// Sun direction as `[x, y, z]` unit vector.
    pub sun_dir: [f32; 3],
    /// Sun intensity in `[0, 1]`.
    pub sun_intensity: f32,
    /// Viewport dimensions at capture time.
    pub viewport: [u32; 2],
    /// Snapshot of frame performance counters at capture time.
    pub perf: PerfInfo,
    /// The terrain feature the camera was pointed at (if `--find` was used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_target: Option<FeatureTargetInfo>,
}

/// Camera position and orientation at capture time.
#[derive(Debug, Serialize)]
pub struct CameraInfo {
    /// World-space eye position `[x, y, z]`.
    pub eye_world: [f32; 3],
    /// Horizontal rotation in degrees (same convention as `--look`).
    pub yaw_deg: f32,
    /// Vertical elevation in degrees (positive = up).
    pub pitch_deg: f32,
}

/// Surface column data directly beneath the camera.
#[derive(Debug, Serialize)]
pub struct GroundInfo {
    /// Column world X and Z coordinates.
    pub column: [i32; 2],
    /// Biome name at this column.
    pub biome: String,
    /// Surface height in world Y blocks.
    pub height: i32,
    /// Y of the topmost water voxel, or `null` if dry land.
    pub water_surface_y: Option<i32>,
    /// True if the column has a cliff face.
    pub is_cliff: bool,
}

/// Frame performance counters at capture time.
#[derive(Debug, Serialize)]
pub struct PerfInfo {
    /// Milliseconds of CPU work (excludes vsync wait) for the last captured frame.
    pub work_ms: f32,
    /// Active draw calls in the last opaque pass.
    pub draw_calls: u32,
    /// LOD0 mesh slots held by the renderer.
    pub chunks_rendered: usize,
    /// Chunks in `Stored` state (gen + lighting complete).
    pub chunks_loaded: usize,
    /// Chunks still waiting for gen/lighting to complete.
    pub chunks_pending: usize,
    /// Relight backlog depth at capture time.
    pub light_queue: usize,
}

/// The terrain feature the camera was aimed at, if `--find` was used.
#[derive(Debug, Serialize)]
pub struct FeatureTargetInfo {
    /// Feature kind (e.g. `"Lava"`, `"CaveEntrance"`).
    pub kind: String,
    /// World-space position of the feature `[x, y, z]`.
    pub pos: [i32; 3],
    /// Horizontal distance from the origin to the feature in blocks.
    pub distance_blocks: f32,
}

impl CaptureMeta {
    /// Write `self` as pretty-printed JSON to `path`.
    pub fn write(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
