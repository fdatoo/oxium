//! AppState: aggregates session + UI state + frame timing.

use crate::session::Session;
use crate::world::stream::StreamRadius;
use oxium::worldgen::config::WorldgenConfig;
use std::time::Instant;

pub struct AppState {
    pub session: Session,
    /// Right-mouse button held — drives look (fly cam) / orbit (orbit cam).
    pub mouse_down: bool,
    /// Middle-mouse button held — drives pan (both cams).
    pub mmb_down: bool,
    pub last_cursor: Option<(f64, f64)>,
    pub last_frame: Instant,
    pub last_regen_ms: Option<f32>,
    /// Process-start clock anchor. Used by the shader's pinned-chunk
    /// pulse animation so it ticks against monotonic time, not frame
    /// dt, and stays smooth across pauses.
    pub start_time: Instant,
    /// Total chunks resident in the GPU scene HashMap last frame.
    /// Cached from `SceneRenderer::chunk_count()` so the layout can
    /// read it without a SceneRenderer reference.
    pub scene_chunks_total: usize,
    /// Chunks that passed frustum culling last frame.
    pub scene_chunks_visible: usize,
}

impl AppState {
    pub fn new(seed: u64, config: WorldgenConfig, radius: StreamRadius) -> Self {
        Self {
            session: Session::new(seed, config, radius),
            mouse_down: false,
            mmb_down: false,
            last_cursor: None,
            last_frame: Instant::now(),
            last_regen_ms: None,
            start_time: Instant::now(),
            scene_chunks_total: 0,
            scene_chunks_visible: 0,
        }
    }

    pub fn dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        dt.min(0.1)
    }
}
