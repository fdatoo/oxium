//! AppState: aggregates session + UI state + frame timing.

use crate::session::Session;
use oxium::worldgen::config::WorldgenConfig;
use std::time::Instant;

pub struct AppState {
    pub session: Session,
    pub mouse_down: bool,
    pub last_cursor: Option<(f64, f64)>,
    pub last_frame: Instant,
    pub last_regen_ms: Option<f32>,
}

impl AppState {
    pub fn new(seed: u64, config: WorldgenConfig) -> Self {
        Self {
            session: Session::new(seed, config),
            mouse_down: false,
            last_cursor: None,
            last_frame: Instant::now(),
            last_regen_ms: None,
        }
    }

    pub fn dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        dt.min(0.1)
    }
}
