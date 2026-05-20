//! AppState: aggregates session + UI state + frame timing.

use crate::preset::PresetUi;
use crate::session::Session;
use crate::world::Region;
use oxium::worldgen::config::WorldgenConfig;
use std::time::Instant;

/// Which visualisation tab is active in the right-panel header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightTab {
    /// 2D top-down map with stage-selector + legend.
    Map,
    /// Cut-plane cross-section.
    CrossSection,
}

impl Default for RightTab {
    fn default() -> Self {
        RightTab::Map
    }
}

pub struct AppState {
    pub session: Session,
    /// Right-mouse button held — drives look (fly cam) / orbit (orbit cam).
    pub mouse_down: bool,
    /// Middle-mouse button held — drives pan (both cams).
    pub mmb_down: bool,
    pub last_cursor: Option<(f64, f64)>,
    pub last_frame: Instant,
    pub last_regen_ms: Option<f32>,
    pub right_tab: RightTab,
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
    pub presets: PresetUi,
    /// World-Y above which the fragment shader discards. A very
    /// large default disables the cutaway. Lower to shave off the
    /// surface and expose caves underneath.
    pub cutaway_max_y: f32,
}

impl AppState {
    pub fn new(seed: u64, config: WorldgenConfig, region: Region) -> Self {
        Self {
            session: Session::new(seed, config, region),
            mouse_down: false,
            mmb_down: false,
            last_cursor: None,
            last_frame: Instant::now(),
            last_regen_ms: None,
            start_time: Instant::now(),
            scene_chunks_total: 0,
            scene_chunks_visible: 0,
            right_tab: RightTab::default(),
            presets: PresetUi::new(),
            cutaway_max_y: 1e9,
        }
    }

    pub fn dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        dt.min(0.1)
    }
}
