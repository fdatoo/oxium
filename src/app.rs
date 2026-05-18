//! Top-level App: holds renderer, world, ECS, jobs, time.
//!
//! `AppState` owns everything that lives across frames. It runs the
//! per-frame schedule (`step`) and is held by the `winit` event handler in
//! `main.rs`.
//!
//! The schedule, spelled out:
//!
//! ```text
//! input  →  movement  →  (physics)  →  (interaction)  →  (world_stream)
//!       →  (mesh_upload)  →  render  →  clear_input_buf
//! ```
//!
//! Parentheses mark systems that arrive in later milestones (M3, M6, M7).

use std::sync::Arc;
use std::time::Instant;
use winit::window::Window;

use crate::ecs::systems::input::InputBuf;
use crate::ecs::GameEcs;
use crate::render::Renderer;

/// Global per-session state. Constructed once at startup and stepped once
/// per redraw.
pub struct AppState {
    pub window: Arc<Window>,
    pub renderer: Renderer,
    pub ecs: GameEcs,
    pub input_buf: InputBuf,
    /// Wall-clock time of the previous `step`; used to derive `dt`.
    pub last_tick: Instant,
}

impl AppState {
    /// Spin up renderer + ECS for the given window. The player spawns at
    /// `(16, 12, 40)` — above the M1 test slab so we can see it.
    pub fn new(window: Arc<Window>) -> Self {
        let renderer = Renderer::new(window.clone());
        let ecs = GameEcs::new(glam::Vec3::new(16.0, 12.0, 40.0));
        Self {
            window,
            renderer,
            ecs,
            input_buf: InputBuf::default(),
            last_tick: Instant::now(),
        }
    }

    /// Run one frame: drain input, step systems, render, then clear the
    /// per-frame edge flags in the input buffer.
    pub fn step(&mut self) {
        // Bound dt to absorb hitches and pauses (e.g. window dragged). A
        // 0.1 s clamp keeps the worst-case displacement per axis to half a
        // block at max walk speed, well within the swept-collision regime.
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
        self.last_tick = now;

        crate::ecs::systems::input::apply_input(&mut self.ecs, &self.input_buf);
        crate::ecs::systems::movement::movement(&mut self.ecs, dt);

        if let Err(e) = crate::ecs::systems::render::render(&self.ecs, &self.renderer) {
            // Swapchain hiccups (Lost / Outdated) are recoverable on the
            // next resize; just log and continue.
            log::warn!("render error: {e:?}");
        }

        self.input_buf.clear_per_frame();
    }
}
