//! Top-level App: holds renderer, world, ECS, jobs, time.
//!
//! `AppState` is the single object that lives across frames. It owns the
//! renderer, the voxel world, the ECS, the background job pool, and the
//! shared generator + block registry. The per-frame `step` runs the
//! complete schedule:
//!
//! ```text
//! input  →  movement  →  world_stream  →  drain_jobs  →  world_unload  →  render
//!        →  clear_input_buf
//! ```
//!
//! Each system is a free function in `ecs::systems`; the ordering here is
//! the *only* place we declare it — easy to debug and reason about.

use std::sync::Arc;
use std::time::Instant;
use winit::window::Window;

use crate::ecs::systems::input::InputBuf;
use crate::ecs::GameEcs;
use crate::jobs::Jobs;
use crate::render::Renderer;
use crate::voxel::block::BlockRegistry;
use crate::voxel::world::World;
use crate::worldgen::Generator;

/// Global per-session state. Constructed once at startup and stepped once
/// per redraw.
pub struct AppState {
    pub window: Arc<Window>,
    pub renderer: Renderer,
    pub ecs: GameEcs,
    /// All currently-loaded chunks, keyed by `ChunkCoord`. Not in the ECS
    /// (chunk data is too cache-hot and read by worker threads).
    pub world: World,
    /// Worker pool + result channel for chunk gen / mesh / lighting jobs.
    pub jobs: Jobs,
    /// World generator, shared with worker threads via `Arc`.
    pub generator: Arc<Generator>,
    /// Block registry, shared with worker threads via `Arc`.
    pub registry: Arc<BlockRegistry>,
    pub input_buf: InputBuf,
    /// Wall-clock time of the previous `step`; used to derive `dt`.
    pub last_tick: Instant,
}

impl AppState {
    /// Build all subsystems. The player spawns at `(16, 96, 16)` — above
    /// the average terrain height (≈ 64) so the world streams in
    /// beneath the player rather than around them.
    pub fn new(window: Arc<Window>) -> Self {
        let seed = 42;
        let renderer = Renderer::new(window.clone());
        let ecs = GameEcs::new(glam::Vec3::new(16.0, 96.0, 16.0));
        let world = World::new(seed);
        let jobs = Jobs::new();
        let generator = Arc::new(Generator::new(seed));
        let registry = Arc::new(BlockRegistry::new());
        Self {
            window,
            renderer,
            ecs,
            world,
            jobs,
            generator,
            registry,
            input_buf: InputBuf::default(),
            last_tick: Instant::now(),
        }
    }

    /// Run one frame: input → movement → world_stream → drain_jobs →
    /// world_unload → render. See module docs for ordering rationale.
    pub fn step(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
        self.last_tick = now;

        crate::ecs::systems::input::apply_input(&mut self.ecs, &self.input_buf);
        crate::ecs::systems::movement::movement(&mut self.ecs, dt);
        crate::ecs::systems::world_stream::world_stream(
            &self.ecs,
            &mut self.world,
            &self.jobs,
            &self.generator,
        );
        crate::ecs::systems::mesh_upload::drain_jobs(
            &mut self.world,
            &self.jobs,
            &mut self.renderer,
            &self.registry,
        );
        crate::ecs::systems::world_stream::world_unload(
            &self.ecs,
            &mut self.world,
            &mut self.renderer,
        );

        if let Err(e) = crate::ecs::systems::render::render(&self.ecs, &self.renderer) {
            log::warn!("render error: {e:?}");
        }
        self.input_buf.clear_per_frame();
    }
}
