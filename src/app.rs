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

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::window::Window;

use crate::ecs::systems::input::InputBuf;
use crate::ecs::GameEcs;
use crate::jobs::Jobs;
use crate::persistence::thread::{PersistRequest, Persistence};
use crate::render::Renderer;
use crate::voxel::block::BlockRegistry;
use crate::voxel::world::{ChunkSlot, World};
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
    /// Dedicated I/O thread for chunk save/load.
    pub persistence: Persistence,
    /// Root directory for region files this session writes to.
    pub saves_dir: PathBuf,
    /// Wall-clock time of the previous autosave tick. Autosave runs every
    /// `AUTOSAVE_INTERVAL` seconds.
    pub last_autosave: Instant,
    pub input_buf: InputBuf,
    /// Wall-clock time of the previous `step`; used to derive `dt`.
    pub last_tick: Instant,
    /// Real time when AppState was created — used to compute `time`
    /// for shader animation (water shimmer, etc).
    pub start_time: Instant,
}

/// How often the autosave system flushes modified chunks to disk.
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);

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

        // One save folder per binary; could grow into a world-picker UI.
        let saves_dir = PathBuf::from("saves/default");
        std::fs::create_dir_all(&saves_dir).ok();
        let persistence = Persistence::spawn(saves_dir.clone());

        Self {
            window,
            renderer,
            ecs,
            world,
            jobs,
            generator,
            registry,
            persistence,
            saves_dir,
            last_autosave: Instant::now(),
            input_buf: InputBuf::default(),
            last_tick: Instant::now(),
            start_time: Instant::now(),
        }
    }

    /// Run one frame: input → movement → world_stream → drain_jobs →
    /// world_unload → render. See module docs for ordering rationale.
    pub fn step(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
        self.last_tick = now;

        crate::ecs::systems::input::apply_input(&mut self.ecs, &self.input_buf);
        crate::ecs::systems::time_of_day::advance(&mut self.ecs, dt);
        crate::ecs::systems::movement::movement(&mut self.ecs, dt);
        crate::ecs::systems::physics::physics(&mut self.ecs, &self.world, dt);

        // Interaction: raycast + place/break. Returns chunks the edit
        // dirtied; we immediately spawn relight (followed by remesh) on
        // each so the player sees the result within a frame or two.
        let dirty_chunks =
            crate::ecs::systems::interaction::interaction(&mut self.ecs, &mut self.world);
        for c in dirty_chunks {
            use crate::voxel::world::ChunkSlot;
            let needs_light = match self.world.chunks.get(&c) {
                Some(ChunkSlot::Stored { meta, .. }) => meta.dirty.light,
                _ => false,
            };
            let Some(ChunkSlot::Stored { data, .. }) = self.world.chunks.get(&c) else {
                continue;
            };
            let data_arc = std::sync::Arc::new(data.clone());
            let neighbors =
                crate::ecs::systems::mesh_upload::gather_neighbors(&self.world, c);
            if needs_light {
                self.jobs
                    .spawn_relight(c, data_arc, neighbors, self.registry.clone());
            } else {
                self.jobs
                    .spawn_mesh_lod0(c, data_arc, neighbors, self.registry.clone());
            }
        }
        crate::ecs::systems::world_stream::world_stream(
            &self.ecs,
            &mut self.world,
            &self.jobs,
            &self.generator,
            &self.registry,
            &self.persistence,
            &self.saves_dir,
        );
        crate::ecs::systems::mesh_upload::drain_jobs(
            &mut self.world,
            &self.jobs,
            &mut self.renderer,
            &self.registry,
        );
        crate::ecs::systems::mesh_upload::drain_persistence(
            &mut self.world,
            &self.jobs,
            &self.persistence,
            &self.generator,
            &self.registry,
        );
        crate::ecs::systems::world_stream::world_unload(
            &self.ecs,
            &mut self.world,
            &mut self.renderer,
            &self.persistence,
        );

        // Autosave: every AUTOSAVE_INTERVAL, push every modified chunk
        // through to the persistence thread. Cheap if no chunks are
        // modified.
        if self.last_autosave.elapsed() >= AUTOSAVE_INTERVAL {
            self.last_autosave = Instant::now();
            self.flush_modified();
        }

        let time = self.start_time.elapsed().as_secs_f32();
        if let Err(e) =
            crate::ecs::systems::render::render(&self.ecs, &mut self.renderer, time)
        {
            log::warn!("render error: {e:?}");
        }
        self.input_buf.clear_per_frame();
    }

    /// Send every currently-modified chunk through the persistence thread.
    /// Used by both autosave and the `Drop` flush-on-close path.
    fn flush_modified(&self) {
        for (c, slot) in &self.world.chunks {
            if let ChunkSlot::Stored { data, meta } = slot
                && meta.modified
            {
                let _ = self.persistence.req_tx.send(PersistRequest::Save {
                    coord: *c,
                    data: data.clone(),
                });
            }
        }
    }
}

impl Drop for AppState {
    /// On a clean shutdown, push any modified chunks out to disk and
    /// politely tell the persistence thread to exit. We send a
    /// `Shutdown` *after* the saves so the channel drains in order.
    fn drop(&mut self) {
        self.flush_modified();
        let _ = self.persistence.req_tx.send(PersistRequest::Shutdown);
    }
}
