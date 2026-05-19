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
    /// Rolling FPS meter — sampled every `step` and read by the HUD.
    pub fps_meter: FpsMeter,
    /// Latest perf snapshot — what the HUD prints below FPS/XYZ. Set
    /// once per step from the bookkeeping numbers the other systems
    /// hand back (relight queue size, GPU-uploaded mesh count, etc).
    pub perf: PerfSnapshot,
}

/// Per-frame counters shown in the debug HUD. Cheap to keep around;
/// the values are recomputed every step from whatever the systems
/// hand back, so there's no stale-data hazard.
#[derive(Debug, Default, Clone, Copy)]
pub struct PerfSnapshot {
    /// Chunks currently flagged `dirty.light` and waiting for the
    /// relight pump. Non-zero means lighting is still converging — a
    /// number that holds steady (instead of trending toward zero) is
    /// the smoking gun for a cascade that doesn't terminate.
    pub light_queue: usize,
    /// LOD0 mesh slots currently held by the renderer. Useful as a
    /// proxy for "is the world fully streamed in yet."
    pub chunks_rendered: usize,
    /// Milliseconds of *work* (not wall-clock between frames) the
    /// last step took. With vsync on, frame time = work + vsync
    /// wait; work-time alone tells you whether a low FPS reading is
    /// a real perf problem or just the display refresh dropping to
    /// 60Hz under low demand. ProMotion displays often park at 60Hz
    /// after a burst of activity even when the app's per-step work
    /// is still well under 8ms.
    pub work_ms: f32,
}

/// How often the autosave system flushes modified chunks to disk.
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);

/// Rolling window of recent frame durations, used to compute a smooth
/// FPS readout for the HUD. A short window (1 s of frames) reacts
/// quickly to genuine slowdowns without flickering between samples.
pub struct FpsMeter {
    samples: std::collections::VecDeque<f32>,
    capacity: usize,
}

impl FpsMeter {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record one frame's duration (seconds). Drops the oldest sample
    /// when the ring is full so the average tracks the most recent
    /// `capacity` frames.
    pub fn record(&mut self, dt: f32) {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(dt.max(1.0e-5));
    }

    /// Smoothed frames-per-second over the current window. Returns 0
    /// before any frames have been recorded so the HUD doesn't display
    /// "NaN" on the first frame.
    pub fn fps(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().sum();
        self.samples.len() as f32 / sum
    }
}

impl AppState {
    /// Build all subsystems. The player spawns at `(16, 96, 16)` — above
    /// the average terrain height (≈ 64) so the world streams in
    /// beneath the player rather than around them.
    pub fn new(window: Arc<Window>) -> Self {
        Self::new_with_spawn(window, glam::Vec3::new(16.0, 96.0, 16.0), false)
    }

    /// Like [`new`] but accepts an explicit spawn point. Used by the
    /// CLI `--spawn` / `--find-water` flags. `uncapped = true`
    /// switches the swapchain to `PresentMode::Immediate` so HUD FPS
    /// shows actual throughput instead of being capped to the
    /// display refresh rate.
    pub fn new_with_spawn(window: Arc<Window>, spawn: glam::Vec3, uncapped: bool) -> Self {
        let seed = 42;
        let present_mode = if uncapped {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };
        let renderer = Renderer::new_with_present_mode(window.clone(), present_mode);
        let ecs = GameEcs::new(spawn);
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
            fps_meter: FpsMeter::new(60),
            perf: PerfSnapshot::default(),
        }
    }

    /// Run one frame: input → movement → world_stream → drain_jobs →
    /// world_unload → render. See module docs for ordering rationale.
    pub fn step(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
        self.last_tick = now;
        self.fps_meter.record(dt);
        // Start of the per-step CPU work; stopped just before the
        // render `present()` call so the measurement excludes
        // wall-clock time we spend waiting on vsync.
        let work_start = now;

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
        // Relight pump runs after the two job-drain stages so it picks
        // up the `dirty.light` flags those handlers just set on newly
        // loaded/generated chunks. Each frame queues a bounded number
        // of relight jobs; over a few seconds the world converges to
        // a fixed lighting state with correct cross-chunk propagation.
        // The return value is the *total* (not just dispatched) count
        // of `dirty.light` chunks, which the HUD prints so we can see
        // whether the cascade is terminating.
        self.perf.light_queue = crate::ecs::systems::mesh_upload::relight_pump(
            &mut self.world,
            &self.jobs,
            &self.registry,
        );
        self.perf.chunks_rendered = self.renderer.chunk_mesh_count();
        // Sample work-time just before the render call so the metric
        // captures every system except the GPU present (which is
        // where vsync blocks). On vsync-capped frames this number
        // will be much lower than `1000 / FPS`; on perf-bound frames
        // they'll be roughly equal.
        self.perf.work_ms = work_start.elapsed().as_secs_f32() * 1000.0;
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
        let fps = self.fps_meter.fps();
        if let Err(e) = crate::ecs::systems::render::render(
            &self.ecs,
            &mut self.renderer,
            &self.registry,
            fps,
            time,
            &self.perf,
        ) {
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
