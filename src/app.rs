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
    /// Optional per-frame profiler. `Some` when launched with
    /// `--profile <path>` on the binary. Each step wraps its systems
    /// in `profiler::time(...)` calls; one CSV row per frame lands in
    /// the file. The HUD path doesn't read this — it's strictly for
    /// offline post-mortem analysis.
    pub profiler: Option<crate::profiler::Profiler>,
    /// Counter of edit actions (place/break) this frame, recorded
    /// into the profile CSV so we can grep for the frames where
    /// the user triggered an edit.
    pub frame_edit_count: u32,
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
    /// Number of chunk draws issued in the last opaque pass. With no
    /// frustum culling we draw every chunk in the load radius, so this
    /// tracks roughly `chunks_rendered` — but it's the actual count
    /// the renderer fed to wgpu, useful for spotting "the CPU is
    /// spending most of the frame in `wgpu::draw_indexed` overhead".
    pub draw_calls: u32,
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
    /// Build all subsystems. The player spawns at the supplied
    /// world-space coord (the CLI `--spawn` / `--find-water` flags
    /// pick this; the default mid-air location is set in `main`).
    /// `uncapped = true` switches the swapchain to
    /// `PresentMode::Immediate` so HUD FPS shows actual throughput
    /// instead of being capped to the display refresh rate.
    /// `profile_path = Some(p)` opens `crate::profiler::Profiler`
    /// and starts writing per-frame CSV.
    pub fn new_with_spawn(
        window: Arc<Window>,
        spawn: glam::Vec3,
        uncapped: bool,
        profile_path: Option<&std::path::Path>,
    ) -> Self {
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
            profiler: profile_path.and_then(|p| match crate::profiler::Profiler::open(p) {
                Ok(prof) => {
                    log::info!("profiling enabled, writing to {}", p.display());
                    Some(prof)
                }
                Err(e) => {
                    log::warn!("failed to open profile path {}: {e:?}", p.display());
                    None
                }
            }),
            frame_edit_count: 0,
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

        let prof = self.profiler.as_ref();
        use crate::profiler::time;

        time(prof, "input", || {
            crate::ecs::systems::input::apply_input(&mut self.ecs, &self.input_buf)
        });
        time(prof, "time_of_day", || {
            crate::ecs::systems::time_of_day::advance(&mut self.ecs, dt)
        });
        time(prof, "movement", || {
            crate::ecs::systems::movement::movement(&mut self.ecs, dt)
        });
        time(prof, "physics", || {
            crate::ecs::systems::physics::physics(&mut self.ecs, &self.world, dt)
        });

        // Interaction: raycast + place/break. Returns chunks the edit
        // dirtied; we immediately spawn relight (followed by remesh) on
        // each so the player sees the result within a frame or two.
        let dirty_chunks = time(prof, "interaction", || {
            crate::ecs::systems::interaction::interaction(&mut self.ecs, &mut self.world)
        });
        self.frame_edit_count = dirty_chunks.len() as u32;
        // Run the EDIT's relight + LOD0 mesh + upload INLINE on
        // the main thread. Without this, the player-edit jobs go
        // to the back of the rayon queue behind hundreds of
        // streaming gen/mesh jobs from the initial fly-in, and
        // the visible "block didn't break" lag becomes
        // multi-second. ~10 ms one-frame stall is far better
        // than that wait — and avoids the out-of-order race where
        // streaming meshes for the same chunk overwrite the edit's
        // mesh. The mesh-version tag still protects against the
        // latter for the cascade-triggered jobs that *do* go to
        // the pool.
        //
        // Not wrapped in `time(...)` because the closure would need
        // `&mut self` while `prof` is still borrowed; the cost
        // shows up in the next step's `WMS` reading instead.
        let edit_start = std::time::Instant::now();
        for c in &dirty_chunks {
            Self::apply_edit_inline(
                &mut self.world,
                &mut self.renderer,
                &self.registry,
                *c,
            );
        }
        if let Some(p) = self.profiler.as_ref() {
            p.record(
                "edit_inline",
                edit_start.elapsed().as_micros().min(u32::MAX as u128) as u32,
            );
        }
        time(prof, "world_stream", || {
            crate::ecs::systems::world_stream::world_stream(
                &self.ecs,
                &mut self.world,
                &self.jobs,
                &self.generator,
                &self.registry,
                &self.persistence,
                &self.saves_dir,
            )
        });
        time(prof, "drain_jobs", || {
            crate::ecs::systems::mesh_upload::drain_jobs(
                &mut self.world,
                &self.jobs,
                &mut self.renderer,
                &self.registry,
            )
        });
        time(prof, "drain_persistence", || {
            crate::ecs::systems::mesh_upload::drain_persistence(
                &mut self.world,
                &self.jobs,
                &self.persistence,
                &self.generator,
                &self.registry,
            )
        });
        // Relight pump runs after the two job-drain stages so it picks
        // up the `dirty.light` flags those handlers just set on newly
        // loaded/generated chunks. Each frame queues a bounded number
        // of relight jobs; over a few seconds the world converges to
        // a fixed lighting state with correct cross-chunk propagation.
        // The return value is the *total* (not just dispatched) count
        // of `dirty.light` chunks, which the HUD prints so we can see
        // whether the cascade is terminating.
        self.perf.light_queue = time(prof, "relight_pump", || {
            crate::ecs::systems::mesh_upload::relight_pump(
                &mut self.world,
                &self.jobs,
                &self.registry,
            )
        }) as u32 as _;
        self.perf.chunks_rendered = self.renderer.chunk_mesh_count();
        self.perf.draw_calls = self.renderer.last_draw_calls();
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

        let now_secs = self.start_time.elapsed().as_secs_f32();
        let fps = self.fps_meter.fps();
        time(prof, "render", || {
            if let Err(e) = crate::ecs::systems::render::render(
                &self.ecs,
                &mut self.renderer,
                &self.registry,
                fps,
                now_secs,
                &self.perf,
            ) {
                log::warn!("render error: {e:?}");
            }
        });
        self.input_buf.clear_per_frame();
        // Sample full per-step time AFTER the render call so WMS
        // captures GPU command encoding + the present (or whatever
        // wgpu blocks on under Immediate present mode). The HUD on
        // the *next* step reads this — 1 frame stale, which is
        // imperceptible for perf debugging.
        self.perf.work_ms = work_start.elapsed().as_secs_f32() * 1000.0;
        // Flush this frame's profiler row, if profiling enabled.
        if let Some(p) = &self.profiler {
            p.finish_frame(crate::profiler::FrameCounters {
                fps,
                work_ms: self.perf.work_ms,
                draw_calls: self.perf.draw_calls,
                light_queue: self.perf.light_queue as u32,
                chunks_rendered: self.perf.chunks_rendered as u32,
                edits: self.frame_edit_count,
            });
        }
        self.frame_edit_count = 0;
    }

    /// Run relight + LOD0 mesh + GPU upload for a single edit-dirtied
    /// chunk synchronously on the main thread. Called once per
    /// affected chunk in the same step that the edit happened. The
    /// total cost is roughly one chunk decompress + one BFS + one
    /// greedy mesh + one wgpu buffer upload — about 5-10 ms even on
    /// the worst case — which beats every alternative that puts the
    /// work on a worker pool already saturated with stream-in jobs.
    fn apply_edit_inline(
        world: &mut crate::voxel::world::World,
        renderer: &mut crate::render::Renderer,
        registry: &crate::voxel::block::BlockRegistry,
        coord: crate::voxel::coords::ChunkCoord,
    ) {
        use crate::voxel::chunk::{ChunkDirty, ChunkState, DenseChunk, Neighbors, PalettedChunk};
        use crate::voxel::world::ChunkSlot;

        let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get(&coord) else {
            return;
        };
        let needs_light = meta.dirty.light;
        let data_arc = data.clone();

        // Decompress chunk + neighbours up-front so the greedy mesher
        // and the BFS (if relighting) can share them.
        let mut dense = data_arc.decompress();
        let neighbor_arcs =
            crate::ecs::systems::mesh_upload::gather_neighbors(world, coord);
        let neighbor_dense: Vec<Option<DenseChunk>> = neighbor_arcs
            .iter()
            .map(|opt| opt.as_ref().map(|p| p.decompress()))
            .collect();
        let neighbor_refs: [Option<&DenseChunk>; 6] = [
            neighbor_dense[0].as_ref(),
            neighbor_dense[1].as_ref(),
            neighbor_dense[2].as_ref(),
            neighbor_dense[3].as_ref(),
            neighbor_dense[4].as_ref(),
            neighbor_dense[5].as_ref(),
        ];
        let ns = Neighbors {
            chunks: neighbor_refs,
        };

        // Relight the edited chunk in-place. Skipped when the edit
        // only dirtied a neighbour's mesh (no `dirty.light` set).
        if needs_light {
            crate::lighting::recompute_chunk(&mut dense, &ns, registry);
        }

        // Mesh from the (possibly relit) dense data.
        let mesh = crate::mesher::greedy::mesh_greedy(&dense, &neighbor_refs, registry);

        // Swap the relit data back into the chunk slot if we did
        // relight; bump the version so any in-flight cascade mesh
        // for this chunk gets discarded on completion.
        if needs_light {
            let new_data = std::sync::Arc::new(PalettedChunk::compress(&dense));
            if let Some(ChunkSlot::Stored {
                data: cur,
                meta,
            }) = world.chunks.get_mut(&coord)
            {
                *cur = new_data;
                meta.dirty = ChunkDirty {
                    mesh: true,
                    light: false,
                };
                meta.state = ChunkState::Generated;
                meta.mesh_version = meta.mesh_version.wrapping_add(1);
            }
        }

        // Upload the freshly-built mesh. No version check needed —
        // we just generated it from the current data on the main
        // thread, so by definition it's the latest.
        renderer.upload_chunk_mesh(coord, 0, &mesh);
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
