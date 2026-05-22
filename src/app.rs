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

use crate::ecs::GameEcs;
use crate::ecs::systems::input::InputBuf;
use crate::jobs::Jobs;
use crate::persistence::SaveIndex;
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
    /// File watcher for `assets/worldgen/default.ron`. Must be kept
    /// alive for the watcher thread to keep running; dropping it
    /// shuts the watcher down. The watcher swaps a new
    /// `WorldgenConfig` into the generator's `ConfigHolder` on every
    /// file change.
    pub _worldgen_watcher: Box<dyn std::any::Any + Send + Sync>,
    /// Block registry, shared with worker threads via `Arc`.
    pub registry: Arc<BlockRegistry>,
    /// Dedicated I/O thread for chunk save/load.
    pub persistence: Persistence,
    /// In-memory mirror of which region slots have data on disk. The
    /// streaming system consults this on every chunk it considers
    /// requesting from disk; without it, every empty sibling slot in
    /// a region file the player has touched would round-trip the
    /// single-threaded persistence worker for a `NotPresent`
    /// response. See [`crate::persistence::SaveIndex`] for the full
    /// rationale.
    pub save_index: SaveIndex,
    /// Root directory for region files this session writes to.
    pub saves_dir: PathBuf,
    /// Wall-clock time of the previous autosave tick. Autosave runs every
    /// `AUTOSAVE_INTERVAL` seconds.
    pub last_autosave: Instant,
    pub input_buf: InputBuf,
    pub input_state: crate::ecs::systems::input::InputState,
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
    /// UI state machine: pause menu, chat log, command dispatcher.
    /// When `ui.is_playing()` is false, `step` skips the game schedule
    /// (full freeze); the renderer still draws the last frame plus the
    /// UI overlay so the menu/chat is visible.
    pub ui: crate::ui::Ui,
    /// Per-frame `world_stream` scratch: the sorted candidate-chunk
    /// list, cached across frames and only rebuilt when the player
    /// crosses a chunk boundary. See
    /// [`crate::ecs::systems::world_stream::WorldStreamCache`].
    pub world_stream_cache: crate::ecs::systems::world_stream::WorldStreamCache,
    /// When true, the opaque shader renders all geometry at full
    /// brightness (skips the direct+block+sky-ambient composition).
    /// Toggled by pressing B in-game. Default: false.
    pub fullbright: bool,
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
    /// Chunks currently in `ChunkSlot::Stored` state — i.e., gen
    /// completed and world.data is populated. Compare against
    /// `chunks_rendered`: if `chunks_loaded` is much larger, mesh
    /// jobs haven't completed yet; if they're roughly equal, the
    /// world really is fully streamed and any voids are something
    /// else (cull, frustum, …).
    pub chunks_loaded: usize,
    /// Chunks currently in `ChunkSlot::Pending` — gen spawned but
    /// hasn't returned. If this stays high, the worker pool is
    /// stuck (panic, deadlock, or saturated).
    pub chunks_pending: usize,
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
    /// Graph-engine op queue depth entering this frame (pre-tick snapshot).
    /// Non-zero during initial stream-in; approaches zero as light converges.
    /// Always 0 under `legacy-lighting` (use `light_queue` there instead).
    pub light_ops_pending: usize,
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
        seed_override: Option<u64>,
    ) -> Self {
        // One save folder per binary; could grow into a world-picker UI.
        let saves_dir = PathBuf::from("saves/default");
        std::fs::create_dir_all(&saves_dir).ok();

        // Load or initialise the world manifest. New worlds get a
        // freshly-rolled seed; legacy saves get the synthetic
        // legacy fallback so already-explored terrain stays in its
        // original frame.
        let manifest = crate::persistence::manifest::load_or_init(&saves_dir)
            .expect("failed to load or initialise world manifest");
        // TEMP for worldgen testing: force a known seed so every
        // launch shows the same terrain. Drop this override when the
        // generator is locked in and we want fresh worlds again.
        const TEST_SEED_OVERRIDE: Option<u64> = Some(42);
        let seed = seed_override
            .or(TEST_SEED_OVERRIDE)
            .unwrap_or(manifest.seed);

        if seed != manifest.seed {
            log::warn!(
                "seed mismatch (saved={}, effective={}) — discarding save at {}",
                manifest.seed,
                seed,
                saves_dir.display()
            );
            let regions_dir = saves_dir.join("regions");
            if regions_dir.exists() {
                std::fs::remove_dir_all(&regions_dir)
                    .expect("failed to clear regions on seed change");
            }
            let mut fresh = crate::persistence::manifest::WorldManifest::fresh();
            fresh.seed = seed;
            crate::persistence::manifest::save(&saves_dir, &fresh)
                .expect("failed to write manifest on seed change");
        }

        log::info!(
            "world manifest loaded: seed={} (manifest seed={}, version={})",
            seed,
            manifest.seed,
            manifest.worldgen_version,
        );

        let present_mode = if uncapped {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };
        let renderer = Renderer::new_with_present_mode(window.clone(), present_mode);
        let ecs = GameEcs::new(spawn);
        let world = World::new(seed);
        let jobs = Jobs::new();
        // Build a hot-reloadable WorldgenConfig and spawn a file
        // watcher on assets/worldgen/default.ron. The watcher
        // atomically swaps in a new config on every change; newly
        // generated chunks reflect it. Existing chunks keep their
        // original generation (full chunk-cache invalidation is a
        // future-PR concern).
        let cfg = crate::worldgen::config::WorldgenConfig::bundled_default()
            .expect("bundled default.ron must parse");
        let holder = crate::worldgen::config::ConfigHolder::new(cfg);
        let watcher_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("worldgen")
            .join("default.ron");
        let watcher = crate::worldgen::config::spawn_watcher(watcher_path, holder.clone())
            .expect("file watcher must start");
        let generator = Arc::new(Generator::with_config(seed, holder));
        let registry = Arc::new(BlockRegistry::new());

        let persistence = Persistence::spawn(saves_dir.clone());

        // Surface the world seed in chat so the player can copy it
        // out of the in-game log (alongside the `world manifest
        // loaded: seed=…` line in stdout/log file).
        let mut ui = crate::ui::Ui::new();
        ui.log.push_system(format!("World seed: {seed}"));

        Self {
            window,
            renderer,
            ecs,
            world,
            jobs,
            generator,
            _worldgen_watcher: Box::new(watcher),
            registry,
            persistence,
            save_index: SaveIndex::new(),
            saves_dir,
            last_autosave: Instant::now(),
            input_buf: InputBuf::default(),
            input_state: crate::ecs::systems::input::InputState::default(),
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
            ui,
            world_stream_cache: crate::ecs::systems::world_stream::WorldStreamCache::default(),
            fullbright: false,
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
        use crate::profiler::time;

        self.ui.tick(dt);

        if self.ui.is_playing() {
            let prof = self.profiler.as_ref();

            time(prof, "input", || {
                crate::ecs::systems::input::apply_input(
                    &mut self.ecs,
                    &self.input_buf,
                    &mut self.input_state,
                )
            });
            // B toggles fullbright: all opaque geometry renders at full
            // brightness, skipping the lighting composition. Useful for
            // cave spelunking where dim block-light obscures structure.
            if self
                .input_buf
                .key_pressed_this_frame
                .contains(&winit::keyboard::KeyCode::KeyB)
            {
                self.fullbright = !self.fullbright;
            }
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
                Self::apply_edit_inline(&mut self.world, &mut self.renderer, &self.registry, *c);
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
                    &mut self.save_index,
                    &self.saves_dir,
                    &mut self.world_stream_cache,
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
            let light_pending = self.world.light_engine.pending_ops_count();
            let light_budget = if self.perf.chunks_pending > 100 || light_pending > 10_000 {
                500_000
            } else {
                50_000
            };
            self.perf.light_ops_pending = light_pending;
            time(prof, "light_engine_tick", || {
                self.world.light_engine_tick(light_budget);
            });
            time(prof, "upload_dirty_light_volumes", || {
                crate::ecs::systems::mesh_upload::upload_dirty_light_volumes(
                    &mut self.world,
                    &mut self.renderer,
                );
            });
            // Relight pump runs after the two job-drain stages so it picks
            // up the `dirty.light` flags those handlers just set on newly
            // loaded/generated chunks. Each frame queues a bounded number
            // of relight jobs; over a few seconds the world converges to
            // a fixed lighting state with correct cross-chunk propagation.
            // The return value is the *total* (not just dispatched) count
            // of `dirty.light` chunks, which the HUD prints so we can see
            // whether the cascade is terminating.
            #[cfg(feature = "legacy-lighting")]
            {
                self.perf.light_queue = time(prof, "relight_pump", || {
                    crate::ecs::systems::mesh_upload::relight_pump(
                        &mut self.world,
                        &self.jobs,
                        &self.registry,
                    )
                }) as u32 as _;
            }
            self.perf.chunks_rendered = self.renderer.chunk_mesh_count();
            self.perf.draw_calls = self.renderer.last_draw_calls();
            // Walk the world chunks once to count Stored vs Pending so
            // the HUD can show whether the missing chunks are simply
            // un-generated yet vs generated-but-not-meshed.
            let mut stored = 0usize;
            let mut pending = 0usize;
            for slot in self.world.chunks.values() {
                match slot {
                    crate::voxel::world::ChunkSlot::Stored { .. } => stored += 1,
                    crate::voxel::world::ChunkSlot::Pending => pending += 1,
                }
            }
            self.perf.chunks_loaded = stored;
            self.perf.chunks_pending = pending;
            crate::ecs::systems::world_stream::world_unload(
                &self.ecs,
                &mut self.world,
                &mut self.renderer,
                &self.persistence,
                &mut self.save_index,
            );

            // Autosave: every AUTOSAVE_INTERVAL, push every modified chunk
            // through to the persistence thread. Cheap if no chunks are
            // modified.
            //
            // `flush_modified` takes `&mut self` because it stamps
            // `save_index`, which conflicts with the immutable borrow
            // of `self.profiler` that the `prof` binding above holds.
            // We rebind `prof` afterwards so the render pass below can
            // still time itself; NLL narrows the original binding's
            // scope to end at the rebind.
            if self.last_autosave.elapsed() >= AUTOSAVE_INTERVAL {
                self.last_autosave = Instant::now();
                self.flush_modified();
            }
        }

        // Drain UI effects every frame (so menu actions work while paused).
        // Collect first so `apply_ui_effect`'s `&mut self` doesn't conflict
        // with the `prof` borrow that the render block below needs.
        let effects: Vec<_> = self.ui.drain_effects();
        for eff in effects {
            self.apply_ui_effect(eff);
        }

        let prof = self.profiler.as_ref();
        let now_secs = self.start_time.elapsed().as_secs_f32();
        let fps = self.fps_meter.fps();
        // Underwater detection: probe the block at the camera's eye
        // position. `get_block` returns `None` for unloaded chunks
        // (shouldn't happen at the player's own coord but be defensive)
        // so we conservatively assume air in that case. The shaders
        // own the smoothing; here we just hand them a 0/1 step value.
        {
            use crate::ecs::components::{Camera, Position};
            let mut q = self
                .ecs
                .world
                .query_one::<(&Position, &Camera)>(self.ecs.player)
                .unwrap();
            let (pos, cam) = q.get().unwrap();
            let eye = pos.0 + cam.eye_offset;
            let block_pos = crate::voxel::coords::BlockPos(glam::IVec3::new(
                eye.x.floor() as i32,
                eye.y.floor() as i32,
                eye.z.floor() as i32,
            ));
            let in_water = matches!(
                self.world.get_block(block_pos),
                Some(crate::voxel::block::Block::Water)
            );
            self.renderer
                .set_underwater(if in_water { 1.0 } else { 0.0 });
        }
        time(prof, "render", || {
            if let Err(e) = crate::ecs::systems::render::render(
                &self.ecs,
                &mut self.renderer,
                &self.registry,
                fps,
                now_secs,
                &self.perf,
                &self.ui,
                &self.generator,
                &self.world,
                self.fullbright,
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
                chunks_loaded: self.perf.chunks_loaded as u32,
                chunks_pending: self.perf.chunks_pending as u32,
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
        #[cfg(feature = "legacy-lighting")]
        use crate::voxel::chunk::{ChunkDirty, ChunkState, PalettedChunk};
        use crate::voxel::chunk::{DenseChunk, Neighbors};
        use crate::voxel::world::ChunkSlot;

        let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get(&coord) else {
            return;
        };
        #[cfg(feature = "legacy-lighting")]
        let needs_light = meta.dirty.light;
        #[cfg(not(feature = "legacy-lighting"))]
        let _ = meta;
        let data_arc = data.clone();

        // Decompress chunk + neighbours up-front so the greedy mesher
        // and the BFS (if relighting) can share them.
        let mut dense = data_arc.decompress();
        let neighbor_arcs = crate::ecs::systems::mesh_upload::gather_neighbors(world, coord);
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
        // In the default (engine) build, the graph engine handles
        // light propagation — the inline path just re-meshes.
        #[cfg(feature = "legacy-lighting")]
        if needs_light {
            crate::lighting::recompute_chunk(&mut dense, &ns, registry);
        }

        // Mesh from the (possibly relit) dense data.
        let mesh = crate::mesher::greedy::mesh_greedy(&dense, &neighbor_refs, registry);

        // Swap the relit data back into the chunk slot if we did
        // relight; bump the version so any in-flight cascade mesh
        // for this chunk gets discarded on completion.
        #[cfg(feature = "legacy-lighting")]
        if needs_light {
            let new_data = std::sync::Arc::new(PalettedChunk::compress(&dense));
            if let Some(ChunkSlot::Stored { data: cur, meta }) = world.chunks.get_mut(&coord) {
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
        // Upload the light volume too — without this the GPU keeps
        // the pre-edit values and broken/placed blocks render against
        // stale lighting (pits stay pitch-black, torches don't bleed
        // their RGB into neighbours). Cheap (one wgpu queue write of
        // ~144 KB) and the path is hot enough that the cost is fine.
        let blob = crate::voxel::chunk::build_light_volume_blob(&dense, &ns);
        renderer.upload_chunk_light_volume(coord, blob.as_ref());
    }

    /// Apply one UI-emitted intent. Each variant maps to a small piece
    /// of game-state mutation (or a process-level action like Quit).
    /// Kept small so adding a command is one match arm here plus one
    /// variant on `UiEffect`.
    fn apply_ui_effect(&mut self, eff: crate::ui::effect::UiEffect) {
        use crate::ui::effect::UiEffect;
        match eff {
            UiEffect::Quit => {
                self.ui.wants_quit = true;
            }
            UiEffect::Save => self.flush_modified(),
            UiEffect::Teleport(p) => {
                use crate::ecs::components::{Position, Velocity};
                if let Ok(mut q) = self
                    .ecs
                    .world
                    .query_one::<(&mut Position, &mut Velocity)>(self.ecs.player)
                {
                    if let Some((pos, vel)) = q.get() {
                        pos.0 = p;
                        vel.0 = glam::Vec3::ZERO;
                    }
                }
            }
            UiEffect::SetTime(t) => {
                let t = t.clamp(0.0, 1.0);
                for (_, tod) in self
                    .ecs
                    .world
                    .query::<&mut crate::ecs::components::TimeOfDay>()
                    .iter()
                {
                    tod.t = t;
                }
            }
            UiEffect::ToggleFly => {
                use crate::ecs::components::{Movement, MovementMode};
                if let Ok(mut q) = self.ecs.world.query_one::<&mut Movement>(self.ecs.player) {
                    if let Some(mv) = q.get() {
                        mv.mode = match mv.mode {
                            MovementMode::Walk => MovementMode::Fly,
                            MovementMode::Fly => MovementMode::Walk,
                        };
                    }
                }
            }
            UiEffect::ToggleNoclip => {
                use crate::ecs::components::{Movement, MovementMode};
                let mut new_state: Option<(bool, bool)> = None;
                if let Ok(mut q) = self.ecs.world.query_one::<&mut Movement>(self.ecs.player) {
                    if let Some(mv) = q.get() {
                        mv.noclip = !mv.noclip;
                        new_state = Some((mv.noclip, matches!(mv.mode, MovementMode::Fly)));
                    }
                }
                if let Some((on, in_fly)) = new_state {
                    let msg = if on {
                        if in_fly {
                            "noclip ON".to_string()
                        } else {
                            "noclip ON (engages once you /fly)".to_string()
                        }
                    } else {
                        "noclip OFF".to_string()
                    };
                    self.ui.log.push_system(msg);
                }
            }
            UiEffect::PostMessage(msg) => {
                self.ui.log.push_system(msg);
            }
            UiEffect::ClearChat => {
                self.ui.log.clear();
            }
        }
    }

    /// Send every currently-modified chunk through the persistence thread.
    /// Used by both autosave and the `Drop` flush-on-close path.
    ///
    /// Also stamps each saved coord into `save_index` so the next
    /// `world_stream` pass that considers the chunk routes it through
    /// the persistence thread instead of (incorrectly) regenerating
    /// it from seed.
    fn flush_modified(&mut self) {
        for (c, slot) in &self.world.chunks {
            if let ChunkSlot::Stored { data, meta } = slot
                && meta.modified
            {
                let _ = self.persistence.req_tx.send(PersistRequest::Save {
                    coord: *c,
                    data: data.clone(),
                });
                self.save_index.mark(*c);
            }
        }
    }
}

impl Drop for AppState {
    /// On a clean shutdown, push any modified chunks out to disk. The
    /// `persistence` field's own `Drop` (see
    /// [`crate::persistence::thread::Persistence`]) sends `Shutdown`
    /// and joins the I/O thread once this method returns and field
    /// teardown reaches it — that join is what guarantees every
    /// queued save reaches disk before the process exits.
    fn drop(&mut self) {
        self.flush_modified();
    }
}
