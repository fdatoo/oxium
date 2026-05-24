//! Top-level App: holds renderer, world, ECS, jobs, time.
//!
//! `AppState` is the single object that lives across frames. It owns the
//! renderer, the voxel world, the ECS, the background job pool, and the
//! shared generator + block registry. The per-frame `step` runs the
//! complete schedule:
//!
//! ```text
//! input  →  ui  →  movement  →  world_stream  →  drain_jobs  →  world_unload  →  render
//!        →  clear_input_frame
//! ```
//!
//! Each system is a free function in `ecs::systems`; the ordering here is
//! the *only* place we declare it — easy to debug and reason about.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::window::Window;
use wgpu;

use crate::audio::{AmbientProbe, AudioEngine, AudioEvent};
use crate::ecs::GameEcs;
use crate::jobs::Jobs;
use crate::persistence::SaveIndex;
use crate::persistence::thread::{PersistRequest, Persistence};
use crate::render::Renderer;
use crate::voxel::block::{Block, BlockRegistry};
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
    pub input: crate::input_engine::InputEngine,
    pub input_state: crate::ecs::systems::input::InputState,
    /// File watcher for `assets/input/default.ron`. Kept alive for hot reload.
    pub _input_watcher: Box<dyn std::any::Any + Send + Sync>,
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
    /// Paused UI skips the game schedule; chat keeps it running while
    /// routing keyboard input to the text editor.
    pub ui: crate::ui::Ui,
    pub clipboard: crate::ui::Clipboard,
    /// Runtime audio backend and decoded/static sound cache. This is
    /// binary-owned because it holds OS audio device state.
    pub audio: AudioEngine,
    audio_events: Vec<AudioEvent>,
    /// Per-frame `world_stream` scratch: the sorted candidate-chunk
    /// list, cached across frames and only rebuilt when the player
    /// crosses a chunk boundary. See
    /// [`crate::ecs::systems::world_stream::WorldStreamCache`].
    pub world_stream_cache: crate::ecs::systems::world_stream::WorldStreamCache,
    /// When true, the opaque shader renders all geometry at full
    /// brightness (skips the direct+block+sky-ambient composition).
    /// Toggled by the debug `ToggleFullbright` action. Default: false.
    pub fullbright: bool,
}

/// Per-frame counters shown in the debug HUD. Cheap to keep around;
/// the values are recomputed every step from whatever the systems
/// hand back, so there's no stale-data hazard.
#[derive(Debug, Default, Clone, Copy)]
pub struct PerfSnapshot {
    /// Chunks waiting for the relight pump. Non-zero means lighting is
    /// still converging.
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
    /// Relight backlog mirrored into profiler fields that historically
    /// reported lighting work depth.
    pub light_ops_pending: usize,
}

/// How often the autosave system flushes modified chunks to disk.
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);
const LANDING_SOUND_MIN_FALL_SPEED: f32 = 0.75;

#[derive(Debug, Clone, Copy)]
struct PlayerAudioState {
    pos: glam::Vec3,
    vel: glam::Vec3,
    half: glam::Vec3,
    bob_phase: f32,
    grounded: bool,
    walking: bool,
}

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
        let (input_holder, input_path) = crate::input_engine::load_default_holder()
            .expect("bundled input default.ron must parse");
        let input_watcher = crate::input_engine::spawn_watcher(input_path, input_holder.clone())
            .expect("input file watcher must start");
        let registry = Arc::new(BlockRegistry::new());

        let persistence = Persistence::spawn(saves_dir.clone());

        // Surface the world seed in chat so the player can copy it
        // out of the in-game log (alongside the `world manifest
        // loaded: seed=…` line in stdout/log file).
        let mut ui = crate::ui::Ui::new();
        ui.log.push_system(format!("World seed: {seed}"));
        let audio =
            AudioEngine::from_default_config().expect("bundled audio default.ron must parse");

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
            input: crate::input_engine::InputEngine::new(input_holder),
            input_state: crate::ecs::systems::input::InputState::default(),
            _input_watcher: Box::new(input_watcher),
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
            clipboard: crate::ui::Clipboard::new(),
            audio,
            audio_events: Vec::with_capacity(16),
            world_stream_cache: crate::ecs::systems::world_stream::WorldStreamCache::default(),
            fullbright: false,
        }
    }

    /// Run one frame with a caller-supplied timestep and optional shader-time
    /// override.
    ///
    /// # Parameters
    ///
    /// - `dt`: physics timestep in seconds (capped to 0.1 by the caller for
    ///   wall-clock mode; passed raw for sim mode).
    /// - `sim_time_override`: when `Some(t)`, the renderer's shader `time`
    ///   uniform (used by water shimmer, sun movement, etc.) is driven by `t`
    ///   instead of `self.start_time.elapsed()`. Pass `None` from the
    ///   [`step`] wrapper so the wall-clock path computes shader time at the
    ///   same point in the frame as before, preserving byte-identical PNG
    ///   output vs. the un-refactored version.
    ///
    /// The probe's sim-burst mode calls this directly with
    /// `Some(sim_time)` so every burst run is fully reproducible: same seed,
    /// same `sim_dt`, same frame index → identical pixels.
    pub fn step_with_dt(&mut self, dt: f32, sim_time_override: Option<f32>) {
        self.fps_meter.record(dt);
        // Start of the per-step CPU work; stopped just before the
        // render `present()` call so the measurement excludes
        // wall-clock time we spend waiting on vsync.
        let work_start = Instant::now();
        use crate::profiler::time;

        self.ui.tick(dt);
        let input_mode = self.ui.input_mode();
        let was_playing = self.ui.is_playing();
        let actions = self.input.resolve(input_mode);
        self.ui.apply_actions(&actions);
        if was_playing != self.ui.is_playing() {
            self.input.clear_all();
        }

        if self.ui.should_step_game() {
            let prof = self.profiler.as_ref();

            time(prof, "input", || {
                crate::ecs::systems::input::apply_input(
                    &mut self.ecs,
                    &actions,
                    &mut self.input_state,
                )
            });
            // B toggles fullbright: all opaque geometry renders at full
            // brightness, skipping the lighting composition. Useful for
            // cave spelunking where dim block-light obscures structure.
            if actions.pressed(crate::input_engine::InputAction::ToggleFullbright) {
                self.fullbright = !self.fullbright;
            }
            time(prof, "time_of_day", || {
                crate::ecs::systems::time_of_day::advance(&mut self.ecs, dt)
            });
            time(prof, "movement", || {
                crate::ecs::systems::movement::movement(&mut self.ecs, dt)
            });
            let player_audio_before = self.player_audio_state();
            time(prof, "physics", || {
                crate::ecs::systems::physics::physics(&mut self.ecs, &self.world, dt)
            });
            if let Some(before) = player_audio_before {
                let events = Self::player_audio_events(&self.ecs, &self.world, before);
                self.audio_events.extend(events);
            }

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
            if !dirty_chunks.is_empty() {
                let dirty_set: std::collections::HashSet<_> =
                    dirty_chunks.iter().copied().collect();
                for c in &dirty_chunks {
                    Self::apply_edit_inline(
                        &mut self.world,
                        &mut self.renderer,
                        &self.registry,
                        *c,
                        Some(&dirty_set),
                        false,
                        false,
                    );
                }
                for c in &dirty_chunks {
                    Self::apply_edit_inline(
                        &mut self.world,
                        &mut self.renderer,
                        &self.registry,
                        *c,
                        None,
                        true,
                        false,
                    );
                }
                for c in &dirty_chunks {
                    Self::refresh_edit_upload_inline(
                        &mut self.world,
                        &mut self.renderer,
                        &self.registry,
                        *c,
                    );
                }
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
            // Relight pump runs after the two job-drain stages so it picks
            // up the lighting states those handlers just set on newly
            // loaded/generated chunks. Each frame queues a bounded number
            // of relight jobs; over a few seconds the world converges to
            // a fixed lighting state with correct cross-chunk propagation.
            let relight_priority = {
                let mut q = self
                    .ecs
                    .world
                    .query_one::<&crate::ecs::components::Position>(self.ecs.player)
                    .unwrap();
                q.get()
                    .map(|pos| crate::ecs::systems::world_stream::player_chunk(pos.0))
            };
            self.perf.light_queue = time(prof, "relight_pump", || {
                crate::ecs::systems::mesh_upload::relight_pump(
                    &mut self.world,
                    &self.jobs,
                    &self.registry,
                    relight_priority,
                )
            });
            self.perf.light_ops_pending = self.perf.light_queue;
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
        let time_of_day = self.time_of_day();
        self.audio_events.push(AudioEvent::MusicState {
            time_of_day,
            paused: !self.ui.is_playing(),
        });
        if let Some(probe) = self.ambient_probe(time_of_day) {
            self.audio_events.push(AudioEvent::AmbientProbe(probe));
        }
        if !self.ui.is_playing() {
            self.audio_events.push(AudioEvent::FootstepState {
                pos: glam::Vec3::ZERO,
                block: None,
                horizontal_speed: 0.0,
                bob_phase: 0.0,
                walking: false,
            });
        }
        self.audio.drain_events(self.audio_events.drain(..));

        let prof = self.profiler.as_ref();
        // Shader time: use the caller-supplied value when running under a
        // deterministic sim clock; otherwise read wall-clock elapsed time at
        // the same point in the frame as the original code (preserves
        // byte-identical output on the normal path).
        let now_secs = sim_time_override.unwrap_or_else(|| self.start_time.elapsed().as_secs_f32());
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
                self.input.debug_enabled(),
            ) {
                log::warn!("render error: {e:?}");
            }
        });
        self.input.clear_frame();
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
                light_ops: self.perf.light_ops_pending as u32,
                chunks_rendered: self.perf.chunks_rendered as u32,
                chunks_loaded: self.perf.chunks_loaded as u32,
                chunks_pending: self.perf.chunks_pending as u32,
                edits: self.frame_edit_count,
            });
        }
        self.frame_edit_count = 0;
    }

    /// Run one frame driven by real wall-clock time.
    ///
    /// Computes `dt` from the gap since the last call, caps it at 100 ms to
    /// prevent large physics leaps after a pause or background stall, then
    /// delegates to [`step_with_dt`] with `sim_time_override = None` so the
    /// shader time reads from `self.start_time` at the same place as the
    /// original single-method code did — preserving byte-identical PNG output.
    ///
    /// This is the path used by the normal game window and wall-clock burst
    /// capture. The probe's sim-burst mode bypasses this and calls
    /// [`step_with_dt`] directly.
    pub fn step(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
        self.last_tick = now;
        self.step_with_dt(dt, None);
    }

    /// Run relight for a single edit-dirtied chunk synchronously on
    /// the main thread, optionally followed by LOD0 mesh + GPU upload.
    /// Edits use two relight passes over the dirty neighbourhood: first
    /// without importing light from other dirty chunks to clear removed
    /// sources, then with the freshly recomputed neighbours to restore
    /// valid cross-border propagation.
    fn apply_edit_inline(
        world: &mut crate::voxel::world::World,
        renderer: &mut crate::render::Renderer,
        registry: &crate::voxel::block::BlockRegistry,
        coord: crate::voxel::coords::ChunkCoord,
        excluded_light_neighbors: Option<
            &std::collections::HashSet<crate::voxel::coords::ChunkCoord>,
        >,
        force_light: bool,
        upload_now: bool,
    ) {
        use crate::voxel::chunk::{ChunkDirty, ChunkState, PalettedChunk};
        use crate::voxel::chunk::{DenseChunk, Neighbors};
        use crate::voxel::world::ChunkSlot;

        let Some(ChunkSlot::Stored { data, meta }) = world.chunks.get(&coord) else {
            return;
        };
        let needs_light = force_light || meta.dirty.light;
        let light_inputs = meta.light_inputs.clone();
        let data_arc = data.clone();

        // Decompress chunk + neighbours up-front so the greedy mesher
        // and the BFS (if relighting) can share them.
        let mut dense = data_arc.decompress();
        let neighbor_arcs = crate::ecs::systems::mesh_upload::gather_neighbors(world, coord);
        let neighbor_dense: Vec<Option<DenseChunk>> = neighbor_arcs
            .iter()
            .map(|opt| opt.as_ref().map(|p| p.decompress()))
            .collect();
        let mesh_neighbor_refs: [Option<&DenseChunk>; 6] = [
            neighbor_dense[0].as_ref(),
            neighbor_dense[1].as_ref(),
            neighbor_dense[2].as_ref(),
            neighbor_dense[3].as_ref(),
            neighbor_dense[4].as_ref(),
            neighbor_dense[5].as_ref(),
        ];
        let neighbor_coords = crate::ecs::systems::mesh_upload::neighbor_coords(coord);
        let neighbor_lit: [bool; 6] = std::array::from_fn(|i| {
            if excluded_light_neighbors.is_some_and(|set| set.contains(&neighbor_coords[i])) {
                return false;
            }
            crate::ecs::systems::mesh_upload::chunk_light_usable(world, neighbor_coords[i])
        });
        let light_neighbor_refs: [Option<&DenseChunk>; 6] = [
            neighbor_lit[0]
                .then_some(())
                .and(neighbor_dense[0].as_ref()),
            neighbor_lit[1]
                .then_some(())
                .and(neighbor_dense[1].as_ref()),
            neighbor_lit[2]
                .then_some(())
                .and(neighbor_dense[2].as_ref()),
            neighbor_lit[3]
                .then_some(())
                .and(neighbor_dense[3].as_ref()),
            neighbor_lit[4]
                .then_some(())
                .and(neighbor_dense[4].as_ref()),
            neighbor_lit[5]
                .then_some(())
                .and(neighbor_dense[5].as_ref()),
        ];
        let light_ns = Neighbors {
            chunks: light_neighbor_refs,
        };

        // Relight the edited chunk in-place. Skipped when the edit
        // only dirtied a neighbour's mesh (no `dirty.light` set).
        if needs_light {
            crate::lighting::recompute_chunk_with_inputs(
                &mut dense,
                &light_ns,
                &light_inputs,
                registry,
            );
        }

        // Swap the relit data back into the chunk slot if we did
        // relight; bump the version so any in-flight cascade mesh
        // for this chunk gets discarded on completion.
        if needs_light {
            let new_data = std::sync::Arc::new(PalettedChunk::compress(&dense));
            if let Some(ChunkSlot::Stored { data: cur, meta }) = world.chunks.get_mut(&coord) {
                *cur = new_data;
                meta.dirty = ChunkDirty {
                    mesh: true,
                    light: false,
                };
                meta.state = ChunkState::Generated;
                meta.light_state = crate::voxel::chunk::LightState::Lit {
                    version: meta.light_version,
                };
                meta.unresolved_borders = crate::voxel::chunk::FaceMask::NONE;
                meta.mesh_version = meta.mesh_version.wrapping_add(1);
            }
        }

        if !upload_now {
            return;
        }

        // Mesh from the (possibly relit) dense data.
        let mesh = crate::mesher::greedy::mesh_greedy(&dense, &mesh_neighbor_refs, registry);
        // Upload the light volume before the mesh so the new chunk bind
        // group never has to point at the renderer placeholder.
        let blob = crate::voxel::chunk::build_light_volume_blob(&dense, &light_ns);
        renderer.upload_chunk_light_volume(coord, blob.as_ref());

        // Upload the freshly-built mesh. No version check needed —
        // we just generated it from the current data on the main
        // thread, so by definition it's the latest.
        renderer.upload_chunk_mesh(coord, 0, &mesh);
        if let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get_mut(&coord) {
            meta.state = ChunkState::Ready;
            meta.dirty.mesh = false;
        }
    }

    fn refresh_edit_upload_inline(
        world: &mut crate::voxel::world::World,
        renderer: &mut crate::render::Renderer,
        registry: &crate::voxel::block::BlockRegistry,
        coord: crate::voxel::coords::ChunkCoord,
    ) {
        use crate::voxel::chunk::{DenseChunk, Neighbors};
        use crate::voxel::world::ChunkSlot;

        let Some(ChunkSlot::Stored { data, .. }) = world.chunks.get(&coord) else {
            return;
        };
        let dense = data.decompress();
        let neighbor_arcs = crate::ecs::systems::mesh_upload::gather_neighbors(world, coord);
        let neighbor_dense: Vec<Option<DenseChunk>> = neighbor_arcs
            .iter()
            .map(|opt| opt.as_ref().map(|p| p.decompress()))
            .collect();
        let mesh_neighbor_refs: [Option<&DenseChunk>; 6] = [
            neighbor_dense[0].as_ref(),
            neighbor_dense[1].as_ref(),
            neighbor_dense[2].as_ref(),
            neighbor_dense[3].as_ref(),
            neighbor_dense[4].as_ref(),
            neighbor_dense[5].as_ref(),
        ];
        let neighbor_coords = crate::ecs::systems::mesh_upload::neighbor_coords(coord);
        let neighbor_lit: [bool; 6] = std::array::from_fn(|i| {
            crate::ecs::systems::mesh_upload::chunk_light_usable(world, neighbor_coords[i])
        });
        let light_neighbor_refs: [Option<&DenseChunk>; 6] = [
            neighbor_lit[0]
                .then_some(())
                .and(neighbor_dense[0].as_ref()),
            neighbor_lit[1]
                .then_some(())
                .and(neighbor_dense[1].as_ref()),
            neighbor_lit[2]
                .then_some(())
                .and(neighbor_dense[2].as_ref()),
            neighbor_lit[3]
                .then_some(())
                .and(neighbor_dense[3].as_ref()),
            neighbor_lit[4]
                .then_some(())
                .and(neighbor_dense[4].as_ref()),
            neighbor_lit[5]
                .then_some(())
                .and(neighbor_dense[5].as_ref()),
        ];
        let light_ns = Neighbors {
            chunks: light_neighbor_refs,
        };

        let blob = crate::voxel::chunk::build_light_volume_blob(&dense, &light_ns);
        renderer.upload_chunk_light_volume(coord, blob.as_ref());
        let mesh = crate::mesher::greedy::mesh_greedy(&dense, &mesh_neighbor_refs, registry);
        renderer.upload_chunk_mesh(coord, 0, &mesh);
        if let Some(ChunkSlot::Stored { meta, .. }) = world.chunks.get_mut(&coord) {
            meta.state = crate::voxel::chunk::ChunkState::Ready;
            meta.dirty.mesh = false;
            meta.mesh_version = meta.mesh_version.wrapping_add(1);
        }
    }

    fn player_audio_state(&self) -> Option<PlayerAudioState> {
        use crate::ecs::components::{
            Aabb, Camera, Grounded, Movement, MovementMode, Position, Velocity,
        };
        let mut q = self
            .ecs
            .world
            .query_one::<(&Position, &Velocity, &Aabb, &Camera, &Grounded, &Movement)>(
                self.ecs.player,
            )
            .ok()?;
        let (pos, vel, aabb, camera, grounded, movement) = q.get()?;
        Some(PlayerAudioState {
            pos: pos.0,
            vel: vel.0,
            half: aabb.half,
            bob_phase: camera.bob_phase,
            grounded: grounded.0,
            walking: matches!(movement.mode, MovementMode::Walk),
        })
    }

    fn player_audio_events(
        ecs: &GameEcs,
        world: &World,
        before: PlayerAudioState,
    ) -> Vec<AudioEvent> {
        let Some(after) = Self::player_audio_state_from(ecs) else {
            return Vec::new();
        };
        let mut events = Vec::with_capacity(3);
        if before.walking && before.grounded && !after.grounded && after.vel.y > 0.0 {
            events.push(AudioEvent::Jump);
        }
        let surface_block = block_under_feet(world, after.pos, after.half);
        if !before.grounded && after.grounded && before.vel.y < -LANDING_SOUND_MIN_FALL_SPEED {
            events.push(AudioEvent::Land {
                impact: before.vel.y.abs(),
                block: surface_block,
            });
        }

        let horizontal_speed = glam::Vec3::new(after.vel.x, 0.0, after.vel.z).length();
        let walking = after.walking && after.grounded && horizontal_speed > 0.5;
        events.push(AudioEvent::FootstepState {
            pos: after.pos,
            block: walking.then_some(surface_block).flatten(),
            horizontal_speed,
            bob_phase: after.bob_phase,
            walking,
        });
        events
    }

    fn player_audio_state_from(ecs: &GameEcs) -> Option<PlayerAudioState> {
        use crate::ecs::components::{
            Aabb, Camera, Grounded, Movement, MovementMode, Position, Velocity,
        };
        let mut q = ecs
            .world
            .query_one::<(&Position, &Velocity, &Aabb, &Camera, &Grounded, &Movement)>(ecs.player)
            .ok()?;
        let (pos, vel, aabb, camera, grounded, movement) = q.get()?;
        Some(PlayerAudioState {
            pos: pos.0,
            vel: vel.0,
            half: aabb.half,
            bob_phase: camera.bob_phase,
            grounded: grounded.0,
            walking: matches!(movement.mode, MovementMode::Walk),
        })
    }

    fn time_of_day(&self) -> f32 {
        self.ecs
            .world
            .query::<&crate::ecs::components::TimeOfDay>()
            .iter()
            .next()
            .map(|(_, tod)| tod.t)
            .unwrap_or_default()
    }

    fn ambient_probe(&self, time_of_day: f32) -> Option<AmbientProbe> {
        let state = self.player_audio_state()?;
        let nearby_water = water_contact_factor(&self.world, state.pos, state.half);
        let nearby_lava = nearby_lava_factor(&self.world, state.pos);
        Some(AmbientProbe {
            listener_pos: state.pos,
            time_of_day,
            nearby_water,
            nearby_lava,
            undergroundness: undergroundness(state.pos),
        })
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
                let mut teleported = false;
                if let Ok(mut q) = self
                    .ecs
                    .world
                    .query_one::<(&mut Position, &mut Velocity)>(self.ecs.player)
                    && let Some((pos, vel)) = q.get()
                {
                    pos.0 = p;
                    vel.0 = glam::Vec3::ZERO;
                    teleported = true;
                }
                if teleported {
                    self.ui
                        .log
                        .push_system(format!("Teleported to {:.1}, {:.1}, {:.1}.", p.x, p.y, p.z));
                }
            }
            UiEffect::SetTime(t) => {
                let t = t.clamp(0.0, 1.0);
                let mut changed = false;
                for (_, tod) in self
                    .ecs
                    .world
                    .query::<&mut crate::ecs::components::TimeOfDay>()
                    .iter()
                {
                    tod.t = t;
                    changed = true;
                }
                if changed {
                    self.ui.log.push_system(format!("Time set to {t:.2}."));
                }
            }
            UiEffect::ToggleFly => {
                use crate::ecs::components::{Movement, MovementMode};
                let mut enabled = None;
                if let Ok(mut q) = self.ecs.world.query_one::<&mut Movement>(self.ecs.player)
                    && let Some(mv) = q.get()
                {
                    mv.mode = match mv.mode {
                        MovementMode::Walk => MovementMode::Fly,
                        MovementMode::Fly => MovementMode::Walk,
                    };
                    enabled = Some(matches!(mv.mode, MovementMode::Fly));
                }
                if let Some(on) = enabled {
                    self.ui
                        .log
                        .push_system(if on { "fly ON" } else { "fly OFF" });
                }
            }
            UiEffect::ToggleNoclip => {
                use crate::ecs::components::{Movement, MovementMode};
                let mut new_state: Option<(bool, bool)> = None;
                if let Ok(mut q) = self.ecs.world.query_one::<&mut Movement>(self.ecs.player)
                    && let Some(mv) = q.get()
                {
                    mv.noclip = !mv.noclip;
                    new_state = Some((mv.noclip, matches!(mv.mode, MovementMode::Fly)));
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
            UiEffect::PlayUiSound => {
                self.audio_events.push(AudioEvent::UiClick);
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

fn block_under_feet(world: &World, pos: glam::Vec3, half: glam::Vec3) -> Option<Block> {
    let y = (pos.y - 0.05).floor() as i32;
    let y_candidates = [y, y - 1];
    let x = (half.x - 0.02).max(0.0);
    let z = (half.z - 0.02).max(0.0);
    let samples = [
        (0.0, 0.0),
        (x, 0.0),
        (-x, 0.0),
        (0.0, z),
        (0.0, -z),
        (x, z),
        (x, -z),
        (-x, z),
        (-x, -z),
    ];
    let mut stone = None;
    for y in y_candidates {
        for (dx, dz) in samples {
            let block_pos = crate::voxel::coords::BlockPos(glam::IVec3::new(
                (pos.x + dx).floor() as i32,
                y,
                (pos.z + dz).floor() as i32,
            ));
            match world.get_block(block_pos) {
                Some(Block::Air) | None => {}
                Some(Block::Stone) => stone = Some(Block::Stone),
                Some(block) => return Some(block),
            }
        }
    }
    stone
}

fn water_contact_factor(world: &World, pos: glam::Vec3, half: glam::Vec3) -> f32 {
    let x = (half.x - 0.02).max(0.0);
    let z = (half.z - 0.02).max(0.0);
    let samples = [
        (0.0, 0.0),
        (x, 0.0),
        (-x, 0.0),
        (0.0, z),
        (0.0, -z),
        (x, z),
        (x, -z),
        (-x, z),
        (-x, -z),
    ];
    for y in [pos.y + 0.1, pos.y + 0.9] {
        for (dx, dz) in samples {
            let p = crate::voxel::coords::BlockPos(glam::IVec3::new(
                (pos.x + dx).floor() as i32,
                y.floor() as i32,
                (pos.z + dz).floor() as i32,
            ));
            if matches!(world.get_block(p), Some(Block::Water)) {
                return 1.0;
            }
        }
    }
    0.0
}

fn nearby_lava_factor(world: &World, pos: glam::Vec3) -> f32 {
    let base = crate::voxel::coords::BlockPos(glam::IVec3::new(
        pos.x.floor() as i32,
        pos.y.floor() as i32,
        pos.z.floor() as i32,
    ));
    let mut lava = 0.0_f32;
    const RADIUS: i32 = 6;
    for dy in -3..=3 {
        for dz in -RADIUS..=RADIUS {
            for dx in -RADIUS..=RADIUS {
                let dist2 = (dx * dx + dy * dy + dz * dz).max(1) as f32;
                let weight = 1.0 / dist2.sqrt();
                let p = crate::voxel::coords::BlockPos(base.0 + glam::IVec3::new(dx, dy, dz));
                if let Some(Block::Lava) = world.get_block(p) {
                    lava = lava.max(weight);
                }
            }
        }
    }
    lava.clamp(0.0, 1.0)
}

fn undergroundness(pos: glam::Vec3) -> f32 {
    ((72.0 - pos.y) / 48.0).clamp(0.0, 1.0)
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

// ── Capture helpers ───────────────────────────────────────────────────────────
//
// Shared between the main binary's `--screenshot-and-exit` path and the
// `oxium-probe capture` subcommand.

/// Pull `(eye, yaw, pitch)` from the player entity in the ECS.
///
/// Eye is in world space (position + camera eye offset).
/// Yaw and pitch are in radians, matching the convention used by
/// [`Renderer::render_to_view`].
pub fn camera_from_ecs(ecs: &crate::ecs::GameEcs) -> (glam::Vec3, f32, f32) {
    use crate::ecs::components::{Camera, Position};
    let mut q = ecs
        .world
        .query_one::<(&Position, &Camera)>(ecs.player)
        .unwrap();
    let (pos, cam) = q.get().unwrap();
    (pos.0 + cam.eye_offset, cam.yaw, cam.pitch)
}

/// Apply a look direction to the player's camera component in the ECS.
///
/// `yaw` and `pitch` are in radians. This is the programmatic counterpart
/// of mouse-look; use it from capture tools that need to point the camera at
/// a specific target without going through `InputEngine`.
pub fn set_camera_look(ecs: &mut crate::ecs::GameEcs, yaw: f32, pitch: f32) {
    use crate::ecs::components::Camera;
    for (_, cam) in ecs.world.query::<&mut Camera>().iter() {
        cam.yaw = yaw;
        cam.pitch = pitch.clamp(-1.553, 1.553);
    }
}

/// Render one frame to an offscreen texture matching the surface format
/// and save it as a PNG.
///
/// The shader animation clock (`time`) should be zeroed for deterministic
/// captures so cloud/water/caustic animation phases don't vary between runs.
#[allow(clippy::too_many_arguments)]
pub fn capture_offscreen(
    renderer: &crate::render::Renderer,
    path: &std::path::Path,
    eye: glam::Vec3,
    yaw: f32,
    pitch: f32,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    time: f32,
    ui: Option<&crate::ui::Ui>,
) -> anyhow::Result<()> {
    use crate::render::hud::build_hud;
    use crate::render::screenshot::capture_texture_to_png;

    let width = renderer.gpu.surface_cfg.width;
    let height = renderer.gpu.surface_cfg.height;
    let format = renderer.gpu.surface_cfg.format;

    let texture = renderer
        .gpu
        .device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let aspect = width as f32 / height.max(1) as f32;
    let registry = crate::voxel::block::BlockRegistry::new();
    let perf = PerfSnapshot::default();
    let mut hud = build_hud((width, height), 60.0, eye, 0, &registry, &perf, None);
    if let Some(ui) = ui {
        ui.draw_overlay((width, height), &mut hud);
    }
    renderer.render_to_view(
        &view,
        eye,
        yaw,
        pitch,
        aspect,
        sun_dir,
        sun_intensity,
        time,
        Some(&hud),
    );
    capture_texture_to_png(
        &renderer.gpu.device,
        &renderer.gpu.queue,
        &texture,
        format,
        width,
        height,
        path,
    )
}
