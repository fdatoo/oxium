//! `oxium-probe capture` subcommand implementation.
//!
//! Boots a hidden winit window, creates an `AppState`, runs the warmup +
//! quiesce loop (matching the `--screenshot-and-exit` path in `main.rs`),
//! then captures one or more PNGs with per-frame sidecar JSON files and an
//! aggregate `metrics.csv`.
//!
//! ## Single-shot mode (`--frames 1`, the default)
//!
//! `--out path/to/shot.png` is the PNG destination. A sidecar
//! `path/to/shot.png.json` and a `path/to/shot.png.metrics.csv` are written
//! alongside it.
//!
//! ## Burst mode (`--frames N` where N > 1)
//!
//! `--out path/to/burst_dir/` must be a directory path (it is created if
//! missing). Screenshots land as `burst_dir/0001.png`, `burst_dir/0002.png`,
//! …; sidecars as `burst_dir/0001.png.json`, …; aggregate metrics as
//! `burst_dir/metrics.csv`.
//!
//! `--interval-ms T` sets the minimum wall-clock gap between consecutive
//! captures; 0 (default) captures on every rendered frame.
//!
//! ## Feature search (`--find <kind>`)
//!
//! Runs a worldgen probe *before* the renderer starts (cheap, no GPU needed)
//! to locate a feature, then uses that position as the spawn. When the
//! feature is underground, the spawn is offset both vertically (+20 blocks)
//! *and* horizontally (+20 blocks on both X and Z axes) so `--look-at-feature`
//! produces an angled view instead of looking straight down.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use oxium::app::{AppState, camera_from_ecs, capture_offscreen, set_camera_look};
use oxium::ecs::systems::time_of_day::sun_state;
use oxium::worldgen::features::{self, FeatureHit, FeatureKind};
use oxium::worldgen::Generator;

use crate::cli::{BurstMode, CaptureArgs, parse_xz, parse_xyz};
use crate::sidecar::{CaptureMeta, CameraInfo, FeatureTargetInfo, GroundInfo, PerfInfo};

// How many consecutive frames the chunk-mesh count must be stable
// before we consider streaming settled. Mirrors the constant in main.rs.
const SCREENSHOT_QUIESCE_FRAMES: u32 = 60;

// ── Entry point ───────────────────────────────────────────────────────────────

/// Run the `capture` subcommand: spin up a hidden window, warm up, capture.
pub fn run(seed: u64, args: CaptureArgs) -> anyhow::Result<()> {
    // Phase 1 (no GPU needed): resolve the feature hit if --find was requested.
    let feature_hit: Option<FeatureHit> = if let Some(ref kind_str) = args.find {
        let kind: FeatureKind = kind_str.parse()?;
        let (ox, oz) = parse_xz(&args.origin).context("--origin: expected WX,WZ")?;
        let g = Generator::new(seed);
        let hits = features::find(&g, ox, oz, kind, args.max_radius, 1)?;
        hits.into_iter().next()
    } else {
        None
    };

    // Resolve spawn position: --spawn > feature hit > default.
    let spawn = if let Some(ref s) = args.spawn {
        let (x, y, z) = parse_xyz(s).context("--spawn: expected X,Y,Z")?;
        glam::Vec3::new(x as f32, y as f32, z as f32)
    } else if let Some(ref hit) = feature_hit {
        // Spawn above the feature's column surface so the camera has a clear
        // sightline. Use a throwaway Generator to look up the surface height;
        // the worldgen LRU will be warm again when AppState starts.
        let g = Generator::new(seed);
        let col = g.column_data(hit.pos[0], hit.pos[2]);

        if hit.pos[1] < col.height {
            // Underground feature (lava, cave). Offset both vertically and
            // horizontally so --look-at-feature produces an angled view
            // (~45°) instead of looking straight down.
            glam::Vec3::new(
                hit.pos[0] as f32 + 20.0,
                hit.pos[1] as f32 + 20.0,
                hit.pos[2] as f32 + 20.0,
            )
        } else {
            // Surface feature — spawn above the terrain directly overhead.
            glam::Vec3::new(
                hit.pos[0] as f32,
                col.height as f32 + 30.0,
                hit.pos[2] as f32,
            )
        }
    } else {
        glam::Vec3::new(16.0, 250.0, 16.0)
    };

    // Resolve look direction: --look-at-feature > --look-at > --look > default.
    let look_override = resolve_look(&args, spawn, feature_hit.as_ref());

    // Resolve output path and validate burst/single-shot consistency.
    let out_path = args
        .out
        .clone()
        .context("--out <path> is required for capture")?;

    if args.frames > 1 && out_path.extension().is_some() {
        // Guard: burst output must be a directory, not a .png file.
        anyhow::bail!(
            "--out {:?} looks like a file path but --frames {} > 1 requires a directory path \
             (no extension). Pass a directory like --out /tmp/burst/",
            out_path,
            args.frames
        );
    }
    if args.mode == BurstMode::Sim && args.interval_ms > 0 {
        anyhow::bail!(
            "--interval-ms {} is meaningless with --mode sim (sim clock is not tied to wall \
             time). Remove --interval-ms or switch to --mode wallclock.",
            args.interval_ms
        );
    }

    // Resolve metrics CSV path:
    //   single-shot: <out_png>.metrics.csv
    //   burst:       <out_dir>/metrics.csv
    let metrics_path = if args.frames <= 1 {
        // e.g. /tmp/probe.png → /tmp/probe.png.metrics.csv
        let mut s = out_path.as_os_str().to_owned();
        s.push(".metrics.csv");
        PathBuf::from(s)
    } else {
        out_path.join("metrics.csv")
    };

    // Phase 2: run the event loop.
    let driver = ProbeCapture {
        seed,
        spawn,
        look_override,
        time_of_day: args.time,
        out_path: out_path.clone(),
        window_size: parse_window_size(args.window_size.as_deref())?,
        warmup_frames: args.warmup_frames,
        burst_total: args.frames,
        burst_interval: Duration::from_millis(args.interval_ms),
        burst_mode: args.mode,
        sim_dt: args.sim_dt,
        metrics_path,
        feature_hit,
        state: None,
        frames_drawn: 0,
        screenshot_last_chunk_count: 0,
        screenshot_stable_frames: 0,
        burst_done: 0,
        burst_next_at: None,
        sim_time: 0.0,
    };

    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut { driver })?;
    Ok(())
}

// ── ApplicationHandler ────────────────────────────────────────────────────────

struct ProbeCapture {
    seed: u64,
    spawn: glam::Vec3,
    /// Pre-resolved `(yaw_rad, pitch_rad)` to apply after spawn, or `None`
    /// to leave the camera at its default orientation.
    look_override: Option<(f32, f32)>,
    time_of_day: Option<f32>,
    /// Output path: a `.png` file for single-shot, or a directory for burst.
    out_path: PathBuf,
    window_size: Option<(u32, u32)>,
    warmup_frames: u32,
    /// Total number of frames to capture (1 = single shot).
    burst_total: u32,
    /// Minimum wall-clock gap between consecutive captures (wallclock mode).
    burst_interval: Duration,
    /// Whether to use wall-clock or sim timing for the burst.
    burst_mode: BurstMode,
    /// Fixed timestep in seconds for sim mode.
    sim_dt: f32,
    /// Accumulated sim clock. Starts at 0 when quiesce completes; advances by
    /// `sim_dt` each captured frame. Used as the shader `time` uniform so
    /// water animation and sun position are deterministic.
    sim_time: f32,
    /// Where the per-frame CSV profiler writes.
    metrics_path: PathBuf,
    feature_hit: Option<FeatureHit>,
    state: Option<AppState>,
    frames_drawn: u32,
    screenshot_last_chunk_count: usize,
    screenshot_stable_frames: u32,
    /// How many frames have been captured so far (0 until quiesce completes).
    burst_done: u32,
    /// Wall-clock deadline for the next capture (wallclock mode).
    /// `None` until quiesce is done; set to `Some(Instant::now())` on quiesce.
    burst_next_at: Option<Instant>,
}

impl ApplicationHandler for ProbeCapture {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the burst output directory now (before AppState, so any I/O
        // error surfaces before we spend time warming up the renderer).
        if self.burst_total > 1 {
            if let Err(e) = std::fs::create_dir_all(&self.out_path) {
                log::error!("failed to create burst directory {:?}: {e}", self.out_path);
                event_loop.exit();
                return;
            }
        } else if let Some(parent) = self.out_path.parent() {
            // For single-shot, create the parent directory if needed.
            let _ = std::fs::create_dir_all(parent);
        }
        // Also ensure the parent of the metrics CSV exists (it's always in the
        // same directory as the output, so `create_dir_all` above covers it).

        let mut attrs = WindowAttributes::default()
            .with_title("oxium-probe")
            .with_visible(false); // always hidden — we just need the GPU surface
        if let Some((w, h)) = self.window_size {
            attrs = attrs.with_inner_size(PhysicalSize::new(w, h));
        }
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let mut state = AppState::new_with_spawn(
            window,
            self.spawn,
            /* uncapped */ true,
            /* profile_path */ Some(&self.metrics_path),
            Some(self.seed),
        );

        // Enable noclip + fly so physics doesn't drop the player out of
        // position. The probe never has user input, so movement is driven only
        // by the camera teleport set above.
        {
            use oxium::ecs::components::{Movement, MovementMode};
            for (_, mv) in state.ecs.world.query::<&mut Movement>().iter() {
                mv.mode = MovementMode::Fly;
                mv.noclip = true;
            }
        }

        // Apply time-of-day override before the first frame.
        if let Some(t) = self.time_of_day {
            for (_, tod) in state
                .ecs
                .world
                .query::<&mut oxium::ecs::components::TimeOfDay>()
                .iter()
            {
                tod.t = t;
            }
        }

        // Apply look direction override.
        if let Some((yaw, pitch)) = self.look_override {
            set_camera_look(&mut state.ecs, yaw, pitch);
        }

        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.renderer.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                // ── Advance the simulation ────────────────────────────────────
                // Quiesce always uses the wall-clock step; once quiesce is
                // done, sim mode switches to step_with_dt so the shader clock
                // is deterministic for the captured frames.
                let in_sim_burst = self.burst_mode == BurstMode::Sim
                    && self.burst_next_at.is_some();

                if in_sim_burst {
                    // Fixed timestep, shader time = self.sim_time.
                    state.step_with_dt(self.sim_dt, Some(self.sim_time));
                } else {
                    state.step();
                }
                self.frames_drawn = self.frames_drawn.saturating_add(1);

                // ── Phase: quiescing ──────────────────────────────────────────
                // Track chunk-mesh count stability. Once warmup has elapsed and
                // the count has been stable for SCREENSHOT_QUIESCE_FRAMES
                // consecutive frames, the world is fully streamed in.
                if self.burst_next_at.is_none() {
                    let count = state.renderer.chunk_mesh_count();
                    if count > self.screenshot_last_chunk_count {
                        self.screenshot_last_chunk_count = count;
                        self.screenshot_stable_frames = 0;
                    } else {
                        self.screenshot_stable_frames =
                            self.screenshot_stable_frames.saturating_add(1);
                    }

                    if self.frames_drawn > self.warmup_frames
                        && self.screenshot_stable_frames >= SCREENSHOT_QUIESCE_FRAMES
                    {
                        // Quiesce complete — start the burst clock.
                        self.burst_next_at = Some(Instant::now());
                        log::info!(
                            "quiesce complete after {} frames ({} chunks); starting {} burst \
                             ({} frames{})",
                            self.frames_drawn,
                            count,
                            match self.burst_mode {
                                BurstMode::Wallclock => "wallclock",
                                BurstMode::Sim => "sim",
                            },
                            self.burst_total,
                            if self.burst_mode == BurstMode::Wallclock {
                                format!(", interval {}ms", self.burst_interval.as_millis())
                            } else {
                                format!(", sim_dt {:.4}s", self.sim_dt)
                            },
                        );
                    }
                }

                // ── Phase: bursting ───────────────────────────────────────────
                if self.burst_next_at.is_some() && self.burst_done < self.burst_total {
                    let should_capture = match self.burst_mode {
                        // Wall-clock: respect the inter-frame interval.
                        BurstMode::Wallclock => {
                            let next_at = self.burst_next_at.unwrap();
                            Instant::now() >= next_at
                        }
                        // Sim: capture every step — the fixed dt IS the interval.
                        BurstMode::Sim => true,
                    };

                    if should_capture {
                        // Re-apply look override so frame-0 drift doesn't affect aim.
                        if let Some((yaw, pitch)) = self.look_override {
                            set_camera_look(&mut state.ecs, yaw, pitch);
                        }

                        // Shader time for the offscreen capture:
                        //   wallclock → 0.0 (deterministic baseline, same as main.rs)
                        //   sim       → current sim_time (animation advances across frames)
                        let shader_time = match self.burst_mode {
                            BurstMode::Wallclock => 0.0,
                            BurstMode::Sim => self.sim_time,
                        };

                        let frame_idx = self.burst_done + 1; // 1-based for filenames
                        let out_png = burst_frame_path(&self.out_path, self.burst_total, frame_idx);
                        let result = do_capture(
                            state,
                            self.seed,
                            self.time_of_day,
                            &out_png,
                            self.feature_hit.as_ref(),
                            shader_time,
                        );
                        match result {
                            Ok(()) => {
                                self.burst_done += 1;
                                match self.burst_mode {
                                    BurstMode::Wallclock => {
                                        // Advance the wall-clock deadline.
                                        let prev = self.burst_next_at.unwrap();
                                        self.burst_next_at = Some(prev + self.burst_interval);
                                    }
                                    BurstMode::Sim => {
                                        // Advance the sim clock.
                                        self.sim_time += self.sim_dt;
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!("capture failed: {e:?}");
                                event_loop.exit();
                                return;
                            }
                        }
                    }

                    if self.burst_done >= self.burst_total {
                        log::info!("burst complete ({} frames captured)", self.burst_done);
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}


// ── Capture logic (free function to avoid borrow conflicts) ───────────────────

/// Render and save one screenshot + sidecar JSON. Called from the event loop
/// after warmup + quiesce. Separated from `ProbeCapture` so it can borrow
/// `state` independently of the driver struct.
///
/// `shader_time` drives the `time` uniform for the offscreen render (water
/// shimmer, etc.). Pass `0.0` for wall-clock captures (deterministic baseline);
/// pass the current `sim_time` for sim-burst captures so animation advances
/// across frames while remaining reproducible across runs.
fn do_capture(
    state: &AppState,
    seed: u64,
    time_of_day: Option<f32>,
    out_path: &Path,
    feature_hit: Option<&FeatureHit>,
    shader_time: f32,
) -> anyhow::Result<()> {
    let (eye, yaw, pitch) = camera_from_ecs(&state.ecs);
    let (sun_dir, sun_intensity) = sun_state(&state.ecs);

    capture_offscreen(
        &state.renderer,
        out_path,
        eye,
        yaw,
        pitch,
        sun_dir,
        sun_intensity,
        shader_time,
        Some(&state.ui),
    )?;

    log::info!(
        "captured {} ({} chunks; lod={:?})",
        out_path.display(),
        state.renderer.chunk_mesh_count(),
        state.renderer.chunk_mesh_lod_counts(),
    );

    // Write sidecar JSON.
    let sidecar = sidecar_path(out_path);
    let col = state.generator.column_data(eye.x as i32, eye.z as i32);
    let (vw, vh) = state.renderer.framebuffer_size();
    let meta = CaptureMeta {
        schema: 1,
        seed,
        camera: CameraInfo {
            eye_world: [eye.x, eye.y, eye.z],
            yaw_deg: yaw.to_degrees(),
            pitch_deg: pitch.to_degrees(),
        },
        ground: GroundInfo {
            column: [eye.x as i32, eye.z as i32],
            biome: format!("{:?}", col.biome),
            height: col.height,
            water_surface_y: col.water_surface_y,
            is_cliff: col.is_cliff,
        },
        time_of_day: time_of_day.unwrap_or(0.5),
        sun_dir,
        sun_intensity,
        viewport: [vw, vh],
        perf: PerfInfo {
            work_ms: state.perf.work_ms,
            draw_calls: state.perf.draw_calls,
            chunks_rendered: state.perf.chunks_rendered,
            chunks_loaded: state.perf.chunks_loaded,
            chunks_pending: state.perf.chunks_pending,
            light_queue: state.perf.light_queue,
        },
        feature_target: feature_hit.map(|h| FeatureTargetInfo {
            kind: format!("{:?}", h.kind),
            pos: h.pos,
            distance_blocks: h.distance_blocks,
        }),
    };
    meta.write(&sidecar)?;
    log::info!("sidecar written to {}", sidecar.display());
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute the sidecar path for a screenshot: `foo/bar.png` → `foo/bar.png.json`.
fn sidecar_path(png_path: &Path) -> PathBuf {
    let mut s = png_path.as_os_str().to_owned();
    s.push(".json");
    PathBuf::from(s)
}

/// Resolve the output path for burst frame `idx` (1-based).
///
/// - Single-shot (`total == 1`): returns `base_or_dir` unchanged (it's the PNG).
/// - Burst (`total > 1`): returns `base_or_dir / "{idx:04}.png"`.
fn burst_frame_path(base_or_dir: &Path, total: u32, idx: u32) -> PathBuf {
    if total <= 1 {
        base_or_dir.to_path_buf()
    } else {
        base_or_dir.join(format!("{idx:04}.png"))
    }
}

/// Resolve the camera look direction from the CLI args and optional feature hit.
///
/// Priority: `--look-at-feature` > `--look-at WX,WY,WZ` > `--look yaw,pitch`.
/// Returns `(yaw_rad, pitch_rad)` or `None` if no override was requested.
fn resolve_look(
    args: &CaptureArgs,
    eye: glam::Vec3,
    feature_hit: Option<&FeatureHit>,
) -> Option<(f32, f32)> {
    if args.look_at_feature {
        let hit = feature_hit?;
        let target = glam::Vec3::new(
            hit.pos[0] as f32,
            hit.pos[1] as f32,
            hit.pos[2] as f32,
        );
        return Some(look_at_angles(eye, target));
    }
    if let Some(ref s) = args.look_at
        && let Ok((tx, ty, tz)) = parse_xyz(s)
    {
        let target = glam::Vec3::new(tx as f32, ty as f32, tz as f32);
        return Some(look_at_angles(eye, target));
    }
    if let Some(ref s) = args.look
        && let Some((yaw_deg, pitch_deg)) = parse_look_degrees(s)
    {
        return Some((yaw_deg.to_radians(), pitch_deg.to_radians()));
    }
    None
}

/// Compute `(yaw_rad, pitch_rad)` so the camera at `eye` faces `target`.
///
/// Yaw is the horizontal angle: 0 rad = facing +X, π/2 rad = facing +Z.
/// Pitch is the elevation: positive = looking up.
fn look_at_angles(eye: glam::Vec3, target: glam::Vec3) -> (f32, f32) {
    let delta = target - eye;
    let horiz = (delta.x * delta.x + delta.z * delta.z).sqrt();
    let yaw = delta.z.atan2(delta.x);
    let pitch = delta.y.atan2(horiz);
    (yaw, pitch)
}

fn parse_look_degrees(s: &str) -> Option<(f32, f32)> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return None;
    }
    let yaw: f32 = parts[0].trim().parse().ok()?;
    let pitch: f32 = parts[1].trim().parse().ok()?;
    Some((yaw, pitch))
}

fn parse_window_size(s: Option<&str>) -> anyhow::Result<Option<(u32, u32)>> {
    let Some(s) = s else {
        return Ok(None);
    };
    let parts: Vec<&str> = s.split('x').collect();
    anyhow::ensure!(parts.len() == 2, "window size must be WxH, got {:?}", s);
    let w: u32 = parts[0].parse().context("window width")?;
    let h: u32 = parts[1].parse().context("window height")?;
    Ok(Some((w, h)))
}
