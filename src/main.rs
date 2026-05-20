//! oxium — entry point.
//!
//! M3 onwards: the world is streamed in around the player by background
//! generation jobs. No more hardcoded test chunk.
//!
//! Hidden flag `--screenshot-and-exit <path>` is still here: it renders one
//! offscreen frame from the live camera state and saves a PNG.

// Pure engine modules live in the `oxium` library crate so integration
// tests can drive them headlessly. Re-export them at the binary's crate
// root so `crate::voxel::…` references in our `app` / `ecs` / `render`
// submodules continue to resolve without rewriting paths everywhere.
pub use oxium::{jobs, lighting, mesher, persistence, physics, voxel, worldgen};

// Binary-only modules — they import `winit`/`wgpu` directly and so
// aren't part of the library surface.
mod app;
mod ecs;
mod profiler;
mod render;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, WindowAttributes, WindowId};

use app::AppState;
use glam::Vec3;

/// Parsed CLI flags.
struct CliOptions {
    /// Render one offscreen frame, save a PNG, and exit.
    /// `--screenshot-and-exit <path>`.
    screenshot_path: Option<PathBuf>,
    /// Number of normal frames to render before the screenshot capture.
    /// Tunable via `OXIUM_SCREENSHOT_WARMUP_FRAMES` (default 60).
    warmup_frames: u32,
    /// Override the player spawn point. `--spawn x,y,z`.
    spawn: Option<Vec3>,
    /// Override the camera orientation, *degrees*. `--look yaw,pitch`.
    /// Applied at screenshot time only (so it doesn't fight the
    /// physics-driven walking camera in a real session).
    look: Option<(f32, f32)>,
    /// Auto-locate a water column and use it as the spawn — useful for
    /// "show me the water" screenshots without having to compute a
    /// coordinate by hand. `--find-water`.
    find_water: bool,
    /// Override the `TimeOfDay.t` value, 0..=1. 0/1 = midnight,
    /// 0.25 = sunrise, 0.5 = noon, 0.75 = sunset. `--time 0.85` would
    /// show the night sky with stars + moon.
    time_of_day: Option<f32>,
    /// Disable vsync (run with `PresentMode::Immediate`). Lets the HUD
    /// FPS readout reflect actual CPU/GPU throughput instead of being
    /// capped by the display refresh — important for diagnosing
    /// "FPS stuck at 60" on ProMotion displays that drop refresh rate
    /// under low demand.
    uncapped: bool,
    /// Enable per-frame profiling and write CSV rows to this path.
    /// `--profile <path>` records every system's per-step timing
    /// alongside the perf-counter snapshot. Open the file in a
    /// spreadsheet (or `awk`) to see where the frame budget goes.
    profile_path: Option<PathBuf>,
    /// Hidden test hook: drop the UI into `paused` or `chat` before the
    /// screenshot frame. No effect during normal play.
    ui_state: Option<String>,
}

impl CliOptions {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut screenshot_path = None;
        let mut spawn = None;
        let mut look = None;
        let mut find_water = false;
        let mut time_of_day: Option<f32> = None;
        let mut uncapped = false;
        let mut profile_path = None;
        let mut ui_state: Option<String> = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--screenshot-and-exit" => {
                    let path = args
                        .next()
                        .expect("--screenshot-and-exit requires a path argument");
                    screenshot_path = Some(PathBuf::from(path));
                }
                "--spawn" => {
                    let v = args.next().expect("--spawn requires `x,y,z`");
                    let parts: Vec<f32> = v.split(',').map(|s| s.parse().unwrap()).collect();
                    assert_eq!(parts.len(), 3, "--spawn expects three comma-separated floats");
                    spawn = Some(Vec3::new(parts[0], parts[1], parts[2]));
                }
                "--look" => {
                    let v = args.next().expect("--look requires `yaw_deg,pitch_deg`");
                    let parts: Vec<f32> = v.split(',').map(|s| s.parse().unwrap()).collect();
                    assert_eq!(parts.len(), 2, "--look expects two comma-separated floats");
                    look = Some((parts[0].to_radians(), parts[1].to_radians()));
                }
                "--find-water" => {
                    find_water = true;
                }
                "--time" => {
                    let v = args.next().expect("--time requires a 0..=1 value");
                    time_of_day = Some(v.parse().unwrap());
                }
                "--uncapped" => {
                    uncapped = true;
                }
                "--profile" => {
                    let path = args
                        .next()
                        .expect("--profile requires a path argument");
                    profile_path = Some(PathBuf::from(path));
                }
                "--ui" => {
                    let v = args.next().expect("--ui requires `paused` or `chat`");
                    ui_state = Some(v);
                }
                _ => {}
            }
        }
        let warmup_frames = std::env::var("OXIUM_SCREENSHOT_WARMUP_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        Self {
            screenshot_path,
            warmup_frames,
            spawn,
            look,
            find_water,
            time_of_day,
            uncapped,
            profile_path,
            ui_state,
        }
    }
}

/// Scan world generation around `(0, 0)` for the lowest-height column
/// (which will be flooded with water by the sea-level pass). Returns a
/// position 2 blocks above the water surface so the camera lands just
/// over the waves.
fn find_water_spawn() -> Vec3 {
    use worldgen::{Generator, SEA_LEVEL};
    let g = Generator::new(42);
    let mut dense = voxel::chunk::DenseChunk::empty();
    let mut best: Option<(f32, Vec3)> = None;
    // Sweep a handful of chunks around the origin. We only need to find
    // one tile with height < SEA_LEVEL.
    for cx in -3..=3 {
        for cz in -3..=3 {
            for cy in 1..=2 {
                let coord = voxel::coords::ChunkCoord(glam::IVec3::new(cx, cy, cz));
                g.fill_chunk(coord, &mut dense);
                let origin = coord.origin().0;
                for lx in 0..32 {
                    for lz in 0..32 {
                        // Scan the column top-down for the topmost
                        // *non-air* block. If it's Water we found a
                        // sea-surface cell.
                        for ly in (0..32).rev() {
                            let lp = voxel::coords::LocalPos(glam::UVec3::new(lx, ly, lz));
                            let b = dense.blocks[lp.to_index()];
                            if b == voxel::block::Block::Air {
                                continue;
                            }
                            if b == voxel::block::Block::Water {
                                let wx = origin.x as f32 + lx as f32;
                                let wy = origin.y as f32 + ly as f32;
                                let wz = origin.z as f32 + lz as f32;
                                let dist =
                                    (wx * wx + wz * wz + (wy - SEA_LEVEL as f32).powi(2)).sqrt();
                                if best.map(|(d, _)| dist < d).unwrap_or(true) {
                                    best = Some((dist, Vec3::new(wx, wy + 2.0, wz)));
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
        .unwrap_or_else(|| Vec3::new(0.0, SEA_LEVEL as f32 + 2.0, 0.0))
}

/// winit application handler. `state` is created lazily in `resumed`.
struct App {
    state: Option<AppState>,
    cli: CliOptions,
    frames_drawn: u32,
    /// Frame pacing target: the instant at which the next redraw is
    /// allowed to fire. Initialised on the first frame and advanced
    /// by [`Self::frame_budget`] each frame. Ignored when
    /// `cli.uncapped` is set.
    next_frame_target: Option<Instant>,
}

/// Target frame budget when the FPS cap is on. 60 FPS = 16.666… ms
/// per frame; we use exact integer nanos for monotonic advancement.
const FRAME_BUDGET_60_FPS: Duration = Duration::from_nanos(16_666_667);

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Hide the window in screenshot mode so the renderer runs truly
        // headless — no surface flash on macOS, no focus-stealing while
        // capturing a baseline. The window still exists (wgpu's Surface
        // needs a winit window on every platform), it's just never shown.
        let attrs = WindowAttributes::default()
            .with_title("oxium")
            .with_visible(self.cli.screenshot_path.is_none());
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        // Pick spawn: explicit --spawn wins, else --find-water locates
        // a known wet column, else the default mid-air spawn.
        let spawn = self
            .cli
            .spawn
            .or_else(|| self.cli.find_water.then(find_water_spawn));
        let uncapped = self.cli.uncapped;
        let profile = self.cli.profile_path.as_deref();
        let mut state = match spawn {
            Some(p) => AppState::new_with_spawn(window.clone(), p, uncapped, profile),
            None => AppState::new_with_spawn(
                window.clone(),
                // High default spawn so you can see the whole load
                // radius from above for diagnostics. Use --spawn
                // x,y,z to pick a different one.
                glam::Vec3::new(16.0, 250.0, 16.0),
                uncapped,
                profile,
            ),
        };
        // CLI `--time` overrides the TimeOfDay sun-cycle value. Apply
        // after spawn so the Sun entity already exists.
        if let Some(t) = self.cli.time_of_day {
            for (_, tod) in state
                .ecs
                .world
                .query::<&mut crate::ecs::components::TimeOfDay>()
                .iter()
            {
                tod.t = t;
            }
        }

        if let Some(ref kind) = self.cli.ui_state {
            use crate::ui::state::{MenuNav, UiState};
            use crate::ui::chat::ChatInput;
            state.ui.state = match kind.as_str() {
                "paused" => UiState::Paused { menu: MenuNav::Top { hovered: 0 } },
                "chat"   => UiState::Chat { input: ChatInput::new("/he") },
                other    => panic!("--ui: expected 'paused' or 'chat', got {other}"),
            };
            // Pre-seed a few chat lines so the chat snapshot shows content.
            state.ui.log.push_system("System: hello there");
            state.ui.log.push_player("a friendly note");
            state.ui.log.push_echo("/help");
        }

        // Skip cursor grab when running in screenshot mode so the helper
        // doesn't steal cursor focus on the host system.
        if self.cli.screenshot_path.is_none() {
            grab_cursor(&window);
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
            WindowEvent::KeyboardInput { event: ke, .. } => {
                if let PhysicalKey::Code(code) = ke.physical_key {
                    let text = ke.text.as_deref();
                    let disp = state.ui.on_key(code, ke.state, text);
                    if disp == crate::ui::input::InputDisposition::Forward {
                        state.input_buf.on_key(code, ke.state);
                    }
                }
            }
            WindowEvent::MouseInput {
                button,
                state: bstate,
                ..
            } => {
                if state.ui.is_playing() {
                    state.input_buf.on_mouse_button(button, bstate);
                } else {
                    // The UI consumes mouse clicks while paused/chatting
                    // (menu activation is wired in Task 9).
                    state.ui.on_mouse_button(button, bstate);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if !state.ui.is_playing() {
                    let (w, h) = state.renderer.framebuffer_size();
                    state.ui.on_mouse_move(position.x as f32, position.y as f32, (w, h));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // winit reports either lines (mouse wheel) or pixels
                // (trackpad scroll). Treat one line ≈ one detent; the
                // pixel branch scales down so trackpad scrolling
                // doesn't fly through the hotbar.
                use winit::event::MouseScrollDelta;
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                if state.ui.is_playing() {
                    state.input_buf.on_scroll(lines);
                }
                // Discarded while paused/chatting — chat doesn't scroll yet.
            }
            WindowEvent::RedrawRequested => {
                state.step();
                self.frames_drawn = self.frames_drawn.saturating_add(1);

                if state.ui.cursor_state_changed {
                    state.ui.cursor_state_changed = false;
                    // Snap to centre on every Playing⇄!Playing transition,
                    // BEFORE changing grab mode. On resume this becomes
                    // the position CursorGrabMode::Locked pins to; on
                    // pause this is the position the cursor will appear
                    // at when we show it. Doing the warp first also
                    // avoids macOS quirks where set_cursor_position on a
                    // freshly-grabbed window sometimes silently no-ops.
                    let (w, h) = state.renderer.framebuffer_size();
                    let _ = state.window.set_cursor_position(
                        winit::dpi::PhysicalPosition::new(w as f64 / 2.0, h as f64 / 2.0),
                    );
                    if state.ui.is_playing() {
                        grab_cursor(&state.window);
                    } else {
                        release_cursor(&state.window);
                    }
                }

                // Re-assert cursor invisibility every gameplay frame.
                // macOS only honours `set_cursor_visible(false)` while
                // it considers the cursor "over" the window; the moment
                // motion suggests the cursor has wandered, the OS
                // flashes the system pointer back on. Cheap to call
                // (no-op when already hidden), and unlike a warp it
                // doesn't perturb DeviceEvent::MouseMotion.
                if state.ui.is_playing() {
                    state.window.set_cursor_visible(false);
                }

                if state.ui.wants_quit {
                    event_loop.exit();
                    return;
                }

                if let Some(path) = self.cli.screenshot_path.clone()
                    && self.frames_drawn > self.cli.warmup_frames
                {
                    let (eye, mut yaw, mut pitch) = camera_from_ecs(&state.ecs);
                    if let Some((y, p)) = self.cli.look {
                        yaw = y;
                        pitch = p;
                    }
                    let (sun_dir, sun_intensity) =
                        ecs::systems::time_of_day::sun_state(&state.ecs);
                    let time = state.start_time.elapsed().as_secs_f32();
                    match capture_offscreen(
                        &state.renderer,
                        &path,
                        eye,
                        yaw,
                        pitch,
                        sun_dir,
                        sun_intensity,
                        time,
                    ) {
                        Ok(()) => log::info!(
                            "screenshot saved to {} ({} entries; lod counts={:?})",
                            path.display(),
                            state.renderer.chunk_mesh_count(),
                            state.renderer.chunk_mesh_lod_counts(),
                        ),
                        Err(e) => log::error!("screenshot failed: {e:?}"),
                    }
                    event_loop.exit();
                    return;
                }

                // Frame pacing is driven from `about_to_wait` —
                // calling request_redraw here would defeat
                // WaitUntil (winit treats the pending redraw as a
                // reason to wake immediately on most platforms).
            }
            _ => {}
        }
    }

    /// Called by winit when there are no more events to process and
    /// the loop is about to enter its wait state. This is the
    /// canonical hook for frame pacing: we decide here whether to
    /// request the next redraw (cap reached) or set WaitUntil to
    /// sleep until the next 60 FPS slot.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_ref() else {
            return;
        };
        if self.cli.uncapped {
            // Uncapped path: poll as fast as possible.
            event_loop.set_control_flow(ControlFlow::Poll);
            state.window.request_redraw();
            return;
        }
        let now = Instant::now();
        let target = self.next_frame_target.unwrap_or(now);
        if now >= target {
            // Time to draw this frame. Advance the target for the
            // next one; clamp to `now` if we fell behind so the
            // engine doesn't run a burst of catch-up frames after a
            // stall.
            state.window.request_redraw();
            let next = (target + FRAME_BUDGET_60_FPS).max(now);
            self.next_frame_target = Some(next);
            event_loop.set_control_flow(ControlFlow::WaitUntil(next));
        } else {
            // Not yet — sleep until the next frame slot.
            event_loop.set_control_flow(ControlFlow::WaitUntil(target));
        }
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            // Only buffer motion while playing. Otherwise any cursor
            // movement during a menu / chat session accumulates in
            // InputBuf and gets applied as a camera shake on the first
            // resume frame — visible as a sudden snap back toward
            // where the cursor was last moved in the menu.
            if state.ui.is_playing() {
                state.input_buf.on_mouse_motion(dx, dy);
            }
        }
    }
}

/// Pull `(eye, yaw, pitch)` from the player entity in the ECS.
fn camera_from_ecs(ecs: &crate::ecs::GameEcs) -> (Vec3, f32, f32) {
    use crate::ecs::components::{Camera, Position};
    let mut q = ecs
        .world
        .query_one::<(&Position, &Camera)>(ecs.player)
        .unwrap();
    let (pos, cam) = q.get().unwrap();
    (pos.0 + cam.eye_offset, cam.yaw, cam.pitch)
}

/// Render one frame to an offscreen texture matching the surface format
/// and save it as a PNG.
#[allow(clippy::too_many_arguments)]
fn capture_offscreen(
    renderer: &render::Renderer,
    path: &std::path::Path,
    eye: Vec3,
    yaw: f32,
    pitch: f32,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    time: f32,
) -> anyhow::Result<()> {
    let width = renderer.gpu.surface_cfg.width;
    let height = renderer.gpu.surface_cfg.height;
    let format = renderer.gpu.surface_cfg.format;

    let texture = renderer.gpu.device.create_texture(&wgpu::TextureDescriptor {
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
    // Capture the HUD too so screenshots can verify HUD layout.
    // Build a representative HudFrame using a fixed-ish FPS readout
    // (the real metric requires multiple live frames; the screenshot
    // path runs in one shot after warmup).
    let registry = crate::voxel::block::BlockRegistry::new();
    let perf = crate::app::PerfSnapshot::default();
    let hud = crate::render::hud::build_hud(
        (width, height),
        60.0,
        eye,
        0,
        &registry,
        &perf,
    );
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

    render::screenshot::capture_texture_to_png(
        &renderer.gpu.device,
        &renderer.gpu.queue,
        &texture,
        format,
        width,
        height,
        path,
    )
}

fn grab_cursor(window: &winit::window::Window) {
    // Prefer Locked (cursor is pinned, doesn't move at all). If the
    // platform refuses (some Linux setups, older macOS), fall back to
    // Confined which at least keeps the cursor inside the window.
    match window.set_cursor_grab(CursorGrabMode::Locked) {
        Ok(()) => log::debug!("cursor grab: Locked"),
        Err(e_locked) => match window.set_cursor_grab(CursorGrabMode::Confined) {
            Ok(()) => log::info!(
                "cursor grab: Confined (Locked unavailable: {e_locked:?})"
            ),
            Err(e_confined) => log::warn!(
                "cursor grab failed (Locked: {e_locked:?}; Confined: {e_confined:?})"
            ),
        },
    }
    window.set_cursor_visible(false);
}

fn release_cursor(window: &winit::window::Window) {
    let _ = window.set_cursor_grab(CursorGrabMode::None);
    window.set_cursor_visible(true);
}

fn main() {
    env_logger::init();
    let cli = CliOptions::parse();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        state: None,
        cli,
        frames_drawn: 0,
        next_frame_target: None,
    };
    event_loop.run_app(&mut app).unwrap();
}
