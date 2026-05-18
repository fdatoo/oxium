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
mod render;

use std::path::PathBuf;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, WindowEvent};
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
}

impl CliOptions {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut screenshot_path = None;
        let mut spawn = None;
        let mut look = None;
        let mut find_water = false;
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default().with_title("oxium");
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        // Pick spawn: explicit --spawn wins, else --find-water locates
        // a known wet column, else the default mid-air spawn.
        let spawn = self
            .cli
            .spawn
            .or_else(|| self.cli.find_water.then(find_water_spawn));
        let state = match spawn {
            Some(p) => AppState::new_with_spawn(window.clone(), p),
            None => AppState::new(window.clone()),
        };

        // Skip cursor grab when running in screenshot mode so the helper
        // doesn't steal cursor focus on the host system.
        if self.cli.screenshot_path.is_none() {
            if let Err(e) = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            {
                log::warn!("cursor grab failed: {e:?}");
            }
            window.set_cursor_visible(false);
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
                    state.input_buf.on_key(code, ke.state);
                    if code == winit::keyboard::KeyCode::Escape
                        && ke.state == ElementState::Pressed
                    {
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::MouseInput {
                button,
                state: bstate,
                ..
            } => {
                state.input_buf.on_mouse_button(button, bstate);
            }
            WindowEvent::RedrawRequested => {
                state.step();
                self.frames_drawn = self.frames_drawn.saturating_add(1);

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

                state.window.request_redraw();
            }
            _ => {}
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
            state.input_buf.on_mouse_motion(dx, dy);
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
    renderer.render_to_view(&view, eye, yaw, pitch, aspect, sun_dir, sun_intensity, time);

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

fn main() {
    env_logger::init();
    let cli = CliOptions::parse();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        state: None,
        cli,
        frames_drawn: 0,
    };
    event_loop.run_app(&mut app).unwrap();
}
