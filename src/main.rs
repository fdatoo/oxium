//! oxium — entry point.
//!
//! M3 onwards: the world is streamed in around the player by background
//! generation jobs. No more hardcoded test chunk.
//!
//! Hidden flag `--screenshot-and-exit <path>` is still here: it renders one
//! offscreen frame from the live camera state and saves a PNG.

mod app;
mod ecs;
mod jobs;
mod lighting;
mod mesher;
mod persistence;
mod physics;
mod render;
mod voxel;
mod worldgen;

use std::path::PathBuf;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, WindowAttributes, WindowId};

use app::AppState;
use glam::Vec3;

/// Parsed CLI flags. Only the screenshot flag is meaningful right now.
struct CliOptions {
    /// If set, render one offscreen frame, save a PNG, and exit.
    /// Activated by `--screenshot-and-exit <path>`.
    screenshot_path: Option<PathBuf>,
    /// Number of normal frames to render before the screenshot capture.
    /// Tunable via `OXIUM_SCREENSHOT_WARMUP_FRAMES` (default 60 — enough
    /// to let the streaming system generate the nearby chunks).
    warmup_frames: u32,
}

impl CliOptions {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut screenshot_path = None;
        while let Some(arg) = args.next() {
            if arg == "--screenshot-and-exit" {
                let path = args
                    .next()
                    .expect("--screenshot-and-exit requires a path argument");
                screenshot_path = Some(PathBuf::from(path));
            }
        }
        let warmup_frames = std::env::var("OXIUM_SCREENSHOT_WARMUP_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        Self {
            screenshot_path,
            warmup_frames,
        }
    }
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
        let state = AppState::new(window.clone());

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

                if let Some(path) = self.cli.screenshot_path.clone() {
                    if self.frames_drawn > self.cli.warmup_frames {
                        let (eye, yaw, pitch) = camera_from_ecs(&state.ecs);
                        let (sun_dir, sun_intensity) =
                            ecs::systems::time_of_day::sun_state(&state.ecs);
                        match capture_offscreen(
                            &state.renderer,
                            &path,
                            eye,
                            yaw,
                            pitch,
                            sun_dir,
                            sun_intensity,
                        ) {
                            Ok(()) => log::info!(
                                "screenshot saved to {} ({} chunks)",
                                path.display(),
                                state.renderer.chunk_mesh_count()
                            ),
                            Err(e) => log::error!("screenshot failed: {e:?}"),
                        }
                        event_loop.exit();
                        return;
                    }
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
fn capture_offscreen(
    renderer: &render::Renderer,
    path: &std::path::Path,
    eye: Vec3,
    yaw: f32,
    pitch: f32,
    sun_dir: [f32; 3],
    sun_intensity: f32,
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
    renderer.render_to_view(&view, eye, yaw, pitch, aspect, sun_dir, sun_intensity);

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
