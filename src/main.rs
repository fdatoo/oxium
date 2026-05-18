//! oxium — entry point.
//!
//! M1 wires a static fly-cam pointed at a single hardcoded "stone slab with
//! grass on top" chunk and presents it through `wgpu`.
//!
//! The binary also supports a hidden `--screenshot-and-exit <path>` flag
//! used for headless visual verification: it renders one frame into an
//! offscreen RGBA8 texture and saves the result to a PNG, then exits.

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
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use glam::{UVec3, Vec3};
use mesher::naive::mesh_chunk_no_neighbors;
use render::Renderer;
use voxel::block::{Block, BlockRegistry};
use voxel::chunk::DenseChunk;
use voxel::coords::{LocalPos, CHUNK_DIM_U};

/// Hardcoded M1 test chunk: 8 blocks tall (stone/dirt/grass layers),
/// 32 wide and 32 deep. Replaced in M3 by procedural worldgen.
fn build_test_chunk() -> DenseChunk {
    let mut c = DenseChunk::empty();
    for z in 0..CHUNK_DIM_U {
        for x in 0..CHUNK_DIM_U {
            for y in 0..8 {
                let b = if y == 7 {
                    Block::Grass
                } else if y >= 4 {
                    Block::Dirt
                } else {
                    Block::Stone
                };
                c.set(LocalPos(UVec3::new(x, y, z)), b);
            }
        }
    }
    c
}

/// Parsed command-line options. The M1 binary recognises only the
/// screenshot flag; everything else is ignored.
struct CliOptions {
    /// If set, render one frame, save a PNG to this path, and exit.
    /// Activated by `--screenshot-and-exit <path>`.
    screenshot_path: Option<PathBuf>,
    /// How many frames to render *before* the screenshot one. Lets the
    /// surface/swap-chain settle on systems that need an extra frame.
    /// Configurable via `OXIUM_SCREENSHOT_WARMUP_FRAMES`; default 2.
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
            .unwrap_or(2);
        Self {
            screenshot_path,
            warmup_frames,
        }
    }
}

/// The `winit` application handler. Owns the window + renderer once `resumed`
/// has fired and, for the screenshot path, the post-warmup capture state.
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    cli: CliOptions,
    frames_drawn: u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default().with_title("oxium");
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let mut renderer = Renderer::new(window.clone());

        // Build and upload the M1 hardcoded chunk so there's something to draw.
        let registry = BlockRegistry::new();
        let mesh = mesh_chunk_no_neighbors(&build_test_chunk(), &registry);
        renderer.upload_test_mesh(&mesh);

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let (Some(window), Some(renderer)) = (self.window.as_ref(), self.renderer.as_mut()) else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                // M1 uses a static camera. The view is from a high corner
                // looking down into the slab at a ~45° angle.
                let eye = Vec3::new(40.0, 18.0, 40.0);
                let yaw = -3.0 * std::f32::consts::FRAC_PI_4;
                let pitch = -0.35;

                if let Err(e) = renderer.render(eye, yaw, pitch) {
                    log::warn!("render error: {e:?}");
                }
                self.frames_drawn = self.frames_drawn.saturating_add(1);

                // Screenshot path: after the warmup frames, take one offscreen
                // capture and exit cleanly.
                if let Some(path) = self.cli.screenshot_path.clone() {
                    if self.frames_drawn > self.cli.warmup_frames {
                        match capture_offscreen(renderer, &path, eye, yaw, pitch) {
                            Ok(()) => log::info!("screenshot saved to {}", path.display()),
                            Err(e) => log::error!("screenshot failed: {e:?}"),
                        }
                        event_loop.exit();
                        return;
                    }
                }

                window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Render one frame to an offscreen `Rgba8UnormSrgb` texture and save it to
/// `path`. Used by the `--screenshot-and-exit` flag. Decoupled from the swap
/// chain so the saved PNG has a known format and never reflects compositor
/// quirks.
fn capture_offscreen(
    renderer: &Renderer,
    path: &std::path::Path,
    eye: Vec3,
    yaw: f32,
    pitch: f32,
) -> anyhow::Result<()> {
    let width = renderer.gpu.surface_cfg.width;
    let height = renderer.gpu.surface_cfg.height;
    // Match the surface format so the offscreen path reuses the same
    // pipeline as the live render; the screenshot helper handles
    // BGRA→RGBA reordering if needed.
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
    renderer.render_to_view(&view, eye, yaw, pitch, aspect);

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
    // Poll = run as fast as the renderer + present mode allow; right for
    // a game. `Wait` is wrong because it would idle between OS events.
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        renderer: None,
        cli,
        frames_drawn: 0,
    };
    event_loop.run_app(&mut app).unwrap();
}
