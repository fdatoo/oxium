//! Worldgen tuning visualizer — PR 1 (streaming skeleton).

mod app;
mod camera;
mod layout;
mod overlays;
mod paint;
mod probe;
mod render;
mod session;
mod widgets;
mod world;

use crate::app::AppState;
use crate::render::scene::SceneRenderer;
use crate::render::RenderState;
use crate::session::CamKind;
use clap::Parser;
use oxium::worldgen::config::WorldgenConfig;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    /// World seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Run one frame and exit (for CI smoke tests).
    #[arg(long, default_value_t = false)]
    check: bool,
    /// Stream radius in chunks (XZ, camera-relative). Default 3 = 7×7
    /// chunk grid ≈ 224 blocks visible horizontally. Larger = more
    /// world visible but slower regen on every config edit.
    #[arg(long, default_value_t = 3)]
    radius_xz: i32,
    /// Stream Y half-range in chunks, world-anchored (NOT camera-
    /// relative). Default 4 = chunks `cy ∈ -4..=4` ≈ blocks -128..=160,
    /// which covers the full default Oxium world height regardless of
    /// camera elevation.
    #[arg(long, default_value_t = 4)]
    radius_y: i32,
}

struct VizApp {
    window: Option<Arc<Window>>,
    render: Option<RenderState>,
    scene: Option<SceneRenderer>,
    state: AppState,
    /// True between W/A/S/D press and release.
    keys: KeyState,
    /// One-frame flag set by --check to terminate after first redraw.
    quit_after_render: bool,
}

#[derive(Default)]
struct KeyState {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
    q: bool,
    e: bool,
    shift: bool,
}

impl VizApp {
    fn new(cli: Cli) -> Self {
        let config = WorldgenConfig::bundled_default().expect("bundled default.ron");
        let radius = crate::world::stream::StreamRadius {
            xz: cli.radius_xz.max(0),
            y: cli.radius_y.max(0),
        };
        Self {
            state: AppState::new(cli.seed, config, radius),
            window: None,
            render: None,
            scene: None,
            keys: KeyState::default(),
            quit_after_render: cli.check,
        }
    }
}

impl ApplicationHandler for VizApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Oxium worldgen visualizer (PR 1)")
            .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 1000.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let render = RenderState::new(window.clone());
        let scene = SceneRenderer::new(
            &render.device,
            render.surface_config.format,
            render.surface_config.width,
            render.surface_config.height,
        );
        self.window = Some(window);
        self.render = Some(render);
        self.scene = Some(scene);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(render), Some(scene)) =
            (self.window.as_ref(), self.render.as_mut(), self.scene.as_mut())
        else {
            return;
        };
        let _ = render.egui_state.on_window_event(window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                render.resize(size.width, size.height);
                scene.resize(&render.device, size.width, size.height);
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.state.mouse_down {
                    if let Some((px, py)) = self.state.last_cursor {
                        let dx = (position.x - px) as f32;
                        let dy = (position.y - py) as f32;
                        match self.state.session.cam_kind {
                            CamKind::Fly => self.state.session.fly.look(dx, dy),
                            CamKind::Orbit => {
                                self.state.session.orbit.yaw -= dx * 0.005;
                                self.state.session.orbit.pitch =
                                    (self.state.session.orbit.pitch + dy * 0.005).clamp(-1.5, 1.5);
                            }
                        }
                    }
                }
                self.state.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.state.mouse_down = state == ElementState::Pressed;
                } else if button == MouseButton::Left
                    && state == ElementState::Released
                    && !render.egui_ctx.wants_pointer_input()
                {
                    // Left-click in the 3D viewport → raycast to the
                    // first solid voxel and pin its column. We gate on
                    // `!wants_pointer_input()` so clicks inside any
                    // egui panel (config sliders, map, probe) don't
                    // double-fire as pickers.
                    if let Some((mx, my)) = self.state.last_cursor {
                        let w = render.surface_config.width as f32;
                        let h = render.surface_config.height as f32;
                        let aspect = w / h.max(1.0);
                        let vp = self.state.session.camera().view_proj(aspect);
                        let (origin, dir) = mouse_to_ray((mx, my), (w, h), vp);
                        if let Some((wx, wz)) =
                            self.state.session.world.raycast_column(origin, dir, 4096.0)
                        {
                            let gen_arc = self.state.session.generator.clone();
                            self.state.session.probe.pin(&gen_arc, wx, wz);
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amt = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 4.0,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.5,
                };
                if matches!(self.state.session.cam_kind, CamKind::Orbit) {
                    self.state.session.orbit.distance =
                        (self.state.session.orbit.distance - amt).clamp(50.0, 768.0);
                } else {
                    self.state.session.fly.speed =
                        (self.state.session.fly.speed + amt).clamp(2.0, 200.0);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.keys.w = pressed,
                        KeyCode::KeyA => self.keys.a = pressed,
                        KeyCode::KeyS => self.keys.s = pressed,
                        KeyCode::KeyD => self.keys.d = pressed,
                        KeyCode::KeyQ => self.keys.q = pressed,
                        KeyCode::KeyE => self.keys.e = pressed,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.shift = pressed,
                        KeyCode::KeyO if pressed => self.state.session.toggle_camera(),
                        KeyCode::KeyR if pressed => self.state.session.invalidator.bump(),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let dt = self.state.dt();

                // Camera input (fly cam only).
                if matches!(self.state.session.cam_kind, CamKind::Fly) {
                    let fwd = (self.keys.w as i32 - self.keys.s as i32) as f32;
                    let strafe = (self.keys.d as i32 - self.keys.a as i32) as f32;
                    let vert = (self.keys.e as i32 - self.keys.q as i32) as f32;
                    self.state
                        .session
                        .fly
                        .translate(fwd, strafe, vert, self.keys.shift, dt);
                }

                // Stream + invalidate.
                let t0 = Instant::now();
                if self.state.session.invalidator.take_pending() {
                    self.state.session.world.wipe();
                    self.state.session.probe.refresh(&self.state.session.generator);
                }
                // Every frame: re-request every chunk currently on the
                // GPU. Once they're cached the call is a no-op; right
                // after a wipe, this is the only path that actually
                // refills chunks outside the streaming radius (e.g.
                // orbit cam zoomed out, or chunks meshed at earlier
                // camera positions). max_in_flight bounds how many
                // spawn per frame — the rest queue for next frame.
                let visible = scene.chunk_coords();
                self.state.session.world.request_chunks(&visible);
                let cam_pos = self.state.session.camera().position();
                self.state.session.world.request_around(cam_pos);
                let landed = self.state.session.world.drain_results();
                for (coord, mesh) in landed {
                    scene.upload_chunk(&render.device, coord, &mesh);
                }
                if t0.elapsed().as_secs_f32() > 0.001 {
                    self.state.last_regen_ms = Some(t0.elapsed().as_secs_f32() * 1000.0);
                }

                // egui pass.
                let raw_input = render.egui_state.take_egui_input(window);
                let mut reset_cam = false;
                let mut force_regen = false;
                let app_ref = &mut self.state;
                let full_output = render.egui_ctx.clone().run(raw_input, |ctx| {
                    let r = crate::layout::dashboard(ctx, app_ref);
                    reset_cam = r.reset_camera;
                    force_regen = r.force_regen;
                });
                if reset_cam {
                    self.state.session.fly = crate::camera::FlyCamera::new();
                    self.state.session.orbit = crate::camera::OrbitCamera::new();
                }
                if force_regen {
                    self.state.session.invalidator.bump();
                }
                render
                    .egui_state
                    .handle_platform_output(window, full_output.platform_output.clone());

                // Camera uniform + frame composition.
                let aspect = render.surface_config.width as f32
                    / render.surface_config.height.max(1) as f32;
                scene.update_camera(&render.queue, self.state.session.camera(), aspect);
                if let Err(e) = render_frame(render, scene, full_output) {
                    eprintln!("render: {e:?}");
                }

                if self.quit_after_render {
                    event_loop.exit();
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Inverse-project a mouse click into a world-space ray. NDC space is
/// (-1, -1) at bottom-left, (+1, +1) at top-right; mouse pixels are
/// (0, 0) at top-left with Y growing downward — hence the Y flip.
fn mouse_to_ray(
    mouse: (f64, f64),
    screen: (f32, f32),
    view_proj: glam::Mat4,
) -> (glam::Vec3, glam::Vec3) {
    let nx = (mouse.0 as f32 / screen.0).clamp(0.0, 1.0) * 2.0 - 1.0;
    let ny = 1.0 - (mouse.1 as f32 / screen.1).clamp(0.0, 1.0) * 2.0;
    let inv = view_proj.inverse();
    let near = inv.project_point3(glam::Vec3::new(nx, ny, 0.0));
    let far = inv.project_point3(glam::Vec3::new(nx, ny, 1.0));
    (near, (far - near).normalize_or_zero())
}

fn render_frame(
    render: &mut RenderState,
    scene: &SceneRenderer,
    full_output: egui::FullOutput,
) -> Result<(), wgpu::SurfaceError> {
    let frame = render.surface.get_current_texture()?;
    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = render.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("viz frame"),
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.05,
                        b: 0.08,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: scene.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        scene.render(&mut pass);
    }

    let paint_jobs = render
        .egui_ctx
        .tessellate(full_output.shapes, full_output.pixels_per_point);
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [render.surface_config.width, render.surface_config.height],
        pixels_per_point: full_output.pixels_per_point,
    };
    for (id, image_delta) in &full_output.textures_delta.set {
        render.egui_renderer.update_texture(&render.device, &render.queue, *id, image_delta);
    }
    render.egui_renderer.update_buffers(&render.device, &render.queue, &mut encoder, &paint_jobs, &screen);
    {
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime();
        render.egui_renderer.render(&mut pass, &paint_jobs, &screen);
    }

    render.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
    for id in &full_output.textures_delta.free {
        render.egui_renderer.free_texture(id);
    }
    Ok(())
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = VizApp::new(cli);
    event_loop.run_app(&mut app).expect("run loop");
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn check_flag_parses() {
        let cli = Cli::parse_from(["worldgen_viz", "--check"]);
        assert!(cli.check);
        assert_eq!(cli.seed, 42);
    }

    #[test]
    fn seed_flag_parses() {
        let cli = Cli::parse_from(["worldgen_viz", "--seed", "1337"]);
        assert_eq!(cli.seed, 1337);
        assert!(!cli.check);
    }
}
