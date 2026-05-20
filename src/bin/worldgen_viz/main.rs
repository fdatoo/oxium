//! Worldgen tuning visualizer.

mod cross;
mod render;
mod scene;
mod spline_widget;
mod ui;
mod worldgen_bridge;

use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::render::RenderState;

struct App {
    window: Option<Arc<Window>>,
    render: Option<RenderState>,
    scene: Option<scene::SceneRender>,
    camera: scene::OrbitCamera,
    config: oxium::worldgen::config::WorldgenConfig,
    dirty: bool,
    mouse_down: bool,
    last_cursor: Option<(f64, f64)>,
    cross_texture: Option<egui::TextureHandle>,
    last_regen_ms: Option<f32>,
    auto_regen: bool,
}

impl App {
    fn new() -> Self {
        let config = oxium::worldgen::config::WorldgenConfig::bundled_default()
            .expect("bundled default.ron must parse");
        Self {
            window: None,
            render: None,
            scene: None,
            camera: scene::OrbitCamera::new(),
            config,
            dirty: true,
            mouse_down: false,
            last_cursor: None,
            cross_texture: None,
            last_regen_ms: None,
            auto_regen: true,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Oxium worldgen visualizer")
            .with_inner_size(winit::dpi::LogicalSize::new(1400.0, 900.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let render = RenderState::new(window.clone());
        let scene = scene::SceneRender::new(
            &render.device,
            render.surface_config.format,
            render.surface_config.width,
            render.surface_config.height,
        );
        self.window = Some(window);
        self.render = Some(render);
        self.scene = Some(scene);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let (Some(window), Some(render)) = (self.window.as_ref(), self.render.as_mut()) else {
            return;
        };
        let _ = render.egui_state.on_window_event(window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                render.resize(size.width, size.height);
                if let Some(s) = self.scene.as_mut() {
                    s.resize(&render.device, size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_down {
                    if let Some((px, py)) = self.last_cursor {
                        let dx = (position.x - px) as f32;
                        let dy = (position.y - py) as f32;
                        self.camera.yaw -= dx * 0.005;
                        self.camera.pitch = (self.camera.pitch + dy * 0.005).clamp(-1.5, 1.5);
                    }
                }
                self.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Right {
                    self.mouse_down = state == winit::event::ElementState::Pressed;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amt = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y * 4.0,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.5,
                };
                self.camera.distance = (self.camera.distance - amt).clamp(8.0, 512.0);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.state == winit::event::ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        match code {
                            KeyCode::KeyR => self.dirty = true,
                            KeyCode::Space => self.auto_regen = !self.auto_regen,
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let raw_input = render.egui_state.take_egui_input(window);
                let mut dirty_local = self.dirty;
                let cfg = &mut self.config;
                let cam_info = (
                    self.camera.yaw,
                    self.camera.pitch,
                    self.camera.distance,
                );
                // Regenerate top-down texture before egui frame (so we
                // have a TextureHandle to display).
                if self.dirty || self.cross_texture.is_none() {
                    let img = crate::cross::render_topdown(42, cfg);
                    let tex = render.egui_ctx.load_texture(
                        "cross_topdown",
                        img,
                        egui::TextureOptions::LINEAR,
                    );
                    self.cross_texture = Some(tex);
                }
                let cross_tex = self.cross_texture.clone();
                let last_regen_ms = self.last_regen_ms;
                let mut auto_regen_local = self.auto_regen;
                let mut force_regen = false;
                let full_output = render.egui_ctx.clone().run(raw_input, |ctx| {
                    egui::SidePanel::left("config_panel")
                        .resizable(true)
                        .default_width(360.0)
                        .show(ctx, |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                if crate::ui::density_panel(ui, &mut cfg.density) {
                                    dirty_local = true;
                                }
                                ui.separator();
                                if crate::ui::preset_panel(ui, cfg) {
                                    dirty_local = true;
                                }
                            });
                        });
                    egui::SidePanel::right("cross_panel")
                        .resizable(true)
                        .default_width(420.0)
                        .show(ctx, |ui| {
                            ui.heading("Top-down (h_target)");
                            if let Some(tex) = cross_tex.as_ref() {
                                ui.image((tex.id(), egui::vec2(400.0, 400.0)));
                            }
                            ui.label("256×256 px, 4 blocks/px → 1024 blocks/side");
                        });
                    egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!(
                                "yaw {:.2} pitch {:.2} dist {:.0}",
                                cam_info.0, cam_info.1, cam_info.2
                            ));
                            ui.separator();
                            if let Some(ms) = last_regen_ms {
                                ui.label(format!("last regen: {:.1} ms", ms));
                            }
                            ui.separator();
                            ui.label(if auto_regen_local {
                                "auto-regen: ON"
                            } else {
                                "auto-regen: OFF (press R to regen)"
                            });
                            ui.separator();
                            if ui.button("[R] Regen").clicked() {
                                force_regen = true;
                            }
                            if ui
                                .button(if auto_regen_local { "Pause" } else { "Resume" })
                                .clicked()
                            {
                                auto_regen_local = !auto_regen_local;
                            }
                        });
                    });
                    egui::CentralPanel::default().show(ctx, |ui| {
                        ui.label(format!(
                            "yaw {:.2} pitch {:.2} dist {:.0} | dirty: {}",
                            cam_info.0, cam_info.1, cam_info.2, dirty_local
                        ));
                    });
                });
                self.dirty = dirty_local || force_regen;
                self.auto_regen = auto_regen_local;
                render
                    .egui_state
                    .handle_platform_output(window, full_output.platform_output.clone());

                // Regen mesh if config is dirty AND auto_regen is on
                // (or the user just pressed [R] / clicked Regen).
                if self.dirty && self.auto_regen {
                    let t0 = std::time::Instant::now();
                    let (verts, idxs) =
                        worldgen_bridge::regen_region_mesh(42, &self.config);
                    if let Some(scene) = self.scene.as_mut() {
                        scene.upload_mesh(&render.device, &verts, &idxs);
                    }
                    let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
                    self.last_regen_ms = Some(elapsed_ms);
                    eprintln!(
                        "viz regen: {:.1} ms ({} verts, {} idxs)",
                        elapsed_ms,
                        verts.len(),
                        idxs.len()
                    );
                    self.dirty = false;
                }

                let scene_ref = self.scene.as_ref().expect("scene");
                let aspect = render.surface_config.width as f32
                    / render.surface_config.height.max(1) as f32;
                scene_ref.update_camera(&render.queue, &self.camera, aspect);
                if let Err(e) = render_frame(render, scene_ref, full_output) {
                    eprintln!("render: {:?}", e);
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn render_frame(
    render: &mut RenderState,
    scene: &scene::SceneRender,
    full_output: egui::FullOutput,
) -> Result<(), wgpu::SurfaceError> {
    let frame = render.surface.get_current_texture()?;
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = render
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viz frame"),
        });

    // 3D scene pass
    {
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
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
                    view: &scene.depth_view,
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

    // egui pass (overlay)
    let paint_jobs = render
        .egui_ctx
        .tessellate(full_output.shapes, full_output.pixels_per_point);
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [render.surface_config.width, render.surface_config.height],
        pixels_per_point: full_output.pixels_per_point,
    };
    for (id, image_delta) in &full_output.textures_delta.set {
        render
            .egui_renderer
            .update_texture(&render.device, &render.queue, *id, image_delta);
    }
    render.egui_renderer.update_buffers(
        &render.device,
        &render.queue,
        &mut encoder,
        &paint_jobs,
        &screen,
    );
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
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run loop");
}
