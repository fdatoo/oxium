//! Worldgen tuning visualizer.

mod render;

use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::render::RenderState;

struct App {
    window: Option<Arc<Window>>,
    render: Option<RenderState>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            render: None,
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
        self.window = Some(window);
        self.render = Some(render);
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
            }
            WindowEvent::RedrawRequested => {
                let raw_input = render.egui_state.take_egui_input(window);
                let full_output = render.egui_ctx.clone().run(raw_input, |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        ui.heading("Oxium worldgen visualizer");
                        ui.label("Hello, world. Sliders + mesh viewport coming in later tasks.");
                    });
                });
                render
                    .egui_state
                    .handle_platform_output(window, full_output.platform_output.clone());
                if let Err(e) = render_egui_frame(render, full_output) {
                    eprintln!("render error: {:?}", e);
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn render_egui_frame(
    render: &mut RenderState,
    full_output: egui::FullOutput,
) -> Result<(), wgpu::SurfaceError> {
    let frame = render.surface.get_current_texture()?;
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = render
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viz encoder"),
        });

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
                label: Some("viz pass"),
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
