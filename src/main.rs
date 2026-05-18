//! oxium — entry point.
//!
//! The binary owns the [`winit`] event loop. `winit` 0.30 uses a callback-style
//! API where the loop drives an [`ApplicationHandler`] implementor. For now this
//! implementor just opens a window and idles; subsequent milestones layer on
//! GPU initialization, ECS scheduling, and rendering.

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

/// Minimal handler that owns the OS window. Replaced in M2 by `AppState`.
struct App {
    /// The OS window. `Option` because winit 0.30 only hands it to us once the
    /// event loop reaches the `Resumed` lifecycle event.
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Resumed fires once on startup (and again after iconification on
        // some platforms). It is the only place to call `create_window`.
        let attrs = WindowAttributes::default().with_title("oxium");
        self.window = Some(event_loop.create_window(attrs).unwrap());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                // Keep the window asking for redraws so it doesn't stall when
                // we have no animation yet. Real rendering shows up in M1.
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    // Poll = run the loop as fast as possible (game loop). The alternative,
    // `Wait`, blocks until an OS event arrives — fine for a desktop app, but
    // wrong for a game that wants to render continuously.
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { window: None };
    event_loop.run_app(&mut app).unwrap();
}
