# Worldgen Visualizer — V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone binary (`cargo run --bin worldgen_viz`) that opens a window with an `egui` sidebar exposing every `WorldgenConfig` field as a live slider. Edits regenerate a small worldgen region (4 chunks) and re-mesh in real time, showing the result in a 3D viewport. Includes a 2D cross-section view and a custom spline-plot widget for tuning Hermite knots visually. Builds the user's geometric intuition for the migration.

**Architecture:** New `[[bin]]` target at `src/bin/worldgen_viz/main.rs`. Reuses `oxium::worldgen::*` (Generator, WorldgenConfig, ConfigHolder), `oxium::mesher::*` (greedy mesher), and `oxium::voxel::*` types. Uses `egui` + `egui-wgpu` + `egui-winit` for the UI layer on top of a fresh `wgpu` device (does NOT reuse the game's full PBR renderer — the visualizer needs only a debug shader for solid-colored meshes). State is held in a `VisualizerApp` struct that owns the wgpu device, egui context, current `WorldgenConfig`, a `Generator`, a mesh cache, and an orbit camera.

**Tech Stack:**
- Rust 2024 edition
- `egui = "0.27"` — immediate-mode UI
- `egui-wgpu = "0.27"` — wgpu rendering backend for egui
- `egui-winit = "0.27"` — winit integration for egui input
- Existing: `wgpu`, `winit`, `pollster`, `glam`, `bytemuck`, `oxium::worldgen`, `oxium::mesher`

**Reference:** Design rationale in `docs/superpowers/specs/2026-05-19-minecraft-worldgen-research.md`. Discussion of the visualizer's purpose is at the end of the in-session conversation log.

**Depends on:** PR 2 — Worldgen Density Foundation. Specifically requires `WorldgenConfig`, `DensityConfig`, `ConfigHolder`, `Generator::with_config(seed, holder)`. Implement PR 2 first.

---

### Task 1: Cargo dependencies + new bin target

**Files:**
- Modify: `Cargo.toml`
- Create: `src/bin/worldgen_viz/main.rs`

- [ ] **Step 1.1: Add egui dependencies and bin target**

Edit `Cargo.toml`. Add to `[dependencies]` (place near the existing graphics group):

```toml
# Visualizer UI — used by src/bin/worldgen_viz only.
egui = "0.27"
egui-wgpu = "0.27"
egui-winit = "0.27"
```

Add at the end of the file (after `[profile.dev]`):

```toml
[[bin]]
name = "worldgen_viz"
path = "src/bin/worldgen_viz/main.rs"
```

- [ ] **Step 1.2: Create the stub binary**

Create `src/bin/worldgen_viz/main.rs` with a minimal entry that prints a message and exits:

```rust
//! Worldgen tuning visualizer — see docs/superpowers/plans/2026-05-19-worldgen-visualizer.md
//!
//! Run with `cargo run --release --bin worldgen_viz`. Opens a window
//! with a slider panel for WorldgenConfig and a live-updating 3D
//! mesh viewport showing the worldgen output.

fn main() {
    println!("worldgen_viz: stub. Window and UI in later tasks.");
}
```

- [ ] **Step 1.3: Verify compilation**

Run: `cargo build --bin worldgen_viz 2>&1 | tail -5`

Expected: `Finished` line.

Run: `cargo run --bin worldgen_viz 2>&1 | tail -3`

Expected: stub message printed.

- [ ] **Step 1.4: Commit**

```bash
git add Cargo.toml Cargo.lock src/bin/worldgen_viz/main.rs
git commit -m "$(cat <<'EOF'
build(viz): add worldgen_viz bin target + egui deps

New binary src/bin/worldgen_viz for the live-tuning visualizer.
Adds egui + egui-wgpu + egui-winit. Stub main() until task 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Window + wgpu + egui shell

**Files:**
- Create: `src/bin/worldgen_viz/render.rs`
- Modify: `src/bin/worldgen_viz/main.rs`

- [ ] **Step 2.1: Add `render.rs` with wgpu + egui setup**

Create `src/bin/worldgen_viz/render.rs`:

```rust
//! wgpu device + surface + egui rendering glue.

use egui_wgpu::ScreenDescriptor;
use std::sync::Arc;
use winit::window::Window;

pub struct RenderState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub egui_state: egui_winit::State,
    pub egui_ctx: egui::Context,
    pub egui_renderer: egui_wgpu::Renderer,
}

impl RenderState {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = unsafe {
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(
                instance.create_surface(window.as_ref()).expect("create surface"),
            )
        };
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("request adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("worldgen_viz device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )).expect("request device");

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats.iter().find(|f| f.is_srgb())
            .copied().unwrap_or(surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1);

        Self {
            device,
            queue,
            surface,
            surface_config,
            egui_state,
            egui_ctx,
            egui_renderer,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn render_egui(
        &mut self,
        window: &Window,
        full_output: egui::FullOutput,
    ) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("worldgen_viz encoder"),
        });

        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &paint_jobs, &screen);

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("worldgen_viz pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer.render(&mut pass.forget_lifetime(), &paint_jobs, &screen);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        Ok(())
    }
}
```

- [ ] **Step 2.2: Rewrite `main.rs` to open a window**

Replace `src/bin/worldgen_viz/main.rs`:

```rust
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
        Self { window: None, render: None }
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
                render.egui_state.handle_platform_output(window, full_output.platform_output.clone());
                if let Err(e) = render.render_egui(window, full_output) {
                    eprintln!("render error: {:?}", e);
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run loop");
}
```

- [ ] **Step 2.3: Run and confirm a window opens**

Run: `cargo run --release --bin worldgen_viz`

Expected: a 1400×900 dark window labeled "Oxium worldgen visualizer" with a centered hello-world panel. Close-button works. Resize-able.

If wgpu fails to initialize, check graphics driver and adapter availability (try `wgpu::Backends::PRIMARY` instead of default if needed).

- [ ] **Step 2.4: Commit**

```bash
git add src/bin/worldgen_viz/
git commit -m "$(cat <<'EOF'
feat(viz): window + wgpu + egui shell

Opens a 1400x900 window with an egui context backed by wgpu.
CentralPanel placeholder; sliders and mesh viewport in later tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: WorldgenConfig editing panel

**Files:**
- Create: `src/bin/worldgen_viz/ui.rs`
- Modify: `src/bin/worldgen_viz/main.rs`

- [ ] **Step 3.1: Create `ui.rs` with the config sidebar**

Create `src/bin/worldgen_viz/ui.rs`:

```rust
//! egui panels for editing WorldgenConfig live.

use egui::Ui;
use oxium::worldgen::config::{DensityConfig, WorldgenConfig};

/// Returns true if the user changed any value (caller marks `dirty`).
pub fn density_panel(ui: &mut Ui, cfg: &mut DensityConfig) -> bool {
    let snapshot = cfg.clone();
    ui.heading("Density");
    ui.collapsing("World range", |ui| {
        ui.add(egui::Slider::new(&mut cfg.y_min, -256..=0).text("y_min"));
        ui.add(egui::Slider::new(&mut cfg.y_max, 0..=512).text("y_max"));
        ui.add(egui::Slider::new(&mut cfg.y_gradient_amplitude, 0.1..=5.0).text("y_gradient_amplitude"));
    });
    ui.collapsing("Composition", |ui| {
        ui.add(egui::Slider::new(&mut cfg.composition_scale, 0.5..=16.0).text("composition_scale"));
        ui.add(egui::Slider::new(&mut cfg.above_surface_softening, 0.0..=1.0).text("above_surface_softening"));
        ui.add(egui::Slider::new(&mut cfg.factor, 0.1..=10.0).text("factor"));
    });
    ui.collapsing("Base 3D noise", |ui| {
        ui.add(egui::Slider::new(&mut cfg.base_3d_period, 4.0..=128.0).text("base_3d_period"));
        ui.add(egui::Slider::new(&mut cfg.base_3d_amplitude, 0.0..=4.0).text("base_3d_amplitude"));
        ui.add(egui::Slider::new(&mut cfg.base_3d_y_scale, 0.1..=2.0).text("base_3d_y_scale"));
    });
    ui.collapsing("Slides", |ui| {
        ui.add(egui::Slider::new(&mut cfg.slide_top_blocks, 0..=64).text("slide_top_blocks"));
        ui.add(egui::Slider::new(&mut cfg.slide_top_target, -1.0..=1.0).text("slide_top_target"));
        ui.add(egui::Slider::new(&mut cfg.slide_bottom_blocks, 0..=64).text("slide_bottom_blocks"));
        ui.add(egui::Slider::new(&mut cfg.slide_bottom_target, -1.0..=1.0).text("slide_bottom_target"));
    });
    // For PR 2, offset_spline is Constant(0.0) — show as read-only.
    ui.collapsing("Offset spline (PR 3)", |ui| {
        ui.label(format!("{:?}", cfg.offset_spline));
        ui.label("Spline editing widget lives in task 8.");
    });
    *cfg != snapshot
}

/// Save / load preset buttons. Returns true if the config was
/// replaced from disk (caller marks `dirty`).
pub fn preset_panel(ui: &mut Ui, cfg: &mut WorldgenConfig) -> bool {
    let mut loaded = false;
    ui.heading("Presets");
    if ui.button("Save current → assets/worldgen/scratch.ron").clicked() {
        if let Ok(s) = ron::ser::to_string_pretty(cfg, ron::ser::PrettyConfig::default()) {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("assets").join("worldgen").join("scratch.ron");
            if let Err(e) = std::fs::write(&path, s) {
                eprintln!("save preset: {e}");
            } else {
                eprintln!("saved to {path:?}");
            }
        }
    }
    if ui.button("Reload default.ron").clicked() {
        if let Ok(new) = WorldgenConfig::bundled_default() {
            *cfg = new;
            loaded = true;
        }
    }
    loaded
}
```

- [ ] **Step 3.2: Wire up the panel in `main.rs`**

In `src/bin/worldgen_viz/main.rs`, add `mod ui;` near the top and replace the `egui::CentralPanel` block with a sidebar + central:

```rust
// Add to App struct:
struct App {
    window: Option<Arc<Window>>,
    render: Option<RenderState>,
    config: oxium::worldgen::config::WorldgenConfig,
    dirty: bool,
}

impl App {
    fn new() -> Self {
        let config = oxium::worldgen::config::WorldgenConfig::bundled_default()
            .expect("bundled default.ron must parse");
        Self {
            window: None,
            render: None,
            config,
            dirty: true,
        }
    }
}

// In the RedrawRequested branch, replace the closure body:
let full_output = render.egui_ctx.clone().run(raw_input, |ctx| {
    egui::SidePanel::left("config_panel")
        .resizable(true)
        .default_width(360.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                if crate::ui::density_panel(ui, &mut self.config.density) {
                    self.dirty = true;
                }
                ui.separator();
                if crate::ui::preset_panel(ui, &mut self.config) {
                    self.dirty = true;
                }
            });
        });
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.heading("Mesh viewport");
        ui.label(if self.dirty {
            format!("dirty (regen pending; factor = {:.2})", self.config.density.factor)
        } else {
            format!("clean (factor = {:.2})", self.config.density.factor)
        });
    });
});
```

- [ ] **Step 3.3: Run and confirm sliders react**

Run: `cargo run --release --bin worldgen_viz`

Expected: a sidebar appears on the left with collapsible groups (World range, Composition, Base 3D noise, Slides, Offset spline). Sliders move. The CentralPanel label updates to show `dirty (regen pending; factor = X.XX)` whenever a slider changes.

- [ ] **Step 3.4: Commit**

```bash
git add src/bin/worldgen_viz/
git commit -m "$(cat <<'EOF'
feat(viz): WorldgenConfig slider sidebar + presets

Sidebar exposes every DensityConfig field as an egui slider.
Save/load preset buttons (writes to assets/worldgen/scratch.ron).
Dirty flag set on any change; mesh regen wires up in task 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: 3D mesh viewport with orbit camera (no worldgen yet)

**Files:**
- Create: `src/bin/worldgen_viz/scene.rs`
- Modify: `src/bin/worldgen_viz/main.rs`, `src/bin/worldgen_viz/render.rs`

- [ ] **Step 4.1: Create `scene.rs` with orbit camera and a placeholder cube mesh**

Create `src/bin/worldgen_viz/scene.rs`:

```rust
//! 3D scene state: orbit camera + a single mesh + a debug-color shader.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self {
            target: Vec3::new(0.0, 64.0, 0.0),
            yaw: 0.5,
            pitch: 0.3,
            distance: 96.0,
        }
    }
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let eye = self.target + Vec3::new(
            self.distance * self.yaw.cos() * self.pitch.cos(),
            self.distance * self.pitch.sin(),
            self.distance * self.yaw.sin() * self.pitch.cos(),
        );
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.5, 1024.0);
        proj * view
    }
}

pub struct SceneRender {
    pub pipeline: wgpu::RenderPipeline,
    pub depth_view: wgpu::TextureView,
    pub depth_format: wgpu::TextureFormat,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl SceneRender {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera ubo"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bg"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viz shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viz pl"),
            bind_group_layouts: &[&camera_layout],
            push_constant_ranges: &[],
        });
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viz pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Placeholder cube mesh (24 verts, 36 indices). Task 5 replaces this.
        let (verts, idxs) = unit_cube_mesh();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viz vbo"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viz ibo"),
            contents: bytemuck::cast_slice(&idxs),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let depth_view = create_depth(device, depth_format, width, height);

        Self {
            pipeline,
            depth_view,
            depth_format,
            camera_buffer,
            camera_bind_group,
            vertex_buffer,
            index_buffer,
            index_count: idxs.len() as u32,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth_view = create_depth(device, self.depth_format, width, height);
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, camera: &OrbitCamera, aspect: f32) {
        let vp = camera.view_proj(aspect);
        let u = CameraUniform { view_proj: vp.to_cols_array_2d() };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[u]));
    }

    pub fn upload_mesh(&mut self, device: &wgpu::Device, verts: &[Vertex], idxs: &[u32]) {
        self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viz vbo"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        self.index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viz ibo"),
            contents: bytemuck::cast_slice(idxs),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        self.index_count = idxs.len() as u32;
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

fn create_depth(device: &wgpu::Device, format: wgpu::TextureFormat, w: u32, h: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viz depth"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

const SHADER: &str = r#"
struct Camera { view_proj: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> cam: Camera;

struct VsIn { @location(0) pos: vec3<f32>, @location(1) color: vec3<f32>, };
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec3<f32>, };

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.pos = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

fn unit_cube_mesh() -> (Vec<Vertex>, Vec<u32>) {
    // 8 corners of a [0,32]^3 cube at world origin, faces colored
    // by normal direction for visual confirmation.
    let s = 32.0;
    let v = |p: [f32; 3], c: [f32; 3]| Vertex { position: p, color: c };
    let verts = vec![
        v([0.,0.,0.], [0.4,0.4,0.4]), v([s,0.,0.], [0.4,0.4,0.4]),
        v([s,0.,s], [0.4,0.4,0.4]), v([0.,0.,s], [0.4,0.4,0.4]),
        v([0.,s,0.], [0.8,0.8,0.8]), v([s,s,0.], [0.8,0.8,0.8]),
        v([s,s,s], [0.8,0.8,0.8]), v([0.,s,s], [0.8,0.8,0.8]),
    ];
    let idxs = vec![
        0,2,1, 0,3,2, // bottom
        4,5,6, 4,6,7, // top
        0,1,5, 0,5,4, // -z
        2,3,7, 2,7,6, // +z
        1,2,6, 1,6,5, // +x
        3,0,4, 3,4,7, // -x
    ];
    (verts, idxs)
}
```

- [ ] **Step 4.2: Wire scene into render loop**

In `src/bin/worldgen_viz/main.rs`, add `mod scene;` and integrate. Replace the App struct + RedrawRequested branch:

```rust
struct App {
    window: Option<Arc<Window>>,
    render: Option<RenderState>,
    scene: Option<scene::SceneRender>,
    camera: scene::OrbitCamera,
    config: oxium::worldgen::config::WorldgenConfig,
    dirty: bool,
    mouse_down: bool,
    last_cursor: Option<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            render: None,
            scene: None,
            camera: scene::OrbitCamera::new(),
            config: oxium::worldgen::config::WorldgenConfig::bundled_default()
                .expect("bundled default.ron must parse"),
            dirty: true,
            mouse_down: false,
            last_cursor: None,
        }
    }
}
```

In `resumed`, after creating `RenderState`, also build the scene:

```rust
let scene = scene::SceneRender::new(
    &render.device,
    render.surface_config.format,
    render.surface_config.width,
    render.surface_config.height,
);
self.scene = Some(scene);
```

In `WindowEvent::Resized`, also call `scene.resize(&render.device, w, h)`.

In `RedrawRequested`, render the scene BEFORE egui (so egui draws on top):

```rust
WindowEvent::RedrawRequested => {
    // 1. Tick UI.
    let raw_input = render.egui_state.take_egui_input(window);
    let full_output = render.egui_ctx.clone().run(raw_input, |ctx| {
        egui::SidePanel::left("config_panel")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if crate::ui::density_panel(ui, &mut self.config.density) {
                        self.dirty = true;
                    }
                    ui.separator();
                    if crate::ui::preset_panel(ui, &mut self.config) {
                        self.dirty = true;
                    }
                });
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(format!(
                "yaw {:.2} pitch {:.2} distance {:.1} | dirty: {}",
                self.camera.yaw, self.camera.pitch, self.camera.distance, self.dirty
            ));
        });
    });
    render.egui_state.handle_platform_output(window, full_output.platform_output.clone());

    // 2. Render scene + egui.
    let scene_ref = self.scene.as_ref().expect("scene");
    let aspect = render.surface_config.width as f32 / render.surface_config.height.max(1) as f32;
    scene_ref.update_camera(&render.queue, &self.camera, aspect);
    if let Err(e) = render_frame(render, scene_ref, full_output) {
        eprintln!("render: {:?}", e);
    }
    window.request_redraw();
}
```

Add a `render_frame` free function (in main.rs or render.rs — your choice; the depth attachment + scene pass + egui pass go here):

```rust
fn render_frame(
    render: &mut RenderState,
    scene: &scene::SceneRender,
    full_output: egui::FullOutput,
) -> Result<(), wgpu::SurfaceError> {
    let frame = render.surface.get_current_texture()?;
    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = render.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("viz frame"),
    });

    // 3D scene pass
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 }),
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
    let paint_jobs = render.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [render.surface_config.width, render.surface_config.height],
        pixels_per_point: full_output.pixels_per_point,
    };
    for (id, image_delta) in &full_output.textures_delta.set {
        render.egui_renderer.update_texture(&render.device, &render.queue, *id, image_delta);
    }
    render.egui_renderer.update_buffers(&render.device, &render.queue, &mut encoder, &paint_jobs, &screen);
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        });
        render.egui_renderer.render(&mut pass.forget_lifetime(), &paint_jobs, &screen);
    }

    render.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
    for id in &full_output.textures_delta.free {
        render.egui_renderer.free_texture(id);
    }
    Ok(())
}
```

Add `mod scene;` near the top of main.rs.

- [ ] **Step 4.3: Wire camera input**

Add to `WindowEvent` matching (above the existing arms):

```rust
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
```

- [ ] **Step 4.4: Run and confirm camera works**

Run: `cargo run --release --bin worldgen_viz`

Expected:
- A grey cube at world origin (32×32×32).
- Right-click drag rotates the camera.
- Mouse wheel zooms.

- [ ] **Step 4.5: Commit**

```bash
git add src/bin/worldgen_viz/
git commit -m "$(cat <<'EOF'
feat(viz): 3D viewport with orbit camera + placeholder cube

scene.rs owns the wgpu render pipeline, depth attachment, camera
uniform, and a debug shader (per-vertex color, no lighting).
Camera: right-click drag yaw/pitch, mouse wheel zoom.

Renders a 32x32x32 cube at the world origin as a placeholder.
Task 5 replaces this with a meshed chunk-region from the worldgen.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Worldgen → mesh → viewport pipeline

**Files:**
- Create: `src/bin/worldgen_viz/worldgen_bridge.rs`
- Modify: `src/bin/worldgen_viz/main.rs`

- [ ] **Step 5.1: Explore the existing mesher API**

Run: `grep -rn "pub fn" src/mesher/ --include='*.rs' | head -10`

Expected: function signatures for the chunk meshing entry point. Note the function name and signature.

If the mesher's output type is `Vec<MesherVertex>` or similar, adapt the bridge code below to call it. The shape is: take a `&DenseChunk` (and possibly neighbors for face culling), return vertex + index buffers.

- [ ] **Step 5.2: Create the bridge**

Create `src/bin/worldgen_viz/worldgen_bridge.rs`:

```rust
//! Worldgen → mesh adapter for the visualizer.
//!
//! Regenerates a small fixed-size chunk region from a WorldgenConfig
//! and meshes it. Returns vertex + index buffers in the visualizer's
//! `scene::Vertex` format.

use crate::scene::Vertex;
use glam::Vec3;
use oxium::voxel::{Block, DenseChunk};
use oxium::voxel::coords::{ChunkCoord, CHUNK_DIM_U};
use oxium::worldgen::config::{ConfigHolder, WorldgenConfig};
use oxium::worldgen::Generator;

/// Visualizer-fixed region: a 2×2 grid of vertical chunk-stacks,
/// 4 chunks tall each. Total: 4 columns × 4 levels = 16 chunks.
/// At 32^3 voxels per chunk: ~500k voxels — meshes in ~30-50ms.
pub const REGION_CHUNKS_X: i32 = 2;
pub const REGION_CHUNKS_Z: i32 = 2;
pub const REGION_CHUNKS_Y_MIN: i32 = -2;
pub const REGION_CHUNKS_Y_MAX: i32 = 2; // exclusive

/// Generate the region and return mesh vertices + indices for the
/// visualizer's debug shader.
pub fn regen_region_mesh(
    seed: u64,
    config: &WorldgenConfig,
) -> (Vec<Vertex>, Vec<u32>) {
    let holder = ConfigHolder::new(config.clone());
    let gen = Generator::with_config(seed, holder);

    let mut verts = Vec::with_capacity(4096);
    let mut idxs = Vec::with_capacity(6144);

    for cy in REGION_CHUNKS_Y_MIN..REGION_CHUNKS_Y_MAX {
        for cz in 0..REGION_CHUNKS_Z {
            for cx in 0..REGION_CHUNKS_X {
                let mut chunk = DenseChunk::empty();
                let coord = ChunkCoord(glam::IVec3::new(cx, cy, cz));
                gen.fill_chunk(coord, &mut chunk);
                emit_chunk_faces(&chunk, cx, cy, cz, &mut verts, &mut idxs);
            }
        }
    }
    (verts, idxs)
}

/// Minimal naive face mesher: emit a face only when the block is
/// solid and the neighbor in that direction is not solid. Within-chunk
/// only — no neighbor lookups across chunk boundaries (the seams will
/// have over-meshing; acceptable for visualizer V1).
fn emit_chunk_faces(
    chunk: &DenseChunk,
    cx: i32,
    cy: i32,
    cz: i32,
    verts: &mut Vec<Vertex>,
    idxs: &mut Vec<u32>,
) {
    let dim = CHUNK_DIM_U as i32;
    let origin = Vec3::new(
        (cx * dim) as f32,
        (cy * dim) as f32,
        (cz * dim) as f32,
    );
    for lz in 0..dim {
        for ly in 0..dim {
            for lx in 0..dim {
                let local = oxium::voxel::coords::LocalPos(glam::UVec3::new(
                    lx as u32, ly as u32, lz as u32,
                ));
                let block = chunk.blocks[local.to_index()];
                if !is_solid(block) {
                    continue;
                }
                let p = origin + Vec3::new(lx as f32, ly as f32, lz as f32);
                let color = color_for(block, ly + cy * dim);
                // Six face directions; emit if neighbor is non-solid.
                for (nx, ny, nz, face) in &[
                    (1i32, 0, 0, Face::PosX),
                    (-1, 0, 0, Face::NegX),
                    (0, 1, 0, Face::PosY),
                    (0, -1, 0, Face::NegY),
                    (0, 0, 1, Face::PosZ),
                    (0, 0, -1, Face::NegZ),
                ] {
                    let nlx = lx + nx;
                    let nly = ly + ny;
                    let nlz = lz + nz;
                    let neighbor_solid = if nlx < 0 || nlx >= dim
                        || nly < 0 || nly >= dim
                        || nlz < 0 || nlz >= dim
                    {
                        false // chunk-edge: render as if neighbor empty
                    } else {
                        let nl = oxium::voxel::coords::LocalPos(glam::UVec3::new(
                            nlx as u32, nly as u32, nlz as u32,
                        ));
                        is_solid(chunk.blocks[nl.to_index()])
                    };
                    if !neighbor_solid {
                        emit_face(p, color, *face, verts, idxs);
                    }
                }
            }
        }
    }
}

#[derive(Copy, Clone)]
enum Face { PosX, NegX, PosY, NegY, PosZ, NegZ }

fn emit_face(p: Vec3, color: [f32; 3], face: Face, verts: &mut Vec<Vertex>, idxs: &mut Vec<u32>) {
    let base = verts.len() as u32;
    let quad: [Vec3; 4] = match face {
        Face::PosX => [Vec3::new(1.,0.,0.), Vec3::new(1.,0.,1.), Vec3::new(1.,1.,1.), Vec3::new(1.,1.,0.)],
        Face::NegX => [Vec3::new(0.,0.,1.), Vec3::new(0.,0.,0.), Vec3::new(0.,1.,0.), Vec3::new(0.,1.,1.)],
        Face::PosY => [Vec3::new(0.,1.,0.), Vec3::new(1.,1.,0.), Vec3::new(1.,1.,1.), Vec3::new(0.,1.,1.)],
        Face::NegY => [Vec3::new(0.,0.,1.), Vec3::new(1.,0.,1.), Vec3::new(1.,0.,0.), Vec3::new(0.,0.,0.)],
        Face::PosZ => [Vec3::new(1.,0.,1.), Vec3::new(0.,0.,1.), Vec3::new(0.,1.,1.), Vec3::new(1.,1.,1.)],
        Face::NegZ => [Vec3::new(0.,0.,0.), Vec3::new(1.,0.,0.), Vec3::new(1.,1.,0.), Vec3::new(0.,1.,0.)],
    };
    for v in &quad {
        verts.push(Vertex { position: (p + *v).into(), color });
    }
    idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn is_solid(b: Block) -> bool {
    !matches!(b, Block::Air | Block::Water)
}

fn color_for(b: Block, _y: i32) -> [f32; 3] {
    match b {
        Block::Stone => [0.55, 0.55, 0.55],
        Block::Dirt => [0.50, 0.32, 0.18],
        Block::Grass => [0.30, 0.65, 0.25],
        Block::Sand => [0.92, 0.85, 0.62],
        Block::Snow => [0.95, 0.95, 0.97],
        _ => [0.4, 0.4, 0.4],
    }
}
```

- [ ] **Step 5.3: Hook regen into the dirty-flag pipeline**

In `main.rs`, add `mod worldgen_bridge;`. In the App's resumed handler (after the scene is built), trigger an initial regen by uploading the mesh. In each frame after the egui pass, if `self.dirty` is true, regen and upload:

```rust
// After the egui closure, before render_frame:
if self.dirty {
    let (verts, idxs) = worldgen_bridge::regen_region_mesh(42, &self.config);
    if let Some(scene) = self.scene.as_mut() {
        scene.upload_mesh(&render.device, &verts, &idxs);
    }
    self.dirty = false;
}
```

Adjust the camera target to be near the chunk region center:

```rust
// In OrbitCamera::new, change target to:
target: Vec3::new(32.0, 60.0, 32.0),
distance: 128.0,
```

- [ ] **Step 5.4: Run and confirm**

Run: `cargo run --release --bin worldgen_viz`

Expected:
- A visible terrain region (16 chunks worth) in the viewport. Stone, dirt, grass colors. Surface around y=60-80.
- Drag a slider (e.g., `factor`). The mesh regenerates within a second or two. Visible change in surface sharpness.
- Camera works.

If regen feels slow, profile with `cargo run --release` (NOT debug). The 16-chunk region should mesh in well under 1s.

- [ ] **Step 5.5: Commit**

```bash
git add src/bin/worldgen_viz/
git commit -m "$(cat <<'EOF'
feat(viz): worldgen-driven mesh viewport (live regen)

worldgen_bridge::regen_region_mesh runs a fresh Generator with the
current WorldgenConfig over a 2x2x4 chunk region and emits a naive
per-face mesh into the viewport. Triggered automatically when any
slider moves (dirty flag).

Naive mesher (one quad per exposed face) instead of greedy meshing
for V1 — simpler, no neighbor lookups across chunk seams.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: 2D cross-section viewer (top-down map)

**Files:**
- Create: `src/bin/worldgen_viz/cross.rs`
- Modify: `src/bin/worldgen_viz/main.rs`

- [ ] **Step 6.1: Create `cross.rs` — generates an `egui::ColorImage` from a config**

Create `src/bin/worldgen_viz/cross.rs`:

```rust
//! 2D cross-section: top-down map of column heights from the
//! visualizer's current WorldgenConfig.

use oxium::worldgen::Generator;
use oxium::worldgen::config::{ConfigHolder, WorldgenConfig};

pub const CROSS_RES: usize = 256;
/// Each pixel = 4 blocks. 256 × 4 = 1024 blocks per side.
pub const CROSS_BLOCKS_PER_PX: i32 = 4;

/// Render a top-down heightmap of `col.height` over a 1024×1024 block region.
pub fn render_topdown(seed: u64, config: &WorldgenConfig) -> egui::ColorImage {
    let holder = ConfigHolder::new(config.clone());
    let gen = Generator::with_config(seed, holder);
    let mut pixels = vec![egui::Color32::BLACK; CROSS_RES * CROSS_RES];
    for iz in 0..CROSS_RES {
        for ix in 0..CROSS_RES {
            let wx = (ix as i32 - CROSS_RES as i32 / 2) * CROSS_BLOCKS_PER_PX;
            let wz = (iz as i32 - CROSS_RES as i32 / 2) * CROSS_BLOCKS_PER_PX;
            let col = gen.column_data(wx, wz);
            let h = col.height as f32;
            let t = ((h + 50.0) / 200.0).clamp(0.0, 1.0);
            let g = (t * 255.0) as u8;
            pixels[iz * CROSS_RES + ix] = if h <= 62 {
                egui::Color32::from_rgb(20, 40, g.max(40))
            } else {
                egui::Color32::from_rgb(g, g, g)
            };
        }
    }
    egui::ColorImage {
        size: [CROSS_RES, CROSS_RES],
        pixels,
    }
}
```

- [ ] **Step 6.2: Wire into the egui closure**

In `main.rs`, after the side panel, add a right-side panel showing the cross-section. Cache the texture handle across frames; regenerate when `dirty`:

```rust
// Add to App struct:
cross_texture: Option<egui::TextureHandle>,

// In App::new(), initialize to None.

// In the egui closure, AFTER the side panel:
let cross_dirty_local = self.dirty;
egui::SidePanel::right("cross_panel")
    .resizable(true)
    .default_width(420.0)
    .show(ctx, |ui| {
        ui.heading("Top-down (h_target)");
        if cross_dirty_local || self.cross_texture.is_none() {
            let img = crate::cross::render_topdown(42, &self.config);
            let tex = ctx.load_texture("cross_topdown", img, egui::TextureOptions::LINEAR);
            self.cross_texture = Some(tex);
        }
        if let Some(tex) = self.cross_texture.as_ref() {
            ui.image((tex.id(), egui::vec2(400.0, 400.0)));
        }
        ui.label("256×256 px, 4 blocks/px → 1024 blocks per side");
    });
```

- [ ] **Step 6.3: Run and verify**

Run: `cargo run --release --bin worldgen_viz`

Expected: a right sidebar shows a 400×400 grayscale heightmap of a 1024×1024 block region. Oceans dark blue, land lighter. Drag a slider that affects heightmap (e.g., once PR 3 lands, `offset_spline` knots) — the map updates.

- [ ] **Step 6.4: Commit**

```bash
git add src/bin/worldgen_viz/
git commit -m "$(cat <<'EOF'
feat(viz): 2D top-down heightmap cross-section

cross::render_topdown samples col.height over a 1024x1024 block
region at 4 blocks/pixel and renders it as a grayscale egui image.
Right side panel. Regens on dirty flag like the 3D viewport.

Lets you see large-scale continent/biome boundaries at a glance,
which the 3D viewport's 64-block window can't show.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Spline plot widget (custom egui widget)

**Files:**
- Create: `src/bin/worldgen_viz/spline_widget.rs`
- Modify: `src/bin/worldgen_viz/ui.rs`

- [ ] **Step 7.1: Create the widget**

Create `src/bin/worldgen_viz/spline_widget.rs`:

```rust
//! Custom egui widget that draws a CubicSpline curve and lets the
//! user drag knots with the mouse.

use egui::{Color32, Pos2, Sense, Stroke, Vec2, Widget};
use oxium::worldgen::spline::{CubicSpline, Knot};

pub struct SplineEditor<'a> {
    spline: &'a mut CubicSpline,
    x_range: (f32, f32),
    y_range: (f32, f32),
    desired_size: Vec2,
}

impl<'a> SplineEditor<'a> {
    pub fn new(spline: &'a mut CubicSpline) -> Self {
        Self {
            spline,
            x_range: (-1.1, 1.1),
            y_range: (-1.5, 1.5),
            desired_size: Vec2::new(380.0, 220.0),
        }
    }
    pub fn x_range(mut self, lo: f32, hi: f32) -> Self {
        self.x_range = (lo, hi); self
    }
    pub fn y_range(mut self, lo: f32, hi: f32) -> Self {
        self.y_range = (lo, hi); self
    }
}

impl<'a> Widget for SplineEditor<'a> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let (rect, response) = ui.allocate_exact_size(self.desired_size, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let to_screen = |loc: f32, val: f32| -> Pos2 {
            let nx = (loc - self.x_range.0) / (self.x_range.1 - self.x_range.0);
            let ny = (val - self.y_range.0) / (self.y_range.1 - self.y_range.0);
            Pos2::new(
                rect.min.x + nx * rect.width(),
                rect.max.y - ny * rect.height(),
            )
        };
        let from_screen = |p: Pos2| -> (f32, f32) {
            let nx = (p.x - rect.min.x) / rect.width();
            let ny = (rect.max.y - p.y) / rect.height();
            (
                self.x_range.0 + nx * (self.x_range.1 - self.x_range.0),
                self.y_range.0 + ny * (self.y_range.1 - self.y_range.0),
            )
        };

        // Background grid.
        painter.rect_filled(rect, 4.0, Color32::from_rgb(20, 20, 26));
        for i in 0..=4 {
            let t = i as f32 / 4.0;
            let y = rect.min.y + t * rect.height();
            painter.line_segment(
                [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
                Stroke::new(0.5, Color32::from_gray(60)),
            );
        }

        // Draw the spline curve.
        let mut prev: Option<Pos2> = None;
        for i in 0..=120 {
            let t = i as f32 / 120.0;
            let x = self.x_range.0 + t * (self.x_range.1 - self.x_range.0);
            let y = self.spline.evaluate(x);
            let p = to_screen(x, y);
            if let Some(prev_p) = prev {
                painter.line_segment([prev_p, p], Stroke::new(1.5, Color32::from_rgb(100, 200, 255)));
            }
            prev = Some(p);
        }

        // Draw + drag knots.
        if let CubicSpline::Multipoint(knots) = self.spline {
            let mut drag_target: Option<usize> = None;
            for (i, knot) in knots.iter().enumerate() {
                let p = to_screen(knot.loc, knot.val);
                let hit = response.hover_pos().map(|hp| (hp - p).length() < 8.0).unwrap_or(false);
                let color = if hit { Color32::YELLOW } else { Color32::from_rgb(220, 180, 80) };
                painter.circle_filled(p, 5.0, color);
                if hit && response.dragged() {
                    drag_target = Some(i);
                }
            }
            if let Some(i) = drag_target {
                if let Some(pos) = response.interact_pointer_pos() {
                    let (lx, ly) = from_screen(pos);
                    knots[i].loc = lx.clamp(self.x_range.0, self.x_range.1);
                    knots[i].val = ly.clamp(self.y_range.0, self.y_range.1);
                }
            }
        }

        response
    }
}
```

- [ ] **Step 7.2: Wire the widget into the sidebar**

In `ui.rs`, extend `density_panel` (or add a new function) to render the spline editor for `offset_spline`:

```rust
ui.collapsing("Offset spline", |ui| {
    ui.add(crate::spline_widget::SplineEditor::new(&mut cfg.offset_spline));
    if let oxium::worldgen::spline::CubicSpline::Multipoint(knots) = &cfg.offset_spline {
        ui.label(format!("{} knots", knots.len()));
    }
    if ui.button("Convert to Multipoint with 4 knots").clicked() {
        cfg.offset_spline = oxium::worldgen::spline::CubicSpline::Multipoint(vec![
            oxium::worldgen::spline::Knot { loc: -1.0, val: -0.5, slope: 0.0 },
            oxium::worldgen::spline::Knot { loc: -0.3, val: -0.2, slope: 0.0 },
            oxium::worldgen::spline::Knot { loc:  0.2, val:  0.1, slope: 0.0 },
            oxium::worldgen::spline::Knot { loc:  1.0, val:  0.4, slope: 0.0 },
        ]);
    }
});
```

Add `mod spline_widget;` to `main.rs`.

- [ ] **Step 7.3: Run and verify**

Run: `cargo run --release --bin worldgen_viz`

Expected:
- Expand "Offset spline" in the sidebar.
- Click "Convert to Multipoint with 4 knots". A spline curve appears.
- Drag knots with the mouse. The curve reshapes. Terrain re-meshes within ~1s reflecting the change (PR 2's `offset_spline` field is currently constant-only — for PR 2 testing this won't drive terrain shape yet; full integration lands in PR 3).

- [ ] **Step 7.4: Commit**

```bash
git add src/bin/worldgen_viz/
git commit -m "$(cat <<'EOF'
feat(viz): SplineEditor widget — visual spline knot dragging

Custom egui widget that plots a CubicSpline as a curve and renders
draggable knot circles. Mouse drag updates knot loc/val. Connects
to cfg.offset_spline (and future spline fields from PR 3 onward).

For PR 2, the offset spline isn't consumed by terrain yet (it's a
Constant placeholder); the widget proves out the editing UX for
PR 3 where the spline drives real terrain.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Polish — keyboard shortcuts, regen toggle, status bar

**Files:**
- Modify: `src/bin/worldgen_viz/main.rs`, `src/bin/worldgen_viz/ui.rs`

- [ ] **Step 8.1: Add a status bar with regen timing**

In the egui closure, add a bottom panel:

```rust
egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
    ui.horizontal(|ui| {
        ui.label(format!("yaw {:.2} pitch {:.2} dist {:.0}",
            self.camera.yaw, self.camera.pitch, self.camera.distance));
        ui.separator();
        if let Some(last_ms) = self.last_regen_ms {
            ui.label(format!("last regen: {:.1} ms", last_ms));
        }
        ui.separator();
        ui.label(if self.auto_regen { "auto-regen: ON" } else { "auto-regen: OFF (press R to regen)" });
        ui.separator();
        if ui.button("[R] Regen").clicked() {
            self.dirty = true;
        }
        if ui.button(if self.auto_regen { "Pause" } else { "Resume" }).clicked() {
            self.auto_regen = !self.auto_regen;
        }
    });
});
```

Add fields to App:

```rust
last_regen_ms: Option<f32>,
auto_regen: bool,
```

Initialize: `last_regen_ms: None, auto_regen: true`.

- [ ] **Step 8.2: Time the regen and respect auto_regen**

Wrap the regen call:

```rust
if self.dirty && self.auto_regen {
    let t0 = std::time::Instant::now();
    let (verts, idxs) = worldgen_bridge::regen_region_mesh(42, &self.config);
    if let Some(scene) = self.scene.as_mut() {
        scene.upload_mesh(&render.device, &verts, &idxs);
    }
    self.last_regen_ms = Some(t0.elapsed().as_secs_f32() * 1000.0);
    self.dirty = false;
}
```

- [ ] **Step 8.3: Keyboard shortcuts (R for regen, Space for pause)**

In `WindowEvent::KeyboardInput`:

```rust
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
```

- [ ] **Step 8.4: Run and verify**

Run: `cargo run --release --bin worldgen_viz`

Expected:
- Status bar at the bottom showing yaw/pitch/dist + regen timing + auto-regen toggle.
- Press `R`: forces a regen (works even if auto_regen is off).
- Press `Space`: toggles auto_regen.
- Drag sliders with auto_regen off — no regen until `R` pressed (useful when you want to tweak multiple things before committing).

- [ ] **Step 8.5: Commit**

```bash
git add src/bin/worldgen_viz/
git commit -m "$(cat <<'EOF'
feat(viz): status bar + keyboard shortcuts + auto-regen toggle

Bottom status bar shows camera state, last regen timing, and
auto-regen state. Buttons + keys:
- R: force regen
- Space: toggle auto-regen

Auto-regen-off mode lets you batch multiple slider edits before
paying the regen cost, useful for spline editing where you might
move 4-5 knots in sequence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Verification

**Files:** none (verification only)

- [ ] **Step 9.1: Run the visualizer through its full feature set**

Run: `cargo run --release --bin worldgen_viz`

Manual checklist:
- [ ] Window opens at ~1400×900.
- [ ] 3D viewport shows a meshed terrain region (16 chunks).
- [ ] Right-click drag rotates the camera; mouse wheel zooms.
- [ ] Sliders for every `DensityConfig` field move freely.
- [ ] Moving a slider triggers a regen within ~1s.
- [ ] Status bar reports regen time (target: <500ms in release build).
- [ ] Top-down cross-section panel shows a 256×256 heightmap.
- [ ] "Save preset" writes to `assets/worldgen/scratch.ron`.
- [ ] "Reload default.ron" reverts the config.
- [ ] Spline editor: convert to Multipoint, drag knots, curve reshapes.
- [ ] `R` key forces regen.
- [ ] `Space` toggles auto-regen.

- [ ] **Step 9.2: Verify the existing lib tests still pass**

Run: `cargo test 2>&1 | tail -10`

Expected: every test passes (the visualizer is a separate bin target and doesn't affect lib tests).

- [ ] **Step 9.3: Build in release mode and check size**

Run: `cargo build --release --bin worldgen_viz 2>&1 | tail -5`

Expected: `Finished release` line. Binary at `target/release/worldgen_viz`.

- [ ] **Step 9.4: Final commit if any polish needed**

If steps 9.1–9.3 surfaced any issues, fix and commit. Otherwise the visualizer is complete.

---

## Out of scope for V1 (deferred to V2)

- **Climate space explorer.** A 2D slice of the 6D biome hyperbox showing biome regions. Add after PR 4 lands the biome table.
- **Probe line widget.** Click two points in the 3D view, see density / biome / all noise channels charted along the line. Great for "why is this voxel wrong?" debugging.
- **Side-by-side before/after.** Two viewports showing the same region under two different configs. Useful for A/B comparison.
- **Cave structure visualization.** Show graph cave chambers + tunnels as overlay lines/spheres, separate from the meshed terrain.
- **Greedy meshing.** V1 uses a naive face mesher; greedy meshing would emit ~10× fewer triangles. Worth the swap if perf becomes a constraint.
- **Cross-chunk neighbor lookups.** V1's naive mesher over-emits faces at chunk seams (renders the chunk-boundary face on both sides). Acceptable for a debug visualizer; fix when greedy meshing lands.
- **Sun light direction control.** V1 has no lighting (per-vertex debug colors). Adding directional light + normal calculation is V2 polish.
- **Sliders for `WorldgenConfig` fields added by future PRs.** PR 3-8 will add fields; the `density_panel` (and new panels) need extending. Each migration PR's plan should include a "extend visualizer UI" task if it adds tunable values.

## Plan self-review notes

- All 9 tasks have concrete code in every step. No "TODO" or "see X" placeholders.
- File paths are exact: `src/bin/worldgen_viz/{main,render,scene,ui,worldgen_bridge,cross,spline_widget}.rs`.
- Type names match between tasks: `RenderState`, `SceneRender`, `OrbitCamera`, `Vertex`, `CameraUniform`, `App`, `SplineEditor`.
- Each task ends with a clean commit boundary.
- Depends explicitly on PR 2's types (`WorldgenConfig`, `DensityConfig`, `ConfigHolder`, `Generator::with_config`, `CubicSpline`, `Knot`) — PR 2 must land first.
- No tests inside the visualizer binary (it's a tool, not a library) — verification is manual + integration via `cargo build --bin`.
- `RenderState::new` uses `mem::transmute` to fake-extend the surface lifetime — this is a known wgpu pattern when the surface is parametrized on the window's lifetime but the app holds the window in an Arc. If this trips clippy lints, replace with the wgpu 0.20+ `Surface<'static>` constructor pattern.

## Implementation order suggestion (if working sequentially)

Tasks 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9.

Tasks 6 (cross-section) and 7 (spline widget) can swap order — they're independent. Task 8 (polish) depends on 1-5 being done.
