//! Bootstraps `wgpu`: instance → surface → adapter → device → queue.
//!
//! Everything here runs once at startup. The surface is configured with
//! `RENDER_ATTACHMENT | COPY_SRC` so the screenshot path can copy the
//! current swap-chain texture out to a CPU buffer for PNG export.
//!
//! `wgpu::Surface<'static>` requires the surface to outlive its window;
//! we satisfy that by holding `Arc<Window>` in the surface (a stable
//! pattern on `wgpu` 23).

use std::sync::Arc;
use winit::window::Window;

/// The wgpu depth-buffer format used everywhere in oxium. 32-bit float depth
/// is overkill for view distance 1000 but avoids precision artefacts at the
/// horizon and is widely supported.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Owns the long-lived `wgpu` objects. Shared by reference everywhere.
pub struct Gpu {
    /// Instance — the wgpu entry point. Held so the surface (which borrows
    /// from it conceptually) remains valid.
    pub instance: wgpu::Instance,
    /// Drawable window surface. Lifetime is `'static` because it keeps the
    /// `Arc<Window>` alive internally.
    pub surface: wgpu::Surface<'static>,
    /// The chosen physical GPU.
    pub adapter: wgpu::Adapter,
    /// Logical device — the interface for resource creation.
    pub device: wgpu::Device,
    /// Command-submission stream.
    pub queue: wgpu::Queue,
    /// Current configuration (size, format, present mode). Updated on resize.
    pub surface_cfg: wgpu::SurfaceConfiguration,
}

impl Gpu {
    /// Initialize the GPU and configure the surface for the given window.
    ///
    /// The init steps use `pollster::block_on` to drive `wgpu`'s async
    /// `request_adapter`/`request_device` calls on the main thread. wgpu's
    /// "async" here is mostly an artefact of WebGPU compatibility — on
    /// native it completes immediately, so the block is cheap.
    pub fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            // Prefer the discrete GPU on laptops with both an iGPU and a dGPU.
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no compatible adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("oxium-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            // Trace path: pass a directory to capture an api-trace replay
            // dump. We never want this on for normal runs.
            None,
        ))
        .expect("device request failed");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        // Prefer an sRGB-encoded format — the shaders write linear colour and
        // hardware converts to sRGB on store automatically.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let surface_cfg = wgpu::SurfaceConfiguration {
            // COPY_SRC is added so the screenshot path can copy the swap
            // texture to a download buffer.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            // FIFO == v-sync. Trades latency for tear-free output and is
            // universally supported. Optimizing this is M10's job, not M1's.
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_cfg);

        Self {
            instance,
            surface,
            adapter,
            device,
            queue,
            surface_cfg,
        }
    }

    /// Reconfigure the surface for a new window size. Called from
    /// `Renderer::resize` whenever winit reports a resize.
    pub fn resize(&mut self, w: u32, h: u32) {
        self.surface_cfg.width = w.max(1);
        self.surface_cfg.height = h.max(1);
        self.surface.configure(&self.device, &self.surface_cfg);
    }
}

/// Allocate a depth texture matching the given dimensions.
///
/// We treat the depth texture as a sized-by-current-window resource — it's
/// recreated by `Renderer::resize`.
pub fn make_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth-texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}
