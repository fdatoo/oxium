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

/// MSAA sample count for the world-rendering pass (opaque + water + sky +
/// cursor). Higher than 1 smooths every chunk edge by averaging coverage
/// across this many sub-samples per pixel; 4× is the standard sweet spot
/// — anti-aliasing reads as crisp without the GPU cost of 8×. The HUD
/// pass stays at 1× because it draws after the MSAA resolve happens.
pub const MSAA_SAMPLES: u32 = 4;

/// Owns the long-lived `wgpu` objects. Shared by reference everywhere.
pub struct Gpu {
    /// Instance — the wgpu entry point. Held so the surface (which borrows
    /// from it conceptually) remains valid.
    #[allow(dead_code)] // kept-alive owner for `surface`
    pub instance: wgpu::Instance,
    /// Drawable window surface. Lifetime is `'static` because it keeps the
    /// `Arc<Window>` alive internally.
    pub surface: wgpu::Surface<'static>,
    /// The chosen physical GPU.
    #[allow(dead_code)] // retained for `request_device` lifetime + future diagnostics
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
    /// Pick the swapchain present mode here — `Fifo` for v-sync
    /// (default in normal play), `Immediate` for "uncapped" perf
    /// measurement so the HUD's FPS readout reflects actual
    /// throughput rather than the display's refresh rate.
    pub fn new_with_present_mode(window: Arc<Window>, present_mode: wgpu::PresentMode) -> Self {
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
                // Bump max_bind_groups from the default 4 to 5 so the
                // water pipeline can carry both the scene-depth
                // sampler (group 3) and the planar-reflection sampler
                // (group 4). The hard cap on most desktop GPUs is 8;
                // anything ≤ 8 is portable.
                required_limits: wgpu::Limits {
                    max_bind_groups: 5,
                    ..wgpu::Limits::default()
                },
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
            // Default Fifo == v-sync; --uncapped passes Immediate to
            // remove the display-rate cap for perf measurement. If the
            // adapter doesn't support the requested mode, fall back to
            // whatever's first in `present_modes`.
            present_mode: if caps.present_modes.contains(&present_mode) {
                present_mode
            } else {
                caps.present_modes[0]
            },
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

/// Allocate a depth texture matching the given dimensions and the world
/// pass's MSAA sample count. Recreated by `Renderer::resize` whenever
/// the surface dimensions change.
///
/// `COPY_SRC` is included so the depth contents can be copied into a
/// sampleable depth texture (see [`make_depth_sample_texture`]) for
/// the water pass to read terrain depth and compute foam / depth tint.
/// Returns both the texture handle (needed for the copy command) and
/// a view (needed for the render-pass attachment).
pub fn make_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth-texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        // Depth must match the colour target's sample count — otherwise
        // wgpu rejects the pipeline binding at draw time.
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Allocate the sampleable single-sample colour texture the water
/// shader reads to sample the planar reflection. The reflection pass
/// renders into a multisampled colour target and resolves into this
/// texture at end of pass; the water shader then samples it via
/// screen-space UVs with wave-normal distortion.
///
/// Lower-than-screen resolution would be fine for perf (wave
/// distortion hides reflection detail), but full res keeps the
/// implementation simple and our scene isn't fragment-bound.
pub fn make_reflection_color_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::TextureView, wgpu::Texture, wgpu::TextureView) {
    // The MSAA render target the reflection pass actually draws into.
    let msaa = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("reflection-color-msaa"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    // The single-sample resolve target. Sampled by the water shader.
    let resolve = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("reflection-color-resolve"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let msaa_view = msaa.create_view(&wgpu::TextureViewDescriptor::default());
    let resolve_view = resolve.create_view(&wgpu::TextureViewDescriptor::default());
    (msaa_view, resolve, resolve_view)
}

/// Allocate a dedicated depth texture for the reflection pass.
/// We can't reuse the main depth (it's owned by the main world
/// pass which runs concurrently in the same encoder); reflection
/// needs its own depth attachment for the opaque draws to depth-
/// test against each other.
pub fn make_reflection_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("reflection-depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Allocate the sampleable depth-copy texture the water pass reads
/// from. Same format + sample count as the live depth attachment so
/// `copy_texture_to_texture` between them is a 1:1 byte transfer.
///
/// A texture can't be simultaneously bound as a depth attachment and
/// as a shader sampler in the same render pass — the live depth
/// stays the attachment for the water pass's depth test, and this
/// copy is what the shader actually reads to compute terrain depth
/// vs water depth.
pub fn make_depth_sample_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth-sample-texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        // RENDER_ATTACHMENT is required by wgpu for any multisampled
        // texture, even one we never actually render to — the
        // texture is otherwise written exclusively by
        // `copy_texture_to_texture` from the live depth attachment.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Allocate the multisampled colour render target the world pass draws
/// into. Resolved automatically into the swapchain texture each frame
/// by the `resolve_target` field of the render pass colour attachment.
pub fn make_msaa_color_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("msaa-color-texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}
