//! Offscreen HDR target used by the 3D world passes (opaque, water, sky).
//! The composite pass samples this target and resolves to the swapchain.
//!
//! Why HDR: emissive surfaces (torches, sun-bright sky) push values >1.0;
//! the bloom pass in PR 4 needs those pre-tonemap values. `Rgba16Float`
//! gives us the dynamic range without the precision loss of `R11G11B10`.

use wgpu::TextureFormat;

/// Color format used by every render target between the world passes
/// and composite.
pub const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Owns the offscreen color texture sized to the swapchain. Re-created
/// on resize via `recreate`.
pub struct HdrTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl HdrTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width,
            height,
        }
    }

    /// Re-create the underlying texture at the new size. Old view is dropped.
    pub fn recreate(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        *self = Self::new(device, width, height);
    }
}
