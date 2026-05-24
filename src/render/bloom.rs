//! Bloom mip chain: five `Rgba16Float` textures sized ½, ¼, ⅛, 1/16, 1/32
//! of the swapchain. Used by the bloom post pass (PR 4 of the lighting
//! overhaul) to spread HDR-bright pixels into a soft halo.
//!
//! All mips share one linear-filtering sampler (`Rgba16Float` is
//! filterable in wgpu without a device feature). Each mip carries its
//! own `TextureView` because the down/upsample passes bind a specific
//! level as both read source and write target.

use wgpu::TextureFormat;

pub const BLOOM_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const BLOOM_MIP_COUNT: u32 = 5;
pub const BLOOM_MIN_DIM: u32 = 4;

/// One mip of the bloom chain — its texture, the view used to bind it
/// as a write target, and the cached dimensions. The view doubles as
/// the read-source view in down/upsample passes; mips are written as
/// whole-texture render targets, so no level-of-detail subview is
/// needed.
pub struct BloomMip {
    // `texture` and dimensions are kept for future resize/recreation use.
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
}

/// Owned five-mip bloom chain + linear sampler. Recreated on resize via
/// `recreate`.
pub struct BloomChain {
    pub mips: [BloomMip; BLOOM_MIP_COUNT as usize],
    pub sampler: wgpu::Sampler,
}

impl BloomChain {
    pub fn new(device: &wgpu::Device, swapchain_w: u32, swapchain_h: u32) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let mips = std::array::from_fn(|i| make_mip(device, swapchain_w, swapchain_h, i as u32));
        Self { mips, sampler }
    }

    /// Re-create every mip at the new swapchain size. Sampler is reused.
    pub fn recreate(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        for (i, mip) in self.mips.iter_mut().enumerate() {
            *mip = make_mip(device, w, h, i as u32);
        }
    }
}

fn make_mip(device: &wgpu::Device, w: u32, h: u32, level: u32) -> BloomMip {
    // Mip 0 is ½ res; mip 1 is ¼; etc. The +1 in the shift compensates.
    let divisor = 1u32 << (level + 1);
    let width = (w / divisor).max(BLOOM_MIN_DIM);
    let height = (h / divisor).max(BLOOM_MIN_DIM);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("bloom-mip-{level}")),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: BLOOM_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    BloomMip {
        texture,
        view,
        width,
        height,
    }
}
