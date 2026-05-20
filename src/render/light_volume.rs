//! Per-chunk 3D light texture. Resolution is 33³ — one extra cell per
//! axis beyond the 32³ chunk so trilinear sampling at the chunk's
//! +X/+Y/+Z boundary picks up the neighbor cell rather than clamping.
//!
//! Layout: Rgba8Unorm. R/G/B encode block_red/green/blue / 15.0;
//! A encodes sky_light / 15.0. The shader multiplies back as needed.
//!
//! Lifecycle: created on first Meshed or Relit completion for a chunk;
//! re-uploaded on every subsequent Relit; dropped when the chunk
//! leaves the load radius.

use wgpu::TextureFormat;

pub const LIGHT_VOLUME_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
pub const LIGHT_VOLUME_DIM: u32 = 33;
pub const LIGHT_VOLUME_BYTES: usize = 33 * 33 * 33 * 4;

pub struct ChunkLightVolume {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl ChunkLightVolume {
    /// Create + upload from a freshly-built `[u8; LIGHT_VOLUME_BYTES]` blob.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, blob: &[u8]) -> Self {
        assert_eq!(
            blob.len(),
            LIGHT_VOLUME_BYTES,
            "light volume blob size mismatch ({} vs {})",
            blob.len(),
            LIGHT_VOLUME_BYTES
        );
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("chunk-light-volume"),
            size: wgpu::Extent3d {
                width: LIGHT_VOLUME_DIM,
                height: LIGHT_VOLUME_DIM,
                depth_or_array_layers: LIGHT_VOLUME_DIM,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: LIGHT_VOLUME_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            blob,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(LIGHT_VOLUME_DIM * 4),
                rows_per_image: Some(LIGHT_VOLUME_DIM),
            },
            wgpu::Extent3d {
                width: LIGHT_VOLUME_DIM,
                height: LIGHT_VOLUME_DIM,
                depth_or_array_layers: LIGHT_VOLUME_DIM,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }

    /// Reupload contents into the existing texture (no realloc).
    pub fn update(&self, queue: &wgpu::Queue, blob: &[u8]) {
        assert_eq!(
            blob.len(),
            LIGHT_VOLUME_BYTES,
            "light volume blob size mismatch"
        );
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            blob,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(LIGHT_VOLUME_DIM * 4),
                rows_per_image: Some(LIGHT_VOLUME_DIM),
            },
            wgpu::Extent3d {
                width: LIGHT_VOLUME_DIM,
                height: LIGHT_VOLUME_DIM,
                depth_or_array_layers: LIGHT_VOLUME_DIM,
            },
        );
    }
}
