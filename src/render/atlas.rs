//! Block texture atlas.
//!
//! A *texture atlas* is a single big image that holds many smaller tiles
//! packed into a regular grid. The GPU prefers one bound texture over
//! many because every state change between draws costs a tiny amount of
//! CPU work, and our voxel meshes draw all blocks in a single
//! `draw_indexed` per chunk. By packing every block-face image into one
//! atlas we keep that "one texture per chunk draw" property.
//!
//! Layout: a fixed `ATLAS_TILE_COUNT_PER_AXIS² = 16` slot grid, each
//! `TILE_PX × TILE_PX` pixels, total atlas size `ATLAS_PX × ATLAS_PX`.
//! Slots are filled by [`Tile`]'s variant order; unfilled slots stay
//! transparent black and act as the "missing tile" fallback (which
//! should never appear in normal rendering).
//!
//! Filtering is `Nearest` because the textures are pixel art — bilinear
//! filtering would soften the crisp 1-pixel features Design painted in.
//! We also skip mipmaps for v1: cross-tile bleed from a naive
//! `generate_mipmaps` would smear adjacent tiles into each other at
//! distance, producing more obvious artefacts than the slight far-field
//! shimmering we accept by going mipless. (A future iteration can build
//! per-tile mip pyramids and pack those into separate atlas levels.)

use anyhow::{Context, Result};
use image::ImageReader;
use std::path::Path;

/// Pixels along one side of a single tile. 16 matches Minecraft's
/// classic resolution; the engine itself only depends on this constant
/// being consistent between atlas + shader.
pub const TILE_PX: u32 = 16;
/// Tile slots along one side of the square atlas grid. 4 × 4 = 16
/// supported tiles — comfortably more than v1's 9.
pub const ATLAS_TILE_COUNT_PER_AXIS: u32 = 4;
/// Side length of the atlas image in pixels.
pub const ATLAS_PX: u32 = TILE_PX * ATLAS_TILE_COUNT_PER_AXIS;

/// Filename (under the textures directory) for each [`Tile`] slot, in
/// `Tile` discriminant order. The atlas builder iterates this list once
/// at startup; missing files yield a transparent slot.
fn tile_files() -> &'static [&'static str] {
    &[
        "stone.png",
        "dirt.png",
        "grass_block_top.png",
        "grass_block_side.png",
        "sand.png",
        "oak_log.png",
        "oak_log_top.png",
        "oak_leaves.png",
        "water_still.png",
        "snow.png",
        "lava_still.png",
    ]
}

/// Raw RGBA8 atlas plus the metadata needed to upload it as a wgpu
/// texture. The bytes are stored row-major, top-left origin, with one
/// byte per channel — exactly the layout `Rgba8UnormSrgb` expects.
pub struct AtlasImage {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Load every `tile_files()` entry from `textures_dir`, pack them into
/// a fresh atlas, and return the result. Missing files produce a hot
/// magenta tile so the gap is visible in the rendered world — matches
/// the registry's "missing block info" sentinel convention.
pub fn build_atlas(textures_dir: &Path) -> Result<AtlasImage> {
    let mut atlas = vec![0u8; (ATLAS_PX * ATLAS_PX * 4) as usize];

    for (idx, file) in tile_files().iter().enumerate() {
        let path = textures_dir.join(file);
        let tile_rgba = match load_tile(&path) {
            Ok(bytes) => bytes,
            Err(err) => {
                log::warn!(
                    "atlas: missing or unreadable tile {} ({err}); using magenta fallback",
                    path.display()
                );
                magenta_tile()
            }
        };
        blit_tile(&mut atlas, idx as u32, &tile_rgba);
    }

    Ok(AtlasImage {
        rgba: atlas,
        width: ATLAS_PX,
        height: ATLAS_PX,
    })
}

/// Load a `TILE_PX × TILE_PX` RGBA PNG. The PNG is decoded into the
/// crate's native pixel layout, which we then convert to a flat RGBA8
/// buffer regardless of the source channel order. Returns an error if
/// the file's dimensions don't match `TILE_PX × TILE_PX` so a typoed
/// resolution fails loudly rather than silently producing garbage.
fn load_tile(path: &Path) -> Result<Vec<u8>> {
    let img = ImageReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .decode()
        .with_context(|| format!("decoding {}", path.display()))?
        .to_rgba8();
    anyhow::ensure!(
        img.width() == TILE_PX && img.height() == TILE_PX,
        "{} is {}×{}, expected {TILE_PX}×{TILE_PX}",
        path.display(),
        img.width(),
        img.height()
    );
    Ok(img.into_raw())
}

/// Fallback tile for missing assets: hot magenta with full alpha,
/// stamped flat across the whole tile. Same convention as the
/// "missing block info" magenta in `BlockRegistry`.
fn magenta_tile() -> Vec<u8> {
    let mut buf = Vec::with_capacity((TILE_PX * TILE_PX * 4) as usize);
    for _ in 0..(TILE_PX * TILE_PX) {
        buf.extend_from_slice(&[255, 0, 255, 255]);
    }
    buf
}

/// Copy one `TILE_PX × TILE_PX` RGBA tile into atlas slot `index`,
/// row-by-row. Row stride differs between the source (one tile wide)
/// and the destination (atlas wide), so we can't memcpy the whole
/// buffer at once.
fn blit_tile(atlas: &mut [u8], index: u32, tile_rgba: &[u8]) {
    let tile_col = index % ATLAS_TILE_COUNT_PER_AXIS;
    let tile_row = index / ATLAS_TILE_COUNT_PER_AXIS;
    let dst_x_px = tile_col * TILE_PX;
    let dst_y_px = tile_row * TILE_PX;

    let atlas_row_stride = (ATLAS_PX * 4) as usize;
    let tile_row_stride = (TILE_PX * 4) as usize;

    for row in 0..TILE_PX {
        let src_offset = (row as usize) * tile_row_stride;
        let dst_offset = ((dst_y_px + row) as usize) * atlas_row_stride + (dst_x_px as usize) * 4;
        atlas[dst_offset..dst_offset + tile_row_stride]
            .copy_from_slice(&tile_rgba[src_offset..src_offset + tile_row_stride]);
    }
}

/// Built-and-uploaded atlas: a wgpu texture + its sampler + the
/// `BindGroup` the opaque pipeline will reach for in `set_bind_group(2,
/// …)`. Owns the bind-group layout too so the pipeline build can wire
/// it in without reaching back into `Renderer`.
pub struct AtlasGpu {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    /// Public so other pipelines (e.g. the HUD) can build their own
    /// bind groups against the same atlas image. Bind-group layouts
    /// are per-pipeline in wgpu, so a single bind group can't serve
    /// every consumer — but the underlying texture is reusable.
    pub texture: wgpu::Texture,
    _view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
}

/// Bind-group layout slots used by the atlas:
/// - binding 0: the atlas texture (2D, float-sampled)
/// - binding 1: a sampler (Nearest, clamp-to-edge)
pub fn make_atlas_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("atlas-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    })
}

/// Upload an [`AtlasImage`] as a wgpu texture and construct the
/// matching bind group. Uses sRGB-format storage so the gamma curve
/// matches the rest of the pipeline (we authored the PNGs in sRGB and
/// the surface is sRGB too — converting once on read keeps lighting
/// linear-correct).
///
/// `Nearest` filtering preserves the pixel-art crispness; mipmaps are
/// deliberately omitted (see module docs).
pub fn upload_atlas(device: &wgpu::Device, queue: &wgpu::Queue, image: &AtlasImage) -> AtlasGpu {
    let size = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("atlas-texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
        &image.rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(image.width * 4),
            rows_per_image: Some(image.height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("atlas-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    let bind_group_layout = make_atlas_bind_group_layout(device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("atlas-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    AtlasGpu {
        bind_group_layout,
        bind_group,
        texture,
        _view: view,
        _sampler: sampler,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_dimensions_are_consistent() {
        // Compile-time-ish sanity: a 4×4 grid of 16×16 tiles is 64×64.
        assert_eq!(ATLAS_PX, 64);
        assert!(
            tile_files().len() <= (ATLAS_TILE_COUNT_PER_AXIS * ATLAS_TILE_COUNT_PER_AXIS) as usize,
            "too many tiles for the atlas grid"
        );
    }

    #[test]
    fn blit_tile_writes_correct_pixels() {
        // Fill an "all red" 16×16 tile into slot 5 (row 1, col 1) and
        // check that the tile lands at the right corner of the atlas.
        let mut atlas = vec![0u8; (ATLAS_PX * ATLAS_PX * 4) as usize];
        let mut red = Vec::with_capacity((TILE_PX * TILE_PX * 4) as usize);
        for _ in 0..(TILE_PX * TILE_PX) {
            red.extend_from_slice(&[255, 0, 0, 255]);
        }
        blit_tile(&mut atlas, 5, &red);

        // Slot 5 = (col=1, row=1), so top-left atlas pixel is (16, 16).
        let stride = (ATLAS_PX * 4) as usize;
        let pixel = |x: u32, y: u32| {
            let off = (y as usize) * stride + (x as usize) * 4;
            [atlas[off], atlas[off + 1], atlas[off + 2], atlas[off + 3]]
        };
        assert_eq!(pixel(16, 16), [255, 0, 0, 255]);
        assert_eq!(pixel(31, 31), [255, 0, 0, 255]);
        // Untouched pixels stay zero.
        assert_eq!(pixel(0, 0), [0, 0, 0, 0]);
        assert_eq!(pixel(32, 32), [0, 0, 0, 0]);
    }

    #[test]
    fn missing_directory_yields_magenta() {
        // Pointing at a non-existent path returns Ok (the loader's
        // per-tile error path replaces missing files with magenta) and
        // every tile slot is filled with the fallback colour.
        let atlas = build_atlas(Path::new("/nonexistent-textures-dir-12345"))
            .expect("build_atlas should not fail when tiles are merely missing");
        assert_eq!(atlas.width, ATLAS_PX);
        assert_eq!(atlas.height, ATLAS_PX);
        // Probe the centre of tile 0: should be magenta.
        let stride = (ATLAS_PX * 4) as usize;
        let off = (TILE_PX as usize / 2) * stride + (TILE_PX as usize / 2) * 4;
        assert_eq!(&atlas.rgba[off..off + 4], &[255, 0, 255, 255]);
    }
}
