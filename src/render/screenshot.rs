//! Read back the current swap-chain texture into a PNG file.
//!
//! Used by the `--screenshot-and-exit <path>` flag for headless visual
//! verification. Because the renderer's surface is configured with
//! [`wgpu::TextureUsages::COPY_SRC`] (see `gpu.rs`), we can copy the
//! presented texture into a `MAP_READ` buffer and decode the rows.
//!
//! GPU → CPU copy gotchas this module handles:
//!
//! 1. **Row stride padding.** Buffer rows must start on a 256-byte boundary
//!    (`COPY_BYTES_PER_ROW_ALIGNMENT`), so we allocate a padded buffer and
//!    strip the padding bytes when assembling the final PNG rows.
//! 2. **Synchronisation.** `buffer.map_async` is asynchronous; we drive it
//!    to completion synchronously by polling the device with
//!    `Maintain::Wait`.
//! 3. **Format.** Most swap chains hand us BGRA8 sRGB on macOS; we flip the
//!    channels to RGBA before passing to the `image` crate's PNG encoder.

use anyhow::{Result, anyhow};
use std::path::Path;

/// Read the most recently-rendered swap-chain texture out of the GPU into a
/// PNG file at `path`.
///
/// Call **after** a successful `Renderer::render` but **before**
/// `frame.present()` consumed the texture — for our screenshot path the
/// renderer's frame is re-rendered into an offscreen target instead, so
/// this helper just walks the supplied texture+config.
///
/// `format`/`width`/`height` describe the source texture; they must match
/// the texture being copied.
pub fn capture_texture_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Texture,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    path: &Path,
) -> Result<()> {
    // Rows must be aligned to 256 bytes per the wgpu spec.
    const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let bytes_per_pixel: u32 = 4;
    let unpadded_row_bytes = width * bytes_per_pixel;
    let padded_row_bytes = unpadded_row_bytes.div_ceil(ALIGN) * ALIGN;
    let buffer_size = (padded_row_bytes as u64) * (height as u64);

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot-readback-buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("screenshot-copy-encoder"),
    });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));

    // Map the buffer for read and synchronously drive the GPU until it
    // completes.
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| anyhow!("map_async channel dropped: {e}"))?
        .map_err(|e| anyhow!("buffer map failed: {e:?}"))?;

    // Strip padding row-by-row and convert BGRA→RGBA if needed.
    let raw = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded_row_bytes as usize) * (height as usize));
    let bgra = is_bgra(format);
    for row in 0..height as usize {
        let start = row * padded_row_bytes as usize;
        let end = start + unpadded_row_bytes as usize;
        let row_bytes = &raw[start..end];
        if bgra {
            for px in row_bytes.chunks_exact(4) {
                pixels.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
            }
        } else {
            pixels.extend_from_slice(row_bytes);
        }
    }
    drop(raw);
    buffer.unmap();

    image::save_buffer(path, &pixels, width, height, image::ColorType::Rgba8)?;
    Ok(())
}

/// Is the wgpu surface format BGRA-ordered? macOS swap-chains routinely are.
fn is_bgra(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    )
}
