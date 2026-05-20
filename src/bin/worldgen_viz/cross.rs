//! 2D cross-section: top-down map of column heights from the
//! visualizer's current WorldgenConfig.

use oxium::worldgen::config::{ConfigHolder, WorldgenConfig};
use oxium::worldgen::Generator;

pub const CROSS_RES: usize = 256;
/// Each pixel = 4 blocks. 256 × 4 = 1024 blocks per side.
pub const CROSS_BLOCKS_PER_PX: i32 = 4;

/// Render a top-down heightmap of `col.height` over a 1024×1024 block region.
pub fn render_topdown(seed: u64, config: &WorldgenConfig) -> egui::ColorImage {
    let holder = ConfigHolder::new(config.clone());
    let generator = Generator::with_config(seed, holder);
    let mut pixels = vec![egui::Color32::BLACK; CROSS_RES * CROSS_RES];
    for iz in 0..CROSS_RES {
        for ix in 0..CROSS_RES {
            let wx = (ix as i32 - CROSS_RES as i32 / 2) * CROSS_BLOCKS_PER_PX;
            let wz = (iz as i32 - CROSS_RES as i32 / 2) * CROSS_BLOCKS_PER_PX;
            let col = generator.column_data(wx, wz);
            let h = col.height as f32;
            let t = ((h + 50.0) / 200.0).clamp(0.0, 1.0);
            let g = (t * 255.0) as u8;
            pixels[iz * CROSS_RES + ix] = if col.height <= 62 {
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
