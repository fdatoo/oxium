//! Snapshot mode: render a (zoom × width) top-down image of one
//! pipeline stage and write it as a PNG.

use image::{ImageBuffer, Rgba};
use oxium::viz_render;
use oxium::worldgen::{Generator, probe::Stage};
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<(), String> {
    let opts = parse(args)?;
    let stage = parse_stage(&opts.stage)?;
    let generator = Generator::new(opts.seed);

    let w = opts.width;
    let h = opts.width; // square images for now; can add --height later
    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);

    let half = (w as i32) / 2;
    for py in 0..h {
        for px in 0..w {
            // Pixel (px, py) → world coordinate.
            // px=0 → wx = center.0 - half*zoom; py=0 → wz = center.1 - half*zoom.
            let wx = opts.center.0 + (px as i32 - half) * (opts.zoom as i32);
            let wz = opts.center.1 + (py as i32 - half) * (opts.zoom as i32);
            let rgba = viz_render::render_pixel(&generator, stage, wx, wz);
            img.put_pixel(px, py, Rgba(rgba));
        }
    }

    img.save(&opts.output)
        .map_err(|e| format!("write {:?}: {e}", opts.output))?;
    Ok(())
}

#[derive(Debug)]
struct Opts {
    seed: u64,
    center: (i32, i32),
    zoom: u32,
    stage: String,
    width: u32,
    output: PathBuf,
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let mut seed: Option<u64> = None;
    let mut center: Option<(i32, i32)> = None;
    let mut zoom: Option<u32> = None;
    let mut stage: Option<String> = None;
    let mut width: Option<u32> = None;
    let mut output: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let val = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--seed" => seed = Some(val.parse().map_err(|_| format!("bad seed: {val}"))?),
            "--center" => {
                let parts: Vec<&str> = val.split(',').collect();
                if parts.len() != 2 {
                    return Err(format!("--center expects wx,wz, got {val}"));
                }
                let wx: i32 = parts[0]
                    .parse()
                    .map_err(|_| format!("bad wx: {}", parts[0]))?;
                let wz: i32 = parts[1]
                    .parse()
                    .map_err(|_| format!("bad wz: {}", parts[1]))?;
                center = Some((wx, wz));
            }
            "--zoom" => zoom = Some(val.parse().map_err(|_| format!("bad zoom: {val}"))?),
            "--stage" => stage = Some(val.clone()),
            "--width" => width = Some(val.parse().map_err(|_| format!("bad width: {val}"))?),
            "--output" => output = Some(PathBuf::from(val)),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 2;
    }

    Ok(Opts {
        seed: seed.ok_or("--seed required")?,
        center: center.ok_or("--center required")?,
        zoom: zoom.ok_or("--zoom required")?,
        stage: stage.ok_or("--stage required")?,
        width: width.ok_or("--width required")?,
        output: output.ok_or("--output required")?,
    })
}

fn parse_stage(s: &str) -> Result<Stage, String> {
    match s {
        "continentalness" => Ok(Stage::Continentalness),
        "plate-id" => Ok(Stage::PlateId),
        "temperature" => Ok(Stage::Temperature),
        "humidity" => Ok(Stage::Humidity),
        "desertness" => Ok(Stage::Desertness),
        "weirdness" => Ok(Stage::Weirdness),
        "h-pre" => Ok(Stage::HPre),
        "valley-carve" => Ok(Stage::ValleyCarve),
        "h-target" => Ok(Stage::HTarget),
        "flow-accum" => Ok(Stage::FlowAccum),
        "biome-id" => Ok(Stage::BiomeId),
        "aquifer-y" => Ok(Stage::AquiferY),
        "aquifer-substance" => Ok(Stage::AquiferSubstance),
        other => Err(format!("unknown stage: {other}")),
    }
}
