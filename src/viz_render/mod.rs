//! Engine-authoritative rendering helpers shared between the `worldgen_viz`
//! debug app and the `doc_render` snapshot binary. Pure layer on top of
//! `crate::worldgen::Generator` + `crate::worldgen::probe::Stage`.

pub mod colormap;
pub mod stages;

pub use stages::{pixel, render_pixel};
