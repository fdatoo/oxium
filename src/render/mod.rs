//! GPU rendering backend, built on `wgpu` (a portable Vulkan-shaped abstraction).
//!
//! `wgpu` exposes:
//!
//! - `Instance` — discovers GPU adapters.
//! - `Adapter` — a particular physical GPU.
//! - `Device` + `Queue` — the logical interface and its command submission stream.
//! - `Surface` — the swapchain (the OS-window-backed framebuffer).
//!
//! Higher level objects (`RenderPipeline`, `BindGroup`, `Buffer`, `Texture`)
//! are owned per subsystem (sky, opaque voxels, water, cursor, HUD).
