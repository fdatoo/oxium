// Wireframe cursor highlight.
//
// Draws the 12 edges of a 1×1×1 cube (slightly inflated so the lines float
// just outside the targeted block's faces). The vertex shader generates
// the line endpoints from `@builtin(vertex_index)`; no vertex buffer
// needed.

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    sun_dir:       vec4<f32>,
    sun_intensity: f32,
    _pad0:         f32,
    _pad1:         f32,
    _pad2:         f32,
    eye:           vec4<f32>,
    inv_view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct CursorUniform {
    // xyz = world-space corner of the targeted block; w = unused.
    block_min: vec4<f32>,
};
@group(1) @binding(0) var<uniform> cursor: CursorUniform;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // 12 edges × 2 endpoints = 24 vertices.
    var edges = array<vec3<f32>, 24>(
        // bottom square
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0),
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0),
        // top square
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
        vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
        // vertical pillars connecting the two squares
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
    );
    let local = edges[idx];
    // Inflate the cube slightly so its edges float just outside the
    // block's faces — avoids z-fighting with the chunk mesh.
    let pad: f32 = 0.005;
    let inflated = local + (local * 2.0 - vec3<f32>(1.0)) * pad;
    let world = cursor.block_min.xyz + inflated;
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.05, 0.05, 0.05, 1.0);
}
