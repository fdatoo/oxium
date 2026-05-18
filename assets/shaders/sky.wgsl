// Procedural sky shader.
//
// Renders a full-screen triangle behind everything else. The fragment
// shader paints a vertical gradient and lerps between three palettes
// (night, dusk/dawn, day) driven by `camera.sun_intensity`.
//
// The triangle covers the screen by using the "big triangle" trick: a
// single 3-vertex triangle larger than the viewport, clipped by the
// rasterizer. That's faster and simpler than a quad.

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    sun_dir:       vec4<f32>,
    sun_intensity: f32,
    _pad0:         f32,
    _pad1:         f32,
    _pad2:         f32,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // Three vertices covering NDC, big enough to clip away the corner.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    // Far-plane depth (0.999) so it sits behind everything but doesn't
    // collide with `depth_compare = LessEqual` for actual far geometry.
    out.clip_pos = vec4<f32>(pos[idx], 0.999, 1.0);
    out.ndc      = pos[idx];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let day   = vec3<f32>(0.45, 0.65, 1.00);
    let dusk  = vec3<f32>(0.95, 0.55, 0.30);
    let night = vec3<f32>(0.04, 0.05, 0.10);

    let i = camera.sun_intensity;
    // Dusk-blend window: kicks in low and fades out before full noon.
    let dusk_w = smoothstep(0.0, 0.25, i) - smoothstep(0.25, 0.7, i);
    let sky_col = mix(night, day, smoothstep(0.0, 0.7, i)) + dusk * dusk_w * 0.6;

    // Slight vertical gradient: brighter near the horizon (low |y|).
    let horizon = 1.0 - abs(in.ndc.y) * 0.4;
    return vec4<f32>(sky_col * horizon, 1.0);
}
