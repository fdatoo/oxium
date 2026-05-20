// Composite pass: samples the HDR color target, applies the post chain,
// writes to the swapchain.
//
// Post chain order:
//   1. ACES tonemap   — moved from world shaders in Task 5
//   2. Underwater grade — moved from world shaders in Task 6
//
// Underwater caustics use screen-space noise instead of the world-space
// noise the world shaders used. Visually near-identical and avoids
// reconstructing world position from depth in this pass.

struct CameraUniform {
    view_proj:         mat4x4<f32>,
    sun_dir:           vec4<f32>,
    sun_color:         vec4<f32>,
    sky_color:         vec4<f32>,
    sun_intensity:     f32,
    time:              f32,
    underwater_factor: f32,
    clip_y_min:        f32,
    eye:               vec4<f32>,
    inv_view_proj:     mat4x4<f32>,
};

@group(0) @binding(0) var          hdr_tex:     texture_2d<f32>;
@group(0) @binding(1) var          hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> camera:      CameraUniform;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle. See vs_main comment in earlier revisions.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);   // 0, 2, 0
    let y = f32(vid & 2u);            // 0, 0, 2
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv       = vec2<f32>(x, y);
    return out;
}

// ACES filmic tone mapping. Fitted approximation by Krzysztof Narkowicz.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

fn uw_hash(p: vec2<f32>) -> f32 {
    let h = sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453;
    return fract(h);
}

fn uw_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = uw_hash(i);
    let b = uw_hash(i + vec2<f32>(1.0, 0.0));
    let c = uw_hash(i + vec2<f32>(0.0, 1.0));
    let d = uw_hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Underwater colour grade — same maths as the previous in-shader copy,
// but the noise domain is screen-space (UV * 20) rather than world-space.
fn underwater_tint(rgb: vec3<f32>, screen_uv: vec2<f32>, t: f32, factor: f32) -> vec3<f32> {
    if (factor <= 0.0) {
        return rgb;
    }
    let water_blue = vec3<f32>(0.10, 0.30, 0.45);
    var tinted = mix(rgb, water_blue, factor * 0.65);
    let domain = screen_uv * 20.0;
    let a = uw_noise(domain * 0.35 + vec2<f32>( 0.18,  0.11) * t);
    let b = uw_noise(domain * 0.27 + vec2<f32>(-0.13,  0.19) * t);
    let caustic = pow(a * b, 2.0) * 0.6;
    let caustic_color = vec3<f32>(0.65, 0.95, 1.0);
    tinted = tinted + caustic_color * caustic * factor;
    return tinted;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr     = textureSample(hdr_tex, hdr_sampler, in.uv).rgb;
    let mapped  = aces_tonemap(hdr);
    let out_rgb = underwater_tint(mapped, in.uv, camera.time, camera.underwater_factor);
    return vec4<f32>(out_rgb, 1.0);
}
