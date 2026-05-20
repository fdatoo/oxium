// Composite pass: samples the HDR color target and writes to the
// swapchain. Task 3 ships this as a pure passthrough — Tasks 5 and 6
// fold in ACES tonemap and underwater tint respectively. Bloom and
// volumetrics land in later PRs.

@group(0) @binding(0) var hdr_tex:     texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle. Three vertices that fully cover the screen with
// the texture spanning u,v = 0..1 over the visible quad. No vertex
// buffer required — the shader picks corner positions from vertex_index.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);   // 0, 2, 0
    let y = f32(vid & 2u);            // 0, 0, 2
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv       = vec2<f32>(x, y);
    return out;
}

// ACES filmic tone mapping — moved here from the world shaders so the
// HDR target stores unclamped linear values that future PRs (bloom)
// can read pre-tonemap. Fitted approximation by Krzysztof Narkowicz.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr    = textureSample(hdr_tex, hdr_sampler, in.uv).rgb;
    let mapped = aces_tonemap(hdr);
    return vec4<f32>(mapped, 1.0);
}
