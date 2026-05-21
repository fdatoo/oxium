// Bloom pass — three fragment entry points fed by a shared fullscreen
// vertex shader. The pipelines in `src/render/pipelines/bloom.rs` pick
// which entry point to use.
//
//   fs_threshold  — pass 0; HDR → bloom[0]. Applies a smoothstep
//                   cutoff so only HDR-bright pixels feed the chain.
//   fs_downsample — passes 1..4; bloom[n] → bloom[n+1]. 13-tap box
//                   filter (the "dual-filter" pattern from the COD
//                   Siggraph 2014 talk).
//   fs_upsample   — passes 5..8; bloom[n+1] → bloom[n] with additive
//                   blend configured at the pipeline level. 3×3 tent.

@group(0) @binding(0) var src_tex:     texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv       = vec2<f32>(x, y);
    return out;
}

const BLOOM_THRESHOLD: f32 = 1.0;
const BLOOM_KNEE:      f32 = 0.5;

// Smoothstep around `threshold` — pixels far above the threshold pass
// through unchanged, pixels at the threshold attenuate to zero. The
// `knee` controls the soft-falloff width.
fn threshold_curve(color: vec3<f32>) -> vec3<f32> {
    let brightness = max(color.r, max(color.g, color.b));
    let soft       = clamp(brightness - BLOOM_THRESHOLD + BLOOM_KNEE, 0.0, 2.0 * BLOOM_KNEE);
    let soft_q     = (soft * soft) / (4.0 * BLOOM_KNEE + 0.0001);
    let mult       = max(soft_q, brightness - BLOOM_THRESHOLD) / max(brightness, 0.0001);
    return color * mult;
}

@fragment
fn fs_threshold(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, src_sampler, in.uv).rgb;
    return vec4<f32>(threshold_curve(c), 1.0);
}

// 13-tap downsample (Jorge Jimenez / Activision 2014). One center +
// four "inner box" 2×2 averages + four corner averages. Reduces
// fireflies vs a 5-tap or 9-tap.
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(src_tex));
    let t = vec2<f32>(1.0) / tex_size;
    let uv = in.uv;

    let a = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-2.0,  2.0)).rgb;
    let b = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 0.0,  2.0)).rgb;
    let c = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 2.0,  2.0)).rgb;
    let d = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-2.0,  0.0)).rgb;
    let e = textureSample(src_tex, src_sampler, uv                              ).rgb;
    let f = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 2.0,  0.0)).rgb;
    let g = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-2.0, -2.0)).rgb;
    let h = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 0.0, -2.0)).rgb;
    let i = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 2.0, -2.0)).rgb;
    let j = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0,  1.0)).rgb;
    let k = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0,  1.0)).rgb;
    let l = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0, -1.0)).rgb;
    let m = textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0, -1.0)).rgb;

    // Weighted average: the inner 2×2 (j,k,l,m) contributes 0.5, the
    // four outer 2×2 box-averages contribute 0.125 each. Sums to 1.0.
    var color = (j + k + l + m) * 0.125;
    color = color + (a + b + d + e) * 0.03125;
    color = color + (b + c + e + f) * 0.03125;
    color = color + (d + e + g + h) * 0.03125;
    color = color + (e + f + h + i) * 0.03125;
    return vec4<f32>(color, 1.0);
}

// 3×3 tent upsample. Output blends additively with the destination via
// pipeline blend state, so the value here is the bloom contribution to
// add, not the final color.
@fragment
fn fs_upsample(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(src_tex));
    let t = vec2<f32>(1.0) / tex_size;
    let uv = in.uv;

    var color = textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0,  1.0)).rgb * 1.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 0.0,  1.0)).rgb * 2.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0,  1.0)).rgb * 1.0;

    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0,  0.0)).rgb * 2.0;
    color = color + textureSample(src_tex, src_sampler, uv                              ).rgb * 4.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0,  0.0)).rgb * 2.0;

    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>(-1.0, -1.0)).rgb * 1.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 0.0, -1.0)).rgb * 2.0;
    color = color + textureSample(src_tex, src_sampler, uv + t * vec2<f32>( 1.0, -1.0)).rgb * 1.0;

    // 1+2+1+2+4+2+1+2+1 = 16
    color = color / 16.0;
    return vec4<f32>(color, 1.0);
}
