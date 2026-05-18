// Opaque chunk shader.
//
// Receives a 16-byte packed vertex as three vec4<u32> attributes (declared in
// `pipelines/opaque.rs`):
//
//   location 0 = pos_ao     :  pos.xyz (u8) + ao (u8)
//   location 2 = color      :  RGBA, normalised by the format Unorm8x4
//   location 3 = face_light :  normal_face (u8) + light (u8) + 2-byte pad
//
// The vertex shader lifts local 0..=32 coords into world space using the
// per-chunk uniform's `origin`, then projects with the view-proj from the
// camera uniform. The fragment shader combines:
//
//   * face-direction tint (top brightest, bottom darkest)
//   * baked vertex AO (0..3 → 0.45..1.0 brightness)
//   * lighting: max(sky × sun_intensity, block)
//   * distance fog: fade to the same sky palette beyond ~80 blocks
//
// `sun_intensity` is supplied by the time-of-day system. At midnight it
// drops to 0 and torches dominate; at noon the sky channel wins.

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    sun_dir:       vec4<f32>,
    sun_intensity: f32,
    time:          f32,
    _pad1:         f32,
    _pad2:         f32,
    eye:           vec4<f32>,
    inv_view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct ChunkUniform {
    origin: vec4<f32>,   // xyz = chunk origin in world space; w unused
};
@group(1) @binding(0) var<uniform> chunk: ChunkUniform;

struct VsIn {
    @location(0) pos_ao:       vec4<u32>,   // xyz = pos (0..32), w = ao (0..3)
    @location(2) color:        vec4<f32>,   // already normalised to [0,1]
    @location(3) face_light:   vec4<u32>,   // x = normal_face, y = light, zw = pad
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) v_color: vec4<f32>,
    @location(1) v_ao:    f32,
    @location(2) v_light: f32,
    @location(3) v_world: vec3<f32>,   // world-space fragment pos for fog
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world_pos = chunk.origin.xyz + vec3<f32>(in.pos_ao.xyz);

    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.v_world = world_pos;

    let face = in.face_light.x;
    var face_mul: f32 = 0.80;
    if (face == 2u) { face_mul = 1.00; }       // +Y top
    else if (face == 3u) { face_mul = 0.55; }  // -Y bottom
    // face_mul applies to RGB only — leaving alpha untouched means the
    // fragment shader can reliably identify water by `v_color.a < 0.95`
    // (water is the only block with alpha != 1.0 at the vertex source).
    // Otherwise face_mul=0.80 on opaque sides + face_mul=0.55 on
    // opaque bottoms would falsely match the water threshold and
    // trigger the water shimmer code path on leaves / stone undersides.
    out.v_color = vec4<f32>(in.color.rgb * face_mul, in.color.a);
    out.v_ao    = f32(in.pos_ao.w) / 3.0;

    // Light byte: high nibble = sky, low nibble = block. Scale each
    // independently to [0,1], then take max — the brighter source wins
    // (e.g. torches at night dominate the near-zero sky channel).
    let light_byte = in.face_light.y;
    let sky_l   = f32((light_byte >> 4u) & 0x0Fu) / 15.0;
    let block_l = f32(light_byte & 0x0Fu) / 15.0;
    out.v_light = max(sky_l * camera.sun_intensity, block_l);
    return out;
}

// Compute the colour of the sky at the horizon, used as the fog tint.
// Mirrors the gradient logic in sky.wgsl so distant terrain dissolves
// seamlessly into the sky's horizon band rather than into a flat grey.
fn horizon_color(sun_intensity: f32) -> vec3<f32> {
    let day   = vec3<f32>(0.65, 0.80, 1.00);   // brighter than zenith
    let dusk  = vec3<f32>(0.98, 0.62, 0.35);
    let night = vec3<f32>(0.05, 0.06, 0.12);
    let i = sun_intensity;
    let dusk_w = smoothstep(0.0, 0.25, i) - smoothstep(0.25, 0.7, i);
    return mix(night, day, smoothstep(0.0, 0.7, i)) + dusk * dusk_w * 0.6;
}

// Two-octave sin/cos wave used to animate water surfaces. Driven by
// world-space x/z and `camera.time` so adjacent quads stay coherent as
// the camera moves.
fn water_shimmer(world: vec3<f32>, t: f32) -> f32 {
    let a = sin(world.x * 0.45 + t * 1.30) * cos(world.z * 0.37 + t * 1.10);
    let b = sin(world.x * 0.18 + world.z * 0.21 + t * 0.55);
    return a * 0.5 + b * 0.5;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ao = mix(0.45, 1.0, in.v_ao);
    let lit = max(0.05, in.v_light);
    let shade = ao * lit;
    var lit_rgb = in.v_color.rgb * shade;

    // Water shimmer: any fragment whose vertex alpha came in below
    // ~0.95 is non-opaque material — water in v0. Modulate brightness
    // *and* bias the colour toward a cooler/warmer tint with the
    // shimmer factor, both keyed off world space + time so adjacent
    // greedy-merged water quads stay coherent.
    if (in.v_color.a < 0.95) {
        let s = water_shimmer(in.v_world, camera.time);
        // Brightness ripple: ±30 % around the lit colour.
        lit_rgb = lit_rgb * (1.0 + 0.30 * s);
        // Hue lean: bright crests get a touch of foam-cyan, troughs
        // a touch of deeper blue.
        let crest = vec3<f32>(0.65, 0.90, 1.00);
        let trough = vec3<f32>(0.05, 0.15, 0.45);
        lit_rgb = mix(lit_rgb, mix(trough, crest, s * 0.5 + 0.5), 0.30);
    }

    // Distance fog: linear ramp between FOG_START and FOG_END.
    let dist = length(in.v_world - camera.eye.xyz);
    let fog_start = 96.0;
    let fog_end   = 360.0;
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    let fog_col = horizon_color(camera.sun_intensity);
    let out_rgb = mix(lit_rgb, fog_col, fog_t);

    return vec4<f32>(out_rgb, in.v_color.a);
}
