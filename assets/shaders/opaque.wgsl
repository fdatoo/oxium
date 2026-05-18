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
//
// `sun_intensity` is supplied by the time-of-day system. At midnight it
// drops to 0 and torches dominate; at noon the sky channel wins.

struct CameraUniform {
    view_proj:     mat4x4<f32>,   //  0  ..  64
    sun_dir:       vec4<f32>,     // 64  ..  80
    sun_intensity: f32,           // 80  ..  84
    _pad0:         f32,           // 84  ..  88
    _pad1:         f32,           // 88  ..  92
    _pad2:         f32,           // 92  ..  96   (struct size: 96 bytes)
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
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world_pos = chunk.origin.xyz + vec3<f32>(in.pos_ao.xyz);

    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);

    let face = in.face_light.x;
    var face_mul: f32 = 0.80;
    if (face == 2u) { face_mul = 1.00; }       // +Y top
    else if (face == 3u) { face_mul = 0.55; }  // -Y bottom
    out.v_color = in.color * face_mul;
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ao = mix(0.45, 1.0, in.v_ao);
    // Tiny ambient floor so the deep dark isn't pitch black — easier to
    // see what we're walking into in caves.
    let lit = max(0.05, in.v_light);
    let shade = ao * lit;
    return vec4<f32>(in.v_color.rgb * shade, in.v_color.a);
}
