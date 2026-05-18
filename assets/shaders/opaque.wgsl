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
// camera uniform. The fragment shader does simple AO + face-direction
// shading; light/sun handling is added in M5.

struct CameraUniform {
    view_proj: mat4x4<f32>,
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
    // Move local coords (0..=32) into world space by adding the chunk origin.
    let world_pos = chunk.origin.xyz + vec3<f32>(in.pos_ao.xyz);

    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);

    // Face-direction directional shade: top brightest, sides medium,
    // bottom darkest. Makes voxel cubes readable even before real lighting.
    let face = in.face_light.x;
    var face_mul: f32 = 1.0;
    if (face == 2u) { face_mul = 1.00; }       // +Y top
    else if (face == 3u) { face_mul = 0.55; }  // -Y bottom
    else { face_mul = 0.80; }                  // any side

    out.v_color = in.color * face_mul;
    out.v_ao    = f32(in.pos_ao.w) / 3.0;
    out.v_light = 1.0;   // M5 will read the real `light` byte here
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Lerp the AO factor into a sensible darkening range so even fully
    // occluded corners stay readable (not pitch-black).
    let ao    = mix(0.55, 1.0, in.v_ao);
    let shade = ao * in.v_light;
    return vec4<f32>(in.v_color.rgb * shade, in.v_color.a);
}
