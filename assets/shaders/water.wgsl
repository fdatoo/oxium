// Water shader.
//
// Draws every fragment whose vertex carries `color.a < 0.95` — the
// non-opaque sentinel the mesher tags water blocks with. Anything else
// gets discarded at the top of `fs_main` so this pipeline only "owns"
// the water surface.
//
// On top of that filter we layer four water-specific effects:
//
//   1. **Wave vertex displacement.** Two-octave sin/cos field over
//      world-space (x, z) pushes every TOP-face vertex (face = +Y) up
//      and down a fraction of a block — enough to make the surface
//      visibly undulate from any angle but small enough to stay
//      grid-aligned at distance.
//   2. **Fresnel-style depth fade.** The shader emits *low alpha* when
//      the view ray hits the water nearly head-on (you see through it
//      to the bottom) and *higher alpha* at grazing angles (the water
//      looks reflective). Real Fresnel for IOR=1.33 lands around
//      F0 ≈ 0.02; we use 0.04 as a slightly punchier default.
//   3. **Sun specular.** Blinn-Phong with a tight high exponent so the
//      sun glint is small and crisp, scaled by the sun's intensity so
//      the glint fades at night.
//   4. **Depth-saturated tint.** Distance from the camera modulates the
//      water colour: shallow water lets the bottom show through with
//      a faint blue cast; deeper water reads progressively darker and
//      more saturated. Cheap stand-in for real underwater absorption
//      until we can read the depth buffer for a true depth-difference.
//
// All four effects are post-tonemapped through the same ACES curve the
// opaque shader uses, so the look is consistent across the pass split.

struct CameraUniform {
    view_proj:         mat4x4<f32>,
    sun_dir:           vec4<f32>,
    sun_intensity:     f32,
    time:              f32,
    underwater_factor: f32,
    _pad2:             f32,
    eye:               vec4<f32>,
    inv_view_proj:     mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct ChunkUniform {
    origin: vec4<f32>,
};
@group(1) @binding(0) var<uniform> chunk: ChunkUniform;

// Atlas bind group is part of the shared pipeline layout so the same
// vertex buffers work in both passes; the water shader doesn't sample
// it, but declaring it keeps the layout identical to the opaque
// pipeline.
@group(2) @binding(0) var atlas_tex:     texture_2d<f32>;
@group(2) @binding(1) var atlas_sampler: sampler;

struct VsIn {
    @location(0) pos_ao:     vec4<u32>,
    @location(2) color:      vec4<f32>,
    @location(3) face_light: vec4<u32>,
    @location(4) tile_uv:    vec4<u32>,
};

struct VsOut {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0) v_color:           vec4<f32>,
    @location(1) v_face:            f32,
    @location(2) v_light:           f32,
    @location(3) v_world:           vec3<f32>,
    @location(4) v_normal:          vec3<f32>,
};

// Multi-octave wave height. Driven by world-space xz and `camera.time`
// so adjacent chunks ripple coherently as the camera moves. Two
// octaves at incommensurate frequencies and phases — keeps the
// surface from looking like a single repeating wavelength.
fn wave_height(world_xz: vec2<f32>, t: f32) -> f32 {
    let a = sin(world_xz.x * 0.45 + t * 1.30) * cos(world_xz.y * 0.37 + t * 1.10);
    let b = sin(world_xz.x * 0.18 + world_xz.y * 0.21 + t * 0.55);
    return a * 0.5 + b * 0.5;
}

// Schlick's Fresnel approximation. `cos_theta` is `dot(view, normal)`.
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m5 = m * m * m * m * m;
    return f0 + (1.0 - f0) * m5;
}

// ACES filmic tonemap — same curve as the opaque shader.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn underwater_tint(rgb: vec3<f32>, factor: f32) -> vec3<f32> {
    let water_blue = vec3<f32>(0.10, 0.30, 0.45);
    return mix(rgb, water_blue, factor * 0.65);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var world_pos = chunk.origin.xyz + vec3<f32>(in.pos_ao.xyz);
    let face = in.face_light.x;

    // Only the +Y top face gets wave displacement. ±X/±Z side faces of
    // water columns stay flat; if they undulated independently the
    // greedy-merged side quads would shear apart visibly. -Y bottom
    // faces also stay flat (you only see them while underwater
    // looking up, where the flat plane is fine).
    if (face == 2u) {
        // Amplitude 0.10 keeps the wave subtle — water still reads as
        // "the top of a block", not a roiling sea.
        let h = wave_height(world_pos.xz, camera.time);
        world_pos.y = world_pos.y + h * 0.10;
    }

    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.v_world = world_pos;
    out.v_color = in.color;
    out.v_face = f32(face);

    // Sky/block light packed in `face_light.y`. Scale each to [0, 1].
    let light_byte = in.face_light.y;
    let sky_l   = f32((light_byte >> 4u) & 0x0Fu) / 15.0;
    let block_l = f32(light_byte & 0x0Fu) / 15.0;
    out.v_light = max(sky_l * camera.sun_intensity, block_l);

    // Outward-facing normal for the lit face. Same ordering as
    // `mesher::Face` — keep in sync if either side changes.
    if (face == 0u)      { out.v_normal = vec3<f32>( 1.0,  0.0,  0.0); }
    else if (face == 1u) { out.v_normal = vec3<f32>(-1.0,  0.0,  0.0); }
    else if (face == 2u) { out.v_normal = vec3<f32>( 0.0,  1.0,  0.0); }
    else if (face == 3u) { out.v_normal = vec3<f32>( 0.0, -1.0,  0.0); }
    else if (face == 4u) { out.v_normal = vec3<f32>( 0.0,  0.0,  1.0); }
    else                 { out.v_normal = vec3<f32>( 0.0,  0.0, -1.0); }

    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Opaque-mesh fragments piggyback on this pipeline (we share one
    // vertex buffer) — toss every non-water fragment so they don't
    // double-shade after the opaque pass already drew them.
    if (in.v_color.a >= 0.95) {
        discard;
    }

    let view_dir = normalize(camera.eye.xyz - in.v_world);
    let cos_theta = clamp(dot(view_dir, in.v_normal), 0.0, 1.0);

    // Fresnel: low at head-on, high at grazing. We cap the *visual*
    // contribution (`rgb_fresnel`) at 0.55 so even glancing water
    // doesn't turn into a mirror — at our pixel-art fidelity a real
    // unclamped Schlick fresnel makes the water read like the sky
    // with a few wave wrinkles, which loses the water character
    // entirely. Alpha uses the unclamped value because the *opacity*
    // really should grow at grazing angles (more vertical column of
    // water absorbs more light).
    let fresnel = fresnel_schlick(cos_theta, 0.04);
    let rgb_fresnel = min(fresnel, 0.55);

    // Base water colour. Saturated deep blue rather than the vertex
    // tint's lighter shade — vertex `color` exists so the mesher /
    // physics can identify water, but the rendered colour is
    // shader-defined. Lit by sky-light so caves don't glow blue.
    let water_body = vec3<f32>(0.05, 0.32, 0.55);
    var base = water_body * max(in.v_light, 0.05);

    // Distance-saturation: shallow-near reads brighter, deep-far
    // reads darker. Stand-in for real depth-buffer absorption.
    let dist = length(in.v_world - camera.eye.xyz);
    let depth_t = clamp(dist / 120.0, 0.0, 1.0);
    let deep    = base * 0.55 + vec3<f32>(0.01, 0.06, 0.12);
    base = mix(base, deep, depth_t);

    // Sun specular. Blinn-Phong, tight exponent → small crisp glint.
    let to_light = -camera.sun_dir.xyz;
    let half_vec = normalize(view_dir + to_light);
    let spec_n   = max(dot(in.v_normal, half_vec), 0.0);
    let specular = pow(spec_n, 120.0) * camera.sun_intensity;

    // Sky reflection colour: tinted toward water blue so even when
    // fresnel hits the cap the surface stays believably watery.
    let sky_reflection = vec3<f32>(0.30, 0.55, 0.85);
    var rgb = mix(base, sky_reflection, rgb_fresnel);

    // Additive sun glint — punches through at any angle where the
    // half-vector lines up with the surface normal, independent of
    // the fresnel blend.
    let sun_glint = vec3<f32>(1.00, 0.95, 0.80);
    rgb = rgb + sun_glint * specular;

    // Distance fog: dissolve into the horizon so far water doesn't
    // strip-band against the sky.
    let fog_start = 96.0;
    let fog_end   = 360.0;
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    let fog_col = vec3<f32>(0.55, 0.72, 0.95) * camera.sun_intensity
                + vec3<f32>(0.04, 0.05, 0.10) * (1.0 - camera.sun_intensity);
    rgb = mix(rgb, fog_col, fog_t);

    rgb = aces_tonemap(rgb);
    rgb = underwater_tint(rgb, camera.underwater_factor);

    // Alpha 0.65..0.88 — never fully transparent (you can always tell
    // there's water), never fully opaque (you can always see *some*
    // of what's underneath).
    let alpha = mix(0.65, 0.88, fresnel);

    return vec4<f32>(rgb, alpha);
}
