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

@group(2) @binding(0) var atlas_tex:     texture_2d<f32>;
@group(2) @binding(1) var atlas_sampler: sampler;

// Atlas geometry — keep in sync with `render::atlas`. The water
// shader samples tile 8 (`water_still.png`) at scrolling UVs to put
// an animated ripple pattern on the surface.
const ATLAS_TILE_COUNT_PER_AXIS: f32 = 4.0;
const TILE_UV_SIZE:              f32 = 1.0 / ATLAS_TILE_COUNT_PER_AXIS;

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
// so adjacent chunks ripple coherently as the camera moves. Three
// octaves at incommensurate frequencies — combines a slow rolling
// swell with a faster choppy detail layer on top, gives the surface
// real motion instead of one repeating wavelength.
fn wave_height(world_xz: vec2<f32>, t: f32) -> f32 {
    let big   = sin(world_xz.x * 0.20 + t * 0.60) * cos(world_xz.y * 0.17 + t * 0.50);
    let med   = sin(world_xz.x * 0.45 + t * 1.30) * cos(world_xz.y * 0.37 + t * 1.10);
    let small = sin(world_xz.x * 0.95 + world_xz.y * 1.05 + t * 2.40);
    return big * 0.50 + med * 0.35 + small * 0.15;
}

// Cheap 2D hash for procedural noise — same shape the sky shader uses.
fn whash2(p: vec2<f32>) -> f32 {
    let h = sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453;
    return fract(h);
}

// Value noise on a unit grid with smoothstep interpolation between
// cell corners. Used for the surface ripple pattern and the
// underwater caustic shimmer.
fn wnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = whash2(i);
    let b = whash2(i + vec2<f32>(1.0, 0.0));
    let c = whash2(i + vec2<f32>(0.0, 1.0));
    let d = whash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Two layers of scrolling value noise. Each layer drifts in a
// different direction so they slide past each other and the
// difference reads as flickering surface ripple — the same trick
// real-time water shaders use for "caustic-like" patterns without
// raymarching.
fn ripple_pattern(world_xz: vec2<f32>, t: f32) -> f32 {
    let a = wnoise(world_xz * 0.32 + vec2<f32>( 0.20,  0.13) * t);
    let b = wnoise(world_xz * 0.21 + vec2<f32>(-0.16,  0.21) * t);
    // Center around 0 and scale to roughly [-0.4, 0.4] so the result
    // can be added directly to the surface brightness term.
    return (a - 0.5) * 0.45 + (b - 0.5) * 0.40;
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

fn underwater_tint(rgb: vec3<f32>, world: vec3<f32>, t: f32, factor: f32) -> vec3<f32> {
    if (factor <= 0.0) {
        return rgb;
    }
    let water_blue = vec3<f32>(0.10, 0.30, 0.45);
    var tinted = mix(rgb, water_blue, factor * 0.65);
    // Reuse the water shader's value-noise primitive for caustics —
    // keeps the underwater pattern coherent with the surface ripple
    // pattern visible from above.
    let a = wnoise(world.xz * 0.35 + vec2<f32>( 0.18,  0.11) * t);
    let b = wnoise(world.xz * 0.27 + vec2<f32>(-0.13,  0.19) * t);
    let caustic = pow(a * b, 2.0) * 0.6;
    let caustic_color = vec3<f32>(0.65, 0.95, 1.0);
    return tinted + caustic_color * caustic * factor;
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
        // Amplitude 0.28 makes the swell visibly roll without
        // breaking the illusion of water sitting at the block grid —
        // anything taller and the wave crests pop above neighbouring
        // sand banks, which reads as broken geometry.
        let h = wave_height(world_pos.xz, camera.time);
        world_pos.y = world_pos.y + h * 0.28;
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

    // Animated surface texture. Sample `water_still.png` (atlas tile
    // 8) at TWO scrolling UVs and combine — the difference reads as
    // glints sliding across the surface, even on still parts of the
    // wave field where the vertex displacement alone wouldn't catch
    // the eye.
    let tile_origin = vec2<f32>(0.0, 2.0) * TILE_UV_SIZE; // tile 8 → (col 0, row 2)
    let uv_a = fract(in.v_world.xz * 0.08 + vec2<f32>( 0.04,  0.03) * camera.time);
    let uv_b = fract(in.v_world.xz * 0.13 + vec2<f32>(-0.05,  0.02) * camera.time);
    let tex_a = textureSampleLevel(atlas_tex, atlas_sampler, tile_origin + uv_a * TILE_UV_SIZE, 0.0).rgb;
    let tex_b = textureSampleLevel(atlas_tex, atlas_sampler, tile_origin + uv_b * TILE_UV_SIZE, 0.0).rgb;
    let surface_tex = (tex_a + tex_b) * 0.5;

    // Surface ripple: animated value-noise pattern modulates the
    // base brightness. Brighter ridges + slightly darker troughs
    // make the surface read as moving even when the vertex wave is
    // gentle.
    let ripple = ripple_pattern(in.v_world.xz, camera.time);
    base = base * (1.0 + ripple * 0.45);
    // Lean the colour toward the sampled water texture so the
    // pixel-art ripple pattern from `water_still.png` is visible on
    // the surface.
    base = mix(base, surface_tex * water_body * 1.8, 0.35);

    // Sun specular. Blinn-Phong, tight exponent → small crisp glint.
    // The ripple noise perturbs the half-vector slightly so the
    // glint *moves* across the surface as the waves roll instead of
    // sitting in a fixed spot.
    let to_light = -camera.sun_dir.xyz;
    let half_vec = normalize(view_dir + to_light);
    let spec_n   = max(dot(in.v_normal, half_vec) + ripple * 0.08, 0.0);
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
    rgb = underwater_tint(rgb, in.v_world, camera.time, camera.underwater_factor);

    // Alpha 0.65..0.88 — never fully transparent (you can always tell
    // there's water), never fully opaque (you can always see *some*
    // of what's underneath).
    let alpha = mix(0.65, 0.88, fresnel);

    return vec4<f32>(rgb, alpha);
}
