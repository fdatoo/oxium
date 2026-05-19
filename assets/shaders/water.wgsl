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

// Multi-octave value-noise wave height. Driven by world-space xz and
// `camera.time` so adjacent chunks ripple coherently as the camera
// moves.
//
// We use value noise rather than the more obvious `sin(x) * cos(z)`
// because the product-of-sines approach produces wave crests that
// run parallel to the world axes — when the sun catches those
// crests, the reflection spreads into long parallel stripes across
// the entire surface (the visible artefact a player will read as
// "lines on the water"). Noise has no axis-aligned structure, so
// the wave normals decorrelate over distance and the sun reflection
// breaks into the chaotic shimmer real water produces.
//
// Sampled per-pixel by the fragment shader to derive a fake surface
// normal — the actual mesh geometry stays nearly flat. Doing this
// per-pixel rather than per-vertex matters because greedy meshing
// merges many block-tops into single huge quads; vertex displacement
// alone gives a quad four wave samples at its corners and bilinear-
// interpolates between them, which reads as flat at any reasonable
// camera distance.
fn wave_height(world_xz: vec2<f32>, t: f32) -> f32 {
    let big   = wnoise(world_xz * 0.05 + vec2<f32>( 0.30,  0.20) * t);
    let med   = wnoise(world_xz * 0.13 + vec2<f32>(-0.18,  0.25) * t);
    let small = wnoise(world_xz * 0.27 + vec2<f32>( 0.15, -0.22) * t);
    // Each layer is centred around 0 (subtract 0.5) and weighted so
    // the slow swell dominates and the fine chop is a small detail
    // term on top.
    return (big - 0.5) * 1.00 + (med - 0.5) * 0.55 + (small - 0.5) * 0.25;
}

// Numerical gradient of `wave_height` over `world_xz`. The two
// finite-difference samples give us dh/dx and dh/dz; the surface
// normal of a heightfield (x, h(x,z), z) is then
// `normalize(-dh/dx, 1, -dh/dz)`. Scaled by `amplitude` so the same
// wave field drives both vertex displacement and the per-pixel
// normal perturbation.
fn wave_normal(world_xz: vec2<f32>, t: f32, amplitude: f32) -> vec3<f32> {
    let eps = 0.5;
    let h_x0 = wave_height(world_xz - vec2<f32>(eps, 0.0), t) * amplitude;
    let h_x1 = wave_height(world_xz + vec2<f32>(eps, 0.0), t) * amplitude;
    let h_z0 = wave_height(world_xz - vec2<f32>(0.0, eps), t) * amplitude;
    let h_z1 = wave_height(world_xz + vec2<f32>(0.0, eps), t) * amplitude;
    let dx = (h_x1 - h_x0) / (2.0 * eps);
    let dz = (h_z1 - h_z0) / (2.0 * eps);
    return normalize(vec3<f32>(-dx, 1.0, -dz));
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
    // Note: NO vertex displacement.
    //
    // We used to push the +Y face up/down by `wave_height * 0.28`
    // here. The wave function evaluates to the same value at any
    // given (world x, world z), so adjacent chunks agreed on the
    // *boundary vertex* Y exactly. But the surface inside each
    // chunk is rasterised as TWO triangles meeting on a diagonal,
    // and each triangle's plane equation interpolates Y inside the
    // chunk from its three corners. The slope perpendicular to a
    // shared chunk-boundary edge is determined by the chunk's
    // *interior* corner, which differs between neighbours — so the
    // surface has a *slope* discontinuity at every chunk seam,
    // visible as a 1-2 pixel dark hairline under MSAA (the
    // rasteriser leaves micro-coverage gaps at the crease because
    // adjacent triangles in different draw calls don't share an
    // edge equation).
    //
    // Real shader packs avoid this by faking waves entirely in the
    // fragment shader's normal field (no vertex motion). We already
    // compute `wave_normal` per-pixel below; that drives fresnel +
    // sun reflection so the surface still reads as rippled. The
    // geometry itself stays perfectly planar across chunks.

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

    // Per-pixel surface normal from the wave gradient. Sampled in
    // world space so neighbouring chunks agree on what the wave is
    // doing at the seam — no visible boundary line. The amplitude
    // here drives BOTH how the surface lights specularly and how
    // much the sky reflection wobbles, which is what gives a real
    // water body its lively shimmer.
    var surface_n = in.v_normal;
    if (in.v_face == 2.0) {
        // Only top faces get wave-perturbed normals — side faces of
        // exposed water columns shouldn't pretend to be wavy.
        surface_n = wave_normal(in.v_world.xz, camera.time, 0.45);
    }
    let cos_theta = clamp(dot(view_dir, surface_n), 0.0, 1.0);

    // Fresnel: low (transparent) at head-on, high (mirror) at
    // grazing. Capped at 0.92 — keeps the very edge of the water
    // from being a *perfect* mirror so the sky-reflection colour
    // doesn't lose every trace of water hue.
    let fresnel = fresnel_schlick(cos_theta, 0.04);
    let rgb_fresnel = min(fresnel, 0.92);

    // Base water body colour. Cool deep blue lit by sky-light. The
    // body is mostly hidden by the sky reflection at glancing
    // angles; only really visible when looking nearly straight down.
    let water_body = vec3<f32>(0.04, 0.20, 0.38) * max(in.v_light, 0.10);

    // Sky reflection: read the horizon colour for the lower half of
    // the visible sky and the zenith for the upper. Tonemapping
    // earlier in the shader doesn't apply here — these values feed
    // into the final tonemap pass at the end.
    let horizon = vec3<f32>(0.65, 0.80, 1.00) * camera.sun_intensity
                + vec3<f32>(0.05, 0.07, 0.12) * (1.0 - camera.sun_intensity);
    let zenith  = vec3<f32>(0.30, 0.50, 0.95) * camera.sun_intensity
                + vec3<f32>(0.02, 0.03, 0.07) * (1.0 - camera.sun_intensity);
    // The "reflected up direction" — how vertical the surface is at
    // this fragment — picks how much zenith vs horizon shows.
    let sky_t = clamp(surface_n.y, 0.0, 1.0);
    let sky_reflection = mix(horizon, zenith, sky_t * sky_t);

    // Sun reflection: reflect the view direction across the surface
    // normal, dot against the "to-sun" direction. A *wide* exponent
    // (around 32, much wider than a tight glint) makes the sun
    // smear into a long shimmer trail rather than a single bright
    // pixel — the readable visual cue every shader-pack water uses.
    // The exact wave normal we perturbed above is what breaks the
    // trail up into the moving shimmer you see in screenshots; with
    // a flat normal it'd be a hard ellipse.
    // `camera.sun_dir` already points *from* the camera *toward* the
    // sun (the sky shader uses it as-is for the disc dot product).
    // No negation needed here.
    let to_sun = normalize(camera.sun_dir.xyz);
    let view_reflected = reflect(-view_dir, surface_n);
    let sun_align = max(dot(view_reflected, to_sun), 0.0);
    // Two terms layered: a moderate-width primary trail (exponent
    // 32) and a much wider glow (exponent 8) that bleeds the sun's
    // colour out across the surrounding water like real
    // atmospheric scatter on a water plane at sunset.
    let trail = pow(sun_align, 32.0) * 1.30 * camera.sun_intensity;
    let halo  = pow(sun_align,  8.0) * 0.25 * camera.sun_intensity;
    let sun_color  = vec3<f32>(1.00, 0.92, 0.70);
    let sun_warm   = vec3<f32>(1.00, 0.78, 0.45);
    let sun_glint  = sun_color * trail + sun_warm * halo;

    // Compose: water body → blend toward sky reflection by fresnel,
    // then add the sun trail on top. The trail is bright enough
    // even at low fresnel that the eye reads it as the dominant
    // surface feature, which is exactly the shader-pack signature.
    var rgb = mix(water_body, sky_reflection, rgb_fresnel) + sun_glint;

    // Distance fog: dissolve into the horizon so far water doesn't
    // strip-band against the sky.
    let dist = length(in.v_world - camera.eye.xyz);
    let fog_start = 96.0;
    let fog_end   = 360.0;
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    rgb = mix(rgb, horizon, fog_t);

    rgb = aces_tonemap(rgb);
    rgb = underwater_tint(rgb, in.v_world, camera.time, camera.underwater_factor);

    // Alpha 0.78..0.97 — much more opaque than before. The previous
    // 0.65..0.88 range let beach and chunk-boundary sand bleed
    // through the surface in long diagonal lines because the water
    // wasn't opaque enough to mask anything past the first few
    // blocks. The shader-pack look depends on the surface itself
    // being the visual subject, not the bottom seen through it.
    let alpha = mix(0.78, 0.97, fresnel);

    return vec4<f32>(rgb, alpha);
}
