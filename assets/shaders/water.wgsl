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
    clip_y_min:        f32,
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

// Group 3: sampleable copy of the opaque pass's MSAA depth buffer.
// `multisampled` because our world pass runs at 4× MSAA; we read
// sample 0 via `textureLoad` — pixel-art-style shorelines don't
// gain meaningfully from a 4-sample resolve.
@group(3) @binding(0) var scene_depth: texture_depth_multisampled_2d;

// Group 4: the planar-reflection texture. The reflection pass renders
// the world from a virtual camera mirrored across the water plane;
// here we sample the result at screen-space UVs distorted by the
// wave normal so the reflected image actually wobbles with the
// waves instead of reading as a perfect mirror.
@group(4) @binding(0) var reflection_tex:     texture_2d<f32>;
@group(4) @binding(1) var reflection_sampler: sampler;

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
    /// Clip-space position of the vertex *before* wave displacement,
    /// in `(x, y, w)` form. The fragment shader uses this for the
    /// reflection lookup so the sampling UV stays locked to the
    /// undisturbed water plane — using the displaced `clip_pos`
    /// would let animated wave Y propagate into the screen-space
    /// UV and shimmer the reflection content every frame even when
    /// the camera was still.
    @location(5) v_undisp_clip:     vec3<f32>,
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
    // Three rotated plane-wave components. Using `sin(dot(p, dir))`
    // produces axis-aligned crests only when `dir` is axis-aligned;
    // we pick three non-orthogonal diagonal directions so the crest
    // ridges run at three different angles and never line up into
    // long parallel bands like the previous sin(x)·cos(z) version
    // did.
    //
    // Frequencies tuned for visible per-block variation: 0.30
    // gives a wavelength of ~21 blocks (long swell), 0.65 gives
    // ~9.5 blocks (medium chop), 1.40 gives ~4.5 blocks (small
    // surface detail). The per-block sampling step is well under
    // Nyquist for all three so the waves don't alias into
    // jagged stair-steps.
    let dir1 = vec2<f32>( 0.71,  0.30);
    let dir2 = vec2<f32>(-0.40,  0.85);
    let dir3 = vec2<f32>( 0.55, -0.55);
    let big   = sin(dot(world_xz, dir1) * 0.30 + t * 1.10);
    let med   = sin(dot(world_xz, dir2) * 0.65 + t * 1.50);
    let small = sin(dot(world_xz, dir3) * 1.40 + t * 2.20);
    return big * 0.55 + med * 0.30 + small * 0.15;
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

// Reconstruct linear-space view-distance from a normalised depth
// value. wgpu uses `[0, 1]` depth (near = 0, far = 1) and our
// camera matrix has `near = 0.05`, `far = 1000.0` — keep this in
// sync with `view_proj` in `render::camera`.
fn linear_depth(d: f32) -> f32 {
    let near = 0.05;
    let far  = 1000.0;
    return near * far / (far - d * (far - near));
}

// Tonemap moved to composite.wgsl as of PR 1 Task 5; the water
// fragment now outputs linear HDR into the Rgba16Float target.

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
    // Wave displacement on the +Y top face. Safe to re-enable now
    // that the mesher emits water tops as 1-block-per-quad (see
    // `emit_water_tops_per_block` in `mesher/greedy.rs`) — at
    // 1-block resolution the slope discontinuities between adjacent
    // quads are tiny and read as part of the wave detail rather
    // than as chunk-boundary hairlines. The earlier 32-block
    // greedy water quads produced visible MSAA seams when
    // displaced; small quads don't.
    // Vertex wave displacement on +Y top faces. Now safe because
    // the mesher emits water tops as 1-block-per-quad
    // (`emit_water_tops_per_block`) — the chunk-boundary slope
    // discontinuities that broke the previous 32-block-greedy
    // version are gone at this resolution, and what was a "seam
    // hairline" becomes part of the wave detail.
    // Capture the *undisplaced* clip-space position now, BEFORE the
    // wave displacement modifies world_pos.y. Used by the fragment
    // shader to compute the screen-space UV for the reflection
    // texture — sampling at the displaced clip position would let
    // animated wave Y propagate into the reflection UV, causing the
    // reflected content to shimmer every frame regardless of camera
    // motion. Keying the lookup off the undisplaced plane keeps the
    // reflection geometrically locked to the world.
    let undisp_clip = camera.view_proj * vec4<f32>(world_pos, 1.0);

    if (face == 2u) {
        // Wave displacement is biased so the *crest* sits at the
        // water-block top and the surface only dips DOWN from
        // there. Otherwise crests rose above the block plane and
        // popped visibly above the surrounding sand bank — you
        // could see the sky between the wave top and the shore.
        //
        // `wave_height` returns roughly [-1, 1]; `(h - 1)` maps
        // that into [-2, 0], scaled by 0.22 → [-0.44, 0] block
        // displacement. Always non-positive, so the surface never
        // exceeds its authoring height.
        let h = wave_height(world_pos.xz, camera.time);
        world_pos.y = world_pos.y + (h - 1.0) * 0.22;
    }

    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.v_undisp_clip = vec3<f32>(undisp_clip.x, undisp_clip.y, undisp_clip.w);
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

    // DEBUG: visualise actual vertex displacement by colour-coding
    // by world Y. If vertices are displaced, we should see Y vary
    // across the surface.
    if (camera.underwater_factor > 0.4 && camera.underwater_factor < 0.6) {
        let dy = in.v_world.y - 62.0;
        let t = (dy + 2.0) * 0.25; // map [-2, 2] -> [0, 1]
        return vec4<f32>(t, 1.0 - t, 0.5, 1.0);
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
        // exposed water columns shouldn't pretend to be wavy. The
        // amplitude here is the *normal-perturbation* strength,
        // separate from the vertex-displacement amplitude.
        // 0.20 gives readable surface ripple without making the
        // reflection content visibly slosh on small camera moves.
        surface_n = wave_normal(in.v_world.xz, camera.time, 0.20);
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

    // Planar reflection sample. The reflection texture was rendered
    // by a virtual camera mirrored across the water plane (see
    // `Renderer::encode_reflection_pass`); the screen-space pixel
    // we're shading corresponds to the same screen-space pixel in
    // the reflection texture. Wave-normal-distorted UVs make the
    // reflection wobble with the surface ripples — without the
    // distortion the reflection would read as a perfect static
    // mirror.
    //
    // `surface_n.xz` carries the wave-induced lateral tilt of the
    // surface normal. Project it through the view space scaled by a
    // small factor to get the distortion vector in NDC.
    // Reflection UV is derived from the *undisplaced* clip-space
    // position so wave Y animation doesn't shimmer the sample point
    // each frame. NDC -> [0,1] UV; Y is flipped because wgpu's NDC
    // has +Y up while texture v=0 is at the top.
    let ndc = in.v_undisp_clip.xy / in.v_undisp_clip.z;
    let base_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    // Distortion magnitude is intentionally small. Wave normals are
    // time-animated, so a large distortion factor multiplied by a
    // changing normal injects a per-frame wobble that reads as a
    // shimmering, motion-amplifying reflection. ~1% of screen width
    // gives just enough ripple to break the perfect-mirror look
    // without making the reflected content slosh around.
    let distort = surface_n.xz * 0.012;
    let refl_uv = clamp(base_uv + distort, vec2<f32>(0.0), vec2<f32>(1.0));
    // Five-tap box blur on the reflection sample. Real water
    // reflections aren't crisp mirrors — there are fine surface
    // capillary waves that scatter light, plus the water column
    // itself diffuses what passes through it. The blur happens
    // *here* (not in the source render) so the cheaper low-res
    // reflection pass stays cheap, and we still control the
    // perceived softness from one place.
    let refl_size = vec2<f32>(textureDimensions(reflection_tex));
    let texel = vec2<f32>(1.0) / refl_size;
    let blur = 1.2; // radius in source-texels — 1.2 ≈ ~3.6 dest pixels
    var refl_sum = textureSampleLevel(reflection_tex, reflection_sampler, refl_uv, 0.0).rgb;
    refl_sum = refl_sum + textureSampleLevel(reflection_tex, reflection_sampler,
        clamp(refl_uv + vec2<f32>( texel.x * blur,  0.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
    refl_sum = refl_sum + textureSampleLevel(reflection_tex, reflection_sampler,
        clamp(refl_uv + vec2<f32>(-texel.x * blur,  0.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
    refl_sum = refl_sum + textureSampleLevel(reflection_tex, reflection_sampler,
        clamp(refl_uv + vec2<f32>( 0.0,  texel.y * blur), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
    refl_sum = refl_sum + textureSampleLevel(reflection_tex, reflection_sampler,
        clamp(refl_uv + vec2<f32>( 0.0, -texel.y * blur), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
    let sky_reflection = refl_sum * 0.2;
    // Fallback for the still-handy horizon colour (used by the
    // distance-fog blend below).
    let horizon = vec3<f32>(0.65, 0.80, 1.00) * camera.sun_intensity
                + vec3<f32>(0.05, 0.07, 0.12) * (1.0 - camera.sun_intensity);

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
    // Three terms layered for a richer reflection:
    //   - `core`:  tight bright centre (exponent 200), the "the sun
    //              is literally reflected here" pixel
    //   - `trail`: medium-width primary trail (exponent 48)
    //   - `halo`:  wide warm glow that scatters the sun's colour
    //              across the surrounding water
    // Tightening `trail` from 32 → 48 and adding the high-exponent
    // core gives the reflection a clear "burning bright in the
    // middle, soft warm edges" shape — much closer to a real
    // shader-pack water glint.
    let core  = pow(sun_align, 200.0) * 2.50 * camera.sun_intensity;
    let trail = pow(sun_align,  48.0) * 1.20 * camera.sun_intensity;
    let halo  = pow(sun_align,   8.0) * 0.25 * camera.sun_intensity;
    let sun_core_color = vec3<f32>(1.00, 0.98, 0.90);
    let sun_color      = vec3<f32>(1.00, 0.92, 0.70);
    let sun_warm       = vec3<f32>(1.00, 0.78, 0.45);
    let sun_glint = sun_core_color * core + sun_color * trail + sun_warm * halo;

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

    // Depth-buffer driven effects. `clip_pos.xy` are already in
    // framebuffer pixel coordinates by the time fs_main runs (wgpu
    // `@builtin(position)` semantics); convert to integer texel
    // coords for `textureLoad`. Sample 0 is fine — pixel-art
    // shorelines don't gain from a 4-sample resolve.
    let pix = vec2<i32>(in.clip_pos.xy);
    let scene_d   = textureLoad(scene_depth, pix, 0);
    let water_d   = in.clip_pos.z;
    let scene_lin = linear_depth(scene_d);
    let water_lin = linear_depth(water_d);
    // `depth_diff` is the world-space distance the camera's view ray
    // travels through water before hitting the bottom. Zero where
    // the water surface IS the bottom (i.e., the camera is grazing
    // a shoreline), grows with depth toward open water.
    let depth_diff = max(0.0, scene_lin - water_lin);

    // Shoreline foam: brighten the surface toward white in a thin
    // ribbon where water meets a shallow bottom. The smoothstep
    // window (0.05..0.6 blocks of vertical separation) is tight on
    // purpose — a wider window flooded shallow rivers entirely
    // (RIVER_CARVE = 3 blocks total, so the whole river center
    // would have been in the foam range). 0.6 blocks gives a fringe
    // about one block wide at the shore, fading cleanly into the
    // depth-tinted body.
    let foam_mask = (1.0 - smoothstep(0.05, 0.6, depth_diff))
                  * (1.0 - fog_t); // fade foam out in the distance
    let foam_color = vec3<f32>(0.95, 0.98, 1.00);
    rgb = mix(rgb, foam_color, foam_mask * 0.85);

    // Depth tint: deeper water reads progressively darker and more
    // saturated, the cheap stand-in for real light absorption.
    //
    // The depth range is tight on purpose. Rivers carve only 3
    // blocks deep and lakes 5 (see `worldgen` constants), so a
    // 14-block range left the tint barely visible. 6 blocks gives
    // the river center a clear shift and the lake center a strong
    // saturated blue. `pow(depth_t, 0.7)` brightens the curve so
    // shallow water leans into the tint earlier without losing the
    // top-out at full depth.
    let depth_t = clamp(depth_diff / 6.0, 0.0, 1.0);
    let depth_curve = pow(depth_t, 0.7);
    let deep_tint = vec3<f32>(0.04, 0.18, 0.32);
    rgb = mix(rgb, deep_tint, depth_curve * 0.75);

    // Tonemap moved to composite; underwater tint stays here for now
    // (Task 6 moves it as well). Output is linear HDR.
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
