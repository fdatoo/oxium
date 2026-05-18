// Procedural sky shader.
//
// Renders a full-screen triangle behind everything else. The fragment
// shader paints a vertical gradient driven by `sun_intensity`, fades in
// a horizon band that matches `opaque.wgsl`'s fog colour, and stamps a
// bright sun disc + soft halo wherever the camera's view ray gets close
// to `sun_dir`.

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

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // Three vertices covering NDC, big enough to clip the corner.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    // Far-plane depth (0.999) so opaque geometry overdraws via the
    // pipeline's `LessEqual` depth compare.
    out.clip_pos = vec4<f32>(pos[idx], 0.999, 1.0);
    out.ndc      = pos[idx];
    return out;
}

// Hash → 0..1 pseudo-random. Deterministic per integer coordinate.
// Cheap, gpu-friendly, no precision issues out to thousands of units.
fn hash2(p: vec2<f32>) -> f32 {
    let h = sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453;
    return fract(h);
}

// Bilinear value noise on a unit grid, with Perlin-style smooth-step
// interpolation between cell corners.
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);  // smoothstep
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 4-octave fractional Brownian motion. Gives the fluffy-edge cloud
// silhouettes you want without needing real Perlin/Simplex code.
fn fbm(p: vec2<f32>) -> f32 {
    var sum: f32 = 0.0;
    var amp: f32 = 0.5;
    var pp = p;
    for (var i: i32 = 0; i < 4; i++) {
        sum += amp * value_noise(pp);
        pp = pp * 2.03;       // slightly non-integer scale → less grid lattice
        amp = amp * 0.5;
    }
    return sum;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Sky gradient palette.
    let zenith  = vec3<f32>(0.30, 0.50, 0.95);
    let horizon = vec3<f32>(0.65, 0.80, 1.00);
    let dusk    = vec3<f32>(0.98, 0.62, 0.35);
    let night   = vec3<f32>(0.05, 0.06, 0.12);

    let i = camera.sun_intensity;
    let dusk_w = smoothstep(0.0, 0.25, i) - smoothstep(0.25, 0.7, i);

    let t = clamp((in.ndc.y + 1.0) * 0.5, 0.0, 1.0);
    let sky_band_day = mix(horizon, zenith, smoothstep(0.0, 0.6, t));
    var sky_lit = mix(night, sky_band_day, smoothstep(0.0, 0.7, i))
                + dusk * dusk_w * (1.0 - t) * 0.7;

    // Reconstruct a world-space ray for this pixel.
    let clip = vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let world = camera.inv_view_proj * clip;
    let world_pos = world.xyz / world.w;
    let ray_dir = normalize(world_pos - camera.eye.xyz);

    // ── Clouds: a single horizontal layer at altitude CLOUD_Y. For
    // each upward-looking fragment, intersect its view ray with the
    // plane y = CLOUD_Y and sample fbm at the hit's (x, z). Wind
    // scrolls the noise origin with `camera.time`. Density is
    // smoothstepped from a threshold to produce soft edges; cloud
    // colour leans warm where the sun glow would hit. Clouds also
    // *occlude the sun*: dense clouds suppress the sun-disc glow at
    // that fragment.
    let cloud_altitude = 140.0;
    var cloud_density: f32 = 0.0;
    var cloud_color = vec3<f32>(1.0);
    if (ray_dir.y > 0.02) {
        let dy = cloud_altitude - camera.eye.y;
        if (dy > 0.0) {
            let t_hit = dy / ray_dir.y;
            // Use eye + ray*t cleanly in vec3 then take xz.
            let hit3 = camera.eye.xyz + ray_dir * t_hit;
            let cloud_pos = hit3.xz;
            let wind = vec2<f32>(3.0, 1.0) * camera.time;
            // Larger noise scale (smaller per-unit value) so clouds
            // span tens of blocks each rather than fragmenting into
            // pixel-sized speckle.
            let n = fbm((cloud_pos + wind) * 0.012);
            cloud_density = smoothstep(0.42, 0.62, n);
            let horiz_falloff = smoothstep(0.02, 0.20, ray_dir.y);
            cloud_density = cloud_density * horiz_falloff;
            let sun_lean = max(0.0, dot(ray_dir, normalize(camera.sun_dir.xyz)));
            let lit_white = mix(vec3<f32>(0.78, 0.80, 0.85),
                                vec3<f32>(1.00, 0.97, 0.90),
                                pow(sun_lean, 1.5));
            cloud_color = lit_white * (0.4 + 0.6 * i);
        }
    }

    // Sun disc (sharper) + warm halo (wider). Sun is occluded by
    // clouds in front of it.
    let sun_axis = normalize(camera.sun_dir.xyz);
    let cos_sun = dot(ray_dir, sun_axis);
    let disc = smoothstep(0.9994, 0.9998, cos_sun);
    let halo = smoothstep(0.97, 1.0, cos_sun);
    let sun_glow = (disc * vec3<f32>(1.00, 0.96, 0.88)
                 +  halo * vec3<f32>(1.00, 0.80, 0.50) * 0.35)
                 * smoothstep(0.0, 0.15, i);
    let sun_visible = sun_glow * (1.0 - cloud_density * 0.85);

    // Mix the cloud over the sky-lit background; add sun_visible on top.
    sky_lit = mix(sky_lit, cloud_color, cloud_density);
    return vec4<f32>(sky_lit + sun_visible, 1.0);
}
