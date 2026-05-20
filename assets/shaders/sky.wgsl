// Procedural sky shader.
//
// Renders a full-screen triangle behind everything else. The fragment
// shader paints a vertical gradient driven by `sun_intensity`, fades in
// a horizon band that matches `opaque.wgsl`'s fog colour, and stamps a
// bright sun disc + soft halo wherever the camera's view ray gets close
// to `sun_dir`.

struct CameraUniform {
    view_proj:         mat4x4<f32>,
    sun_dir:           vec4<f32>,
    sun_color:         vec4<f32>,
    sky_color:         vec4<f32>,
    sun_intensity:     f32,
    time:              f32,
    underwater_factor: f32,
    clip_y_min:        f32,
    eye:               vec4<f32>,
    inv_view_proj:     mat4x4<f32>,
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

// Star field: hash the upper-hemisphere view ray and stamp bright
// pinpoints where the hash crosses a high threshold. Sampling on
// (ray.x, ray.z) / ray.y projects the hemisphere onto a flat plane,
// so star density and size stay roughly uniform with altitude.
fn star_field(ray: vec3<f32>) -> f32 {
    if (ray.y < 0.05) {
        return 0.0;
    }
    // Project the hemisphere onto an XZ plane and sample at a high
    // frequency — each unit corresponds to one potential star cell.
    let p = vec2<f32>(ray.x, ray.z) / ray.y * 140.0;
    let cell = floor(p);
    let f = fract(p) - 0.5;
    let h = hash2(cell);
    // ~3 % of cells host a star; the brightest stars come from cells
    // whose hash crosses well above the threshold.
    let star = step(0.97, h);
    let d = length(f);
    // Tight glow: sharp pinpoint with a tiny falloff so stars are
    // crisp rather than blurry.
    let glow = pow(max(0.0, 1.0 - d * 2.0), 8.0);
    let bright = (h - 0.97) / 0.03;
    return star * glow * bright;
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

// 5-octave fractional Brownian motion. One more octave than the previous
// 4-octave version — the extra detail puts visible wisps on the edges of
// the larger cloud masses so the silhouettes don't look airbrushed.
// Gives the fluffy-edge cloud silhouettes you want without needing real
// Perlin/Simplex code.
fn fbm(p: vec2<f32>) -> f32 {
    var sum: f32 = 0.0;
    var amp: f32 = 0.5;
    var pp = p;
    for (var i: i32 = 0; i < 5; i++) {
        sum += amp * value_noise(pp);
        pp = pp * 2.03;       // slightly non-integer scale → less grid lattice
        amp = amp * 0.5;
    }
    return sum;
}

// Tonemap moved to composite.wgsl as of PR 1 Task 5. The sky shader
// now outputs linear HDR into the Rgba16Float target; the sun disc
// stays unclamped so composite's tonemap can preserve its warm core.

// Underwater tint moved to composite.wgsl (PR 1 Task 6).

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
            let n = fbm((cloud_pos + wind) * 0.025);
            // Sharper threshold window (tighter 0.48..0.62) gives the
            // clouds more defined silhouettes — the previous wider
            // band faded them into uniform haze.
            cloud_density = smoothstep(0.48, 0.62, n);
            let horiz_falloff = smoothstep(0.02, 0.20, ray_dir.y);
            cloud_density = cloud_density * horiz_falloff;
            let sun_lean = max(0.0, dot(ray_dir, normalize(camera.sun_dir.xyz)));
            // Brighter sun-side highlight + slightly darker shaded
            // side so clouds have visible volume rather than reading
            // as a flat overlay.
            let shaded   = vec3<f32>(0.68, 0.71, 0.78);
            let lit_warm = vec3<f32>(1.05, 1.00, 0.92);
            let lit_white = mix(shaded, lit_warm, pow(sun_lean, 1.2));
            cloud_color = lit_white * (0.4 + 0.6 * i);
        }
    }

    // Sun composition: a bright central disc + warm halo + a wide
    // soft bloom that scatters light across half the sky on a clear
    // day. The bloom is what makes the area around the sun feel
    // "glowing" rather than just "yellow circle on blue gradient".
    //
    // - `disc`: tiny crisp solid sun (cos_sun ≈ 1).
    // - `halo`: warm corona of a few degrees.
    // - `bloom`: very wide low-intensity falloff (cos_sun > 0.5),
    //   simulates atmospheric scatter around a bright source.
    let sun_axis = normalize(camera.sun_dir.xyz);
    let cos_sun = dot(ray_dir, sun_axis);
    let disc  = smoothstep(0.9994, 0.9998, cos_sun);
    let halo  = smoothstep(0.97,  1.0,    cos_sun);
    let bloom = pow(max(cos_sun, 0.0), 8.0);
    let sun_glow = (disc  * vec3<f32>(1.20, 1.10, 0.95)
                 +  halo  * vec3<f32>(1.00, 0.80, 0.50) * 0.35
                 +  bloom * vec3<f32>(1.00, 0.85, 0.60) * 0.25)
                 * smoothstep(0.0, 0.15, i);
    let sun_visible = sun_glow * (1.0 - cloud_density * 0.85);

    // Night factor: zero at noon, ramps in as the sun drops. The moon
    // and stars both fade in with this so they don't fight the day sky.
    let night_factor = 1.0 - smoothstep(0.05, 0.35, i);

    // Moon: same disc shape as the sun but in the anti-sun direction
    // and tinted cool. A touch larger (smoothstep at 0.9988) so it
    // reads as the moon's apparent size.
    let moon_axis = -sun_axis;
    let cos_moon = dot(ray_dir, moon_axis);
    let moon_disc = smoothstep(0.9988, 0.9996, cos_moon);
    let moon_halo = smoothstep(0.985, 0.9996, cos_moon);
    let moon_glow = (moon_disc * vec3<f32>(0.95, 0.96, 1.00)
                  +  moon_halo * vec3<f32>(0.45, 0.50, 0.70) * 0.20)
                  * night_factor;
    let moon_visible = moon_glow * (1.0 - cloud_density * 0.85);

    // Stars: high-frequency pinpoint field on the upper hemisphere,
    // multiplied by `night` so they invisible during the day, and by
    // `(1 - cloud_density)` so dense clouds occlude them.
    let s = star_field(ray_dir);
    let stars_rgb = vec3<f32>(0.95, 0.96, 1.00) * s * 2.0 * night_factor * (1.0 - cloud_density);

    // Mix the cloud over the sky-lit background; add the celestial
    // bodies + stars on top.
    sky_lit = mix(sky_lit, cloud_color, cloud_density);
    let rgb = sky_lit + sun_visible + moon_visible + stars_rgb;
    // Tonemap + underwater tint both live in composite.wgsl as of
    // PR 1 Tasks 5/6. Output is linear HDR.
    return vec4<f32>(rgb, 1.0);
}
