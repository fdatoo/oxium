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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Sky gradient palette. `day` is the near-zenith tint; `dusk` warms
    // it when the sun is rising/setting; `night` is the dark fallback.
    let zenith = vec3<f32>(0.30, 0.50, 0.95);
    let horizon = vec3<f32>(0.65, 0.80, 1.00);
    let dusk    = vec3<f32>(0.98, 0.62, 0.35);
    let night   = vec3<f32>(0.05, 0.06, 0.12);

    let i = camera.sun_intensity;
    let dusk_w = smoothstep(0.0, 0.25, i) - smoothstep(0.25, 0.7, i);

    // Vertical gradient: horizon band at the bottom of the screen
    // softly bleeds upward into the zenith colour. `t` is 0 at the
    // horizon (ndc.y = -1) and 1 at the top (ndc.y = +1).
    let t = clamp((in.ndc.y + 1.0) * 0.5, 0.0, 1.0);
    let sky_band_day = mix(horizon, zenith, smoothstep(0.0, 0.6, t));
    let sky_lit = mix(night, sky_band_day, smoothstep(0.0, 0.7, i))
                + dusk * dusk_w * (1.0 - t) * 0.7;

    // Reconstruct a world-space ray for this pixel so we can stamp the
    // sun disc in the correct direction regardless of camera yaw/pitch.
    // Project (ndc.x, ndc.y, +1) (far plane) through inv_view_proj and
    // subtract the camera eye for the direction vector.
    let clip = vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let world = camera.inv_view_proj * clip;
    let world_pos = world.xyz / world.w;
    let ray_dir = normalize(world_pos - camera.eye.xyz);

    // Sun disc: dot product with the sun direction, sharper-step for the
    // disc itself and a wider soft glow around it. Multiplied by
    // `sun_intensity` so the sun fades out below the horizon.
    let sun_axis = normalize(camera.sun_dir.xyz);
    let cos_sun = dot(ray_dir, sun_axis);
    let disc = smoothstep(0.9994, 0.9998, cos_sun);
    let halo = smoothstep(0.97, 1.0, cos_sun);
    let sun_glow = (disc * vec3<f32>(1.00, 0.96, 0.88)
                 +  halo * vec3<f32>(1.00, 0.80, 0.50) * 0.35)
                 * smoothstep(0.0, 0.15, i);

    return vec4<f32>(sky_lit + sun_glow, 1.0);
}
