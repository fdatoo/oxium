struct Camera { view_proj: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> cam: Camera;

struct VsIn { @location(0) pos: vec3<f32>, @location(1) color: vec3<f32>, };
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_y: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.pos = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    out.world_y = in.pos.y;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Mild fake lighting: brighten high blocks slightly. Matches today's
    // viewer aesthetic — replace with real shading in PR 3.
    let lit = in.color * (0.7 + 0.3 * clamp((in.world_y - 40.0) / 100.0, 0.0, 1.0));
    return vec4<f32>(lit, 1.0);
}
