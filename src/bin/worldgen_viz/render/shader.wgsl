struct Camera {
    view_proj: mat4x4<f32>,
    // xyz = selected chunk coord (integer-valued floats so the layout
    // stays std140-friendly). w = selection enabled flag (1.0 if a
    // chunk is currently pinned, 0.0 otherwise).
    selected_chunk: vec4<f32>,
    // x = elapsed seconds, used to animate the chunk-tint pulse.
    // y = cutaway max-Y (fragments with world Y above this discard,
    //     shaving the top off the world so the user can see caves).
    //     A very large value (e.g. 1e9) disables the cutaway.
    // z/w reserved.
    time: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct VsIn { @location(0) pos: vec3<f32>, @location(1) color: vec3<f32>, };
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.pos = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    out.world_pos = in.pos;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Cutaway: discard any fragment whose world Y is above the
    // user-selected cap. Lets you "shave off" the surface to see
    // caves directly. cam.time.y == 1e9 (or any sufficiently large
    // value) disables the cutaway.
    if (in.world_pos.y > cam.time.y) {
        discard;
    }

    // Mild fake lighting: brighten high blocks slightly.
    var lit = in.color * (0.7 + 0.3 * clamp((in.world_pos.y - 40.0) / 100.0, 0.0, 1.0));

    // Pinned-chunk tint: pulse a warm yellow over any fragment whose
    // world XZ falls inside the selected chunk's 32-block footprint,
    // at any Y. (We tint the full vertical column rather than a single
    // chunk so the highlight stays visible no matter where the camera
    // is in the world.) The cosine drives a ~1 s cycle between 0 and 1.
    if (cam.selected_chunk.w > 0.5) {
        let min_x = cam.selected_chunk.x * 32.0;
        let min_z = cam.selected_chunk.z * 32.0;
        let inside =
            in.world_pos.x >= min_x && in.world_pos.x < min_x + 32.0 &&
            in.world_pos.z >= min_z && in.world_pos.z < min_z + 32.0;
        if (inside) {
            let pulse = 0.5 + 0.5 * cos(cam.time.x * 4.0);
            let tint = vec3<f32>(1.0, 0.85, 0.25);
            lit = mix(lit, tint, 0.55 * pulse);
        }
    }

    return vec4<f32>(lit, 1.0);
}
