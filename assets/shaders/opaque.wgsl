// Opaque chunk shader.
//
// Receives a 16-byte packed vertex as four vec4<u32> attributes (declared in
// `pipelines/opaque.rs`):
//
//   location 0 = pos_ao     :  pos.xyz (u8) + ao (u8)
//   location 2 = color      :  RGBA tint, normalised by Unorm8x4
//   location 3 = face_light :  normal_face (u8) + light (u8) + 2-byte pad
//   location 4 = tile_uv    :  tile_index (u8) + u_tile (u8) + v_tile (u8) + pad
//
// The vertex shader lifts local 0..=32 coords into world space using the
// per-chunk uniform's `origin`, then projects with the view-proj from the
// camera uniform. It also computes the per-corner tile-space UV (in
// "tiles", so a 1×1 quad has UVs spanning (0,0)..(1,1)) and looks up the
// tile's origin within the atlas. The fragment shader then:
//
//   * samples the atlas at `tile_origin + fract(uv) * tile_size` — the
//     `fract` is what makes a greedy w×h quad repeat the tile per block
//   * multiplies the texel by the vertex colour (biome tint; grayscale
//     `grass_block_top.png` × green vertex colour = green grass top)
//   * applies face-direction tint (top brightest, bottom darkest)
//   * applies baked vertex AO (0..3 → 0.45..1.0 brightness)
//   * combines lighting: max(sky × sun_intensity, block)
//   * fades into the horizon colour with distance fog
//
// Sentinel tile index 0xFF (= 255) means "untextured" — the shader
// skips the atlas sample and uses the vertex colour directly. That
// path is for blocks like Torch / Air which don't have an atlas entry.

struct CameraUniform {
    view_proj:         mat4x4<f32>,
    sun_dir:           vec4<f32>,
    sun_color:         vec4<f32>,
    sky_color:         vec4<f32>,
    sun_intensity:     f32,
    time:              f32,
    // 0 = camera in air, 1 = submerged. Every fragment colour-grades
    // toward a deep blue tint scaled by this; it's the cheap
    // alternative to a dedicated underwater post-process pass.
    underwater_factor: f32,
    // Minimum world-space Y a fragment may have before being kept.
    // The main pass sets this to a deep negative (no clip); the
    // reflection pass sets it to SEA_LEVEL so anything under water
    // is dropped from the reflected image.
    clip_y_min:        f32,
    eye:               vec4<f32>,
    inv_view_proj:     mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct ChunkUniform {
    origin: vec4<f32>,   // xyz = chunk origin in world space; w unused
};
@group(1) @binding(0) var<uniform> chunk: ChunkUniform;
@group(1) @binding(1) var light_volume:  texture_3d<f32>;
@group(1) @binding(2) var light_sampler: sampler;

@group(2) @binding(0) var atlas_tex: texture_2d<f32>;
@group(2) @binding(1) var atlas_sampler: sampler;

// Atlas layout constants — must stay in lockstep with
// `render::atlas`'s `TILE_PX = 16`, `ATLAS_TILE_COUNT_PER_AXIS = 4`.
// One tile is 1/4 of the atlas in each axis (4×4 grid).
const ATLAS_TILE_COUNT_PER_AXIS: f32 = 4.0;
const TILE_UV_SIZE:              f32 = 1.0 / ATLAS_TILE_COUNT_PER_AXIS;
const UNTEXTURED_TILE:           u32 = 255u;

struct VsIn {
    @location(0) pos_ao:       vec4<u32>,   // xyz = pos (0..32), w = ao (0..3)
    @location(2) color:        vec4<f32>,   // already normalised to [0,1]
    @location(3) face_light:   vec4<u32>,   // x = normal_face, y = light, zw = pad
    @location(4) tile_uv:      vec4<u32>,   // x = tile_index, yz = u_tile/v_tile, w = pad
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) v_color: vec4<f32>,
    @location(1) v_ao:    f32,
    @location(2) v_light: f32,
    @location(3) v_world: vec3<f32>,   // world-space fragment pos for fog
    @location(4) v_uv:    vec2<f32>,   // corner UV in *tile units*
    @location(5) @interpolate(flat) v_tile_index: u32,
    @location(6) @interpolate(flat) v_face_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world_pos = chunk.origin.xyz + vec3<f32>(in.pos_ao.xyz);

    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.v_world = world_pos;

    let face = in.face_light.x;
    var face_normal: vec3<f32>;
    switch (face) {
        case 0u: { face_normal = vec3<f32>( 1.0,  0.0,  0.0); }  // PosX
        case 1u: { face_normal = vec3<f32>(-1.0,  0.0,  0.0); }  // NegX
        case 2u: { face_normal = vec3<f32>( 0.0,  1.0,  0.0); }  // PosY
        case 3u: { face_normal = vec3<f32>( 0.0, -1.0,  0.0); }  // NegY
        case 4u: { face_normal = vec3<f32>( 0.0,  0.0,  1.0); }  // PosZ
        case 5u: { face_normal = vec3<f32>( 0.0,  0.0, -1.0); }  // NegZ
        default: { face_normal = vec3<f32>( 0.0,  1.0,  0.0); }
    }
    out.v_face_normal = face_normal;
    // Pass colour through without face_mul — the fragment shader now derives
    // directional shading from the light volume + wrap-diffuse sun.
    // Alpha is preserved so the fragment shader can still identify water by
    // `v_color.a < 0.95`.
    out.v_color = vec4<f32>(in.color.rgb, in.color.a);
    out.v_ao    = f32(in.pos_ao.w) / 3.0;

    // Light byte: high nibble = sky, low nibble = block. Scale each
    // independently to [0,1], then take max — the brighter source wins
    // (e.g. torches at night dominate the near-zero sky channel).
    let light_byte = in.face_light.y;
    let sky_l   = f32((light_byte >> 4u) & 0x0Fu) / 15.0;
    let block_l = f32(light_byte & 0x0Fu) / 15.0;
    out.v_light = max(sky_l * camera.sun_intensity, block_l);

    // UV in *tile units* — bytes 0..32 become floats 0..32. The
    // rasteriser interpolates this linearly between corners; the
    // fragment shader's `fract` then maps each integer cell back to
    // [0,1) inside its tile, giving free per-block tile repetition
    // across greedy-merged w×h quads.
    out.v_uv         = vec2<f32>(f32(in.tile_uv.y), f32(in.tile_uv.z));
    out.v_tile_index = in.tile_uv.x;
    return out;
}

// Compute the colour of the sky at the horizon, used as the fog tint.
// Mirrors the gradient logic in sky.wgsl so distant terrain dissolves
// seamlessly into the sky's horizon band rather than into a flat grey.
//
// Sun-warming pulls the horizon toward a soft peach when the sun is
// near the horizon (sin(angle) low → sun_intensity low but non-zero),
// the same atmospheric-scattering cue that makes real sunsets read
// as orange before turning into the dusk band proper.
fn horizon_color(sun_intensity: f32) -> vec3<f32> {
    let day   = vec3<f32>(0.65, 0.80, 1.00);   // brighter than zenith
    let dusk  = vec3<f32>(0.98, 0.62, 0.35);
    let night = vec3<f32>(0.05, 0.06, 0.12);
    let peach = vec3<f32>(1.00, 0.78, 0.62);
    let i = sun_intensity;
    let dusk_w = smoothstep(0.0, 0.25, i) - smoothstep(0.25, 0.7, i);
    // Peach warming peaks at low-but-positive intensity — same
    // window as `dusk_w` but a touch wider so the warm cast extends
    // into early morning / late afternoon, not just sunset proper.
    let peach_w = smoothstep(0.05, 0.30, i) - smoothstep(0.30, 0.80, i);
    return mix(night, day, smoothstep(0.0, 0.7, i))
        + dusk * dusk_w * 0.6
        + peach * peach_w * 0.25;
}

// Per-block colour-variation noise. Adds a small low-frequency tint
// modulation to surfaces so large flat areas don't read as uniform
// painted patches. Driven by world-space block coordinates (the
// `floor` snaps the perturbation to block boundaries so neighbouring
// blocks shift independently, mimicking how a stack of distinct
// physical blocks would look). Output is centered on 0 so it can be
// added directly to a tint without changing average brightness.
fn block_variation_hash(p: vec3<f32>) -> f32 {
    let q = floor(p);
    var h = sin(dot(q, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453;
    h = fract(h);
    return (h - 0.5) * 2.0; // map [0,1) → [-1, 1)
}

// Smooth low-frequency value noise on world (x, z) — used to drive
// biome-scale tint variation in `biome_tint_shift`. Output in [0, 1].
fn biome_hash2(p: vec2<f32>) -> f32 {
    let h = sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453;
    return fract(h);
}
fn biome_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = biome_hash2(i);
    let b = biome_hash2(i + vec2<f32>(1.0, 0.0));
    let c = biome_hash2(i + vec2<f32>(0.0, 1.0));
    let d = biome_hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Continuous biome-scale tint shift. Computes a per-world-position
// (humidity, temperature)-like signal from low-frequency noise — same
// continuous structure the worldgen biome model has, just evaluated
// here per fragment. Returns small `(r, g, b)` offsets to add to a
// surface block's tint so grass / sand colour shifts smoothly across
// climate zones instead of stepping at the discrete biome boundary.
//
// Lower-frequency than the per-block jitter so the variation is
// "this whole valley is greener" rather than "this single block is
// brighter" — biome-scale variation vs block-scale variation.
fn biome_tint_shift(world_xz: vec2<f32>) -> vec3<f32> {
    let h = biome_noise(world_xz * 0.0035);  // ~285 block period
    let t = biome_noise(world_xz * 0.0035 + vec2<f32>(50.0, 50.0));
    // h shifts the green/yellow axis (humidity proxy): wet=greener,
    // dry=yellower. t shifts brightness slightly (temperature proxy).
    // Magnitudes kept small (~10% / 5%) so the variation is "this
    // patch reads as a different shade of green" rather than "the
    // grass has gone weird colours".
    let humidity_shift = (h - 0.5) * 0.10;
    let temp_shift     = (t - 0.5) * 0.05;
    return vec3<f32>(
        -humidity_shift + temp_shift,
         humidity_shift + temp_shift,
        -humidity_shift * 0.4
    );
}

// Underwater tint moved to composite.wgsl (PR 1 Task 6) — the caustic
// pattern is now driven by screen-space UV rather than world position.

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Water fragments belong to the dedicated water pipeline. The
    // mesher still emits them into the same vertex buffer (split
    // meshes would cost a refactor we don't need yet), so the opaque
    // pass discards them here and the water pass discards the
    // not-water fragments. Vertex shader work is repeated; fragment
    // work in the overlap region is paid only once thanks to early
    // `discard`.
    if (in.v_color.a < 0.95) {
        discard;
    }

    // Below-clip-plane cull. For the main pass `clip_y_min` is a
    // deep negative (everything renders); for the reflection pass
    // it's SEA_LEVEL, so anything underwater (which has no business
    // appearing in the reflected image of the sky/upper world) gets
    // dropped here.
    if (in.v_world.y < camera.clip_y_min) {
        discard;
    }

    // ── Sample the atlas (or skip for untextured blocks). The tile
    // index is `flat`-interpolated so every fragment inside a quad
    // sees the same integer; rounding here is just defensive.
    var base_rgb = in.v_color.rgb;
    if (in.v_tile_index != UNTEXTURED_TILE) {
        // 4×4 grid → (col, row) from index.
        let row = f32(in.v_tile_index / 4u);
        let col = f32(in.v_tile_index % 4u);
        let tile_origin = vec2<f32>(col, row) * TILE_UV_SIZE;
        // `fract` wraps the per-corner UV (which spans 0..w in tile
        // units across a greedy quad) back into [0,1) so the same
        // tile repeats every block cell. Multiplying by TILE_UV_SIZE
        // then maps that into the tile's slice of the atlas.
        let in_tile = fract(in.v_uv);
        let atlas_uv = tile_origin + in_tile * TILE_UV_SIZE;
        let tex = textureSampleLevel(atlas_tex, atlas_sampler, atlas_uv, 0.0);
        // Vertex colour acts as a tint: grayscale tiles like
        // `grass_block_top.png` pick up the biome's green hue here.
        base_rgb = tex.rgb * in.v_color.rgb;
        // Leaves: the texture has true transparency between leaf
        // clusters. Discard those fragments instead of blending so
        // the silhouette stays crisp and depth-correct.
        if (tex.a < 0.5) {
            discard;
        }
    }

    // ── Per-pixel light volume sample, air-side of the surface.
    //
    // The vertex sits on the face plane: for a PosY face at world-y=11 it
    // shares that y with the air cell at chunk-local index 11 (which
    // spans world [11,12)); for a NegY face at world-y=10 it shares that
    // y with the BLOCK cell at index 10. We need to land sample_local
    // inside the air-side cell so trilinear filtering reads the actual
    // surface illumination, not a 50/50 blend with the opaque cell on
    // the other side of the face. `min(face_normal, 0)` gives 0 for
    // positive normals (vertex is already in the air cell) and -1 for
    // negative normals (step back one cell to the air on that side).
    let sample_world = in.v_world + min(in.v_face_normal, vec3<f32>(0.0));
    let chunk_local  = sample_world - chunk.origin.xyz;
    let uvw          = (chunk_local + vec3<f32>(0.5, 0.5, 0.5)) / 33.0;
    let lvol         = textureSampleLevel(light_volume, light_sampler, uvw, 0.0);
    let block_rgb    = lvol.rgb;        // 0..1 (R/G/B / 15)
    let sky_level    = lvol.a;          // 0..1 (sky_light / 15)

    // ── Wrap-diffuse directional sun with sky-channel occlusion.
    //
    // `camera.sun_dir` is the direction TO the sun (the sky shader treats
    // it that way for the sun-disc dot product). Lambert wants
    // `dot(N, L)` where L points toward the light, so no negation.
    // The previous `-camera.sun_dir` lit the wrong side: at noon the
    // bottoms of blocks got n·L=+1 and the tops clamped to 0.
    let n_dot_l = max(dot(in.v_face_normal, camera.sun_dir.xyz), 0.0);
    let wrap    = (n_dot_l + 0.4) / 1.4;
    // PR 5 introduces real cast shadows; until then `shadow = 1`.
    let shadow  = 1.0;
    let sun_lit = wrap * shadow * sky_level;

    // ── Sky ambient — tinted, scaled by sky exposure.
    let sky_amb = camera.sky_color.rgb * sky_level * 0.35;

    // ── Combine. AO attenuates the AMBIENT terms (sky + indirect)
    // only — direct sources (sun + block-light) pass unattenuated so
    // a torch in a corner doesn't darken its own corner from AO. This
    // is the conventional "AO is ambient occlusion" reading.
    let direct  = camera.sun_color.rgb * sun_lit * camera.sun_intensity;
    let ao_term = mix(0.45, 1.0, in.v_ao);
    let lit     = direct + block_rgb + sky_amb * ao_term;

    let MIN_SHADE = vec3<f32>(0.02, 0.02, 0.02);

    // Per-block brightness jitter: a small ±6% modulation keyed off
    // the world-space block coordinate. Neighbouring blocks (whole
    // integer steps in any axis) get a different jitter; cells
    // *within* a block share the same value, so the variation reads
    // as block-level natural variance rather than per-pixel noise.
    let variation = 1.0 + block_variation_hash(in.v_world) * 0.06;
    // Biome-scale tint shift: low-frequency continuous signal that
    // smoothly varies the hue across hundreds of blocks — the
    // shader-side approximation of "blend biome properties instead
    // of biome IDs". Detected by atlas tile so the rule fires
    // exclusively on grass-top and sand surfaces (not the same-tinted
    // leaves, which would otherwise come along for the ride and look
    // unnaturally pink/yellow).
    //
    // Tile indices must stay in lockstep with `voxel::block::Tile`:
    //   2 = GrassTop, 4 = Sand
    var lit_rgb = base_rgb * (lit + MIN_SHADE) * variation;
    let is_blendable = in.v_tile_index == 2u || in.v_tile_index == 4u;
    if (is_blendable) {
        lit_rgb = lit_rgb + biome_tint_shift(in.v_world.xz) * (lit + MIN_SHADE);
    }

    // Distance fog: linear ramp between FOG_START and FOG_END.
    let dist = length(in.v_world - camera.eye.xyz);
    let fog_start = 96.0;
    let fog_end   = 360.0;
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    // Fog colour: a per-fragment blend between the horizon sky tint
    // and a near-black "cave" colour. Picked by `in.v_light` — the
    // fragment's own sky+block light — so deep-cave fragments fade
    // into darkness instead of the bright sky horizon. Without this
    // the sky-fog leaks into underground views and distant stone
    // walls read as washed-out white. The min(0.05) floor avoids
    // pure black, leaving a hint of colour to silhouette the bulk.
    let sky_fog = horizon_color(camera.sun_intensity);
    let cave_fog = vec3<f32>(0.02, 0.02, 0.03);
    let fog_col = mix(cave_fog, sky_fog, max(in.v_light, 0.05));
    var out_rgb = mix(lit_rgb, fog_col, fog_t);

    // Tonemap + underwater tint both live in composite.wgsl as of
    // PR 1 Tasks 5/6. Output is linear HDR.
    return vec4<f32>(out_rgb, 1.0);
}
