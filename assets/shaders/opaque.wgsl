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
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world_pos = chunk.origin.xyz + vec3<f32>(in.pos_ao.xyz);

    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.v_world = world_pos;

    let face = in.face_light.x;
    var face_mul: f32 = 0.80;
    if (face == 2u) { face_mul = 1.00; }       // +Y top
    else if (face == 3u) { face_mul = 0.55; }  // -Y bottom
    // face_mul applies to RGB only — leaving alpha untouched means the
    // fragment shader can reliably identify water by `v_color.a < 0.95`
    // (water is the only block with alpha != 1.0 at the vertex source).
    // Otherwise face_mul=0.80 on opaque sides + face_mul=0.55 on
    // opaque bottoms would falsely match the water threshold and
    // trigger the water shimmer code path on leaves / stone undersides.
    out.v_color = vec4<f32>(in.color.rgb * face_mul, in.color.a);
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

// Cheap 2D hash matching the sky/water shaders so the underwater
// caustic pattern stays coherent across pipelines.
fn uw_hash(p: vec2<f32>) -> f32 {
    let h = sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453;
    return fract(h);
}
fn uw_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = uw_hash(i);
    let b = uw_hash(i + vec2<f32>(1.0, 0.0));
    let c = uw_hash(i + vec2<f32>(0.0, 1.0));
    let d = uw_hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Apply the underwater colour grade. Two terms:
//   1. Pull every channel toward a deep teal tint by `factor`.
//   2. Overlay an animated caustic-like noise pattern keyed off the
//      fragment's world position + time. The caustics simulate light
//      filtering through ripples above and breaking into shifting
//      bright bands across the underwater scene.
// Composed *after* tonemapping so neither term gets curve-compressed.
fn underwater_tint(rgb: vec3<f32>, world: vec3<f32>, t: f32, factor: f32) -> vec3<f32> {
    if (factor <= 0.0) {
        return rgb;
    }
    let water_blue = vec3<f32>(0.10, 0.30, 0.45);
    var tinted = mix(rgb, water_blue, factor * 0.65);
    // Two scrolling noise layers; their product produces tight
    // caustic-like ridges where both layers are bright at the same
    // place.
    let a = uw_noise(world.xz * 0.35 + vec2<f32>( 0.18,  0.11) * t);
    let b = uw_noise(world.xz * 0.27 + vec2<f32>(-0.13,  0.19) * t);
    let caustic = pow(a * b, 2.0) * 0.6;
    // Tint the caustic pale-aqua so it reads as light from above
    // rather than just a brightness modulation.
    let caustic_color = vec3<f32>(0.65, 0.95, 1.0);
    tinted = tinted + caustic_color * caustic * factor;
    return tinted;
}

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

    let ao = mix(0.45, 1.0, in.v_ao);
    let lit = max(0.05, in.v_light);
    let shade = ao * lit;
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
    var lit_rgb = base_rgb * shade * variation;
    let is_blendable = in.v_tile_index == 2u || in.v_tile_index == 4u;
    if (is_blendable) {
        lit_rgb = lit_rgb + biome_tint_shift(in.v_world.xz) * shade;
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

    // ── Post: underwater tint stays here for now (Task 6 moves it to
    // composite). Tonemap moved to composite already, so this output
    // is unclamped linear HDR — Rgba16Float carries the dynamic range.
    out_rgb = underwater_tint(out_rgb, in.v_world, camera.time, camera.underwater_factor);

    return vec4<f32>(out_rgb, 1.0);
}
