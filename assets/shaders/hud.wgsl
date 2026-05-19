// HUD 2D overlay shader.
//
// One vertex format, one fragment path. The bind group at @group(1)
// holds whichever RGBA texture the current batch sources from (font
// atlas or block atlas — both authored as RGBA8). Multiplying the
// sample by the vertex colour handles every case uniformly:
//
//   - Font glyphs:  atlas reads (1,1,1,coverage); vertex colour is
//                   the tint → output (tint.rgb, tint.a * coverage).
//   - Block icons:  atlas reads (r,g,b,a); vertex colour is white
//                   (or biome tint) → output (rgb * tint, a * tint.a).
//   - Solid rects:  vertex carries sentinel UV (-1, -1) → skip the
//                   sample and return the vertex colour directly,
//                   used for the hotbar background + selection border.
//
// Coordinate convention: vertex positions are *pixels* with origin at
// the top-left of the framebuffer. The screen-size uniform converts
// to NDC, flipping Y so pixel +Y goes downward.

struct Screen {
    // xy = framebuffer pixel size; zw = unused/pad. One uniform shared
    // by every HUD batch this frame.
    size: vec4<f32>,
};
@group(0) @binding(0) var<uniform> screen: Screen;

@group(1) @binding(0) var hud_tex: texture_2d<f32>;
@group(1) @binding(1) var hud_sampler: sampler;

struct VsIn {
    @location(0) pos_px: vec2<f32>,
    @location(1) uv:     vec2<f32>,
    @location(2) color:  vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) v_uv:    vec2<f32>,
    @location(1) v_color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    // Pixel (0, 0) is the top-left of the framebuffer; NDC y points
    // up so we negate. Map x: [0, W] → [-1, 1] and y: [0, H] → [1, -1].
    let ndc = vec2<f32>(
        in.pos_px.x / screen.size.x * 2.0 - 1.0,
        1.0 - in.pos_px.y / screen.size.y * 2.0,
    );
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.v_uv     = in.uv;
    out.v_color  = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.v_uv.x < 0.0) {
        return in.v_color;
    }
    let tex = textureSampleLevel(hud_tex, hud_sampler, in.v_uv, 0.0);
    return tex * in.v_color;
}
