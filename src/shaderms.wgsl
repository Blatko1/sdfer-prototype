struct VertexInput {
    [[builtin(vertex_index)]] vertex_index: u32;
    [[location(0)]] top_left: vec3<f32>;
    [[location(1)]] bottom_right: vec2<f32>;
    [[location(2)]] tex_top_left: vec2<f32>;
    [[location(3)]] tex_bottom_right: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_pos: vec2<f32>;
    [[location(1)]] color: vec3<f32>;
};

struct Matrix {
    matrix: mat4x4<f32>;
};

[[group(0), binding(2)]]
var<uniform> matrix: Matrix;

let glyph_scale: f32 = 0.1;

[[stage(vertex)]]
fn main_vs(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var pos: vec2<f32>;
    var left_x: f32 = in.top_left.x;
    var right_x: f32 = in.bottom_right.x;
    var top_y: f32 = in.top_left.y;
    var bottom_y: f32 = in.bottom_right.y;

    switch (in.vertex_index) {
        case 0: {
            pos = vec2<f32>(left_x, top_y);
            out.tex_pos = in.tex_top_left;
            break;
        }
        case 1: {
            pos = vec2<f32>(right_x, top_y);
            out.tex_pos = vec2<f32>(in.tex_bottom_right.x, in.tex_top_left.y);
            break;
        }
        case 2: {
            pos = vec2<f32>(left_x, bottom_y);
            out.tex_pos = vec2<f32>(in.tex_top_left.x, in.tex_bottom_right.y);
            break;
        }
        case 3: {
            pos = vec2<f32>(right_x, bottom_y);
            out.tex_pos = in.tex_bottom_right;
            break;
        }
        default: {}
    }

    out.clip_position = matrix.matrix * vec4<f32>(pos * glyph_scale, in.top_left.z * glyph_scale, 1.0);
    out.color = vec3<f32>(sin(pos), cos(pos.x));
    return out;
}

struct UnitRange {
    px: f32;
};

[[group(0), binding(0)]]
var texture: texture_multisampled_2d<f32>;
[[group(0), binding(1)]]
var tex_sampler: sampler;
[[group(0), binding(3)]]
var<uniform> unit_range: UnitRange;

let glyph_width: f32 = 0.51;
let smoothing: f32 = 0.02;

fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

let bg_color: vec4<f32> = vec4<f32>(0.3, 0.2, 0.1, 0.0);

fn texture_multisample(coord: vec2<i32>) -> vec4<f32> {
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    for (var i: i32 = 0; i < 4; i=i+1) {
        color = color + textureLoad(texture, coord, i);
    }

    color = color / 4.0;

    return color;
}

[[stage(fragment)]]
fn main_fs(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let tex_size = textureDimensions(texture);
    let tex_coord = vec2<i32>(in.tex_pos * vec2<f32>(tex_size));

    let s = texture_multisample(tex_coord).rgb;
    // Acquire the signed distance
    let d = median(s.r, s.g, s.b) - 0.5;
    // Convert the distance to screen pixels
    let screen_pixels = d/fwidth(d);
    // Opacity
    let w = clamp(screen_pixels + 0.5, 0.0, 1.0);

    let fg_color: vec4<f32> = vec4<f32>(0.6, 0.5, 0.4, 1.0);

    //return mix(bg_color, fg_color, w);
    return vec4<f32>(s, 1.0);
}