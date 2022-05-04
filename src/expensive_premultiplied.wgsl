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
var texture: texture_2d<f32>;
[[group(0), binding(1)]]
var tex_sampler: sampler;

let glyph_width: f32 = 0.51;
let smoothing: f32 = 0.02;

fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

fn screen_px_range(tex_coord: vec2<f32>) -> f32 {
    let range = vec2<f32>(3.0, 3.0) / vec2<f32>(textureDimensions(texture));
    let screen_tex_size: vec2<f32> = vec2<f32>(1.0, 1.0)/fwidth(tex_coord);
    return max(0.5 * dot(range, screen_tex_size), 1.0);
}

fn normalize_safe(v: vec2<f32>) -> vec2<f32>
{
   var len = length( v );
   if(len > 0.0){
        len = 1.0 / len;
    } else {
        len = 0.0;
    };
   return v * len;
}

[[stage(fragment)]]
fn main_fs(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let sample = textureSample(texture, tex_sampler, in.tex_pos).rgba;
    let dist = median(sample.r, sample.g, sample.b) - 0.5;
    let grad_dist = normalize_safe(vec2<f32>(dpdx(dist), dpdy(dist)));

    let uv = in.tex_pos * vec2<f32>(textureDimensions(texture));
    let jdx = dpdx(uv);
    let jdy = dpdy(uv);

    let grad = vec2<f32>(grad_dist.x * jdx.x + grad_dist.y * jdy.x, grad_dist.x * jdx.y + grad_dist.y * jdy.y);
    // Convert the distance to screen pixels
    //let screen_pixels = screen_px_range(in.tex_pos) * d;
    // Opacity
    //let w = clamp(screen_pixels + 0.5, 0.0, 1.0);

    let fg_color: vec4<f32> = vec4<f32>(0.9, 0.5, 0.4, 1.0);
    let bg_color: vec4<f32> = vec4<f32>(0.3, 0.2, 0.1, 0.0);

    let distance_range = 8.0;
    let adjustment = 1.0 / distance_range;
    let afwidth = sqrt(2.0) * 0.5 * length(grad) * adjustment;
    let norm = min(afwidth, 0.5);
    let opacity = smoothStep(0.0 - afwidth, 0.0 + afwidth, dist);

    let gamma = 2.2;
    let premultiply = 1.0;
    let fg_alpha = pow(fg_color.a * opacity, 1.0 / gamma);
    let bg_alpha = pow(bg_color.a * opacity, 1.0 / gamma);
    //return mix(bg_color, fg_color, opacity);
    //result.rgb = mix(fg_color.rgb, fg_color.rgb * alpha, premultiply);
    return vec4<f32>(mix(fg_color.rgb, fg_color.rgb * fg_alpha, premultiply), alpha);
}