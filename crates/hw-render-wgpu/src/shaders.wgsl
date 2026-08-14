// ============================================================
// shaders.wgsl – Full GPU pipeline for handwriting rendering
// ============================================================

// ---- Uniforms ----
struct Uniforms {
    view_proj: mat4x4<f32>,
    ink_color: vec4<f32>,
    wetness: f32,
    _pad1: vec3<f32>,
    texture_width: u32,
    texture_height: u32,
    _pad2: vec2<u32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

// ---- Paper texture + sampler ----
@group(1) @binding(0) var paper_tex: texture_2d<f32>;
@group(1) @binding(1) var paper_sampler: sampler;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;

// ---- Input (base ink) and output (storage) for compute ----
@group(1) @binding(3) var input_tex: texture_2d<f32>;
@group(1) @binding(4) var output_tex: texture_storage_2d<rgba8unorm, read_write>;

// ============================================================
// VERTEX SHADER – renders the ink mesh
// ============================================================
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) pressure: f32,
    @location(3) wetness: f32,
    @location(4) normal: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) pressure: f32,
    @location(2) wetness: f32,
    @location(3) normal: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.pressure = in.pressure;
    out.wetness = in.wetness;
    out.normal = in.normal;
    return out;
}

// ============================================================
// FRAGMENT SHADER – base ink with paper absorption
// ============================================================
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let paper = textureSample(paper_tex, paper_sampler, in.uv);
    let ink = u.ink_color * in.pressure;
    let absorption = 1.0 - paper.r * 0.4;
    let color = mix(paper.rgb, ink.rgb, ink.a * absorption);
    return vec4<f32>(color, 1.0);
}

// ============================================================
// COMPUTE SHADER – anisotropic diffusion (soak‑in effect)
// ============================================================
@compute @workgroup_size(8, 8)
fn cs_diffuse(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = u.texture_width;
    let height = u.texture_height;
    let x = id.x;
    let y = id.y;
    if (x >= width || y >= height) { return; }

    let base_color = textureLoad(input_tex, vec2<i32>(i32(x), i32(y)), 0);
    if (u.wetness < 0.01) {
        textureStore(output_tex, vec2<i32>(i32(x), i32(y)), base_color);
        return;
    }

    // Number of iterations scales with wetness (1..9)
    let iterations = u32(ceil(u.wetness * 8.0)) + 1u;
    let radius = 2;

    // Load paper normal (stored as 0..1, convert to -1..1)
    let normal_raw = textureLoad(normal_tex, vec2<i32>(i32(x), i32(y)), 0).xyz;
    let normal = normal_raw * 2.0 - 1.0;

    var current = base_color;

    for (var iter = 0u; iter < iterations; iter = iter + 1u) {
        var sum = vec4<f32>(0.0);
        var total = 0.0;

        for (var dy = -radius; dy <= radius; dy = dy + 1) {
            for (var dx = -radius; dx <= radius; dx = dx + 1) {
                let px = i32(x) + dx;
                let py = i32(y) + dy;
                if (px < 0 || py < 0 || px >= i32(width) || py >= i32(height)) { continue; }

                let dist2 = f32(dx*dx + dy*dy);
                let gauss = exp(-dist2 / (2.0 * f32(radius * radius)));

                let dir = vec2<f32>(f32(dx), f32(dy));
                let dir_len = length(dir);
                var weight = gauss;

                if (dir_len > 0.001) {
                    let dir_norm = dir / dir_len;
                    let dot = dot(normal.xy, dir_norm);
                    let alignment = max(0.0, dot);
                    let bias = 1.0 + 0.8 * alignment;
                    weight = gauss * bias;
                }

                let neighbor = textureLoad(input_tex, vec2<i32>(px, py), 0);
                sum += neighbor * weight;
                total += weight;
            }
        }

        if (total > 0.0) {
            let blurred = sum / total;
            let soak_factor = 0.15 + 0.05 * f32(iter) / f32(iterations);
            current = mix(current, blurred, soak_factor * u.wetness);
        }
    }

    textureStore(output_tex, vec2<i32>(i32(x), i32(y)), current);
}