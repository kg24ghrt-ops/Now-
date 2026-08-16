// Final assembly: physically-plausible layering + dithered quantization.
// Layering is multiplicative for shadows/stains (light absorption) and additive
// for highlights (hairs catching light). A 1-LSB dither before rgba8unorm
// storage removes banding for human eyes and keeps the histogram's noise floor
// natural for machine-vision/forensic analysis.

@group(0) @binding(0) var<uniform> params: PaperParams;
@group(0) @binding(1) var<storage, read> noise_seed: array<u32>;
@group(0) @binding(2) var paper_tex: texture_storage_2d<rgba16float, read>;
@group(0) @binding(3) var grain_tex: texture_storage_2d<rgba16float, read>;
@group(0) @binding(4) var fiber_tex: texture_storage_2d<rgba16float, read>;
@group(0) @binding(5) var water_tex: texture_storage_2d<rgba16float, read>;
@group(0) @binding(6) var output: texture_storage_2d<rgba8unorm, write>;

struct PaperParams {
    width: u32, height: u32, seed: u32,
    grain_intensity: f32, fiber_density: f32, water_stain_count: u32,
    aging_yellow: f32, fiber_direction: f32, roughness: f32,
    _pad0: f32, _pad1: f32,
};

@compute @workgroup_size(16, 16, 1)
fn composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = vec2<f32>(f32(gid.x) / f32(params.width), f32(gid.y) / f32(params.height));

    var color = textureLoad(paper_tex, coord).rgb;

    // Grain: high-frequency lightness offset
    color += textureLoad(grain_tex, coord).r * 0.14;

    // Fibers: strands cast micro-shadows; hairs catch cool light
    let fiber_val = textureLoad(fiber_tex, coord);
    color *= 1.0 - fiber_val.r * 0.10;
    color += vec3<f32>(0.90, 0.95, 1.00) * fiber_val.g * 0.06;

    // Stains: brown absorption tint, rim darker than interior
    let water_val = textureLoad(water_tex, coord);
    let stain = clamp(water_val.r, 0.0, 1.0);
    let rimness = clamp(water_val.g / max(water_val.r, 1e-4), 0.0, 1.0);
    let tint = mix(vec3<f32>(0.55, 0.42, 0.26), vec3<f32>(0.40, 0.27, 0.13), rimness);
    color = mix(color, color * tint, stain * 0.85);

    // Global aging yellowing
    color = mix(color, color * vec3<f32>(1.02, 0.97, 0.82), clamp(params.aging_yellow, 0.0, 1.0) * 0.5);

    // Gentle large-scale sheen so the sheet reads as a 3D object under a light
    let sheen = sin(uv.x * 3.14159265) * sin(uv.y * 3.14159265);
    color *= 0.985 + 0.030 * sheen;

    // 1-LSB dither from the noise buffer before 8-bit quantization
    let d = f32(noise_seed[(gid.x + gid.y * 517u) & 255u]) * (1.0 / 4294967295.0) - 0.5;
    color += d / 255.0;

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, coord, vec4<f32>(color, 1.0));
}