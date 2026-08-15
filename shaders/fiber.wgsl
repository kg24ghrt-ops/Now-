// Paper fiber: directional cellulose strands
// Creates the wood grain / fiber texture effect

@group(0) @binding(0)
var params: PaperParams;

@group(0) @binding(1)
var noise_seed: array<u32>;

struct PaperParams {
    width: u32,
    height: u32,
    seed: u32,
    grain_intensity: f32,
    fiber_density: f32,
    water_stain_count: u32,
    aging_yellow: f32,
    fiber_direction: f32,
    roughness: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(4)
var fiber_out: texture_storage_2d<rgba16float, write>;

fn hash_u32(n: u32) -> u32 {
    var x = n;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

fn hash_f32(n: u32) -> f32 {
    return f32(hash_u32(n)) / 4294967295.0;
}

fn hash2d(x: u32, y: u32, seed: u32) -> f32 {
    let n = x * 374761393u + y * 668265263u + seed * 1013904243u;
    return hash_f32(n);
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let ix = u32(floor(x));
    let iy = u32(floor(y));
    let fx = fract(x);
    let fy = fract(y);
    let fx_s = fx * fx * (3.0 - 2.0 * fx);
    let fy_s = fy * fy * (3.0 - 2.0 * fy);
    let n00 = hash2d(ix, iy, seed);
    let n10 = hash2d(ix + 1u, iy, seed);
    let n01 = hash2d(ix, iy + 1u, seed);
    let n11 = hash2d(ix + 1u, iy + 1u, seed);
    return mix(mix(n00, n10, fx_s), mix(n01, n11, fx_s), fy_s);
}

@compute @workgroup_size(8, 8, 1)
fn fiber(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let uv = vec2<f32>(
        f32(gid.x) / f32(params.width),
        f32(gid.y) / f32(params.height)
    );

    let dir = vec2<f32>(cos(params.fiber_direction), sin(params.fiber_direction));
    let proj = dot(uv, dir);
    let perp = dot(uv, vec2<f32>(-dir.y, dir.x));

    let density = params.fiber_density * 50.0 + 10.0;
    
    var fiber = 0.0;
    for (var i: u32 = 0u; i < 5u; i = i + 1u) {
        let offset = hash_f32(params.seed + i * 7919u) * 2.0 - 1.0;
        let freq = density * (1.0 + f32(i) * 0.3);
        let line = sin((proj + offset * 0.05 + perp * 0.02 * hash_f32(i + 100u)) * freq);
        fiber += line * line * (0.5 + params.roughness * 0.5);
    }
    
    fiber = fiber * params.fiber_density * 0.15;
    fiber = clamp(fiber, 0.0, 1.0);
    
    textureStore(fiber_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(fiber, fiber, fiber, 1.0));
}
