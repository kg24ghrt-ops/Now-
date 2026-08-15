// Water stains: coffee/tea ring stains and moisture marks

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

@group(0) @binding(5)
var water_out: texture_storage_2d<rgba16float, write>;

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

fn dist_to_stain(uv: vec2<f32>, center: vec2<f32>, radius: f32) -> f32 {
    let d = length(uv - center);
    return smoothstep(radius, 0.0, d);
}

@compute @workgroup_size(8, 8, 1)
fn water(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let uv = vec2<f32>(
        f32(gid.x) / f32(params.width),
        f32(gid.y) / f32(params.height)
    );

    var stain = 0.0;
    let count = min(params.water_stain_count, 20u);
    
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let seed = params.seed + i * 1337u;
        let cx = hash_f32(seed);
        let cy = hash_f32(seed + 1u);
        let radius = 0.03 + hash_f32(seed + 2u) * 0.12;
        let intensity = 0.05 + hash_f32(seed + 3u) * 0.25;
        
        let d = dist_to_stain(uv, vec2<f32>(cx, cy), radius);
        stain += d * intensity;
    }
    
    stain = min(stain, 1.0);
    
    textureStore(water_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(stain, stain * 0.8, stain * 0.6, 1.0));
}
