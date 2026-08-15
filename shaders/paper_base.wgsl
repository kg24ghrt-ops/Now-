// Paper base: off-white with aging, uneven color, subtle mottling
// Uses hash noise for procedural variation — no textures needed

@group(0) @binding(0)
var<uniform> params: PaperParams;

@group(0) @binding(1)
var<storage, read> noise_seed: array<u32>;

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

@group(0) @binding(2)
var paper_out: texture_storage_2d<rgba16float, write>;

// ── Hash functions (GPU-optimized, no sin/cos) ────────────────────

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

fn hash3d(x: u32, y: u32, z: u32, seed: u32) -> f32 {
    let n = x * 374761393u + y * 668265263u + z * 1013904243u + seed * 2013904243u;
    return hash_f32(n);
}

// ── Value noise (smooth interpolation) ────────────────────────────

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let ix = u32(floor(x));
    let iy = u32(floor(y));
    let fx = fract(x);
    let fy = fract(y);
    
    let fx_smooth = fx * fx * (3.0 - 2.0 * fx);
    let fy_smooth = fy * fy * (3.0 - 2.0 * fy);
    
    let n00 = hash2d(ix, iy, seed);
    let n10 = hash2d(ix + 1u, iy, seed);
    let n01 = hash2d(ix, iy + 1u, seed);
    let n11 = hash2d(ix + 1u, iy + 1u, seed);
    
    let nx0 = mix(n00, n10, fx_smooth);
    let nx1 = mix(n01, n11, fx_smooth);
    return mix(nx0, nx1, fy_smooth);
}

fn fbm(x: f32, y: f32, seed: u32, octaves: u32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var freq = 1.0;
    for (var i = 0u; i < octaves; i = i + 1u) {
        value += amplitude * value_noise(x * freq, y * freq, seed + i);
        amplitude *= 0.5;
        freq *= 2.0;
    }
    return value;
}

// ── Main ──────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn paper_base(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    
    let uv = vec2<f32>(
        f32(gid.x) / f32(params.width),
        f32(gid.y) / f32(params.height)
    );
    
    // Base paper color: warm off-white
    var color = vec3<f32>(0.96, 0.94, 0.90);
    
    // Aging: shift toward yellow/brown
    let age = params.aging_yellow;
    color = mix(color, vec3<f32>(0.92, 0.88, 0.75), age * 0.4);
    
    // Large-scale mottling (paper pulp unevenness)
    let mottle = fbm(uv.x * 8.0, uv.y * 8.0, params.seed, 4u);
    color += (mottle - 0.5) * 0.03 * (1.0 + age);
    
    // Medium-scale variation (batch inconsistency)
    let medium = fbm(uv.x * 32.0, uv.y * 32.0, params.seed + 100u, 3u);
    color += (medium - 0.5) * 0.015;
    
    // Subtle edge darkening (older paper effect)
    let edge_dist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let edge_factor = smoothstep(0.0, 0.15, edge_dist);
    color = mix(color * 0.92, color, edge_factor);
    
    // Clamp and store as 16-bit float
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(paper_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(color, 1.0));
}
