// Paper grain: fine cellulose texture, uneven coloring from pulp
// This is the "not-perfect coloring" you wanted

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

@group(0) @binding(3)
var grain_out: texture_storage_2d<rgba16float, write>;

// ── Hash / noise (same as paper_base) ─────────────────────────────

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

fn fbm(x: f32, y: f32, seed: u32, octaves: u32) -> f32 {
    var v = 0.0; var a = 0.5; var f = 1.0;
    for (var i = 0u; i < octaves; i = i + 1u) {
        v += a * value_noise(x * f, y * f, seed + i);
        a *= 0.5; f *= 2.0;
    }
    return v;
}

// ── Voronoi-like cellular noise for grain clumps ──────────────────

fn cellular_noise(x: f32, y: f32, seed: u32) -> f32 {
    let ix = i32(floor(x));
    let iy = i32(floor(y));
    let fx = fract(x);
    let fy = fract(y);
    
    var min_dist = 1.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cx = i32(hash2d(u32(ix + dx), u32(iy + dy), seed)) % 100u;
            let cy = i32(hash2d(u32(ix + dx), u32(iy + dy), seed + 1u)) % 100u;
            let px = f32(dx) + f32(cx) / 100.0;
            let py = f32(dy) + f32(cy) / 100.0;
            let d = length(vec2<f32>(fx - px, fy - py));
            min_dist = min(min_dist, d);
        }
    }
    return min_dist;
}

// ── Main ──────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn grain(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    
    let uv = vec2<f32>(
        f32(gid.x) / f32(params.width),
        f32(gid.y) / f32(params.height)
    );
    
    let intensity = params.grain_intensity;
    let rough = params.roughness;
    
    // Fine grain (cellulose fibers at micro scale)
    let fine = fbm(uv.x * 200.0, uv.y * 200.0, params.seed + 50u, 3u);
    
    // Medium grain (pulp clumps)
    let medium = cellular_noise(uv.x * 40.0, uv.y * 40.0, params.seed + 60u);
    
    // Uneven color patches from imperfect bleaching
    let bleach = fbm(uv.x * 15.0, uv.y * 15.0, params.seed + 70u, 2u);
    
    // Combine into grain mask
    var grain = 0.0;
    grain += (fine - 0.5) * 0.3 * intensity;
    grain += (medium - 0.5) * 0.5 * intensity * (0.5 + rough * 0.5);
    grain += (bleach - 0.5) * 0.2 * intensity;
    
    // Grain is stored as an offset to apply during composite
    let grain_color = vec3<f32>(grain);
    textureStore(grain_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(grain_color, 1.0));
}
