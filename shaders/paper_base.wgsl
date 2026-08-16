// Paper base: warm off-white sheet with pulp mottling, laid/chain lines,
// foxing (age spots) and noisy edge darkening.
// Perceptual model: 1/f spectrum mottle + sparse defects reads as "real paper"
// to human eyes, and keeps CNN/forensic statistics stationary (no tiling).

@group(0) @binding(0) var<uniform> params: PaperParams;

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
    _pad0: f32,               // scalar pads: exact match for Rust [f32; 2]
    _pad1: f32,
};

@group(0) @binding(2) var paper_out: texture_storage_2d<rgba16float, write>;

// ── Fast PCG-style hash: 2 mults, full-period, no sin/cos ───────────
fn h21(p: vec2<u32>, seed: u32) -> f32 {
    var v = p.x * 374761393u + p.y * 668265263u + seed * 1442695041u;
    v = (v ^ (v >> 13u)) * 1274126177u;
    return f32(v ^ (v >> 16u)) * (1.0 / 4294967295.0);
}

fn vnoise(p: vec2<f32>, seed: u32) -> f32 {
    let ip = floor(p);
    let fp = fract(p);
    let u = fp * fp * fp * (fp * (fp * 6.0 - 15.0) + 10.0); // quintic: C2-continuous, no lattice artifacts
    let a = h21(vec2<u32>(u32(ip.x),     u32(ip.y)),     seed);
    let b = h21(vec2<u32>(u32(ip.x)+1u,  u32(ip.y)),     seed);
    let c = h21(vec2<u32>(u32(ip.x),     u32(ip.y)+1u),  seed);
    let d = h21(vec2<u32>(u32(ip.x)+1u,  u32(ip.y)+1u),  seed);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// fbm with rotated octaves → removes axis-aligned "grid" artifacts
fn fbm(p: vec2<f32>, seed: u32, octaves: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.55;
    var q = p;
    let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    for (var i = 0u; i < octaves; i = i + 1u) {
        sum += amp * vnoise(q, seed + i);
        q = rot * (q * 2.03);
        amp *= 0.5;
    }
    return sum;
}

@compute @workgroup_size(16, 16, 1)
fn paper_base(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let uv = vec2<f32>(f32(gid.x) / f32(params.width), f32(gid.y) / f32(params.height));
    let age = clamp(params.aging_yellow, 0.0, 1.0);

    // Warm off-white base + tiny per-sheet temperature shift (no two sheets identical)
    let sheet_tint = h21(vec2<u32>(7u, 9u), params.seed) - 0.5;
    var color = vec3<f32>(0.955, 0.940, 0.905) + vec3<f32>(0.010, 0.004, -0.014) * sheet_tint;

    // Aging → cream/ivory shift
    color = mix(color, vec3<f32>(0.910, 0.865, 0.720), age * 0.45);

    // Domain-warped large mottling (pulp density unevenness)
    let warp = vec2<f32>(
        fbm(uv * 6.0, params.seed + 11u, 3u),
        fbm(uv * 6.0, params.seed + 23u, 3u)) - 0.5;
    let mottle = fbm(uv * 9.0 + warp * 1.7, params.seed, 4u);
    color += (mottle - 0.5) * 0.050 * (0.6 + age);

    // Medium-scale batch variation
    let medium = fbm(uv * 34.0, params.seed + 100u, 3u);
    color += (medium - 0.5) * 0.020;

    // Laid lines (fine horizontal ribs) + chain lines (widely spaced vertical)
    let laid  = sin(uv.y * 3.14159265 * 180.0) * 0.5 + 0.5;
    let chain = smoothstep(0.985, 1.0, sin(uv.x * 3.14159265 * 14.0));
    color -= (laid * 0.012 + chain * 0.020) * (0.4 + 0.6 * params.roughness);

    // Foxing: sparse brown age spots, gated by aging
    let fox = smoothstep(0.86, 0.94, fbm(uv * 60.0, params.seed + 777u, 3u));
    color = mix(color, vec3<f32>(0.62, 0.48, 0.30), fox * age * 0.35);

    // Soft edge darkening with a noisy boundary (hand-made sheet feel)
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let edge_n = edge + (fbm(uv * 40.0, params.seed + 5u, 2u) - 0.5) * 0.02;
    color *= mix(0.93, 1.0, smoothstep(0.0, 0.12, edge_n));

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(paper_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(color, 1.0));
}