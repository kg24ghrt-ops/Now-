// Directional cellulose fibers. Real fibers are short curved strands, not sine
// lines: we use strongly anisotropic, domain-warped noise (long along the fiber
// axis, tiny across it) plus sparse bright "hairs".
// R = strand shadow mask, G = hair highlight mask (composite tints them apart).

@group(0) @binding(0) var<uniform> params: PaperParams;

struct PaperParams {
    width: u32, height: u32, seed: u32,
    grain_intensity: f32, fiber_density: f32, water_stain_count: u32,
    aging_yellow: f32, fiber_direction: f32, roughness: f32,
    _pad0: f32, _pad1: f32,
};

@group(0) @binding(4) var fiber_out: texture_storage_2d<rgba16float, write>;

fn h21(p: vec2<u32>, seed: u32) -> f32 {
    var v = p.x * 374761393u + p.y * 668265263u + seed * 1442695041u;
    v = (v ^ (v >> 13u)) * 1274126177u;
    return f32(v ^ (v >> 16u)) * (1.0 / 4294967295.0);
}

fn vnoise(p: vec2<f32>, seed: u32) -> f32 {
    let ip = floor(p); let fp = fract(p);
    let u = fp * fp * fp * (fp * (fp * 6.0 - 15.0) + 10.0);
    let a = h21(vec2<u32>(u32(ip.x), u32(ip.y)), seed);
    let b = h21(vec2<u32>(u32(ip.x)+1u, u32(ip.y)), seed);
    let c = h21(vec2<u32>(u32(ip.x), u32(ip.y)+1u), seed);
    let d = h21(vec2<u32>(u32(ip.x)+1u, u32(ip.y)+1u), seed);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>, seed: u32, octaves: u32) -> f32 {
    var sum = 0.0; var amp = 0.55; var q = p;
    let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    for (var i = 0u; i < octaves; i = i + 1u) {
        sum += amp * vnoise(q, seed + i);
        q = rot * (q * 2.03);
        amp *= 0.5;
    }
    return sum;
}

@compute @workgroup_size(16, 16, 1)
fn fiber(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let uv = vec2<f32>(f32(gid.x) / f32(params.width), f32(gid.y) / f32(params.height));

    let a = params.fiber_direction;
    let dir  = vec2<f32>(cos(a), sin(a));
    let perp = vec2<f32>(-dir.y, dir.x);
    let along  = dot(uv, dir);
    let across = dot(uv, perp);

    // Strong anisotropy → elongated strand structures; warp bends them organically
    let warp = fbm(vec2<f32>(along * 8.0, across * 40.0), params.seed + 300u, 2u) - 0.5;
    let strands = fbm(vec2<f32>(along * 22.0, across * 160.0) + vec2<f32>(0.0, warp * 3.0),
                      params.seed + 31u, 4u);
    let strand_mask = smoothstep(0.35, 0.75, strands) * params.fiber_density;

    // Sparse bright filaments ("hairs"), ~1–2 px thick at 1080p
    let hair_n = fbm(vec2<f32>(along * 60.0, across * 900.0), params.seed + 320u, 2u);
    let hair = smoothstep(0.80, 0.92, hair_n) * params.fiber_density * (0.3 + 0.7 * params.roughness);

    textureStore(fiber_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(strand_mask, hair, 0.0, 1.0));
}