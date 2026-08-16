// Realistic stains: coffee/tea = dark wavy RIM + faint interior; moisture =
// soft blot. Rim radius is fbm-perturbed so edges are organic, and stains far
// from the current pixel are skipped early (big perf win at high counts).
// R = coverage, G = accumulated rim (used for brown tint in composite).

@group(0) @binding(0) var<uniform> params: PaperParams;

struct PaperParams {
    width: u32, height: u32, seed: u32,
    grain_intensity: f32, fiber_density: f32, water_stain_count: u32,
    aging_yellow: f32, fiber_direction: f32, roughness: f32,
    _pad0: f32, _pad1: f32,
};

@group(0) @binding(5) var water_out: texture_storage_2d<rgba16float, write>;

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
fn water(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let uv = vec2<f32>(f32(gid.x) / f32(params.width), f32(gid.y) / f32(params.height));
    let aspect = vec2<f32>(f32(params.width) / f32(params.height), 1.0); // round stains on any aspect

    var stain = 0.0;
    var rim_acc = 0.0;
    let count = min(params.water_stain_count, 20u);

    for (var i = 0u; i < count; i = i + 1u) {
        let s = params.seed + i * 7919u;
        let center = vec2<f32>(h21(vec2<u32>(i, 1u), s), h21(vec2<u32>(i, 2u), s));
        let radius = 0.04 + h21(vec2<u32>(i, 3u), s) * 0.12;
        let intensity = 0.10 + h21(vec2<u32>(i, 4u), s) * 0.30;
        let is_ring = h21(vec2<u32>(i, 5u), s) > 0.45;

        let d0 = length((uv - center) * aspect);
        if (d0 > radius * 1.6) { continue; }          // early-out: stain can't reach this pixel

        let wob = fbm(uv * 24.0 + f32(i), s, 3u) - 0.5;   // organic wavy boundary
        let r = radius * (1.0 + wob * 0.35);

        if (is_ring) {
            let rim_w = radius * 0.10;
            let rim = smoothstep(r - rim_w, r, d0) * (1.0 - smoothstep(r, r + rim_w, d0));
            let interior = (1.0 - smoothstep(0.0, r, d0)) * 0.30;
            stain   += (rim + interior) * intensity;
            rim_acc += rim * intensity;
        } else {
            let blot = (1.0 - smoothstep(0.0, r, d0)) * (0.6 + 0.4 * (wob + 0.5));
            stain += blot * intensity * 0.6;
        }
    }

    textureStore(water_out, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(min(stain, 1.0), min(rim_acc, 1.0), 0.0, 1.0));
}