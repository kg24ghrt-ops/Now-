// Composite: combine all layers into final output

@group(0) @binding(0)
var params: PaperParams;

@group(0) @binding(2)
var paper_tex: texture_storage_2d<rgba16float, read>;

@group(0) @binding(3)
var grain_tex: texture_storage_2d<rgba16float, read>;

@group(0) @binding(4)
var fiber_tex: texture_storage_2d<rgba16float, read>;

@group(0) @binding(5)
var water_tex: texture_storage_2d<rgba16float, read>;

@group(0) @binding(6)
var output: texture_storage_2d<rgba8unorm, write>;

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

@compute @workgroup_size(8, 8, 1)
fn composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    
    let paper_color = textureLoad(paper_tex, coord);
    let grain_val = textureLoad(grain_tex, coord);
    let fiber_val = textureLoad(fiber_tex, coord);
    let water_val = textureLoad(water_tex, coord);
    
    var color = paper_color.rgb;
    
    color += grain_val.rgb * 0.1;
    
    let fiber_mask = fiber_val.r;
    color *= 1.0 - fiber_mask * 0.15;
    
    let stain = water_val.r;
    color *= 1.0 - stain * 0.3;
    color += vec3<f32>(-0.02, 0.0, 0.02) * stain;
    
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    
    textureStore(output, coord, vec4<f32>(color, 1.0));
}
