//! # Paper Engine
//!
//! A high-performance, procedural paper texture generation engine.
//!
//! This library leverages WebGPU (via [`wgpu`]) compute shaders to generate
//! highly realistic paper textures entirely on the GPU. It is designed to be
//! embedded into mobile applications (e.g., Android via JNI) as a dynamic (`cdylib`)
//! or static library.
//!
//! ## Features
//!
//! - **Procedural Generation**: No image assets required. Textures are generated
//!   mathematically using multi-pass compute shaders.
//! - **Realistic Details**: Simulates cellulose fibers, pulp grain, coffee/tea
//!   water stains, laid/chain lines, and paper aging (foxing).
//! - **High Performance**: Parallelized GPU compute passes with an optimized
//!   workgroup size of 16x16.
//!
//! ## Android Integration
//!
//! This crate is compiled as a `cdylib` (`.so`) for Android. You will typically
//! interact with it via JNI bindings in Kotlin/Java. Ensure you pass the raw
//! `ANativeWindow` pointer to [`paper_engine_create`].

use std::ffi::c_void;
use log::info;
use wgpu::{
    Backends, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor,
    Limits, PowerPreference, Queue, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    StorageTextureAccess, Surface, SurfaceConfiguration, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
};
use raw_window_handle::{AndroidNdkWindowHandle, RawWindowHandle};
use bytemuck::{Pod, Zeroable};

// ── Manual Error Handling for Surface ────────────────────────────────
#[derive(Debug)]
pub enum RenderError {
    Surface(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for RenderError {}

// ── Logging ──────────────────────────────────────────────────────────
#[cfg(target_os = "android")]
pub fn init_logging() {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Debug),
    );
}

#[cfg(not(target_os = "android"))]
pub fn init_logging() {
    env_logger::init();
}

// ── Paper parameters passed from Kotlin ─────────────────────────────
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PaperParams {
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    pub grain_intensity: f32,
    pub fiber_density: f32,
    pub water_stain_count: u32,
    pub aging_yellow: f32,
    pub fiber_direction: f32,
    pub roughness: f32,
    pub _pad: [f32; 2], // Exact match for WGSL scalar padding
}

// ── GPU state ────────────────────────────────────────────────────────
pub struct PaperEngine {
    instance: Instance,
    surface: Option<Surface<'static>>,
    device: wgpu::Device,
    queue: Queue,
    config: SurfaceConfiguration,
    paper_pipeline: wgpu::ComputePipeline,
    grain_pipeline: wgpu::ComputePipeline,
    fiber_pipeline: wgpu::ComputePipeline,
    water_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    noise_buffer: wgpu::Buffer,
    paper_texture: wgpu::Texture,
    grain_texture: wgpu::Texture,
    fiber_texture: wgpu::Texture,
    water_texture: wgpu::Texture,
    output_texture: wgpu::Texture,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl PaperEngine {
    pub async fn new(raw_window: *mut c_void, width: u32, height: u32) -> Self {
        init_logging();
        info!("PaperEngine::new() — width={}, height={}", width, height);

        let mut instance_desc = InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = Backends::VULKAN;
        let instance = Instance::new(instance_desc);

        let surface = unsafe {
            let handle = AndroidNdkWindowHandle::new(
                std::ptr::NonNull::new(raw_window as *mut _).unwrap()
            );
            let raw = RawWindowHandle::AndroidNdk(handle);
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: None,
                raw_window_handle: raw,
            })
            .expect("Failed to create Android surface")
        };

        let adapter = instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }).await.expect("No GPU adapter found");

        let (device, queue) = adapter.request_device(
            &DeviceDescriptor {
                label: Some("paper-device"),
                required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                required_limits: Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            },
        ).await.expect("Failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter()
            .find(|f| **f == TextureFormat::Rgba8UnormSrgb || **f == TextureFormat::Rgba8Unorm)
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let tex_desc = |label: &str, format: TextureFormat, usage: TextureUsages| -> wgpu::Texture {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };

        let paper_texture = tex_desc("paper-base", TextureFormat::Rgba16Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING);
        let grain_texture = tex_desc("grain", TextureFormat::Rgba16Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING);
        let fiber_texture = tex_desc("fiber", TextureFormat::Rgba16Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING);
        let water_texture = tex_desc("water", TextureFormat::Rgba16Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING);
        let output_texture = tex_desc("output", TextureFormat::Rgba8Unorm,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC);

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<PaperParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let noise_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("noise-seed"),
            size: 256 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paper-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paper-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: noise_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&paper_texture.create_view(&TextureViewDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&grain_texture.create_view(&TextureViewDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&fiber_texture.create_view(&TextureViewDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&water_texture.create_view(&TextureViewDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&output_texture.create_view(&TextureViewDescriptor::default())) },
            ],
        });

        let paper_shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("paper-base-shader"), source: ShaderSource::Wgsl(include_str!("../shaders/paper_base.wgsl").into()) });
        let grain_shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("grain-shader"), source: ShaderSource::Wgsl(include_str!("../shaders/grain.wgsl").into()) });
        let fiber_shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("fiber-shader"), source: ShaderSource::Wgsl(include_str!("../shaders/fiber.wgsl").into()) });
        let water_shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("water-shader"), source: ShaderSource::Wgsl(include_str!("../shaders/water.wgsl").into()) });
        let composite_shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("composite-shader"), source: ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()) });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("paper-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0, // Added for wgpu v30 compliance
        });

        let make_pipeline = |shader: &wgpu::ShaderModule, entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            instance,
            surface: Some(surface),
            device, queue, config,
            paper_pipeline: make_pipeline(&paper_shader, "paper_base"),
            grain_pipeline: make_pipeline(&grain_shader, "grain"),
            fiber_pipeline: make_pipeline(&fiber_shader, "fiber"),
            water_pipeline: make_pipeline(&water_shader, "water"),
            composite_pipeline: make_pipeline(&composite_shader, "composite"),
            params_buffer, noise_buffer,
            paper_texture, grain_texture, fiber_texture, water_texture, output_texture,
            bind_group_layout, bind_group,
        }
    }

    fn generate_noise_seed(&self) -> Vec<u32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().hash(&mut hasher);
        let seed = hasher.finish();
        let mut state = seed;
        (0..256).map(|_| {
            state = state.wrapping_mul(747796405).wrapping_add(2891336453);
            (state >> 32) as u32
        }).collect()
    }

    pub fn generate(&mut self, params: &PaperParams) {
        info!("Generating paper: {}x{}, seed={}", params.width, params.height, params.seed);

        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
        let noise = self.generate_noise_seed();
        self.queue.write_buffer(&self.noise_buffer, 0, bytemuck::cast_slice(&noise));

        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paper-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.noise_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.paper_texture.create_view(&TextureViewDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.grain_texture.create_view(&TextureViewDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.fiber_texture.create_view(&TextureViewDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.water_texture.create_view(&TextureViewDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.output_texture.create_view(&TextureViewDescriptor::default())) },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("paper-encoder") });
        
        // Note: Using 16 here to match the new 16x16 workgroup size in the shaders.
        // If you kept the old 8x8 shaders, change 15->7 and 16->8.
        let workgroups_x = (params.width + 15) / 16;
        let workgroups_y = (params.height + 15) / 16;

        let dispatch = |pass_desc: ComputePassDescriptor, pipeline: &wgpu::ComputePipeline, enc: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup| {
            let mut pass = enc.begin_compute_pass(&pass_desc);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        };

        dispatch(ComputePassDescriptor { label: Some("paper-base-pass"), timestamp_writes: None }, &self.paper_pipeline, &mut encoder, &self.bind_group);
        dispatch(ComputePassDescriptor { label: Some("grain-pass"), timestamp_writes: None }, &self.grain_pipeline, &mut encoder, &self.bind_group);
        dispatch(ComputePassDescriptor { label: Some("fiber-pass"), timestamp_writes: None }, &self.fiber_pipeline, &mut encoder, &self.bind_group);
        dispatch(ComputePassDescriptor { label: Some("water-pass"), timestamp_writes: None }, &self.water_pipeline, &mut encoder, &self.bind_group);
        dispatch(ComputePassDescriptor { label: Some("composite-pass"), timestamp_writes: None }, &self.composite_pipeline, &mut encoder, &self.bind_group);

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        info!("Paper generation complete");
    }

    pub fn render(&mut self) -> Result<(), RenderError> {
        let output = match self.surface.as_ref().unwrap().get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.as_ref().unwrap().configure(&self.device, &self.config);
                return self.render();
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.as_ref().unwrap().configure(&self.device, &self.config);
                return self.render();
            }
            e => return Err(RenderError::Surface(format!("Failed to acquire surface texture: {:?}", e))),
        };

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("blit-encoder") });

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.output_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyTextureInfo { texture: &output.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 },
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(output);
        Ok(())
    }
}

// ── FFI Boundary (Exported to C/Kotlin) ─────────────────────────────
#[no_mangle]
pub extern "C" fn paper_engine_create(window: *mut c_void, width: u32, height: u32) -> *mut PaperEngine {
    init_logging();
    let engine = pollster::block_on(PaperEngine::new(window, width, height));
    Box::into_raw(Box::new(engine))
}

#[no_mangle]
pub extern "C" fn paper_engine_generate(engine: *mut PaperEngine, params: *const PaperParams) {
    if engine.is_null() || params.is_null() { return; }
    unsafe { (*engine).generate(&*params); }
}

#[no_mangle]
pub extern "C" fn paper_engine_render(engine: *mut PaperEngine) -> bool {
    if engine.is_null() { return false; }
    unsafe { (*engine).render().is_ok() }
}

#[no_mangle]
pub extern "C" fn paper_engine_destroy(engine: *mut PaperEngine) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine)); }
    }
}