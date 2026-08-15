//! hw-render-wgpu — Vulkan-only wgpu renderer for Android handwriting.
//!
//! Architecture constraints enforced:
//! - Backends::VULKAN only (fixes Mali GLES silent-failure).
//! - GLSL shaders via naga's GLSL front-end (not WGSL).
//! - Validation layer unconditionally in debug builds.
//! - No per-frame allocation: pooled buffers, pre-created pipelines.
//! - Surface management for Android lifecycle (create/destroy/reconfigure).

use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use wgpu::{
    Adapter, Backends, Buffer, BufferDescriptor, BufferUsages, Color, CommandEncoder, Device,
    DeviceDescriptor, Extent3d, Features, FilterMode, Instance, InstanceDescriptor, InstanceFlags,
    Limits, LoadOp, MemoryHints, Operations, PipelineLayout, PowerPreference, PresentMode, Queue,
    RenderPipeline, RequestAdapterOptions, Sampler, SamplerDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, Surface, SurfaceConfiguration,
    SurfaceError, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

// Re-export types for downstream crates
pub use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindingResource, BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState,
    BufferBinding, BufferBindingType, ColorTargetState, ColorWrites, CompareFunction,
    DepthStencilState, Face, FragmentState, FrontFace, IndexFormat, MultisampleState,
    PrimitiveState, PrimitiveTopology, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, SamplerBindingType, StorageTextureAccess,
    TextureSampleType, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
    VertexStepMode,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("no Vulkan adapter found")]
    NoAdapter,
    #[error("device request failed: {0}")]
    RequestDevice(String),
    #[error("surface creation failed: {0}")]
    CreateSurface(String),
    #[error("surface lost or timeout: {0}")]
    Surface(#[from] SurfaceError),
    #[error("shader compilation failed: {0}")]
    ShaderCompile(String),
    #[error("invalid mesh")]
    InvalidMesh,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct RenderConfig {
    pub surface_format: TextureFormat,
    pub physical_size: (u32, u32),
    pub scale_factor: f32,
    pub vsync: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            surface_format: TextureFormat::Rgba8UnormSrgb,
            physical_size: (1080, 2400),
            scale_factor: 2.0,
            vsync: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Vulkan Context (instance + adapter + device + queue)
// ---------------------------------------------------------------------------

pub struct VulkanContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl VulkanContext {
    pub async fn new() -> Result<Self, RenderError> {
        // CRITICAL: Vulkan ONLY. Explicitly exclude GL/GLES.
        let backends = Backends::VULKAN;

        // Validation layer: unconditional in debug, stripped in release.
        // InstanceFlags::debugging() sets VALIDATION + DEBUG bits.
        // On Vulkan backend, VALIDATION enables VK_LAYER_KHRONOS_validation.
        let flags = if cfg!(debug_assertions) {
            InstanceFlags::debugging()
        } else {
            InstanceFlags::empty()
        };

        let instance = Instance::new(&InstanceDescriptor {
            backends,
            flags,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(RenderError::NoAdapter)?;

        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Handwriting Device"),
                    required_features: Features::STORAGE_TEXTURE_FORMAT_RGBA8_UNORM,
                    required_limits: Limits::downlevel_defaults().using_resolution(adapter_limits),
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| RenderError::RequestDevice(e.to_string()))?;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

// ---------------------------------------------------------------------------
// Render Surface (swapchain + depth, recreated on resize)
// ---------------------------------------------------------------------------

pub struct RenderSurface {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub depth_texture: Texture,
    pub depth_view: TextureView,
}

impl RenderSurface {
    pub fn new(
        ctx: &VulkanContext,
        window: impl HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
        config: &RenderConfig,
    ) -> Result<Self, RenderError> {
        let surface = ctx
            .instance
            .create_surface(window)
            .map_err(|e| RenderError::CreateSurface(e.to_string()))?;

        let surface_caps = surface.get_capabilities(&ctx.adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = if config.vsync {
            PresentMode::AutoVsync
        } else {
            PresentMode::AutoNoVsync
        };

        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: config.physical_size.0,
            height: config.physical_size.1,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&ctx.device, &surface_config);

        let (depth_tex, depth_view) = Self::create_depth(&ctx.device, &surface_config);

        Ok(Self {
            surface,
            config: surface_config,
            depth_texture: depth_tex,
            depth_view,
        })
    }

    pub fn reconfigure(&mut self, ctx: &VulkanContext, new_size: (u32, u32)) {
        self.config.width = new_size.0;
        self.config.height = new_size.1;
        self.surface.configure(&ctx.device, &self.config);
        let (depth_tex, depth_view) = Self::create_depth(&ctx.device, &self.config);
        self.depth_texture = depth_tex;
        self.depth_view = depth_view;
    }

    fn create_depth(device: &Device, config: &SurfaceConfiguration) -> (Texture, TextureView) {
        let desc = TextureDescriptor {
            label: Some("depth_texture"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&TextureViewDescriptor::default());
        (texture, view)
    }

    /// Acquire next frame, handling Android surface loss.
    pub fn acquire_frame(&self) -> Result<wgpu::SurfaceTexture, SurfaceError> {
        match self.surface.get_current_texture() {
            Ok(frame) => Ok(frame),
            Err(SurfaceError::Lost) => {
                self.surface.configure(&self.surface, &self.config);
                self.surface.get_current_texture()
            }
            Err(SurfaceError::Outdated) => {
                self.surface.configure(&self.surface, &self.config);
                self.surface.get_current_texture()
            }
            Err(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// GLSL Shaders (via naga front-end)
// ---------------------------------------------------------------------------

pub struct GlslShader {
    pub source: Cow<'static, str>,
    pub stage: naga::ShaderStage,
    pub defines: Vec<(String, Option<String>)>,
}

impl GlslShader {
    pub fn compile(&self, device: &Device, label: &str) -> Result<ShaderModule, RenderError> {
        let desc = ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Glsl {
                shader: Cow::Borrowed(&self.source),
                stage: self.stage,
                defines: self.defines.clone(),
            },
        };
        Ok(device.create_shader_module(desc))
    }
}

// ---------------------------------------------------------------------------
// Pooled GPU Buffers
// ---------------------------------------------------------------------------

pub struct BufferPool {
    pub stroke_points: Buffer,
    pub stroke_indices: Buffer,
    pub uniform_frame: Buffer,
    pub uniform_ink: Buffer,
}

impl BufferPool {
    pub const MAX_STROKE_POINTS: u64 = 16_384;
    pub const MAX_INDICES: u64 = 65_536;
    pub const UNIFORM_SIZE: u64 = 256;

    pub fn new(device: &Device) -> Self {
        let mk = |label: &'static str, size: u64, usage| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        Self {
            stroke_points: mk(
                "stroke_points",
                Self::MAX_STROKE_POINTS * 16,
                BufferUsages::VERTEX | BufferUsages::COPY_DST,
            ),
            stroke_indices: mk(
                "stroke_indices",
                Self::MAX_INDICES * 4,
                BufferUsages::INDEX | BufferUsages::COPY_DST,
            ),
            uniform_frame: mk(
                "uniform_frame",
                Self::UNIFORM_SIZE,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            uniform_ink: mk(
                "uniform_ink",
                Self::UNIFORM_SIZE,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Uniforms (GPU layout)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub viewport: [f32; 2],
    pub time_secs: f32,
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct InkUniforms {
    pub ink_color: [f32; 4],
    pub wetness: f32,
    pub diffusion: f32,
    pub roughness: f32,
    pub _pad2: f32,
}

// ---------------------------------------------------------------------------
// Ink Render Pipeline
// ---------------------------------------------------------------------------

pub struct InkPipeline {
    pub pipeline: RenderPipeline,
    pub bind_group_layout: BindGroupLayout,
}

impl InkPipeline {
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        vert: &ShaderModule,
        frag: &ShaderModule,
    ) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ink_bgl"),
            entries: &[
                // Frame uniform (binding 0)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Ink material uniform (binding 1)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Paper diffuse texture (binding 2)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Paper normal texture (binding 3)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler (binding 4)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ink_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("ink_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: vert,
                entry_point: Some("main"),
                buffers: &[VertexBufferLayout {
                    array_stride: 16,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: 8,
                            shader_location: 1,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: 12,
                            shader_location: 2,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: frag,
                entry_point: Some("main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Paper texture upload (fixed normal map format)
// ---------------------------------------------------------------------------

pub fn upload_paper_textures(
    device: &Device,
    queue: &Queue,
    paper_rgba: &[u8],
    normal_rgba: &[u8], // FIXED: must be RGBA, not 3-channel f32
    width: u32,
    height: u32,
) -> (Texture, TextureView, Texture, TextureView, Sampler) {
    let create_tex = |label, format, data| {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&TextureViewDescriptor::default());
        (tex, view)
    };

    let (paper_tex, paper_view) =
        create_tex("paper_tex", TextureFormat::Rgba8UnormSrgb, paper_rgba);
    let (normal_tex, normal_view) =
        create_tex("normal_tex", TextureFormat::Rgba8Unorm, normal_rgba);

    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("paper_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..Default::default()
    });

    (paper_tex, paper_view, normal_tex, normal_view, sampler)
}

// ---------------------------------------------------------------------------
// Frame encoder helper
// ---------------------------------------------------------------------------

pub struct FrameEncoder<'a> {
    pub encoder: CommandEncoder,
    pub surface_view: TextureView,
    pub depth_view: &'a TextureView,
    pub surface_format: TextureFormat,
}

impl<'a> FrameEncoder<'a> {
    pub fn begin_ink_pass(&mut self, clear_color: Color) -> wgpu::RenderPass<'_> {
        self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("ink_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.surface_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: self.depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    pub fn finish(self) -> wgpu::CommandBuffer {
        self.encoder.finish()
    }
}

// ---------------------------------------------------------------------------
// GLSL Shaders (embedded, architecture doc §3)
// ---------------------------------------------------------------------------

pub const INK_VERTEX_GLSL: &str = r#"
#version 450 core

layout(location = 0) in vec2 a_position;
layout(location = 1) in float a_pressure;
layout(location = 2) in float a_timestamp;

layout(binding = 0) uniform FrameUniform {
    mat4 u_view_proj;
    vec2 u_viewport;
    float u_time;
    float _pad;
};

layout(location = 0) out vec2 v_uv;
layout(location = 1) out float v_pressure;
layout(location = 2) out float v_timestamp;

void main() {
    v_uv = a_position;
    v_pressure = a_pressure;
    v_timestamp = a_timestamp;
    gl_Position = u_view_proj * vec4(a_position, 0.0, 1.0);
}
"#;

pub const INK_FRAGMENT_GLSL: &str = r#"
#version 450 core

layout(location = 0) in vec2 v_uv;
layout(location = 1) in float v_pressure;
layout(location = 2) in float v_timestamp;

layout(binding = 1) uniform InkMaterial {
    vec4 u_ink_color;
    float u_diffusion;
    float u_roughness;
    float u_wetness;
    float _pad;
};

layout(binding = 2) uniform texture2D u_paper_tex;
layout(binding = 3) uniform texture2D u_normal_tex;
layout(binding = 4) uniform sampler u_paper_sampler;

layout(location = 0) out vec4 f_color;

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    float fiber = hash(v_uv * 400.0) * 0.5 + hash(v_uv * 120.0) * 0.3 + 0.2;
    
    float spread = u_diffusion * v_pressure * (1.0 - smoothstep(0.0, 5.0, v_timestamp));
    float edge_softness = 1.0 - smoothstep(0.0, spread + 0.001, length(v_uv - 0.5));
    
    vec3 light_dir = normalize(vec3(0.3, 0.7, 0.5));
    
    // Sample normal from paper normal map
    vec3 normal_raw = texture(sampler2D(u_normal_tex, u_paper_sampler), v_uv).xyz;
    vec3 normal = normalize(normal_raw * 2.0 - 1.0);
    // Perturb by roughness
    normal.xy += (hash(v_uv * 800.0) - 0.5) * u_roughness;
    normal = normalize(normal);
    
    float ndotl = max(dot(normal, light_dir), 0.0);
    float absorption = mix(0.6, 1.0, fiber);
    
    vec4 paper = texture(sampler2D(u_paper_tex, u_paper_sampler), v_uv);
    vec3 ink = u_ink_color.rgb * absorption * (0.5 + 0.5 * ndotl);
    float alpha = u_ink_color.a * edge_softness * (0.3 + 0.7 * v_pressure);
    
    ink *= 1.0 - u_wetness * 0.1;
    f_color = vec4(ink, alpha) + paper * (1.0 - alpha);
}
"#;
