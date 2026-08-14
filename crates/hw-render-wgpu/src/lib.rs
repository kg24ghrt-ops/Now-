//! GPU rendering via wgpu.
//! Provides a renderer that can take an `InkMesh` and render it with a triple‑pass
//! pipeline: base ink, diffusion (compute), and sheen.

use bytemuck::{Pod, Zeroable};
use hw_ink::{InkMesh, InkVertex};
use hw_paper::{PaperParams, PaperTexture};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Errors that can occur during GPU rendering.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("WGPU initialization failed: {0}")]
    InitFailed(String),
    #[error("Shader compilation failed: {0}")]
    ShaderError(String),
    #[error("Failed to create texture: {0}")]
    TextureError(String),
    #[error("Invalid input mesh")]
    InvalidMesh,
}

/// A wgpu‑based renderer.
pub struct Renderer {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Pipelines.
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,

    // Bind group layouts (for sharing).
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    paper_bind_group_layout: wgpu::BindGroupLayout,

    // Cached resources.
    paper_texture: Option<wgpu::Texture>,
    paper_view: Option<wgpu::TextureView>,
    paper_sampler: wgpu::Sampler,

    // Current target size.
    target_width: u32,
    target_height: u32,

    // Offscreen render texture and its view.
    offscreen_texture: Option<wgpu::Texture>,
    offscreen_view: Option<wgpu::TextureView>,
    offscreen_depth_texture: Option<wgpu::Texture>,
    offscreen_depth_view: Option<wgpu::TextureView>,
}

/// Uniforms passed to the shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4], // 4x4 matrix
    ink_color: [f32; 4],      // RGBA
    wetness: f32,
    _padding: [f32; 3],
}

/// Vertex buffer layout (matches `InkVertex`).
const VERTEX_ATTRIBUTES: &[wgpu::VertexAttribute] = &[
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: std::mem::size_of::<[f32; 3]>() as u64,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>()) as u64,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>() + 4) as u64,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>() + 8) as u64,
        shader_location: 4,
    },
];

const VERTEX_BUFFER_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<InkVertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: VERTEX_ATTRIBUTES,
};

impl Renderer {
    /// Create a new renderer. Initializes wgpu with a surface (if `surface` is provided)
    /// or headless for offscreen rendering.
    pub async fn new(surface: Option<wgpu::Surface<'static>>, target_width: u32, target_height: u32) -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN, // Android uses Vulkan; fallback to GLES if needed.
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                ..Default::default()
            })
            .await
            .ok_or_else(|| RenderError::InitFailed("No suitable adapter found".into()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Handwriting Engine GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| RenderError::InitFailed(e.to_string()))?;

        // Compile shaders.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ink Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        // Uniform bind group layout.
        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Paper texture bind group layout.
        let paper_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Paper BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Normal map texture (optional)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Pipeline layout.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &paper_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Render pipeline.
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ink Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VERTEX_BUFFER_LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Compute pipeline for ink diffusion.
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &paper_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ink Diffusion Compute"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_diffuse",
            compilation_options: Default::default(),
            cache: None,
        });

        // Paper sampler.
        let paper_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Paper Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut renderer = Self {
            instance,
            adapter,
            device,
            queue,
            render_pipeline,
            compute_pipeline,
            uniform_bind_group_layout,
            paper_bind_group_layout,
            paper_texture: None,
            paper_view: None,
            paper_sampler,
            target_width,
            target_height,
            offscreen_texture: None,
            offscreen_view: None,
            offscreen_depth_texture: None,
            offscreen_depth_view: None,
        };

        renderer.create_offscreen_targets(target_width, target_height)?;
        Ok(renderer)
    }

    fn create_offscreen_targets(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        // Color texture.
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Depth texture.
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.offscreen_texture = Some(texture);
        self.offscreen_view = Some(view);
        self.offscreen_depth_texture = Some(depth_texture);
        self.offscreen_depth_view = Some(depth_view);
        Ok(())
    }

    /// Load a paper texture from a `PaperTexture` struct.
    pub fn load_paper_texture(&mut self, paper: &PaperTexture) -> Result<(), RenderError> {
        let rgba = paper.as_rgba();
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Paper Texture"),
            size: wgpu::Extent3d {
                width: paper.width,
                height: paper.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(paper.width * 4),
                rows_per_image: Some(paper.height),
            },
            wgpu::Extent3d {
                width: paper.width,
                height: paper.height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Also upload normal map as a separate texture (but we'll store it as a float texture).
        // For simplicity, we'll encode the normal map into a RGBA texture.
        let normal_data: Vec<u8> = paper.normal_map.iter().map(|v| ((v + 1.0) * 127.5) as u8).collect();
        let normal_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Normal Map"),
            size: wgpu::Extent3d {
                width: paper.width,
                height: paper.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &normal_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &normal_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(paper.width * 4),
                rows_per_image: Some(paper.height),
            },
            wgpu::Extent3d {
                width: paper.width,
                height: paper.height,
                depth_or_array_layers: 1,
            },
        );
        let normal_view = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Note: We'll keep both textures, but the shader expects a single bind group with both.
        // We'll store them separately and create a bind group on demand.
        self.paper_texture = Some(texture);
        self.paper_view = Some(view);
        // We'll ignore the normal map for now, or we can combine them.
        // For simplicity, we'll only use the base paper texture.
        Ok(())
    }

    /// Render an ink mesh to the offscreen texture.
    pub fn render_mesh(&mut self, mesh: &InkMesh, ink_color: [f32; 4], wetness: f32) -> Result<(), RenderError> {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return Err(RenderError::InvalidMesh);
        }

        let width = self.target_width;
        let height = self.target_height;

        // Ensure offscreen targets exist.
        if self.offscreen_texture.is_none() {
            self.create_offscreen_targets(width, height)?;
        }

        // Build uniform buffer.
        let uniforms = Uniforms {
            view_proj: [
                [2.0 / width as f32, 0.0, 0.0, 0.0],
                [0.0, -2.0 / height as f32, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0, 1.0],
            ],
            ink_color,
            wetness,
            _padding: [0.0, 0.0, 0.0],
        };
        let uniform_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Vertex and index buffers.
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create bind groups.
        let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform BG"),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        // Paper bind group.
        let paper_bind_group = if let (Some(view), Some(sampler)) = (&self.paper_view, &self.paper_sampler) {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Paper BG"),
                layout: &self.paper_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    // Binding 2 is normal map; we'll just bind the same texture for now.
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
  