//! GPU rendering via wgpu.
//! Two passes: base ink (render) + diffusion (compute).
//! The final image is stored in `diffused_texture`.

use bytemuck::{Pod, Zeroable};
use hw_ink::{InkMesh, InkVertex};
use hw_paper::{PaperParams, PaperTexture};
use wgpu::util::DeviceExt;

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("WGPU init failed: {0}")]
    InitFailed(String),
    #[error("Shader error: {0}")]
    ShaderError(String),
    #[error("Texture error: {0}")]
    TextureError(String),
    #[error("Invalid mesh")]
    InvalidMesh,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    uniform_bgl: wgpu::BindGroupLayout,
    combined_bgl: wgpu::BindGroupLayout,

    paper_texture: wgpu::Texture,
    paper_view: wgpu::TextureView,
    normal_texture: wgpu::Texture,
    normal_view: wgpu::TextureView,
    sampler: wgpu::Sampler,

    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    diffused_texture: wgpu::Texture,
    diffused_view: wgpu::TextureView,

    target_width: u32,
    target_height: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    ink_color: [f32; 4],
    wetness: f32,
    _pad1: [f32; 3],
    texture_width: u32,
    texture_height: u32,
    _pad2: [u32; 2],
}

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
    pub async fn new(width: u32, height: u32) -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .ok_or_else(|| RenderError::InitFailed("No adapter".into()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Handwriting Engine"),
                    required_features: wgpu::Features::STORAGE_TEXTURE_FORMAT_RGBA8_UNORM,
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| RenderError::InitFailed(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ink Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        // ---- Bind group layouts ----
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT
                    | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let combined_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Combined BGL"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // ---- Pipelines ----
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&uniform_bgl, &combined_bgl],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Base Ink"),
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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Diffusion"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_diffuse",
            compilation_options: Default::default(),
            cache: None,
        });

        // ---- Resources ----
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Paper Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Generate default paper (will be replaced later if needed)
        let default_paper = PaperTexture::generate(&PaperParams::default());
        let (paper_tex, paper_view, normal_tex, normal_view) =
            Self::upload_paper(&device, &queue, &default_paper);

        // Offscreen textures
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color Tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Tex"),
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

        let diffused_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Diffused Tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let diffused_view = diffused_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Renderer {
            device,
            queue,
            render_pipeline,
            compute_pipeline,
            uniform_bgl,
            combined_bgl,
            paper_texture: paper_tex,
            paper_view,
            normal_texture: normal_tex,
            normal_view,
            sampler,
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            diffused_texture,
            diffused_view,
            target_width: width,
            target_height: height,
        })
    }

    fn upload_paper(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        paper: &PaperTexture,
    ) -> (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ) {
        let rgba = paper.as_rgba();
        let paper_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Paper Tex"),
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
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &paper_tex,
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
        let paper_view = paper_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let normal_data: Vec<u8> = paper
            .normal_map
            .iter()
            .map(|v| ((v + 1.0) * 127.5) as u8)
            .collect();
        let normal_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Normal Tex"),
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
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &normal_tex,
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
        let normal_view = normal_tex.create_view(&wgpu::TextureViewDescriptor::default());

        (paper_tex, paper_view, normal_tex, normal_view)
    }

    pub fn load_paper_texture(&mut self, paper: &PaperTexture) -> Result<(), RenderError> {
        let (pt, pv, nt, nv) = Self::upload_paper(&self.device, &self.queue, paper);
        self.paper_texture = pt;
        self.paper_view = pv;
        self.normal_texture = nt;
        self.normal_view = nv;
        Ok(())
    }

    pub fn render_mesh(
        &mut self,
        mesh: &InkMesh,
        ink_color: [f32; 4],
        wetness: f32,
    ) -> Result<(), RenderError> {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return Err(RenderError::InvalidMesh);
        }

        let w = self.target_width;
        let h = self.target_height;

        // ---- Uniforms ----
        let view_proj = [
            [2.0 / w as f32, 0.0, 0.0, 0.0],
            [0.0, -2.0 / h as f32, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
        ];
        let uniforms = Uniforms {
            view_proj,
            ink_color,
            wetness,
            _pad1: [0.0; 3],
            texture_width: w,
            texture_height: h,
            _pad2: [0, 0],
        };
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // ---- Vertex / index buffers ----
        let vertex_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertices"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Indices"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        // ---- Bind groups ----
        let uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform BG"),
            layout: &self.uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buf,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let combined_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Combined BG"),
            layout: &self.combined_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.paper_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.diffused_view),
                },
            ],
        });

        // ---- Encode ----
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // 1. Base ink pass
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Base Ink"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &uniform_bg, &[]);
            rpass.set_bind_group(1, &combined_bg, &[]);
            rpass.set_vertex_buffer(0, vertex_buf.slice(..));
            rpass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }

        // 2. Diffusion compute pass
        if wetness > 0.01 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Diffusion"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &uniform_bg, &[]);
            cpass.set_bind_group(1, &combined_bg, &[]);
            let wg_x = (w + 7) / 8;
            let wg_y = (h + 7) / 8;
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        } else {
            // No diffusion – just copy base to diffused texture.
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.color_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.diffused_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Read the final image (after diffusion) as RGBA bytes.
    pub fn read_pixels(&self) -> Result<Vec<u8>, RenderError> {
        let w = self.target_width;
        let h = self.target_height;
        let buf_size = (w * h * 4) as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.diffused_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .unwrap()
            .map_err(|e| RenderError::TextureError(e.to_string()))?;

        let data = slice.get_mapped_range();
        let pixels = data.to_vec();
        drop(data);
        readback.unmap();
        Ok(pixels)
    }
}
