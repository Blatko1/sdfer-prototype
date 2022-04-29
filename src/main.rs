use std::{
    collections::HashMap,
    time::{Duration, Instant, SystemTime}, num::NonZeroU32,
};

use artery_font::Rect;
use bmfont_parser::BMFont;
use image::EncodableLayout;
use nalgebra::{Matrix4, Point3, Vector3};
use pollster::block_on;
use serde_json::Value;
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, KeyboardInput, MouseScrollDelta, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[rustfmt::skip]
const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

struct Text {
    x: f32,
    y: f32,
    z: f32,
    text: String,
}

impl Text {
    pub fn new(text: &str, pos: (f32, f32, f32)) -> Self {
        Self {
            x: pos.0,
            y: pos.1,
            z: pos.2,
            text: text.to_owned(),
        }
    }

    pub fn to_vertices(&self, glyphs: HashMap<u32, Glyph>) -> Vec<Vertex> {
        let mut result = Vec::new();
        let mut chars = self.text.chars();

        let mut temp_right: f32;

        let glyph = glyphs.get(&(chars.next().unwrap() as u32)).unwrap();
        let x1 = self.x + glyph.plane_bounds.left;
        let y1 = self.y + glyph.plane_bounds.top;
        let x2 = self.x + glyph.plane_bounds.right;
        let y2 = self.y + glyph.plane_bounds.bottom;
        let tex_x1 = glyph.atlas_bounds.left;
        let tex_y1 = glyph.atlas_bounds.top;
        let tex_x2 = glyph.atlas_bounds.right;
        let tex_y2 = glyph.atlas_bounds.bottom;
        temp_right = x2;
        let vertex = Vertex {
            top_left: [x1, y1, self.z],
            bottom_right: [x2, y2],
            tex_top_left: [tex_x1, tex_y1],
            tex_bottom_right: [tex_x2, tex_y2],
        };
        result.push(vertex);

        for n in 1..self.text.len() {
            let glyph = glyphs.get(&(chars.next().unwrap() as u32)).unwrap();
            let x1 = temp_right + glyph.advance_x;
            let y1 = self.y + glyph.plane_bounds.top;
            let x2 = temp_right + glyph.advance_x + glyph.plane_bounds.right;
            let y2 = self.y + glyph.plane_bounds.bottom;
            let tex_x1 = glyph.atlas_bounds.left;
            let tex_y1 = glyph.atlas_bounds.top;
            let tex_x2 = glyph.atlas_bounds.right;
            let tex_y2 = glyph.atlas_bounds.bottom;
            temp_right = x2;
            let vertex = Vertex {
                top_left: [x1, y1, self.z],
                bottom_right: [x2, y2],
                tex_top_left: [tex_x1, tex_y1],
                tex_bottom_right: [tex_x2, tex_y2],
            };
            result.push(vertex);
        }

        result
    }
}

struct Glyph {
    advance_x: f32,
    plane_bounds: Rect,
    atlas_bounds: Rect,
}

fn main() {
    let data = artery_font::ArteryFont::read(&include_bytes!("../fonts/arial.arfont")[..]).unwrap();
    let image = data.images.first().unwrap();
    let image_data = &image.data;
    let variants = data.variants.first().unwrap();
    println!("flags: {}, w: {}, h: {}, ch: {}, pixel: {:?}, type: {:?}, child: {}, tex_flags: {}, meta: {}", image.flags, image.width, image.height, image.channels, image.pixel_format, image.image_type, image.child_images, image.texture_flags, image.metadata);
    println!("Distance range: {}", variants.metrics.distance_range);
    let mut glyphs: HashMap<u32, Glyph> = HashMap::new();
    for g in &variants.glyphs {
        let glyph = Glyph {
            advance_x: g.advance.horizontal,
            plane_bounds: g.plane_bounds,
            atlas_bounds: g
                .image_bounds
                .scaled(1.0 / image.width as f32, 1.0 / image.height as f32),
        };
        glyphs.insert(g.codepoint, glyph);
    }

    // let mut rgba = Vec::new();
    // for (i, &b) in image_data.iter().enumerate() {
    //     if i % 3 == 0 {
    //         rgba.push(0);
    //     }
    //     rgba.push(b);
    // }

    let text = Text::new("IHNMLKTVXXESO!\"#$%&/(", (-5.0, 0.0, -10.0));
    let vertices = text.to_vertices(glyphs);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Ray Caster")
        .build(&event_loop)
        .unwrap();

    let mut g = block_on(Graphics::new(&window)).unwrap();




    let extent = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let texture = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Texture"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = g.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    g.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &image_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * image.width),
            rows_per_image: None,
        },
        extent,
    );

    /*let blit_shader = g
        .device
        .create_shader_module(&wgpu::include_wgsl!("blit.wgsl"));
    let mipmap_pipeline = g
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "fs_main",
                targets: &[wgpu::TextureFormat::Rgba8Unorm.into()],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

    let mip_count = 4;

    let mut encoder = g
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mipmap command encoder"),
        });
    
        let bind_group_layout = mipmap_pipeline.get_bind_group_layout(0);
        let t_views = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("mip level {}", mip)),
                    format: None,
                    dimension: None,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: NonZeroU32::new(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                })
            })
            .collect::<Vec<_>>();

        for target_mip in 1..mip_count as usize {
            let bind_group = g
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&t_views[target_mip - 1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &t_views[target_mip],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            pass.set_pipeline(&mipmap_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    g.queue.submit(Some(encoder.finish()));*/





    let mut camera = Camera::new(&g);
    let matrix: [[f32; 4]; 4] = camera.update_global_matrix().into();

    let texture_bind_group_layout =
        g.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
    let matrix_buffer = g
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix Buffer"),
            contents: bytemuck::cast_slice(&matrix),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let px_range = variants.metrics.distance_range / variants.metrics.font_size;
    let texture_bind_group = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: matrix_buffer.as_entire_binding(),
            },
        ],
        label: Some("diffuse_bind_group"),
    });
    let pipeline = pipeline(&g, &texture_bind_group_layout);
    let buffer = g
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

    let mut then = SystemTime::now();
    let mut now = SystemTime::now();
    let mut fps = 0;
    let target_framerate = Duration::from_secs_f64(1.0 / 60.0);
    let mut delta_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            winit::event::Event::DeviceEvent { event: e, .. } => camera.input(&e),
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::Resized(inner_size)
                | winit::event::WindowEvent::ScaleFactorChanged {
                    new_inner_size: &mut inner_size,
                    ..
                } => {
                    g.config.width = inner_size.width.max(1);
                    g.config.height = inner_size.height.max(1);
                    g.surface.configure(&g.device, &g.config);

                    camera.resize(&g);
                }

                winit::event::WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

                winit::event::WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,

                _ => (),
            },

            winit::event::Event::RedrawRequested(_) => {
                camera.update();

                let m: [[f32; 4]; 4] = camera.update_global_matrix().into();
                g.queue
                    .write_buffer(&matrix_buffer, 0, bytemuck::cast_slice(&m));

                // Rendering:
                let frame = g.surface.get_current_texture().unwrap();
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    g.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Command Encoder"),
                        });

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.,
                                }),
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });

                    rpass.set_pipeline(&pipeline);
                    rpass.set_vertex_buffer(0, buffer.slice(..));
                    rpass.set_bind_group(0, &texture_bind_group, &[]);
                    rpass.draw(0..4, 0..vertices.len() as u32);
                }

                g.queue.submit(Some(encoder.finish()));
                frame.present();

                // FPS tracking:
                fps += 1;
                if now.duration_since(then).unwrap().as_millis() > 1000 {
                    window.set_title(&format!("FPS: {}", fps));
                    fps = 0;
                    then = now;
                }
                now = SystemTime::now();
            }

            winit::event::Event::MainEventsCleared => {
                if target_framerate <= delta_time.elapsed() {
                    window.request_redraw();
                    delta_time = Instant::now();
                } else {
                    *control_flow = ControlFlow::WaitUntil(
                        Instant::now() + target_framerate - delta_time.elapsed(),
                    );
                }
            }

            _ => (),
        }
    })
}

fn pipeline(g: &Graphics, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let shader = g
        .device
        .create_shader_module(&wgpu::include_wgsl!("shader.wgsl"));
    let default_layout = g
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Default Render Pipeline Layout"),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });
    g.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Default Render Pipeline"),
            layout: Some(&default_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "main_vs",
                buffers: &[Vertex::buffer_layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "main_fs",
                targets: &[wgpu::ColorTargetState {
                    format: g.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            multiview: None,
        })
}

pub struct Graphics {
    pub device: wgpu::Device,
    pub surface: wgpu::Surface,
    pub adapter: wgpu::Adapter,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    top_left: [f32; 3],
    bottom_right: [f32; 2],
    tex_top_left: [f32; 2],
    tex_bottom_right: [f32; 2],
}

impl Vertex {
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                    shader_location: 3,
                },
            ],
        }
    }
}

impl Graphics {
    pub async fn new(window: &winit::window::Window) -> Result<Self, wgpu::RequestDeviceError> {
        let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
        let instance = wgpu::Instance::new(backends);

        let (size, surface) = unsafe {
            let size = window.inner_size();
            let surface = instance.create_surface(&window);
            (size, surface)
        };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("No adapters found!");

        let adapter_info = adapter.get_info();
        println!(
            "Adapter info: Name: {}, backend: {:?}, device: {:?}",
            adapter_info.name, adapter_info.backend, adapter_info.device_type
        );

        let required_features = wgpu::Features::empty();
        let adapter_features = adapter.features();
        let required_limits = wgpu::Limits::default();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    features: adapter_features & required_features,
                    limits: required_limits,
                },
                None,
            )
            .await?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        Ok(Self {
            device,
            surface,
            adapter,
            queue,
            config,
        })
    }
}

pub struct Camera {
    pub eye: Point3<f32>,
    pub target: Point3<f32>,
    up: Vector3<f32>,
    pub aspect: f32,
    pub fov: f32,
    near: f32,
    far: f32,
    pub controller: CameraController,
}

impl Camera {
    pub fn new(graphics: &Graphics) -> Self {
        let controller = CameraController::new();
        Self {
            eye: Point3::new(0., 0., 1.),
            target: Point3::new(0., 0., -1.),
            up: Vector3::y(),
            aspect: graphics.config.width as f32 / graphics.config.height as f32,
            fov: 60.,
            near: 0.01,
            far: 100.0,
            controller,
        }
    }

    pub fn update_global_matrix(&mut self) -> Matrix4<f32> {
        let target = Point3::new(
            self.eye.x + self.target.x,
            self.eye.y + self.target.y,
            self.eye.z + self.target.z,
        );
        let projection =
            Matrix4::new_perspective(self.aspect, self.fov.to_degrees(), self.near, self.far);
        let view = Matrix4::look_at_rh(&self.eye, &target, &self.up);
        OPENGL_TO_WGPU_MATRIX * projection * view
    }

    pub fn resize(&mut self, graphics: &Graphics) {
        self.aspect = graphics.config.width as f32 / graphics.config.height as f32;
    }

    pub fn update(&mut self) {
        self.fov += self.controller.fov_delta;
        self.controller.fov_delta = 0.;
        self.target = Point3::new(
            self.controller.yaw.to_radians().cos() * self.controller.pitch.to_radians().cos(),
            self.controller.pitch.to_radians().sin(),
            self.controller.yaw.to_radians().sin() * self.controller.pitch.to_radians().cos(),
        );
        let target = Vector3::new(self.target.x, 0.0, self.target.z).normalize();
        self.eye +=
            &target * self.controller.speed * (self.controller.forward - self.controller.backward);
        self.eye += &target.cross(&self.up)
            * self.controller.speed
            * (self.controller.right - self.controller.left);
        self.eye += Vector3::new(0.0, 1.0, 0.0)
            * self.controller.speed
            * (self.controller.up - self.controller.down);
    }

    pub fn input(&mut self, event: &winit::event::DeviceEvent) {
        self.controller.process_input(event);
    }
}

pub struct CameraController {
    speed: f32,
    sensitivity: f64,
    forward: f32,
    backward: f32,
    left: f32,
    right: f32,
    up: f32,
    down: f32,
    pub yaw: f32,
    pub pitch: f32,
    fov_delta: f32,
}

impl CameraController {
    pub fn new() -> Self {
        CameraController {
            speed: 0.04,
            sensitivity: 0.1,
            forward: 0.,
            backward: 0.,
            left: 0.,
            right: 0.,
            up: 0.,
            down: 0.,
            yaw: 270.0,
            pitch: 0.0,
            fov_delta: 0.,
        }
    }

    pub fn process_input(&mut self, event: &winit::event::DeviceEvent) {
        match event {
            /*DeviceEvent::MouseMotion { delta } => {
                self.yaw += (delta.0 * self.sensitivity) as f32;
                self.pitch -= (delta.1 * self.sensitivity) as f32;

                if self.pitch > 89.0 {
                    self.pitch = 89.0;
                } else if self.pitch < -89.0 {
                    self.pitch = -89.0;
                }

                if self.yaw > 360.0 {
                    self.yaw = 0.0;
                } else if self.yaw < 0.0 {
                    self.yaw = 360.0;
                }
            }*/
            DeviceEvent::MouseWheel { delta } => {
                self.fov_delta = match delta {
                    MouseScrollDelta::LineDelta(_, scroll) => *scroll,
                    MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition { y, .. }) => {
                        *y as f32
                    }
                }
            }
            DeviceEvent::Motion { .. } => {}
            DeviceEvent::Button { .. } => {}
            DeviceEvent::Key(KeyboardInput {
                state,
                virtual_keycode,
                ..
            }) => {
                let value: f32;
                if *state == winit::event::ElementState::Pressed {
                    value = 1.
                } else {
                    value = 0.;
                }
                match virtual_keycode.unwrap() {
                    VirtualKeyCode::Space => {
                        self.up = value;
                    }
                    VirtualKeyCode::LShift => {
                        self.down = value;
                    }
                    VirtualKeyCode::W => {
                        self.forward = value;
                    }
                    VirtualKeyCode::S => {
                        self.backward = value;
                    }
                    VirtualKeyCode::A => {
                        self.left = value;
                    }
                    VirtualKeyCode::D => {
                        self.right = value;
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }
}
