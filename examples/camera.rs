use plate::VertexDescription;

#[repr(C)]
struct Vert {
    pos: glam::Vec3,
    uv: glam::Vec2,
}

#[repr(C)]
struct Ubo {
    proj: glam::Mat4,
    view: glam::Mat4,
}

impl VertexDescription for Vert {
    fn binding_descriptions() -> Vec<plate::VertexBindingDescription> {
        vec![
            plate::VertexBindingDescription::new(0, std::mem::size_of::<Self>() as u32, plate::InputRate::VERTEX)
        ]
    }

    fn attribute_descriptions() -> Vec<plate::VertexAttributeDescription> {
        vec![
            plate::VertexAttributeDescription::new(0, 0, memoffset::offset_of!(Self, pos) as u32, plate::Format::R32G32B32_SFLOAT),
            plate::VertexAttributeDescription::new(0, 1, memoffset::offset_of!(Self, uv) as u32, plate::Format::R32G32_SFLOAT),
        ]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new().build(&event_loop)?;

    let instance = plate::Instance::new(&window, &Default::default())?;
    let surface = plate::Surface::new(&instance, &window)?;
    let device = plate::Device::new(instance, surface, &Default::default())?;
    let mut swapchain = plate::swapchain::Swapchain::new(&device, &window, None)?;

    let set_layout = plate::DescriptorSetLayout::new(
        &device,
        &[
            plate::LayoutBinding {
                binding: 0,
                ty: plate::DescriptorType::UNIFORM_BUFFER,
                stage: plate::ShaderStage::VERTEX,
                count: 1,
            },
            plate::LayoutBinding {
                binding: 1,
                ty: plate::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: plate::ShaderStage::FRAGMENT,
                count: 1,
            },
        ],
    )?;
    let pipeline = plate::pipeline::Pipeline::new(
        &device,
        &swapchain,
        vk_shader_macros::include_glsl!("shaders/camera/shader.vert"),
        vk_shader_macros::include_glsl!("shaders/camera/shader.frag"),
        &plate::PipelineParameters {
            vertex_binding_descriptions: Vert::binding_descriptions(),
            vertex_attribute_descriptions: Vert::attribute_descriptions(),
            descriptor_set_layout: Some(&set_layout),
        },
    )?;

    let cmd_pool = plate::CommandPool::new(&device)?;
    let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;

    let vertices = vec![
        Vert { pos: glam::vec3(-0.5, -0.5, 0.5), uv: glam::vec2(1.0, 0.0) },
        Vert { pos: glam::vec3(0.5, -0.5, 0.5), uv: glam::vec2(0.0, 0.0) },
        Vert { pos: glam::vec3(0.5, 0.5, 0.5), uv: glam::vec2(0.0, 1.0) },
        Vert { pos: glam::vec3(-0.5, 0.5, 0.5), uv: glam::vec2(1.0, 1.0) },

        Vert { pos: glam::vec3(-0.5, -0.5, -0.5), uv: glam::vec2(1.0, 0.0) },
        Vert { pos: glam::vec3(0.5, -0.5, -0.5), uv: glam::vec2(0.0, 0.0) },
        Vert { pos: glam::vec3(0.5, 0.5, -0.5), uv: glam::vec2(0.0, 1.0) },
        Vert { pos: glam::vec3(-0.5, 0.5, -0.5), uv: glam::vec2(1.0, 1.0) },
    ];
    let indices = vec![
        0, 1, 2, 2, 3, 0,
        0, 1, 4, 4, 5, 1,
        0, 3, 4, 4, 7, 3,
        2, 3, 7, 7, 6, 2,
        1, 2, 5, 5, 6, 2,
        4, 5, 6, 6, 7, 4,
    ];

    let vert_buffer = plate::VertexBuffer::new(&device, &vertices, &cmd_pool)?;
    let index_buffer = plate::IndexBuffer::new(&device, &indices, &cmd_pool)?;

    let descriptor_pool = plate::DescriptorPool::new(
        &device,
        &[
            plate::PoolSize {
                ty: plate::DescriptorType::UNIFORM_BUFFER,
                count: 1,
            },
            plate::PoolSize {
                ty: plate::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
            },
        ],
        2
    )?;

    let ubo: plate::Buffer<Ubo> = plate::Buffer::new(
        &device,
        1,
        plate::BufferUsageFlags::UNIFORM_BUFFER,
        plate::SharingMode::EXCLUSIVE,
        plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let tex = image::open("examples/texture.jpg")?.flipv();
    let image = plate::Texture::new(&device, &cmd_pool, tex.width(), tex.height(), &tex.to_rgba8().into_raw())?;
    let sampler = plate::Sampler::new(
        &device,
        &plate::SamplerParams {
            address_mode: plate::SamplerAddress::CLAMP_TO_EDGE,
            ..Default::default()
        },
    )?;

    let descriptor_set = plate::DescriptorAllocator::new(&device)
        .add_buffer_binding(0, plate::DescriptorType::UNIFORM_BUFFER, &ubo)
        .add_image_binding(1, plate::DescriptorType::COMBINED_IMAGE_SAMPLER, &image, &sampler)
        .allocate(&set_layout, &descriptor_pool)?;

    let mut ubo = ubo.map()?;

    let fence = plate::Fence::new(&device, plate::FenceFlags::SIGNALED)?;
    let acquire_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    let present_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            winit::event::Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    winit::event::WindowEvent::CloseRequested => {
                        *control_flow = winit::event_loop::ControlFlow::Exit
                    }
                    winit::event::WindowEvent::Resized(_) => { swapchain.recreate(&window).unwrap() }
                    _ => (),
                }
            }

            winit::event::Event::MainEventsCleared => window.request_redraw(),
            winit::event::Event::RedrawRequested(window_id) if window_id == window.id() => {
                fence.wait().unwrap();
                fence.reset().unwrap();

                let (i, _) = swapchain.next_image(&acquire_sem).unwrap();

                ubo.write(&[Ubo {
                    proj: glam::Mat4::perspective_lh(45f32.to_radians(), swapchain.aspect_ratio(), 0.1, 10.0),
                    view: glam::Mat4::look_at_lh(glam::vec3(2.0, 2.0, 2.0), glam::Vec3::ZERO, glam::Vec3::NEG_Y),
                }]);

                cmd_buffer.record(plate::CommandBufferUsageFlags::empty(), || {
                    swapchain.begin_render_pass(&cmd_buffer, i.try_into().unwrap());

                    pipeline.bind(&cmd_buffer, &swapchain);
                    vert_buffer.bind(&cmd_buffer);
                    index_buffer.bind(&cmd_buffer);
                    descriptor_set.bind(&cmd_buffer, pipeline.layout);

                    cmd_buffer.draw_indexed(indices.len() as u32, 1, 0, 0, 0);
                    swapchain.end_render_pass(&cmd_buffer);
                }).unwrap();

                device.queue_submit(
                    device.graphics_queue,
                    &cmd_buffer,
                    plate::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    &acquire_sem,
                    &present_sem,
                    &fence,
                ).unwrap();

                swapchain.present(i, &present_sem).unwrap();
            }

            winit::event::Event::LoopDestroyed => device.wait_idle().unwrap(),
            _ => (),
        }
    })
}
