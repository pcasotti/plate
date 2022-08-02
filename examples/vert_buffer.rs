use plate::VertexDescription;
use plate::plate_macros;

#[repr(C)]
#[derive(plate_macros::Vertex)]
struct Vert {
    #[vertex(loc = 0, format = "R32G32_SFLOAT")]
    pos: glam::Vec2,
    #[vertex(loc = 1, format = "R32G32B32_SFLOAT")]
    color: glam::Vec3,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new().build(&event_loop)?;

    let instance = plate::Instance::new(&window, &Default::default())?;
    let surface = plate::Surface::new(&instance, &window)?;
    let device = plate::Device::new(instance, surface, &Default::default())?;
    let mut swapchain = plate::swapchain::Swapchain::new(&device, &window, None)?;
    let pipeline = plate::pipeline::Pipeline::new(
        &device,
        &swapchain,
        vk_shader_macros::include_glsl!("shaders/vert_buffer/shader.vert"),
        vk_shader_macros::include_glsl!("shaders/vert_buffer/shader.frag"),
        &plate::PipelineParameters {
            vertex_binding_descriptions: Vert::binding_descriptions(),
            vertex_attribute_descriptions: Vert::attribute_descriptions(),
            ..Default::default()
        },
    )?;

    let cmd_pool = plate::CommandPool::new(&device)?;
    let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;

    let vertices = vec![
        Vert { pos: glam::vec2(0.0, -0.5), color: glam::vec3(1.0, 0.0, 0.0) },
        Vert { pos: glam::vec2(0.5, 0.5), color: glam::vec3(0.0, 1.0, 0.0) },
        Vert { pos: glam::vec2(-0.5, 0.5), color: glam::vec3(0.0, 0.0, 1.0) },
    ];
    let vert_buffer = plate::VertexBuffer::new(&device, &vertices, &cmd_pool)?;

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

                cmd_buffer.record(plate::CommandBufferUsageFlags::empty(), || {
                    swapchain.begin_render_pass(*cmd_buffer, i.try_into().unwrap());
                    pipeline.bind(*cmd_buffer, &swapchain);
                    vert_buffer.bind(&cmd_buffer);
                    cmd_buffer.draw(vertices.len() as u32, 1, 0, 0);
                    swapchain.end_render_pass(*cmd_buffer);
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

            winit::event::Event::LoopDestroyed => device.device_wait_idle().unwrap(),
            _ => (),
        }
    })
}
