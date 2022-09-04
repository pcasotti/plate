fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new().build(&event_loop)?;

    let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    let mut e = examples::App::new(&device, &window)?;

    let pipeline = plate::pipeline::Pipeline::new(
        &device,
        &e.render_pass,
        vk_shader_macros::include_glsl!("shaders/triangle/shader.vert"),
        vk_shader_macros::include_glsl!("shaders/triangle/shader.frag"),
        &Default::default(),
    )?;

    let cmd_pool = plate::CommandPool::new(&device)?;
    let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;

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
                    winit::event::WindowEvent::Resized(_) => e.recreate().unwrap(),
                    _ => (),
                }
            }

            winit::event::Event::MainEventsCleared => window.request_redraw(),
            winit::event::Event::RedrawRequested(window_id) if window_id == window.id() => {
                fence.wait().unwrap();
                fence.reset().unwrap();

                let (i, _) = e.swapchain.next_image(&acquire_sem).unwrap();

                cmd_buffer.record(plate::CommandBufferUsageFlags::empty(), || {
                    e.render_pass.begin(&cmd_buffer, &e.framebuffers[i as usize]);
                    pipeline.bind(&cmd_buffer, e.swapchain.extent());
                    cmd_buffer.draw(3, 1, 0, 0);
                    e.render_pass.end(&cmd_buffer);
                }).unwrap();

                device.queue_submit(
                    &cmd_buffer,
                    plate::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    Some(&acquire_sem),
                    Some(&present_sem),
                    Some(&fence),
                ).unwrap();

                e.swapchain.present(i, &present_sem).unwrap();
            }

            winit::event::Event::LoopDestroyed => device.wait_idle().unwrap(),
            _ => (),
        }
    })
}
