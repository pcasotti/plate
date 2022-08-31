fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new().build(&event_loop)?;

    let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;

    let mut depth_image = plate::Image::new(
        &device,
        swapchain.extent().0,
        swapchain.extent().1,
        swapchain.depth_format,
        plate::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        plate::ImageAspectFlags::DEPTH,
    )?;

    let color_attachment = plate::Attachment {
        format: swapchain.surface_format,
        load_op: plate::AttachmentLoadOp::CLEAR,
        store_op: plate::AttachmentStoreOp::STORE,
        initial_layout: plate::ImageLayout::UNDEFINED,
        final_layout: plate::ImageLayout::PRESENT_SRC_KHR,
    };
    let depth_attachment = plate::Attachment {
        format: swapchain.depth_format,
        load_op: plate::AttachmentLoadOp::CLEAR,
        store_op: plate::AttachmentStoreOp::STORE,
        initial_layout: plate::ImageLayout::UNDEFINED,
        final_layout: plate::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    let subpass = plate::SubpassDescription {
        color_attachments: vec![plate::AttachmentReference { attachment: 0, layout: plate::ImageLayout::COLOR_ATTACHMENT_OPTIMAL }],
        depth_attachment: Some(plate::AttachmentReference { attachment: 1, layout: plate::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL }),
        ..Default::default()
    };

    let dependency = plate::SubpassDependency {
        src_subpass: plate::Subpass::EXTERNAL,
        dst_subpass: plate::Subpass(0),
        src_stage_mask: plate::PipelineStage::COLOR_ATTACHMENT_OUTPUT | plate::PipelineStage::EARLY_FRAGMENT_TESTS,
        dst_stage_mask: plate::PipelineStage::COLOR_ATTACHMENT_OUTPUT | plate::PipelineStage::EARLY_FRAGMENT_TESTS,
        src_access_mask: plate::AccessFlags::NONE,
        dst_access_mask: plate::AccessFlags::COLOR_ATTACHMENT_WRITE | plate::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
    };

    let render_pass = plate::RenderPass::new(&device, &[color_attachment, depth_attachment], &[subpass], &[dependency])?;
    let mut framebuffers = swapchain.images
        .iter()
        .map(|image| {
            plate::Framebuffer::new(&device, &render_pass, &[image, &depth_image], swapchain.extent().0, swapchain.extent().1)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let pipeline = plate::pipeline::Pipeline::new(
        &device,
        &render_pass,
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
                    winit::event::WindowEvent::Resized(_) => { (depth_image, framebuffers) = recreate(&device, &window, &mut swapchain, &render_pass).unwrap() }
                    _ => (),
                }
            }

            winit::event::Event::MainEventsCleared => window.request_redraw(),
            winit::event::Event::RedrawRequested(window_id) if window_id == window.id() => {
                fence.wait().unwrap();
                fence.reset().unwrap();

                let (i, _) = swapchain.next_image(&acquire_sem).unwrap();

                cmd_buffer.record(plate::CommandBufferUsageFlags::empty(), || {
                    render_pass.begin(&cmd_buffer, &framebuffers[i as usize]);
                    pipeline.bind(&cmd_buffer, swapchain.extent());
                    cmd_buffer.draw(3, 1, 0, 0);
                    render_pass.end(&cmd_buffer);
                }).unwrap();

                device.queue_submit(
                    &cmd_buffer,
                    plate::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    Some(&acquire_sem),
                    Some(&present_sem),
                    Some(&fence),
                ).unwrap();

                swapchain.present(i, &present_sem).unwrap();
            }

            winit::event::Event::LoopDestroyed => device.wait_idle().unwrap(),
            _ => (),
        }
    })
}

fn recreate(
    device: &std::sync::Arc<plate::Device>,
    window: &winit::window::Window,
    swapchain: &mut plate::Swapchain,
    render_pass: &plate::RenderPass,
) -> Result<(plate::Image, Vec<plate::Framebuffer>), plate::Error> {
    swapchain.recreate(window)?;

    let depth_image = plate::Image::new(
        device,
        swapchain.extent().0,
        swapchain.extent().1,
        swapchain.depth_format,
        plate::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        plate::ImageAspectFlags::DEPTH,
    )?;

    let framebuffers = swapchain.images
        .iter()
        .map(|image| {
            plate::Framebuffer::new(&device, render_pass, &[image, &depth_image], swapchain.extent().0, swapchain.extent().1)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((depth_image, framebuffers))
}
