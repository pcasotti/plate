use std::sync::Arc;

pub struct App {
    device: Arc<plate::Device>,
    pub swapchain: plate::Swapchain,
    pub render_pass: plate::RenderPass,
    pub framebuffers: Vec<plate::Framebuffer>,
    depth_image: plate::Image,
}

impl App {
    pub fn new(device: &Arc<plate::Device>, window: &winit::window::Window) -> Result<Self, plate::Error> {
        let swapchain = plate::swapchain::Swapchain::new(&device, &window)?;

        let depth_image = plate::Image::new(
            device,
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

        let framebuffers = swapchain.images
            .iter()
            .map(|image| {
                plate::Framebuffer::new(&device, &render_pass, &[image, &depth_image], swapchain.extent().0, swapchain.extent().1)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            device: Arc::clone(&device),
            swapchain,
            render_pass,
            framebuffers,
            depth_image,
        })
    }

    pub fn recreate(&mut self) -> Result<(), plate::Error> {
        self.depth_image = plate::Image::new(
            &self.device,
            self.swapchain.extent().0,
            self.swapchain.extent().1,
            self.swapchain.depth_format,
            plate::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            plate::ImageAspectFlags::DEPTH,
        )?;

        self.framebuffers = self.swapchain.images
            .iter()
            .map(|image| {
                plate::Framebuffer::new(&self.device, &self.render_pass, &[image, &self.depth_image], self.swapchain.extent().0, self.swapchain.extent().1)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }
}
