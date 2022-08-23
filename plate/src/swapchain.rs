use std::sync::Arc;

use ash::{extensions::khr, vk};

use crate::{Device, sync::*, image::*, Error, CommandBuffer, Format, rendering::*, PipelineStage, Surface};

/// Errors from the swapchain module.
#[derive(thiserror::Error, Debug)]
pub enum SwapchainError {
    /// None of the available image formats match the depth requirements.
    #[error("No suitable depth format is available")]
    NoSuitableDepthFormat,
}

/// The Swapchain is responsible for providing images to be rendered to the screen.
pub struct Swapchain(pub(crate) Swap, Surface);

impl Swapchain {
    /// Creates a Swapchain.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # struct Vertex(f32);
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// let swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, window: &winit::window::Window) -> Result<Self, Error> {
        let surface = Surface::new(&device.instance, &window)?;
        Ok(Self(Swap::new(device, window, &surface, None)?, surface))
    }

    /// Recreates the swapchain.
    ///
    /// Sould be called if the window was resized or the surface format has changed.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # struct Vertex(f32);
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// swapchain.recreate(&window)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn recreate(&mut self, window: &winit::window::Window) -> Result<(), Error> {
        self.0.device.wait_idle()?;
        Ok(self.0 = Swap::new(&self.0.device, window, &self.1, Some(&self))?)
    }

    /// Returns the aspect ration of the extent.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # struct Vertex(f32);
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// let aspect_ratio = swapchain.aspect_ratio();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn aspect_ratio(&self) -> f32 {
        (self.0.extent.width as f32) / (self.0.extent.height as f32)
    }

    /// Begins the Swapchain render pass.
    ///
    /// To be used when recording a CommandBuffer. Any call do a draw command in between this and a
    /// call to [`end_render_pass()`](Self::end_render_pass()) will draw to this swapchain render pass.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # struct Vertex(f32);
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// # let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let image_index = 0;
    /// // cmd_buffer.record(.., || {
    ///     swapchain.begin_render_pass(&cmd_buffer, image_index);
    ///     // cmd_buffer.draw(..);
    /// // })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn begin_render_pass(&self, command_buffer: &CommandBuffer, image_index: usize) {
        self.0.render_pass.begin(command_buffer, &self.0.framebuffers[image_index])
    }

    /// Ends the Swapchain render pass.
    ///
    /// To be used when recording a CommandBuffer after calling
    /// [`begin_render_pass()`](Self::begin_render_pass()) and the
    /// desired draw commands.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// # let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let image_index = 0;
    /// // cmd_buffer.record(.., || {
    ///     // cmd_buffer.draw(..);
    ///     swapchain.end_render_pass(&cmd_buffer);
    /// // })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn end_render_pass(&self, command_buffer: &CommandBuffer) {
        self.0.render_pass.end(command_buffer)
    }

    /// Acquires the next available swapchain image.
    ///
    /// Returns the index of the next available image from the swapchain and whetherthe swapchain
    /// is suboptimal. Will signal the provided semaphore when done.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let acquire_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// let (image_index, _) = swapchain.next_image(&acquire_sem).unwrap();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn next_image(&self, semaphore: &Semaphore) -> Result<(u32, bool), Error> {
        Ok(unsafe {
            self.0.swapchain_loader.acquire_next_image(
                self.0.swapchain,
                u64::MAX,
                **semaphore,
                vk::Fence::null(),
            )?
        })
    }

    /// Present the image at `image_index` to the screen.
    ///
    /// Will wait on wait_semaphore.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let present_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// let image_index = 0;
    /// swapchain.present(image_index, &present_sem).unwrap();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn present(&self, image_index: u32, wait_semaphore: &Semaphore) -> Result<bool, Error> {
        let swapchains = [self.0.swapchain];
        let wait_semaphores = [**wait_semaphore];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        Ok(unsafe { self.0.swapchain_loader.queue_present(self.0.device.present_queue.queue, &present_info)? })
    }

    /// Returns the depth image format.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let present_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// let depth_format = swapchain.depth_format();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn depth_format(&self) -> Format {
        self.0.depth_format
    }

    /// Returns the color image format.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let present_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// let format = swapchain.image_format();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn image_format(&self) -> Format {
        self.0.image_format
    }

    /// Returns the swapchain extent.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let present_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// let format = swapchain.image_format();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn extent(&self) -> (u32, u32) {
        (self.0.extent.width, self.0.extent.height)
    }

    /// Returns the swapchain [`RenderPass`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let present_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// let render_pass = swapchain.render_pass();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn render_pass(&self) -> &RenderPass {
        &self.0.render_pass
    }
}

pub(crate) struct Swap {
    device: Arc<Device>,

    pub swapchain_loader: khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,

    pub extent: vk::Extent2D,

    #[allow(dead_code)]
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    image_format: Format,

    #[allow(dead_code)]
    depth_image: Image,
    depth_format: Format,

    pub render_pass: RenderPass,
    framebuffers: Vec<Framebuffer>,
}

impl Drop for Swap {
    fn drop(&mut self) {
        unsafe {
            self.image_views
                .iter()
                .for_each(|view| self.device.destroy_image_view(*view, None));
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Swap {
    fn new(
        device: &Arc<Device>,
        window: &winit::window::Window,
        surface: &Surface,
        old_swapchain: Option<&Swapchain>,
    ) -> Result<Self, Error> {
        let surface_capabilities = unsafe {
            surface
                .surface_loader
                .get_physical_device_surface_capabilities(
                    device.physical_device,
                    surface.surface,
                )?
        };
        let surface_formats = unsafe {
            surface
                .surface_loader
                .get_physical_device_surface_formats(
                    device.physical_device,
                    surface.surface,
                )?
        };
        let present_modes = unsafe {
            surface
                .surface_loader
                .get_physical_device_surface_present_modes(
                    device.physical_device,
                    surface.surface,
                )?
        };

        let image_format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::R8G8B8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&surface_formats[0]);

        let present_mode = *present_modes
            .iter()
            .find(|mode| **mode == vk::PresentModeKHR::FIFO)
            .unwrap_or(&present_modes[0]);

        let window_extent = window.inner_size();
        let extent = vk::Extent2D {
            width: window_extent.width.clamp(
                surface_capabilities.min_image_extent.width,
                surface_capabilities.max_image_extent.width,
            ),
            height: window_extent.height.clamp(
                surface_capabilities.min_image_extent.height,
                surface_capabilities.max_image_extent.height,
            ),
        };

        let queue_families = [device.graphics_queue.family, device.present_queue.family];

        let image_count = if surface_capabilities.max_image_count == 0 {
            surface_capabilities.min_image_count + 1
        } else {
            (surface_capabilities.min_image_count + 1).min(surface_capabilities.max_image_count)
        };

        let mut swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.surface)
            .min_image_count(image_count)
            .image_format(image_format.format)
            .image_color_space(image_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        match old_swapchain {
            Some(swapchain) => swapchain_info = swapchain_info.old_swapchain(swapchain.0.swapchain),
            None => (),
        };

        let swapchain_loader = khr::Swapchain::new(&device.instance, &device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None)? };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let image_views = images
            .iter()
            .map(|image| {
                let components = vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                };

                let subresource_range = *vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);

                let view_info = vk::ImageViewCreateInfo::builder()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(image_format.format)
                    .components(components)
                    .subresource_range(subresource_range);

                unsafe { device.create_image_view(&view_info, None) }
            })
            .collect::<Result<Vec<vk::ImageView>, _>>()?;

        let depth_format = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT
        ].into_iter()
            .find(|format| {
                let props = unsafe { device.instance.get_physical_device_format_properties(device.physical_device, *format) };
                props.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            })
            .ok_or(SwapchainError::NoSuitableDepthFormat)?;

        let depth_image = Image::new(
            device,
            extent.width,
            extent.height,
            depth_format,
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ImageAspectFlags::DEPTH,
        )?;

        let color_attachment = Attachment {
            format: image_format.format,
            load_op: AttachmentLoadOp::CLEAR,
            store_op: AttachmentStoreOp::STORE,
            initial_layout: ImageLayout::UNDEFINED,
            final_layout: ImageLayout::PRESENT_SRC_KHR,
        };
        let depth_attachment = Attachment {
            format: depth_format,
            load_op: AttachmentLoadOp::CLEAR,
            store_op: AttachmentStoreOp::STORE,
            initial_layout: ImageLayout::UNDEFINED,
            final_layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpass = SubpassDescription {
            color_attachments: vec![AttachmentReference { attachment: 0, layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL }],
            depth_attachment: Some(AttachmentReference { attachment: 1, layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL }),
            ..Default::default()
        };

        let dependency = SubpassDependency {
            src_subpass: Subpass::EXTERNAL,
            dst_subpass: Subpass(0),
            src_stage_mask: PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS,
            dst_stage_mask: PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS,
            src_access_mask: AccessFlags::NONE,
            dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        };

        let render_pass = RenderPass::new(device, &[color_attachment, depth_attachment], &[subpass], &[dependency])?;

        let framebuffers = image_views
            .iter()
            .map(|view| {
                Framebuffer::from_image_views(device, &render_pass, &[*view, depth_image.view], extent.width, extent.height)
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            device: Arc::clone(&device),
            swapchain_loader,
            swapchain,
            extent,
            images,
            image_views,
            image_format: image_format.format,
            depth_image,
            depth_format,
            render_pass,
            framebuffers,
        })
    }
}
