use std::sync::Arc;

use ash::{extensions::khr, vk};

use crate::{Device, sync::*, image::*};

pub struct Swapchain(Swap);

impl std::ops::Deref for Swapchain {
    type Target = Swap;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Swapchain {
    pub fn new(device: &Arc<Device>, window: &winit::window::Window, old_swapchain: Option<&Self>) -> Result<Self, vk::Result> {
        Ok(Self(Swap::new(device, window, old_swapchain)?))
    }

    pub fn recreate(&mut self, window: &winit::window::Window) -> Result<(), vk::Result> {
        self.0.device.device_wait_idle()?;
        Ok(self.0 = Swap::new(&self.0.device, window, Some(&self))?)
    }
}

pub struct Swap {
    device: Arc<Device>,

    pub swapchain_loader: khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,

    pub extent: vk::Extent2D,

    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,

    depth_image: Image,

    pub render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Drop for Swap {
    fn drop(&mut self) {
        unsafe {
            self.framebuffers.iter().for_each(|framebuffer| {
                self.device.destroy_framebuffer(*framebuffer, None)
            });
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
        old_swapchain: Option<&Swapchain>,
    ) -> Result<Self, vk::Result> {
        let surface_capabilities = unsafe {
            device
                .surface
                .surface_loader
                .get_physical_device_surface_capabilities(
                    device.physical_device,
                    device.surface.surface,
                )?
        };
        let surface_formats = unsafe {
            device
                .surface
                .surface_loader
                .get_physical_device_surface_formats(
                    device.physical_device,
                    device.surface.surface,
                )?
        };
        let present_modes = unsafe {
            device
                .surface
                .surface_loader
                .get_physical_device_surface_present_modes(
                    device.physical_device,
                    device.surface.surface,
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
            .surface(device.surface.surface)
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
            Some(swapchain) => swapchain_info = swapchain_info.old_swapchain(swapchain.swapchain),
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

        //TODO create a device function for this
        let depth_format = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT
        ].into_iter()
            .find(|format| {
                let props = unsafe { device.instance.get_physical_device_format_properties(device.physical_device, *format) };
                props.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            }).unwrap();

        let depth_image = Image::new(
            device,
            extent.width,
            extent.height,
            depth_format,
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ImageAspectFlags::DEPTH,
        )?;


        let color_attachment = vk::AttachmentDescription::builder()
            .format(image_format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let depth_attachment = vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_refs = [*vk::AttachmentReference::builder()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpasses = [*vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)];

        let attachments = [*color_attachment, *depth_attachment];

        let dependencies = [*vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None)? };

        let framebuffers = image_views
            .iter()
            .map(|view| {
                let attachments = [*view, depth_image.view];
                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                unsafe { device.create_framebuffer(&framebuffer_info, None) }
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            device: Arc::clone(&device),
            swapchain_loader,
            swapchain,
            extent,
            images,
            image_views,
            depth_image,
            render_pass,
            framebuffers,
        })
    }

    pub fn aspect_ratio(&self) -> f32 {
        (self.extent.width as f32) / (self.extent.height as f32)
    }

    pub fn begin_render_pass(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                }
            }
        ];

        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &begin_info,
                vk::SubpassContents::INLINE,
            )
        }
    }

    pub fn end_render_pass(&self, command_buffer: vk::CommandBuffer) {
        unsafe { self.device.cmd_end_render_pass(command_buffer) }
    }

    pub fn next_image(&self, semaphore: &Semaphore) -> Result<(u32, bool), vk::Result> {
        unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                **semaphore,
                vk::Fence::null(),
            )
        }
    }

    pub fn present(&self, image_index: u32, wait_semaphore: &Semaphore) -> Result<bool, vk::Result> {
        let swapchains = [self.swapchain];
        let wait_semaphores = [**wait_semaphore];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { self.swapchain_loader.queue_present(self.device.present_queue.queue, &present_info) }
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }
}
