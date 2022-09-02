use std::sync::Arc;

use ash::vk;

use crate::{Device, Error, Format, Image, CommandBuffer, PipelineStage};

pub use vk::AttachmentLoadOp;
pub use vk::AttachmentStoreOp;
pub use vk::ImageLayout;
pub use vk::AccessFlags;

/// Describes [`RenderPass`] Attachment.
pub struct Attachment {
    /// Format of the image.
    pub format: Format,
    /// How the attachment is treated at the beginning of the subpass.
    pub load_op: AttachmentLoadOp,
    /// How the attachment is treated at the end of the subpass.
    pub store_op: AttachmentStoreOp,
    /// Layout of the image when the render pass begins.
    pub initial_layout: ImageLayout,
    /// Layout of the image when the render pass ends.
    pub final_layout: ImageLayout,
}

/// Describes an [`Attachment`] reference.
#[derive(Copy, Clone)]
pub struct AttachmentReference {
    /// Attachment index.
    pub attachment: u32,
    /// Layout of the atatchment during the subpass.
    pub layout: ImageLayout,
}

/// Describes the attachments of a subpass.
pub struct SubpassDescription {
    /// Defines the input attachments.
    pub input_attachments: Vec<AttachmentReference>,
    /// Defines the color attachments.
    pub color_attachments: Vec<AttachmentReference>,
    /// Defines the depth attachment.
    pub depth_attachment: Option<AttachmentReference>,
    /// The indices of attachments to preserve throughout the subpass.
    pub preserve_attachments: Vec<u32>,
    /// Defines the resolve attachments.
    pub resolve_attachments: Vec<AttachmentReference>,
}

impl Default for SubpassDescription {
    fn default() -> Self {
        Self {
            input_attachments: vec![],
            color_attachments: vec![],
            depth_attachment: None,
            preserve_attachments: vec![],
            resolve_attachments: vec![],
        }
    }
}

/// Subpass index.
pub struct Subpass(pub u32);

impl Subpass {
    /// Equivalent to [`vk::SUBPASS_EXTERNAL`], refers to outside of the render pass.
    pub const EXTERNAL: Self = Self(vk::SUBPASS_EXTERNAL);
}

/// Introduces a dependecy between two subpasses.
pub struct SubpassDependency {
    /// First subpass.
    pub src_subpass: Subpass,
    /// Second subpass.
    pub dst_subpass: Subpass,
    /// Wait for first subpass commands to reach this stage.
    pub src_stage_mask: PipelineStage,
    /// Resume second subpass commands from this stage.
    pub dst_stage_mask: PipelineStage,
    /// Access scope of the first subpass.
    pub src_access_mask: AccessFlags,
    /// Access scope of the second subpass.
    pub dst_access_mask: AccessFlags,
}

/// Opaque handle to a [`vk::RenderPass`].
pub struct RenderPass {
    device: Arc<Device>,
    pub(crate) render_pass: vk::RenderPass,
    clear_values: Vec<vk::ClearValue>,
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_render_pass(self.render_pass, None)
        }
    }
}

impl RenderPass {
    /// Creates a RenderPass.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let attachments = [];
    /// # let subpasses = [];
    /// # let dependencies = [];
    /// // let attachments = [..];
    /// // let subpasses = [..];
    /// // let dependencies = [..];
    /// let render_pass = plate::RenderPass::new(&device, &attachments, &subpasses, &dependencies)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, attachments: &[Attachment], subpasses: &[SubpassDescription], dependencies: &[SubpassDependency]) -> Result<Self, Error> {
        let clear_values = attachments.iter()
            .map(|a| {
                match a.format {
                    Format::D16_UNORM | Format::D32_SFLOAT | Format::D16_UNORM_S8_UINT | Format::D24_UNORM_S8_UINT | Format::D32_SFLOAT_S8_UINT => {
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            }
                        }
                    }
                    _ => {
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            }
                        }
                    }
                }
            })
            .collect::<Vec<_>>();

        let atts = attachments.iter()
            .map(|a| {
                *vk::AttachmentDescription::builder()
                    .format(a.format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(a.load_op)
                    .store_op(a.store_op)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(a.initial_layout)
                    .final_layout(a.final_layout)
            })
            .collect::<Vec<_>>();

        let input_attachments = subpasses.iter()
            .map(|s| {
                s.input_attachments.iter()
                    .map(|a| vk::AttachmentReference { attachment: a.attachment, layout: a.layout })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let color_attachments = subpasses.iter()
            .map(|s| {
                s.color_attachments.iter()
                    .map(|a| vk::AttachmentReference { attachment: a.attachment, layout: a.layout })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let depth_attachments = subpasses.iter()
            .filter_map(|s| s.depth_attachment)
            .map(|d| vk::AttachmentReference { attachment: d.attachment, layout: d.layout })
            .collect::<Vec<_>>();
        let preserve_attachments = subpasses.iter()
            .map(|s| s.preserve_attachments.clone())
            .collect::<Vec<_>>();
        let resolve_attachments = subpasses.iter()
            .map(|s| {
                s.resolve_attachments.iter()
                    .map(|a| vk::AttachmentReference { attachment: a.attachment, layout: a.layout })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let subs = subpasses.iter()
            .enumerate()
            .map(|(i, _)| {
                let mut builder = vk::SubpassDescription::builder()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
                if !depth_attachments.is_empty() {
                    builder = builder.depth_stencil_attachment(&depth_attachments[i])
                }
                if !input_attachments[i].is_empty() {
                    builder = builder.input_attachments(&input_attachments[i])
                }
                if !color_attachments[i].is_empty() {
                    builder = builder.color_attachments(&color_attachments[i])
                }
                if !preserve_attachments[i].is_empty() {
                    builder = builder.preserve_attachments(&preserve_attachments[i])
                }
                if !resolve_attachments[i].is_empty() {
                    builder = builder.resolve_attachments(&resolve_attachments[i])
                }
                *builder
            })
            .collect::<Vec<_>>();

        let deps = dependencies.iter()
            .map(|d| {
                *vk::SubpassDependency::builder()
                    .src_subpass(d.src_subpass.0)
                    .dst_subpass(d.dst_subpass.0)
                    .src_stage_mask(d.src_stage_mask)
                    .dst_stage_mask(d.dst_stage_mask)
                    .src_access_mask(d.src_access_mask)
                    .dst_access_mask(d.dst_access_mask)
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
            })
            .collect::<Vec<_>>();

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&atts)
            .subpasses(&subs)
            .dependencies(&deps);

        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None)? };

        Ok(Self {
            device: Arc::clone(device),
            render_pass,
            clear_values,
        })
    }

    /// Begins the renderpass.
    ///
    /// `framebuffer` must contain the attachments described at render pass creation. To be used
    /// when recording a [`CommandBuffer`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// # let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;
    /// # let render_pass = plate::RenderPass::new(&device, &[], &[], &[])?;
    /// # let framebuffer = plate::Framebuffer::new(&device, &render_pass, &[], 0, 0)?;
    /// // cmd_buffer.record(.., || {
    ///     render_pass.begin(&cmd_buffer, &framebuffer);
    /// // })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn begin(&self, cmd_buffer: &CommandBuffer, framebuffer: &Framebuffer) {
        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(framebuffer.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: framebuffer.extent,
            })
            .clear_values(&self.clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                **cmd_buffer,
                &begin_info,
                vk::SubpassContents::INLINE,
            )
        }
    }

    /// Ends the renderpass.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// # let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;
    /// # let render_pass = plate::RenderPass::new(&device, &[], &[], &[])?;
    /// # let framebuffer = plate::Framebuffer::new(&device, &render_pass, &[], 0, 0)?;
    /// // cmd_buffer.record(.., || {
    ///     render_pass.end(&cmd_buffer);
    /// // })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn end(&self, cmd_buffer: &CommandBuffer) {
        unsafe { self.device.cmd_end_render_pass(**cmd_buffer) }
    }
}

/// Reference the attachments used by a [`RenderPass`].
pub struct Framebuffer {
    device: Arc<Device>,
    framebuffer: vk::Framebuffer,
    extent: vk::Extent2D,
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_framebuffer(self.framebuffer, None)
        }
    }
}

impl Framebuffer {
    /// Creates a Framebuffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let image = plate::Image::new(&device, 0, 0, plate::Format::UNDEFINED,
    /// # plate::ImageUsageFlags::empty(), plate::ImageAspectFlags::empty())?;
    /// # let render_pass = plate::RenderPass::new(&device, &[], &[], &[])?;
    /// let framebuffer = plate::Framebuffer::new(
    ///     &device,
    ///     &render_pass,
    ///     &[&image],
    ///     image.width,
    ///     image.height
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, render_pass: &RenderPass, attachments: &[&Image], width: u32, height: u32) -> Result<Self, Error> {
        let attachments = attachments.iter()
            .map(|i| i.view)
            .collect::<Vec<_>>();

        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);

        let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None)? };

        let extent = vk::Extent2D {
            width,
            height,
        };

        Ok(Self {
            device: Arc::clone(device),
            framebuffer,
            extent,
        })
    }
}
