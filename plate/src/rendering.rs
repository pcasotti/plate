use std::sync::Arc;

use ash::vk;

use crate::{Device, Error, Format, Image, CommandBuffer, PipelineStage};

pub use vk::AttachmentLoadOp;
pub use vk::AttachmentStoreOp;
pub use vk::ImageLayout;
pub use vk::AccessFlags;

pub struct Attachment {
    pub format: Format,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
    pub final_layout: ImageLayout,
}

#[derive(Copy, Clone)]
pub struct AttachmentReference {
    pub attachment: u32,
    pub layout: ImageLayout,
}

pub struct SubpassDescription {
    pub input_attachments: Vec<AttachmentReference>,
    pub color_attachments: Vec<AttachmentReference>,
    pub depth_attachment: Option<AttachmentReference>,
    pub preserve_attachments: Vec<u32>,
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

pub struct Subpass(pub u32);

impl Subpass {
    pub const EXTERNAL: Self = Self(vk::SUBPASS_EXTERNAL);
}

pub struct SubpassDependency {
    pub src_subpass: Subpass,
    pub dst_subpass: Subpass,
    pub src_stage_mask: PipelineStage,
    pub dst_stage_mask: PipelineStage,
    pub src_access_mask: AccessFlags,
    pub dst_access_mask: AccessFlags,
}

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
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .depth_stencil_attachment(&depth_attachments[i]);
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

    pub fn end(&self, cmd_buffer: &CommandBuffer) {
        unsafe { self.device.cmd_end_render_pass(**cmd_buffer) }
    }
}

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
    pub fn new(device: &Arc<Device>, render_pass: &RenderPass, attachments: &[&Image], width: u32, height: u32) -> Result<Self, Error> {
        let attachments = attachments.iter()
            .map(|i| i.view)
            .collect::<Vec<_>>();
        Self::from_image_views(device, render_pass, &attachments, width, height)
    }

    pub(crate) fn from_image_views(device: &Arc<Device>, render_pass: &RenderPass, attachments: &[vk::ImageView], width: u32, height: u32) -> Result<Self, Error> {
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
