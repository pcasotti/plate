use std::{ffi, sync::Arc};

use ash::vk;

use crate::{DescriptorSetLayout, Device, Swapchain, Format, Error};

pub use vk::VertexInputRate as InputRate;

pub struct VertexBindingDescription(vk::VertexInputBindingDescription);
pub struct VertexAttributeDescription(vk::VertexInputAttributeDescription);

impl VertexBindingDescription {
    pub fn new(binding: u32, stride: u32, input_rate: InputRate) -> Self {
        Self(
            *ash::vk::VertexInputBindingDescription::builder()
                .binding(binding)
                .stride(stride)
                .input_rate(input_rate)
        )
    }
}

impl VertexAttributeDescription {
    pub fn new(binding: u32, location: u32, offset: u32, format: Format) -> Self {
        Self(
            *ash::vk::VertexInputAttributeDescription::builder()
                .binding(binding)
                .location(location)
                .format(format)
                .offset(offset)
        )
    }
}

pub trait VertexDescription {
    fn binding_descriptions() -> Vec<VertexBindingDescription>;
    fn attribute_descriptions() -> Vec<VertexAttributeDescription>;
}

pub struct PipelineParameters<'a> {
    pub vertex_binding_descriptions: Vec<VertexBindingDescription>,
    pub vertex_attribute_descriptions: Vec<VertexAttributeDescription>,
    pub descriptor_set_layout: Option<&'a DescriptorSetLayout>,
}

impl<'a> Default for PipelineParameters<'_> {
    fn default() -> Self {
        Self {
            vertex_binding_descriptions: vec![],
            vertex_attribute_descriptions: vec![],
            descriptor_set_layout: None,
        }
    }
}

pub struct Pipeline {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    vert_shader: vk::ShaderModule,
    frag_shader: vk::ShaderModule,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.vert_shader, None);
            self.device.destroy_shader_module(self.frag_shader, None);

            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

impl Pipeline {
    pub fn new(
        device: &Arc<Device>,
        swapchain: &Swapchain,
        vert_code: &[u32],
        frag_code: &[u32],
        parameters: &PipelineParameters,
    ) -> Result<Self, Error> {
        let binding_descriptions: Vec<_> = parameters.vertex_binding_descriptions.iter().map(|b| b.0).collect();
        let attribute_descriptions: Vec<_> = parameters.vertex_attribute_descriptions.iter().map(|a| a.0).collect();

        let vert_shader_info = vk::ShaderModuleCreateInfo::builder().code(vert_code);
        let frag_shader_info = vk::ShaderModuleCreateInfo::builder().code(frag_code);

        let vert_shader = unsafe {
            device.create_shader_module(&vert_shader_info, None)?
        };
        let frag_shader = unsafe {
            device.create_shader_module(&frag_shader_info, None)?
        };

        let name = ffi::CString::new("main").expect("Should never fail to build \"main\" string");

        let stage_infos = [
            *vk::PipelineShaderStageCreateInfo::builder()
                .module(vert_shader)
                .stage(vk::ShaderStageFlags::VERTEX)
                .name(&name),
            *vk::PipelineShaderStageCreateInfo::builder()
                .module(frag_shader)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .name(&name),
        ];

        let vertex_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
        }];

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::NONE)
            .polygon_mode(vk::PolygonMode::FILL);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachments = [*vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];

        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);

        let layouts = match &parameters.descriptor_set_layout {
            Some(layout) => vec![layout.layout],
            None => vec![],
        };
        let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
        let layout = unsafe {
            device.create_pipeline_layout(&layout_info, None)?
        };

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stage_infos)
            .vertex_input_state(&vertex_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blend)
            .layout(layout)
            .render_pass(swapchain.render_pass)
            .dynamic_state(&dynamic_state)
            .subpass(0)
            .depth_stencil_state(&stencil_state);

        let pipeline = match unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[*pipeline_info], None) } {
            Ok(p) => Ok(p[0]),
            Err((_, e)) => Err(e)
        }?;

        Ok(Self {
            device: Arc::clone(&device),
            pipeline,
            layout,
            vert_shader,
            frag_shader,
        })
    }

    pub fn bind(&self, command_buffer: vk::CommandBuffer, swapchain: &Swapchain) {
        unsafe {
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline)
        }

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain.extent.width as f32,
            height: swapchain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        unsafe { self.device.cmd_set_viewport(command_buffer, 0, &viewports) };

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        }];
        unsafe { self.device.cmd_set_scissor(command_buffer, 0, &scissors) };
    }
}
