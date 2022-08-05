use std::{ffi, sync::Arc};

use ash::vk;

use crate::{DescriptorSetLayout, Device, Swapchain, Format, Error, CommandBuffer};

pub use vk::VertexInputRate as InputRate;

/// Vertex binding information.
///
/// Describes the size of a vertex and the binding to access it in the shader.
pub struct VertexBindingDescription(vk::VertexInputBindingDescription);

/// Vertex attribute information to pass to the shader.
///
/// Describes the offset of a field of a vector, its format and the corresponding binding and
/// location on the shader.
pub struct VertexAttributeDescription(vk::VertexInputAttributeDescription);

impl VertexBindingDescription {
    /// Creates a VertexBindingDescription.
    ///
    /// # Examples
    ///
    /// ```
    /// struct Vertex(u32);
    /// let binding_description = plate::VertexBindingDescription::new(
    ///     0,
    ///     std::mem::size_of::<Vertex>() as u32,
    ///     plate::InputRate::VERTEX,
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    /// Creates a VertexAttributeDescription.
    ///
    /// # Examples
    ///
    /// ```
    /// struct Vertex{
    ///     f1: u32,
    /// };
    /// plate::VertexAttributeDescription::new(
    ///     0,
    ///     0,
    ///     memoffset::offset_of!(Vertex, f1) as u32,
    ///     plate::Format::R32_UINT,
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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

/// Trait for vertex structs, with binding and attribute descriptions.
pub trait VertexDescription {
    /// Returns a Vec of BindingDescriptions corresponding to a vertex.
    fn binding_descriptions() -> Vec<VertexBindingDescription>;
    /// Returns a Vec of AttributeDescriptions corresponding to the fields of a vertex.
    fn attribute_descriptions() -> Vec<VertexAttributeDescription>;
}

/// Vertex and descriptor information for [`Pipeline`] creation.
pub struct PipelineParameters<'a> {
    /// BindingDescriptions of the vertex to be used by the pipeline.
    pub vertex_binding_descriptions: Vec<VertexBindingDescription>,
    /// AttributeDescriptions of the vertex to be used by the pipeline.
    pub vertex_attribute_descriptions: Vec<VertexAttributeDescription>,
    /// DescriptorSetLayouts to be used by the pipeline.
    pub descriptor_set_layout: &'a [&'a DescriptorSetLayout],
}

impl<'a> Default for PipelineParameters<'_> {
    fn default() -> Self {
        Self {
            vertex_binding_descriptions: vec![],
            vertex_attribute_descriptions: vec![],
            descriptor_set_layout: &[],
        }
    }
}

/// A vulkan graphics pipeline
///
/// The Pipeline is responsible for executing all the operations needed to transform the vertices
/// of a scene into the pixels in the rendered image.
pub struct Pipeline {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
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
    /// Creates a Pipeline.
    ///
    /// The vertex input data in the shaders must match the binding and attribute descriptions
    /// specified in `params`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// # let mut swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// let pipeline = plate::pipeline::Pipeline::new(
    ///     &device,
    ///     &swapchain,
    ///     vk_shader_macros::include_glsl!("shaders/triangle/shader.vert"),
    ///     vk_shader_macros::include_glsl!("shaders/triangle/shader.frag"),
    ///     &Default::default(),
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        device: &Arc<Device>,
        swapchain: &Swapchain,
        vert_code: &[u32],
        frag_code: &[u32],
        params: &PipelineParameters,
    ) -> Result<Self, Error> {
        let binding_descriptions: Vec<_> = params.vertex_binding_descriptions.iter().map(|b| b.0).collect();
        let attribute_descriptions: Vec<_> = params.vertex_attribute_descriptions.iter().map(|a| a.0).collect();

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

        let layouts = params.descriptor_set_layout.into_iter()
            .map(|l| l.layout)
            .collect::<Vec<_>>();
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
            .render_pass(swapchain.0.render_pass)
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

    /// Binds the Pipeline.
    ///
    /// To be used when recording a command buffer.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # struct Vertex(f32);
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// # let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;
    /// # let swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let pipeline = plate::pipeline::Pipeline::new(&device, &swapchain, &[], &[],
    /// # &Default::default())?;
    /// // cmd_buffer.record(.., || {
    ///     pipeline.bind(&cmd_buffer, &swapchain);
    /// // })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn bind(&self, command_buffer: &CommandBuffer, swapchain: &Swapchain) {
        unsafe {
            self.device.cmd_bind_pipeline(**command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline)
        }

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain.0.extent.width as f32,
            height: swapchain.0.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        unsafe { self.device.cmd_set_viewport(**command_buffer, 0, &viewports) };

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.0.extent,
        }];
        unsafe { self.device.cmd_set_scissor(**command_buffer, 0, &scissors) };
    }
}
