use std::sync::Arc;

use ash::vk;

use crate::{image::*, Buffer, CommandBuffer, Device, Error, Pipeline, ImageLayout};

pub use vk::DescriptorType;
pub use vk::ShaderStageFlags as ShaderStage;

/// Errors from the descriptor module.
#[derive(thiserror::Error, Debug)]
pub enum DescriptorError {
    /// The number of provided dynamic offsets is not equal the number of bound dynamic descriptors.
    #[error("The number of provided dynamic offsets is not equal the number of bound dynamic descriptors. Provided {actual} dynamic offsets, but expected {expected}")]
    DynamicOffsetOutOfBounds {
        actual: usize,
        expected: usize,
    },
}

/// A Component for building a descriptor pool.
///
/// Describes the amount of descriptors that can be allocated of a certain type.
pub struct PoolSize {
    /// Type of the descriptor to be allocated
    pub ty: DescriptorType,
    /// Max ammount of descriptors of that type
    pub count: u32,
}

/// Implements the builder pattern for a [`DescriptorPool`].
///
/// # Example
///
/// ```no_run
/// # let event_loop = winit::event_loop::EventLoop::new();
/// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
/// // Create a DescriptorPool capable of allocating at most 2 DescriptorSets of type
/// // UNIFORM_BUFFER or STORAGE_BUFFER
/// let descriptor_pool = plate::DescriptorPool::builder()
///     .add_size(plate::DescriptorType::UNIFORM_BUFFER, 2)
///     .add_size(plate::DescriptorType::STORAGE_BUFFER, 2)
///     .max_sets(2)
///     .build(&device)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct DescriptorPoolBuilder {
    sizes: Vec<PoolSize>,
    max_sets: Option<u32>,
}

impl Default for DescriptorPoolBuilder {
    fn default() -> Self {
        Self {
            sizes: vec![],
            max_sets: None,
        }
    }
}

impl DescriptorPoolBuilder {
    /// Add a [`PoolSize`] component to the DescriptorPool.
    pub fn add_size(&mut self, ty: DescriptorType, count: u32) -> &mut Self {
        self.sizes.push(PoolSize { ty, count });
        self
    }

    /// Set the maximum amount of DescriptorSets to be allocated from the DescriptorPool.
    pub fn max_sets(&mut self, max_sets: u32) -> &mut Self {
        self.max_sets = Some(max_sets);
        self
    }

    /// Builds A DescriptorPool from this builder.
    ///
    /// If the value of `max_sets` is not set, it will default to the sum of all available sets
    /// in the PoolSizes.
    pub fn build(&self, device: &Arc<Device>) -> Result<DescriptorPool, Error> {
        let max_sets = self
            .max_sets
            .unwrap_or(self.sizes.iter().map(|size| size.count).sum());
        DescriptorPool::new(device, &self.sizes, max_sets)
    }
}

/// Holds a vk::DescriptorPool, used to allocate [`DescriptorSets`](DescriptorSet).
pub struct DescriptorPool {
    device: Arc<Device>,
    pool: vk::DescriptorPool,
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

impl DescriptorPool {
    /// Creates a DescriptorPool from [`PoolSizes`](PoolSize) and `max_sets`.
    ///
    /// The `sizes` parameter represent the type of DescriptorSets to be allocated and `max_sets`
    /// is the maximum ammount of DescriptorSets to be allocated from this DescriptorPool.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// // Create a DescriptorPool capable of allocating 1 DescriptorSet of type UNIFORM_BUFFER
    /// let descriptor_pool = plate::DescriptorPool::new(
    ///     &device,
    ///     &[plate::PoolSize {
    ///         ty: plate::DescriptorType::UNIFORM_BUFFER,
    ///         count: 1,
    ///     }],
    ///     1,
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, sizes: &[PoolSize], max_sets: u32) -> Result<Self, Error> {
        let pool_sizes = sizes
            .iter()
            .map(|size| {
                *vk::DescriptorPoolSize::builder()
                    .ty(size.ty)
                    .descriptor_count(size.count)
            })
            .collect::<Vec<_>>();

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets);

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            device: Arc::clone(&device),
            pool,
        })
    }

    /// Returns a [`DescriptorPoolBuilder`] if you prefer to use the builder pattern.
    pub fn builder() -> DescriptorPoolBuilder {
        DescriptorPoolBuilder::default()
    }
}

/// Represents a binding from a descriptor set where the data will be accessible from the shader.
pub struct LayoutBinding {
    /// The actual binding to access from the shader.
    pub binding: u32,
    /// The type of the descriptor. Must match one of the types in the [`DescriptorPool`].
    pub ty: DescriptorType,
    /// The stage in which the data will be acessible.
    pub stage: ShaderStage,
    /// The ammount of descriptors to allocate. Must not exceed the maximum amount of that type
    /// described in the [`DescriptorPool`].
    pub count: u32,
}

/// A DescriptorSetLayout indicates what descriptor types will be allocated from a [`DescriptorPool`].
pub struct DescriptorSetLayout {
    device: Arc<Device>,
    pub(crate) layout: vk::DescriptorSetLayout,
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

impl DescriptorSetLayout {
    /// Creates a DescriptorSetLayout with the specified [`LayoutBindings`](LayoutBinding).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// // Create a layout with an uniform buffer visible in the vertex shader at binding 0
    /// let set_layout = plate::DescriptorSetLayout::new(
    ///     &device,
    ///     &[plate::LayoutBinding {
    ///         binding: 0,
    ///         ty: plate::DescriptorType::UNIFORM_BUFFER,
    ///         stage: plate::ShaderStage::VERTEX,
    ///         count: 1,
    ///     }],
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, bindings: &[LayoutBinding]) -> Result<Self, Error> {
        let bindings = bindings
            .iter()
            .map(|binding| {
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(binding.binding)
                    .descriptor_type(binding.ty)
                    .descriptor_count(binding.count)
                    .stage_flags(binding.stage)
            })
            .collect::<Vec<_>>();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        Ok(Self {
            device: Arc::clone(&device),
            layout,
        })
    }
}

enum WriteDescriptor {
    Buffer {
        binding: u32,
        ty: DescriptorType,
        info: [vk::DescriptorBufferInfo; 1],
        alignment: usize,
    },
    Image {
        binding: u32,
        ty: DescriptorType,
        info: [vk::DescriptorImageInfo; 1],
    },
}

/// Struct used to allocate a [`DescriptorSet`].
pub struct DescriptorAllocator {
    device: Arc<Device>,
    writes: Vec<WriteDescriptor>,
}

impl DescriptorAllocator {
    /// Creates a new DescriptorAllocator.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// let allocator = plate::DescriptorAllocator::new(&device);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            device: Arc::clone(&device),
            writes: vec![],
        }
    }

    /// Binds a [`Buffer`] to a descriptor binding.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let buffer: plate::Buffer<u32> = plate::Buffer::new(
    ///     # &device,
    ///     # 1,
    ///     # plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     # plate::SharingMode::EXCLUSIVE,
    ///     # plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_VISIBLE,
    /// # )?;
    /// let allocator = plate::DescriptorAllocator::new(&device)
    ///     .add_buffer_binding(0, plate::DescriptorType::UNIFORM_BUFFER, &buffer);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_buffer_binding<T>(
        &mut self,
        binding: u32,
        ty: DescriptorType,
        buffer: &Buffer<T>,
    ) -> &mut Self {
        self.add_buffer_index_binding(binding, 0, buffer.instance_count, ty, buffer)
    }

    /// Binds a segment of a [`Buffer`] to a descriptor binding.
    ///
    /// Use the `index` and `instance_count` parameters to specify wich instances of the buffer to
    /// bind.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let buffer: plate::Buffer<u32> = plate::Buffer::new(
    ///     # &device,
    ///     # 1,
    ///     # plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     # plate::SharingMode::EXCLUSIVE,
    ///     # plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_VISIBLE,
    /// # )?;
    /// let allocator = plate::DescriptorAllocator::new(&device)
    ///     .add_buffer_index_binding(0, 0, 1, plate::DescriptorType::UNIFORM_BUFFER, &buffer);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_buffer_index_binding<T>(
        &mut self,
        binding: u32,
        index: usize,
        instance_count: usize,
        ty: DescriptorType,
        buffer: &Buffer<T>,
    ) -> &mut Self {
        let info = [buffer.descriptor_info(index, instance_count)];
        let write = WriteDescriptor::Buffer {
            binding,
            ty,
            info,
            alignment: buffer.alignment_size,
        };
        self.writes.push(write);
        self
    }

    /// Binds a [`Image`] to a descriptor binding.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let image = plate::Image::new(&device, 0, 0, plate::Format::UNDEFINED,
    /// # plate::ImageUsageFlags::empty(), plate::ImageAspectFlags::empty())?;
    /// # let sampler = plate::Sampler::new(&device, &Default::default())?;
    /// let allocator = plate::DescriptorAllocator::new(&device)
    ///     .add_image_binding(
    ///         0, plate::DescriptorType::COMBINED_IMAGE_SAMPLER,
    ///         &image, &sampler,
    ///         plate::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    ///     );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_image_binding(
        &mut self,
        binding: u32,
        ty: DescriptorType,
        image: &Image,
        sampler: &Sampler,
        layout: ImageLayout,
    ) -> &mut Self {
        let info = [image.descriptor_info(sampler, layout)];
        let write = WriteDescriptor::Image {
            binding,
            ty,
            info,
        };
        self.writes.push(write);
        self
    }

    /// Allocates a [`DescriptorSet`] with the added bindings.
    ///
    /// The type and number of the bindings must match the provided DescriptorSetLayout and the
    /// DescriptorPool must have enought capacity for all of them.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let buffer: plate::Buffer<u32> = plate::Buffer::new(
    ///     # &device,
    ///     # 1,
    ///     # plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     # plate::SharingMode::EXCLUSIVE,
    ///     # plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_VISIBLE,
    /// # )?;
    /// # let layout = plate::DescriptorSetLayout::new(&device, &[])?;
    /// # let pool = plate::DescriptorPool::new(&device, &[], 2)?;
    /// // Allocate a DescriptorSet with a uniform buffer at binding 0
    /// let descriptor_set = plate::DescriptorAllocator::new(&device)
    ///     .add_buffer_binding(0, plate::DescriptorType::UNIFORM_BUFFER, &buffer)
    ///     .allocate(&layout, &pool)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn allocate(
        &mut self,
        layout: &DescriptorSetLayout,
        pool: &DescriptorPool,
    ) -> Result<DescriptorSet, Error> {
        let layouts = [layout.layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool.pool)
            .set_layouts(&layouts);

        let set = unsafe { self.device.allocate_descriptor_sets(&alloc_info)?[0] };

        let mut dynamic_sizes = vec![];

        let writes = self
            .writes
            .iter_mut()
            .map(|write| match write {
                WriteDescriptor::Buffer { binding, ty, info, alignment } => {
                    if (*ty == DescriptorType::UNIFORM_BUFFER_DYNAMIC) || (*ty == DescriptorType::STORAGE_BUFFER_DYNAMIC) {
                        dynamic_sizes.push(*alignment as u32);
                    }
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(set)
                        .dst_binding(*binding)
                        .descriptor_type(*ty)
                        .dst_array_element(0)
                        .buffer_info(info)
                }
                WriteDescriptor::Image { binding, ty, info } => {
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(set)
                        .dst_binding(*binding)
                        .descriptor_type(*ty)
                        .dst_array_element(0)
                        .image_info(info)
                }
            })
            .collect::<Vec<_>>();

        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        Ok(DescriptorSet {
            device: Arc::clone(&self.device),
            set,
            dynamic_sizes,
        })
    }
}

/// Holds a vk::DescriptorSet.
pub struct DescriptorSet {
    device: Arc<Device>,
    set: vk::DescriptorSet,
    dynamic_sizes: Vec<u32>,
}

impl DescriptorSet {
    /// Binds the DescriptorSet.
    /// 
    /// To be used when recording a command buffer, should be used after binding the pipeline. The
    /// pipeline should be created with the same [`DescriptorSetLayout`] as this DescriptorSet.
    /// `dynamic_offsets` must have the same length as the number of dynamic descriptors in this
    /// set.
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
    /// # let swapchain = plate::swapchain::Swapchain::new(&device, &window)?;
    /// # let layout = plate::DescriptorSetLayout::new(&device, &[])?;
    /// # let pipeline = plate::pipeline::Pipeline::new(&device, &swapchain.render_pass, &[], &[],
    /// # &Default::default())?;
    /// # let pool = plate::DescriptorPool::new(&device, &[], 2)?;
    /// let descriptor_set = plate::DescriptorAllocator::new(&device).allocate(&layout, &pool)?;
    /// // cmd_buffer.record(.., || {
    ///     // pipeline.bind(..);
    ///     descriptor_set.bind(&cmd_buffer, &pipeline, 0, &[]);
    /// // })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn bind(&self, cmd_buffer: &CommandBuffer, pipeline: &Pipeline, first_set: u32, dynamic_offsets: &[u32]) -> Result<(), Error> {
        if dynamic_offsets.len() != self.dynamic_sizes.len() {
            return Err(DescriptorError::DynamicOffsetOutOfBounds { actual: dynamic_offsets.len(), expected: self.dynamic_sizes.len() }.into())
        }

        let dynamic_offsets = self.dynamic_sizes.iter()
            .enumerate()
            .map(|(i, s)| s * dynamic_offsets[i])
            .collect::<Vec<_>>();

        unsafe {
            self.device.cmd_bind_descriptor_sets(
                **cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                first_set,
                &[self.set],
                &dynamic_offsets,
            )
        };

        Ok(())
    }
}
