use std::{ffi, marker, mem, sync::Arc};

use ash::vk;

use crate::{Device, PipelineStage, command::*, Error, MemoryPropertyFlags};

pub use vk::BufferUsageFlags as BufferUsageFlags;
pub use vk::SharingMode as SharingMode;

/// A struct to hold a vertex buffer.
pub struct VertexBuffer<T>(Buffer<T>);

impl<T> VertexBuffer<T> {
    /// Creates a new VertexBuffer with data from a slice.
    /// 
    /// # Examples
    /// 
    /// ```no_run
    /// struct Vertex(f32);
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// let vertices = [Vertex(0.0), Vertex(1.0)];
    /// let vertex_buffer = plate::VertexBuffer::new(&device, &vertices, &cmd_pool)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, data: &[T], cmd_pool: &CommandPool) -> Result<Self, Error> {
        let size = (mem::size_of::<T>() * data.len()) as u64;
        let staging = Buffer::new(
            device,
            data.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mut mapped = staging.map()?;
        mapped.write(data);
        let staging = mapped.unmap();

        let buffer = Buffer::new(
            device,
            data.len(),
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        staging.copy_to(&buffer, size, cmd_pool)?;

        Ok(Self(buffer))
    }

    /// Binds the VertexBuffer.
    /// 
    /// To be used when recording a command buffer, should be used after binding the pipeline. The
    /// bound pipeline should be created with the proper attribute and binding descriptions from
    /// this struct type T.
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
    /// # let vertices = [Vertex(0.0), Vertex(1.0)];
    /// let vertex_buffer = plate::VertexBuffer::new(&device, &vertices, &cmd_pool)?;
    /// // cmd_buffer.record(.., || {
    ///     // pipeline.bind(..);
    ///     vertex_buffer.bind(&cmd_buffer);
    /// // })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn bind(&self, command_buffer: &CommandBuffer) {
        let buffers = [self.0.buffer];
        unsafe { self.0.device.cmd_bind_vertex_buffers(**command_buffer, 0, &buffers, &[0]) };
    }
}

/// A struct to hold a index buffer.
pub struct IndexBuffer<T>(Buffer<T>);

impl<T> IndexBuffer<T> {
    /// Creates a new IndexBuffer with data from a slice.
    /// 
    /// # Examples
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// let indices = [0, 1, 2];
    /// let index_buffer = plate::IndexBuffer::new(&device, &indices, &cmd_pool)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, data: &[T], cmd_pool: &CommandPool) -> Result<Self, Error> {
        let size = (mem::size_of::<T>() * data.len()) as u64;
        let staging = Buffer::new(
            device,
            data.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mut mapped = staging.map()?;
        mapped.write(data);
        let staging = mapped.unmap();

        let buffer = Buffer::new(
            device,
            data.len(),
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        staging.copy_to(&buffer, size, cmd_pool)?;

        Ok(Self(buffer))
    }

    /// Binds the IndexBuffer.
    /// 
    /// To be used when recording a command buffer, should be used after binding the pipeline and a
    /// vertex buffer. Should also be used before a [`draw_indexed()`](crate::CommandBuffer::draw_indexed()) call, otherwise the draw call
    /// will not make use of the index buffer.
    /// 
    /// # Examples
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// # let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;
    /// # let indices = [0, 1, 2];
    /// let index_buffer = plate::IndexBuffer::new(&device, &indices, &cmd_pool)?;
    /// // cmd_buffer.record(.., || {
    ///     // pipeline.bind(..);
    ///     // vertex_buffer.bind(..);
    ///     index_buffer.bind(&cmd_buffer);
    ///     // cmd_buffer.draw_indexed(..);
    /// // })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn bind(&self, command_buffer: &CommandBuffer) {
        unsafe {
            self.0.device.cmd_bind_index_buffer(
                **command_buffer,
                self.0.buffer,
                0,
                vk::IndexType::UINT32,
            )
        };
    }
}

/// Represents a buffer with memory mapped in the host, capable of performing write operations.
pub struct MappedBuffer<T> {
    buffer: Buffer<T>,
    mapped: *mut ffi::c_void,
}

impl<T> MappedBuffer<T> {
    /// Unmaps the memory from the host and returns the inner [`Buffer`].
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// # let buffer: plate::Buffer<u32> = plate::Buffer::new( // ..
    ///     # &device,
    ///     # 2,
    ///     # plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     # plate::SharingMode::EXCLUSIVE,
    ///     # plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_COHERENT,
    /// # )?;
    /// // Create a MappedBuffer by mapping an existing Buffer
    /// let mapped = buffer.map()?;
    /// // Unmap the memory from the host
    /// let buffer = mapped.unmap();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn unmap(self) -> Buffer<T> {
        unsafe { self.buffer.device.unmap_memory(self.buffer.mem) };
        self.buffer
    }

    /// Writes data from a slice in the mapped memory.
    ///
    /// The given slice must not have length greater than the `instance_count` parameter provided
    /// during the maped Buffer creation.
    ///
    /// # Panics
    ///
    /// Panics if the index is larger than the buffer capacity.
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// # let buffer: plate::Buffer<u32> = plate::Buffer::new( // ..
    ///     # &device,
    ///     # 2,
    ///     # plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     # plate::SharingMode::EXCLUSIVE,
    ///     # plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_COHERENT,
    /// # )?;
    /// let data = [1, 2, 3];
    /// let mut mapped = buffer.map()?;
    /// mapped.write(&data);
    /// mapped.unmap();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write(&mut self, data: &[T]) {
        self.write_index(data, 0);
    }

    /// Writes data from a slice into a specific index from the mapped memory.
    ///
    /// The index + the legth of the data provided must be within range of the
    /// `instance_count` para parameter provided during the mapepd Buffer creation.
    ///
    /// # Panics
    ///
    /// Panics if the length of the data + the index is larger than the buffer capacity.
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// let buffer: plate::Buffer<u32> = plate::Buffer::new(&device, 4, // ..
    ///     # plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     # plate::SharingMode::EXCLUSIVE,
    ///     # plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_COHERENT,
    /// # )?;
    /// // Map an existing buffer created with capacity for 4 instances
    /// let mut mapped = buffer.map()?;
    /// // Write the contents of `data` to the mapped memory, starting from the index 1
    /// let data = [1, 2, 3];
    /// mapped.write_index(&data, 1);
    /// mapped.unmap();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write_index(&mut self, data: &[T], index: usize) {
        assert!(data.len()+index <= self.buffer.instance_count);

        data.iter()
            .enumerate()
            .for_each(|(i, d)| {
                unsafe {
                    (d as *const T as *const u8)
                        .copy_to_nonoverlapping((self.mapped as *mut u8).offset(((i+index) * self.buffer.alignment_size) as isize), mem::size_of::<T>())
                }
            });
    }

    /// Flushes a this Buffer mapped memory.
    ///
    /// # Example
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// let buffer: plate::Buffer<u32> = plate::Buffer::new(&device, 4, // ..
    ///     # plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     # plate::SharingMode::EXCLUSIVE,
    ///     # plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_COHERENT,
    /// # )?;
    /// let mut mapped = buffer.map()?;
    /// mapped.flush()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush(&self) -> Result<(), Error> { self.flush_index(0, self.buffer.instance_count) }

    /// Flushes a range of this Buffer mapped memory.
    ///
    /// # Panics
    ///
    /// Panics if the offset + size is greater than the Buffer capacity.
    ///
    /// # Example
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// let buffer: plate::Buffer<u32> = plate::Buffer::new(&device, 4, // ..
    ///     # plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     # plate::SharingMode::EXCLUSIVE,
    ///     # plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_COHERENT,
    /// # )?;
    /// let mut mapped = buffer.map()?;
    /// mapped.flush_index(2, 1)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flush_index(&self, offset: usize, size: usize) -> Result<(), Error> {
        assert!(offset+size <= self.buffer.instance_count);

        let range = vk::MappedMemoryRange::builder()
            .memory(self.buffer.mem)
            .offset((self.buffer.alignment_size * offset) as u64)
            .size((self.buffer.alignment_size * size) as u64);

        Ok(unsafe { self.buffer.device.flush_mapped_memory_ranges(&[*range])? })
    }
}

/// A struct containing a vk::Buffer.
pub struct Buffer<T> {
    device: Arc<Device>,
    buffer: vk::Buffer,
    mem: vk::DeviceMemory,
    pub(crate) instance_count: usize,
    alignment_size: usize,

    marker: marker::PhantomData<T>,
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.mem, None);
        }
    }
}

impl<T> Buffer<T> {
    /// Creates a Buffer\<T\>.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// // Create a uniform buffer with capacity of 2 instances
    /// let buffer: plate::Buffer<u32> = plate::Buffer::new(
    ///     &device,
    ///     2,
    ///     plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     plate::SharingMode::EXCLUSIVE,
    ///     plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_COHERENT,
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        device: &Arc<Device>,
        instance_count: usize,
        usage: BufferUsageFlags,
        sharing_mode: SharingMode,
        memory_properties: MemoryPropertyFlags,
    ) -> Result<Self, Error> {
        let alignment_size = alignment::<T>(device, usage);
        let size = alignment_size * instance_count;

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(sharing_mode);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type_index = device.memory_type_index(mem_requirements, memory_properties)?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index as u32);

        let mem = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, mem, 0)? };

        Ok(Self {
            device: Arc::clone(&device),
            buffer,
            mem,
            instance_count,
            alignment_size,

            marker: marker::PhantomData,
        })
    }

    /// Maps the memory the host and returns a [`MappedBuffer`].
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// # let surface = plate::Surface::new(&instance, &window)?;
    /// # let device = plate::Device::new(instance, surface, &Default::default())?;
    /// let buffer: plate::Buffer<u32> = plate::Buffer::new( // ..
    ///     # &device,
    ///     # 2,
    ///     # plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     # plate::SharingMode::EXCLUSIVE,
    ///     # plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_COHERENT,
    /// # )?;
    /// let mapped = buffer.map()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn map(self) -> Result<MappedBuffer<T>, Error> {
        let mapped = unsafe {
            self.device.map_memory(
                self.mem,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )?
        };
        
        Ok(MappedBuffer {
            buffer: self,
            mapped,
        })
    }

    pub(crate) fn copy_to(&self, target: &Buffer<T>, size: vk::DeviceSize, cmd_pool: &CommandPool) -> Result<(), Error> {
        let command_buffer = cmd_pool.alloc_cmd_buffer(CommandBufferLevel::PRIMARY)?;
        command_buffer.record(CommandBufferUsageFlags::ONE_TIME_SUBMIT, || {
            let regions = [*vk::BufferCopy::builder().size(size)];
            unsafe {
                self.device.cmd_copy_buffer(
                    *command_buffer,
                    self.buffer,
                    target.buffer,
                    &regions,
                )
            };
        })?;

        self.device.queue_submit(self.device.graphics_queue, &command_buffer, PipelineStage::empty(), None, None, None)?;
        Ok(unsafe { self.device.queue_wait_idle(self.device.graphics_queue.queue)? })
    }

    pub(crate) fn copy_to_image(&self, image: vk::Image, width: u32, height: u32, cmd_pool: &CommandPool) -> Result<(), Error> {
        let cmd_buffer = cmd_pool.alloc_cmd_buffer(CommandBufferLevel::PRIMARY)?;
        cmd_buffer.record(CommandBufferUsageFlags::ONE_TIME_SUBMIT, || {
            let region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { 
                    width,
                    height,
                    depth: 1,
                });

            unsafe { self.device.cmd_copy_buffer_to_image(*cmd_buffer, self.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[*region]) };
        })?;

        self.device.queue_submit(self.device.graphics_queue, &cmd_buffer, PipelineStage::empty(), None, None, None)?;
        Ok(unsafe { self.device.queue_wait_idle(self.device.graphics_queue.queue)? })
    }

    pub(crate) fn descriptor_info(&self, offset: usize, range: usize) -> vk::DescriptorBufferInfo {
        *vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .offset((self.alignment_size * offset) as u64)
            .range((self.alignment_size * range) as u64)
    }
}

fn alignment<T>(device: &Arc<Device>, usage: BufferUsageFlags) -> usize {
    let limits = unsafe { device.instance.get_physical_device_properties(device.physical_device).limits };
    let min_offset = if usage.contains(BufferUsageFlags::UNIFORM_BUFFER) {
        limits.min_uniform_buffer_offset_alignment.max(limits.non_coherent_atom_size)
    } else if usage.contains(BufferUsageFlags::STORAGE_BUFFER) {
        limits.min_storage_buffer_offset_alignment.max(limits.non_coherent_atom_size)
    } else { 1 } as usize;

    let instance_size = mem::size_of::<T>();
    if min_offset > 0 {
        (instance_size + min_offset - 1) & !(min_offset - 1)
    } else { instance_size }
}
