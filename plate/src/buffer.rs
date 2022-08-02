use std::{ffi, marker, mem, sync::Arc};

use ash::vk;

use crate::{Device, PipelineStage, command::*, sync::*, Error, MemoryPropertyFlags};

pub use vk::BufferUsageFlags as BufferUsageFlags;
pub use vk::SharingMode as SharingMode;

/// A struct to hold a vertex buffer.
pub struct VertexBuffer<T>(Buffer<T>);

impl<T> VertexBuffer<T> {
    /// Creates a new VertexBuffer with data from a slice.
    /// 
    /// # Examples
    /// 
    /// ```
    /// struct Vertex(f32);
    /// let vertices = [Vertex(0.0), Vertex(1.0)];
    /// let vertex_buffer = plate::VertexBuffer::new(&device, &vertices, &cmd_pool)?;
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
    /// ```
    /// let vertex_buffer = plate::VertexBuffer::new(..)?;
    /// cmd_buffer.record(.., || {
    ///     pipeline.bind(..);
    ///     vertex_buffer.bind(&command_buffer);
    /// })?;
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
    /// ```
    /// let indices = [0, 1, 2];
    /// let vertex_buffer = plate::IndexBuffer::new(&device, &vertices, &cmd_pool)?;
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
    /// vertex buffer. Should also be used before a draw_indexed() call, otherwise the draw call
    /// will not make use of the index buffer.
    /// 
    /// # Examples
    /// 
    /// ```
    /// let index_buffer = plate::IndexBuffer::new(..)?;
    /// cmd_buffer.record(.., || {
    ///     pipeline.bind(..);
    ///     vertex_buffer.bind(..);
    ///     index_buffer.bind(&command_buffer);
    ///     command_buffer.draw_indexed(..);
    /// })?;
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
    /// Unmaps the memory from the host and returns the inner Buffer.
    /// 
    /// # Example
    /// 
    /// ```
    /// // Create a MappedBuffer by mapping an existing Buffer
    /// let mapped = buffer.map()?;
    /// // Unmap the memory from the host
    /// let buffer = mapped.unmap();
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
    /// ```
    /// let data = [1, 2, 3];
    /// let mapped = buffer.map()?;
    /// mapped.write(&data);
    /// mapped.unmap();
    /// ```
    pub fn write(&mut self, data: &[T]) {
        assert!(data.len() <= self.buffer.instance_count);
        unsafe {
            data.as_ptr()
                .copy_to_nonoverlapping(self.mapped as *mut _, data.len())
        };
    }

    /// Writes data from a slice into a specific index from the mapped memory.
    ///
    /// The index added to the legth of the data provided must be within range of the
    /// `instance_count` para parameter provided during the mapepd Buffer creation.
    ///
    /// # Panics
    ///
    /// Panics if the length of the data + the index is larger than the buffer capacity.
    /// 
    /// # Example
    /// 
    /// ```
    /// let data = [1, 2, 3];
    /// // Map an existing buffer created with capacity for 4 instances
    /// let mapped = buffer.map()?;
    /// // Write the contents of `data` to the mapped memory, starting from the index 1
    /// mapped.write_index(&data, 1);
    /// mapped.unmap();
    /// ```
    pub fn write_index(&mut self, data: &[T], index: usize) {
        assert!(data.len()+index <= self.buffer.instance_count);
        unsafe {
            data.as_ptr()
                .copy_to_nonoverlapping((self.mapped as *mut T).offset(index as isize), data.len())
        };
    }
}

/// A struct containing a vk::Buffer.
pub struct Buffer<T> {
    device: Arc<Device>,
    buffer: vk::Buffer,
    mem: vk::DeviceMemory,
    instance_count: usize,

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
    /// Creates a Buffer\<T\>
    ///
    /// # Examples
    ///
    /// ```
    /// // Create a uniform buffer with capacity of 2 instances
    /// let buffer = plate::Buffer::new(
    ///     &device,
    ///     2,
    ///     plate::BufferUsageFlags::UNIFORM_BUFFER,
    ///     plate::SharingMode::EXCLUSIVE,
    ///     plate::MemoryPropertyFlags::HOST_VISIBLE | plate::MemoryPropertyFlags::HOST_VISIBLE,
    /// )?;
    /// ```
    pub fn new(
        device: &Arc<Device>,
        instance_count: usize,
        usage: BufferUsageFlags,
        sharing_mode: SharingMode,
        memory_properties: MemoryPropertyFlags,
    ) -> Result<Self, Error> {
        let size = mem::size_of::<T>() * instance_count;

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

            marker: marker::PhantomData,
        })
    }

    /// Maps the memory the host and returns a MappedBuffer.
    /// 
    /// # Example
    /// 
    /// ```
    /// let buffer = plate::Buffer::new(..)?;
    /// let mapped = buffer.map()?;
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

        self.device.queue_submit(self.device.graphics_queue, &command_buffer, PipelineStage::empty(), &Semaphore::None, &Semaphore::None, &Fence::None)?;
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

        self.device.queue_submit(self.device.graphics_queue, &cmd_buffer, PipelineStage::empty(), &Semaphore::None, &Semaphore::None, &Fence::None)?;
        Ok(unsafe { self.device.queue_wait_idle(self.device.graphics_queue.queue)? })
    }

    pub(crate) fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        *vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .offset(0)
            .range((mem::size_of::<T>()*self.instance_count) as u64)
    }
}
