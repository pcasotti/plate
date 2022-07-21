use std::{ffi, marker, mem, sync::Arc};

use ash::vk;

use crate::{Device, PipelineStage, command::*, sync::*};

pub use vk::BufferUsageFlags as BufferUsageFlags;
pub use vk::SharingMode as SharingMode;
pub use vk::MemoryPropertyFlags as MemoryPropertyFlags;

pub struct VertexBuffer<T> {
    buffer: Buffer<T>,
}

impl<T> VertexBuffer<T> {
    pub fn new(device: &Arc<Device>, data: &[T], cmd_pool: &CommandPool) -> Result<Self, vk::Result> {
        let size = (mem::size_of::<T>() * data.len()) as u64;
        let mut staging = Buffer::new(
            device,
            data.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        staging.map()?;
        staging.write(data);
        staging.unmap();

        let buffer = Buffer::new(
            device,
            data.len(),
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        staging.copy_to(&buffer, size, cmd_pool)?;

        Ok(Self { buffer })
    }

    pub fn bind(&self, command_buffer: &CommandBuffer) {
        let buffers = [self.buffer.buffer];
        unsafe { self.buffer.device.cmd_bind_vertex_buffers(**command_buffer, 0, &buffers, &[0]) };
    }
}

pub struct IndexBuffer<T> {
    buffer: Buffer<T>,
}

impl<T> IndexBuffer<T> {
    pub fn new(device: &Arc<Device>, data: &[T], cmd_pool: &CommandPool) -> Result<Self, vk::Result> {
        let size = (mem::size_of::<T>() * data.len()) as u64;
        let mut staging = Buffer::new(
            device,
            data.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        staging.map()?;
        staging.write(data);
        staging.unmap();

        let buffer = Buffer::new(
            device,
            data.len(),
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        staging.copy_to(&buffer, size, cmd_pool)?;

        Ok(Self { buffer })
    }

    pub fn bind(&self, command_buffer: &CommandBuffer) {
        unsafe {
            self.buffer.device.cmd_bind_index_buffer(
                **command_buffer,
                self.buffer.buffer,
                0,
                vk::IndexType::UINT32,
            )
        };
    }
}

pub struct Buffer<T> {
    device: Arc<Device>,
    buffer: vk::Buffer,
    mem: vk::DeviceMemory,
    mapped: Option<*mut ffi::c_void>, 
    instance_count: usize,

    marker: marker::PhantomData<T>,
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.unmap();
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.mem, None);
        }
    }
}

impl<T> Buffer<T> {
    pub fn new(
        device: &Arc<Device>,
        instance_count: usize,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<Self, vk::Result> {
        let size = mem::size_of::<T>() * instance_count;

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(sharing_mode);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_properties = unsafe { device.instance.get_physical_device_memory_properties(device.physical_device) };
        let mem_type_index = mem_properties
            .memory_types
            .iter()
            .enumerate()
            .find(|(i, ty)| {
                mem_requirements.memory_type_bits & (1 << i) > 0
                    && ty.property_flags.contains(memory_properties)
            })
            .unwrap()
            .0;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index as u32);

        let mem = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, mem, 0)? };

        Ok(Self {
            device: Arc::clone(&device),
            buffer,
            mem,
            mapped: None,
            instance_count,

            marker: marker::PhantomData,
        })
    }

    pub fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        *vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .offset(0)
            .range((mem::size_of::<T>()*self.instance_count) as u64)
    }

    pub fn map(&mut self) -> Result<(), vk::Result> {
        if self.mapped.is_some() {
            panic!("buffer memory already mapped")
        }
        self.mapped = Some(unsafe {
            self.device.map_memory(
                self.mem,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )?
        });
        Ok(())
    }

    pub fn unmap(&mut self) {
        if self.mapped.is_some() {
            unsafe { self.device.unmap_memory(self.mem) }
            self.mapped = None;
        }
    }

    pub fn write(&mut self, data: &[T]) {
        if self.mapped.is_none() {
            panic!("cannot write to unmapped buffer")
        }
        unsafe {
            data.as_ptr()
                .copy_to_nonoverlapping(self.mapped.unwrap() as *mut _, data.len())
        };
    }

    pub fn write_index(&mut self, data: &[T], index: usize) {
        assert!(index < self.instance_count);
        if self.mapped.is_none() {
            panic!("cannot write to unmapped buffer")
        }
        unsafe {
            data.as_ptr()
                .copy_to_nonoverlapping((self.mapped.unwrap() as *mut T).offset(index as isize), data.len())
        };
    }

    pub fn copy_to(&self, target: &Buffer<T>, size: vk::DeviceSize, cmd_pool: &CommandPool) -> Result<(), vk::Result> {
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

        self.device.queue_submit(self.device.graphics_queue, &command_buffer, PipelineStage::empty(), &Semaphore::None, &Semaphore::None, &Fence::None).unwrap();
        unsafe { self.device.queue_wait_idle(self.device.graphics_queue.queue) }
    }

    //TODO join functions with a trait
    pub fn copy_to_image(&self, image: vk::Image, width: u32, height: u32, cmd_pool: &CommandPool) -> Result<(), vk::Result> {
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

        self.device.queue_submit(self.device.graphics_queue, &cmd_buffer, PipelineStage::empty(), &Semaphore::None, &Semaphore::None, &Fence::None).unwrap();
        unsafe { self.device.queue_wait_idle(self.device.graphics_queue.queue) }
    }
}
