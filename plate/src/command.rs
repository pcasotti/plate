use std::sync::Arc;

use ash::vk;

use crate::{Device, Error};

pub use vk::CommandBufferLevel as CommandBufferLevel;
pub use vk::CommandBufferUsageFlags as CommandBufferUsageFlags;

pub struct CommandPool {
    device: Arc<Device>,
    cmd_pool: vk::CommandPool,
}

impl std::ops::Deref for CommandPool {
    type Target = vk::CommandPool;

    fn deref(&self) -> &Self::Target {
        &self.cmd_pool
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}

impl CommandPool {
    pub fn new(device: &Arc<Device>) -> Result<Self, Error> {
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device.graphics_queue.family);

        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        Ok(Self {
            device: Arc::clone(&device),
            cmd_pool,
        })
    }

    pub fn alloc_cmd_buffers(&self, level: CommandBufferLevel, cmd_buffer_count: u32) -> Result<Vec<CommandBuffer>, Error> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.cmd_pool)
            .level(level)
            .command_buffer_count(cmd_buffer_count);

        let cmd_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        Ok(cmd_buffers.into_iter()
            .map(|cmd_buffer| CommandBuffer { device: Arc::clone(&self.device), cmd_pool: self.cmd_pool, cmd_buffer })
            .collect())
    }

    pub fn alloc_cmd_buffer(&self, level: CommandBufferLevel) -> Result<CommandBuffer, Error> {
        Ok(self.alloc_cmd_buffers(level, 1)?.swap_remove(0))
    }
}

pub struct CommandBuffer {
    device: Arc<Device>,
    cmd_pool: vk::CommandPool,
    cmd_buffer: vk::CommandBuffer,
}

impl std::ops::Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.cmd_buffer
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.free_command_buffers(self.cmd_pool, &[self.cmd_buffer]);
        }
    }
}

impl CommandBuffer {
    pub fn record<F: FnOnce()>(&self, flags: CommandBufferUsageFlags, f: F) -> Result<(), Error> {
        self.begin(flags)?;
        f();
        self.end()?;
        Ok(())
    }

    fn begin(&self, flags: CommandBufferUsageFlags) -> Result<(), Error> {
        let info = vk::CommandBufferBeginInfo::builder().flags(flags);
        unsafe {
            self.device.reset_command_buffer(self.cmd_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(self.cmd_buffer, &info)?;
        }
        Ok(())
    }

    fn end(&self) -> Result<(), Error> {
        unsafe { self.device.end_command_buffer(self.cmd_buffer)? };
        Ok(())
    }

    pub fn draw(&self, vert_count: u32, instance_count: u32, first_vert: u32, first_instance: u32) {
        unsafe { self.device.cmd_draw(self.cmd_buffer, vert_count, instance_count, first_vert, first_instance) }
    }

    pub fn draw_indexed(&self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) {
        unsafe { self.device.cmd_draw_indexed(self.cmd_buffer, index_count, instance_count, first_index, vertex_offset, first_instance) }
    }
}
