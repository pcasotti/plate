use std::sync::Arc;

use ash::vk;

use crate::{Buffer, Device, CommandBuffer, image::*};

pub use vk::DescriptorType;
pub use vk::ShaderStageFlags as ShaderStage;

pub struct PoolSize {
    pub ty: DescriptorType,
    pub count: u32,
}

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
    pub fn add_size(&mut self, ty: DescriptorType, count: u32) -> &mut Self {
        self.sizes.push(PoolSize { ty, count });
        self
    }

    pub fn max_sets(&mut self, max_sets: u32) -> &mut Self {
        self.max_sets = Some(max_sets);
        self
    }

    pub fn build(&self, device: &Arc<Device>) -> Result<DescriptorPool, vk::Result> {
        let max_sets = self.max_sets.unwrap_or(self.sizes.iter().map(|size| size.count).sum());
        DescriptorPool::new(device, &self.sizes, max_sets)
    }
}

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
    pub fn buider() -> DescriptorPoolBuilder {
        DescriptorPoolBuilder::default()
    }

    pub fn new(
        device: &Arc<Device>,
        sizes: &[PoolSize],
        max_sets: u32,
    ) -> Result<Self, vk::Result> {
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
            pool
        })
    }
}

pub struct LayoutBinding {
    pub binding: u32,
    pub ty: DescriptorType,
    pub stage: ShaderStage,
    pub count: u32,
}

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    pub layout: vk::DescriptorSetLayout,
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

impl DescriptorSetLayout {
    pub fn new(device: &Arc<Device>, bindings: &[LayoutBinding]) -> Result<Self, vk::Result> {
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
            layout
        })
    }
}

pub enum DescriptorInfo {
    Buffer(vk::DescriptorBufferInfo),
    Image(vk::DescriptorImageInfo),
}

struct WriteDescriptor {
    binding: u32,
    ty: DescriptorType,
    descriptor_info: DescriptorInfo,
}

pub struct DescriptorAllocator {
    device: Arc<Device>,
    writes: Vec<WriteDescriptor>,
}

impl DescriptorAllocator {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            device: Arc::clone(&device),
            writes: vec![],
        }
    }

    pub fn add_buffer_binding<T>(&mut self, binding: u32, ty: DescriptorType, buffer: &Buffer<T>) -> &mut Self {
        let descriptor_info = DescriptorInfo::Buffer(buffer.descriptor_info());
        let write = WriteDescriptor {
            binding,
            ty,
            descriptor_info,
        };
        self.writes.push(write);
        self
    }

    pub fn add_image_binding(&mut self, binding: u32, ty: DescriptorType, image: &Image, sampler: &Sampler) -> &mut Self {
        let descriptor_info = DescriptorInfo::Image(image.descriptor_info(sampler));
        let write = WriteDescriptor {
            binding,
            ty,
            descriptor_info,
        };
        self.writes.push(write);
        self
    }

    pub fn allocate(
        &mut self,
        layout: &DescriptorSetLayout,
        pool: &DescriptorPool,
    ) -> Result<DescriptorSet, vk::Result> {
        let layouts = [layout.layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool.pool)
            .set_layouts(&layouts);

        let set = unsafe { self.device.allocate_descriptor_sets(&alloc_info)?[0] };
        let writes = self
            .writes
            .iter_mut()
            .map(|write| {
                match write.descriptor_info {
                    DescriptorInfo::Buffer(info) => {
                        let buffer_infos = [info];
                        *vk::WriteDescriptorSet::builder()
                            .dst_set(set)
                            .dst_binding(write.binding)
                            .descriptor_type(write.ty)
                            .dst_array_element(0)
                            .buffer_info(&buffer_infos)
                    },
                    DescriptorInfo::Image(info) => {
                        let image_infos = [info];
                        *vk::WriteDescriptorSet::builder()
                            .dst_set(set)
                            .dst_binding(write.binding)
                            .descriptor_type(write.ty)
                            .dst_array_element(0)
                            .image_info(&image_infos)
                    },
                }
            })
            .collect::<Vec<_>>();

        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        Ok(DescriptorSet {
            device: Arc::clone(&self.device),
            set,
        })
    }
}

pub struct DescriptorSet {
    device: Arc<Device>,
    set: vk::DescriptorSet,
}

impl DescriptorSet {
    pub fn bind(&self, cmd_buffer: &CommandBuffer, layout: vk::PipelineLayout) {
        unsafe { 
            self.device.cmd_bind_descriptor_sets(
                **cmd_buffer,
                ash::vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[self.set],
                &[],
            )
        };
    }
}
