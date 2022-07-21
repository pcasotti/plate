use std::sync::Arc;

use ash::vk;

use crate::{Buffer, Device, command::*, PipelineStage, sync::*};

pub struct Sampler {
    device: Arc<Device>,
    sampler: vk::Sampler,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

impl Sampler {
    pub fn new(device: &Arc<Device>) -> Result<Self, vk::Result> {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        Ok(Self {
            device: Arc::clone(&device),
            sampler,
        })
    }
}

pub struct Image {
    device: Arc<Device>,
    pub image: vk::Image,
    mem: vk::DeviceMemory,
    view: vk::ImageView,
    pub width: u32,
    pub height: u32,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.mem, None);
        }
    }
}

impl Image {
    pub fn new(device: &Arc<Device>, cmd_pool: &CommandPool, width: u32, height: u32, data: &[u8]) -> Result<Self, vk::Result> {
        let mut staging = Buffer::new(
            device,
            (width * height * 4) as usize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        staging.map()?;
        staging.write(data);
        staging.unmap();

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_SRGB)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = unsafe { device.create_image(&image_info, None)? };

        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_properties = unsafe { device.instance.get_physical_device_memory_properties(device.physical_device) };
        let mem_type_index = mem_properties
            .memory_types
            .iter()
            .enumerate()
            .find(|(i, ty)| {
                mem_requirements.memory_type_bits & (1 << i) > 0
                    && ty.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .unwrap()
            .0;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index as u32);

        let mem = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_image_memory(image, mem, 0)? };

        transition_layout(device, image, cmd_pool, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        staging.copy_to_image(image, width, height, cmd_pool)?;
        transition_layout(device, image, cmd_pool, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;

        let components = vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        };

        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .components(components)
            .subresource_range(subresource_range);

        let view = unsafe { device.create_image_view(&view_info, None)? };

        Ok(Self {
            device: Arc::clone(&device),
            image,
            mem,
            view,
            width,
            height,
        })
    }

    pub fn descriptor_info(&self, sampler: &Sampler) -> vk::DescriptorImageInfo {
        *vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(self.view)
            .sampler(sampler.sampler)
    }
}

fn transition_layout(device: &Arc<Device>, image: vk::Image, cmd_pool: &CommandPool, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) -> Result<(), vk::Result> {
    let (src_access, src_stage) = match old_layout {
        vk::ImageLayout::UNDEFINED => (vk::AccessFlags::empty(), vk::PipelineStageFlags::TOP_OF_PIPE),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::TRANSFER),
        _ => unimplemented!(),
    };

    let (dst_access, dst_stage) = match new_layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::TRANSFER),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::AccessFlags::SHADER_READ, vk::PipelineStageFlags::FRAGMENT_SHADER),
        _ => unimplemented!(),
    };

    let cmd_buffer = cmd_pool.alloc_cmd_buffer(CommandBufferLevel::PRIMARY)?;
    cmd_buffer.record(CommandBufferUsageFlags::ONE_TIME_SUBMIT, || {
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
                level_count: 1,
            })
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);

        unsafe { device.cmd_pipeline_barrier(
            *cmd_buffer,
            src_stage, dst_stage, vk::DependencyFlags::empty(),
            &[], &[], &[*barrier]
        ) };
    })?;

    device.queue_submit(device.graphics_queue, &cmd_buffer, PipelineStage::empty(), &Semaphore::None, &Semaphore::None, &Fence::None).unwrap();
    unsafe { device.queue_wait_idle(device.graphics_queue.queue) }
}
