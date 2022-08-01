use std::sync::Arc;

use ash::vk;

use crate::{Buffer, Device, command::*, PipelineStage, sync::*, Format, Error, MemoryPropertyFlags};

pub use vk::Filter as Filter;
pub use vk::SamplerAddressMode as SamplerAddressMode;
pub use vk::ImageUsageFlags as ImageUsageFlags;
pub use vk::ImageAspectFlags as ImageAspectFlags;

pub struct SamplerFilter {
    pub min: Filter,
    pub mag: Filter,
}

impl SamplerFilter {
    pub const LINEAR: Self = Self { min: Filter::LINEAR, mag: Filter::LINEAR };
    pub const NEAREST: Self = Self { min: Filter::NEAREST, mag: Filter::LINEAR };
    pub const CUBIC_IMG: Self = Self { min: Filter::CUBIC_IMG, mag: Filter::CUBIC_IMG };
    pub const CUBIC_EXT: Self = Self { min: Filter::CUBIC_EXT, mag: Filter::CUBIC_EXT };
}

pub struct SamplerAddress {
    pub u: SamplerAddressMode,
    pub v: SamplerAddressMode,
    pub w: SamplerAddressMode,
}

impl SamplerAddress {
    pub const REPEAT: Self = Self::all(SamplerAddressMode::REPEAT);
    pub const MIRRORED_REPEAT: Self = Self::all(SamplerAddressMode::MIRRORED_REPEAT);
    pub const CLAMP_TO_EDGE: Self = Self::all(SamplerAddressMode::CLAMP_TO_EDGE);
    pub const CLAMP_TO_BORDER: Self = Self::all(SamplerAddressMode::CLAMP_TO_BORDER);
    pub const MIRROR_CLAMP_TO_EDGE: Self = Self::all(SamplerAddressMode::MIRROR_CLAMP_TO_EDGE);

    const fn all(mode: SamplerAddressMode) -> Self {
        Self { u: mode, v: mode, w: mode }
    }
}

pub struct SamplerParams {
    pub filter: SamplerFilter,
    pub address_mode: SamplerAddress,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            filter: SamplerFilter::LINEAR,
            address_mode: SamplerAddress::REPEAT,
        }
    }
}

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
    pub fn new(device: &Arc<Device>, params: &SamplerParams) -> Result<Self, vk::Result> {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(params.filter.min)
            .min_filter(params.filter.mag)
            .address_mode_u(params.address_mode.u)
            .address_mode_v(params.address_mode.v)
            .address_mode_w(params.address_mode.w)
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
    pub view: vk::ImageView,
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
    pub fn new(device: &Arc<Device>, width: u32, height: u32, format: Format, usage: ImageUsageFlags, image_aspect: ImageAspectFlags) -> Result<Self, Error> {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = unsafe { device.create_image(&image_info, None)? };

        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_type_index = device.memory_type_index(mem_requirements, MemoryPropertyFlags::DEVICE_LOCAL)?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index as u32);

        let mem = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_image_memory(image, mem, 0)? };

        let components = vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        };

        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(image_aspect)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
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

pub struct Texture(Image);

impl std::ops::Deref for Texture {
    type Target = Image;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Texture {
    pub fn new(device: &Arc<Device>, cmd_pool: &CommandPool, width: u32, height: u32, data: &[u8]) -> Result<Self, Error> {
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

        let image = Image::new(
            device,
            width,
            height,
            Format::R8G8B8A8_SRGB,
            ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
            ImageAspectFlags::COLOR,
        )?;

        transition_layout(device, image.image, cmd_pool, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        staging.copy_to_image(image.image, width, height, cmd_pool)?;
        transition_layout(device, image.image, cmd_pool, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;

        Ok(Self(image))
    }
}

fn transition_layout(device: &Arc<Device>, image: vk::Image, cmd_pool: &CommandPool, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) -> Result<(), Error> {
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

    device.queue_submit(device.graphics_queue, &cmd_buffer, PipelineStage::empty(), &Semaphore::None, &Semaphore::None, &Fence::None)?;
    Ok(unsafe { device.queue_wait_idle(device.graphics_queue.queue)? })
}
