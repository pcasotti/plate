use std::sync::Arc;

use ash::vk;
use crate::{Buffer, Device, command::*, PipelineStage, Format, Error, MemoryPropertyFlags, ImageLayout};
pub use vk::Filter as Filter;
pub use vk::SamplerAddressMode as SamplerAddressMode;
pub use vk::ImageUsageFlags as ImageUsageFlags;
pub use vk::ImageAspectFlags as ImageAspectFlags;

/// Filter mode for a [`Sampler`].
///
/// Describes how to interpolate texels.
pub struct SamplerFilter {
    /// How to handle magnified texels.
    pub min: Filter,
    /// How to handle minified texels.
    pub mag: Filter,
}

impl SamplerFilter {
    /// Linear filtering.
    pub const LINEAR: Self = Self { min: Filter::LINEAR, mag: Filter::LINEAR };
    /// Nearest filtering.
    pub const NEAREST: Self = Self { min: Filter::NEAREST, mag: Filter::LINEAR };
    /// Cubic filtering.
    pub const CUBIC_EXT: Self = Self { min: Filter::CUBIC_EXT, mag: Filter::CUBIC_EXT };
}

/// How to address coordinates outside the image bounds.
pub struct SamplerAddress {
    /// U coordinates address mode.
    pub u: SamplerAddressMode,
    /// V coordinates address mode.
    pub v: SamplerAddressMode,
    /// W coordinates address mode.
    pub w: SamplerAddressMode,
}

impl SamplerAddress {
    /// Repeat the image in all coordinates.
    pub const REPEAT: Self = Self::all(SamplerAddressMode::REPEAT);
    /// Repeat and mirror the image in all coordinates.
    pub const MIRRORED_REPEAT: Self = Self::all(SamplerAddressMode::MIRRORED_REPEAT);
    /// Repeat the edge color in all coordinates.
    pub const CLAMP_TO_EDGE: Self = Self::all(SamplerAddressMode::CLAMP_TO_EDGE);
    /// Repeat the border color (black) in all coordinates.
    pub const CLAMP_TO_BORDER: Self = Self::all(SamplerAddressMode::CLAMP_TO_BORDER);
    /// Repeat the oposite edge color in all coordinates.
    pub const MIRROR_CLAMP_TO_EDGE: Self = Self::all(SamplerAddressMode::MIRROR_CLAMP_TO_EDGE);

    const fn all(mode: SamplerAddressMode) -> Self {
        Self { u: mode, v: mode, w: mode }
    }
}

/// Optional parameters for [`Sampler creation`].
pub struct SamplerParameters {
    /// Filter mode for the sampler.
    pub filter: SamplerFilter,
    /// Address mode for the sampler.
    pub address_mode: SamplerAddress,
}

impl Default for SamplerParameters {
    fn default() -> Self {
        Self {
            filter: SamplerFilter::LINEAR,
            address_mode: SamplerAddress::REPEAT,
        }
    }
}

/// Describes how to sample a image texture.
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
    /// Creates a Sampler.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// let sampler = plate::Sampler::new(&device, &Default::default())?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, params: &SamplerParameters) -> Result<Self, Error> {
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

/// Represents a 2 dimesional array of data.
pub struct Image {
    device: Arc<Device>,
    image: vk::Image,
    mem: Option<vk::DeviceMemory>,
    pub(crate) view: vk::ImageView,
    /// The format of the image.
    pub format: Format,
    /// The width of the image.
    pub width: u32,
    /// The height of the image.
    pub height: u32,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            if let Some(mem) = self.mem {
                self.device.destroy_image(self.image, None);
                self.device.free_memory(mem, None);
            }
        }
    }
}

impl Image {
    /// Creates a Image.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// # let (width, height) = (0, 0);
    /// let image = plate::Image::new(
    ///     &device,
    ///     width,
    ///     height,
    ///     plate::Format::R8G8B8A8_SRGB,
    ///     plate::ImageLayout::UNDEFINED,
    ///     plate::ImageUsageFlags::TRANSFER_DST | plate::ImageUsageFlags::SAMPLED,
    ///     plate::ImageAspectFlags::COLOR,
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, width: u32, height: u32, format: Format, layout: ImageLayout, usage: ImageUsageFlags, image_aspect: ImageAspectFlags) -> Result<Self, Error> {
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
            .initial_layout(ImageLayout::UNDEFINED)
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

        let cmd_pool = CommandPool::new(device)?;
        if layout != ImageLayout::UNDEFINED {
            transition_layout(device, image, &cmd_pool, vk::ImageLayout::UNDEFINED, layout)?;
        }

        Self::from_vk_image(device, image, Some(mem), width, height, format, image_aspect)
    }

    pub(crate) fn from_vk_image(device: &Arc<Device>, image: vk::Image, mem: Option<vk::DeviceMemory>, width: u32, height: u32, format: Format, image_aspect: ImageAspectFlags) -> Result<Self, Error> {
        let view = Self::image_view(device, image, image_aspect, format)?;

        Ok(Self {
            device: Arc::clone(&device),
            image,
            mem,
            view,
            format,
            width,
            height,
        })
    }

    pub(crate) fn descriptor_info(&self, sampler: &Sampler, layout: ImageLayout) -> vk::DescriptorImageInfo {
        *vk::DescriptorImageInfo::builder()
            .image_layout(layout)
            .image_view(self.view)
            .sampler(sampler.sampler)
    }

    fn image_view(device: &Arc<Device>, image: vk::Image, image_aspect: ImageAspectFlags, format: Format) -> Result<vk::ImageView, Error> {
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

        Ok(unsafe { device.create_image_view(&view_info, None)? })
    }
}

/// Holds a [`Image`] with texture data in it.
pub struct Texture(Image);

impl std::ops::Deref for Texture {
    type Target = Image;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Texture {
    /// Creates a Texture from a &[u8].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// # let (width, height) = (0, 0);
    /// # let data = [0];
    /// let image = plate::Texture::new(&device, &cmd_pool, width, height, &data)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, cmd_pool: &CommandPool, width: u32, height: u32, data: &[u8]) -> Result<Self, Error> {
        let staging = Buffer::new(
            device,
            (width * height * 4) as usize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mut mapped = staging.map()?;
        mapped.write(data);
        let staging = mapped.unmap();

        let image = Image::new(
            device,
            width,
            height,
            Format::R8G8B8A8_SRGB,
            ImageLayout::UNDEFINED,
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
        _ => (vk::AccessFlags::empty(), vk::PipelineStageFlags::TOP_OF_PIPE),
    };

    let (dst_access, dst_stage) = match new_layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::TRANSFER),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::AccessFlags::SHADER_READ, vk::PipelineStageFlags::FRAGMENT_SHADER),
        _ => (vk::AccessFlags::empty(), vk::PipelineStageFlags::BOTTOM_OF_PIPE),
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

    device.queue_submit(&cmd_buffer, PipelineStage::empty(), None, None, None)?;
    Ok(unsafe { device.queue_wait_idle(device.queue.queue)? })
}
