use std::{ops, sync::Arc};

use ash::{extensions::khr, vk};

use crate::{Instance, InstanceParameters, CommandBuffer, Semaphore, Fence, Error, MemoryPropertyFlags};

/// Errors from the device module.
#[derive(thiserror::Error, Debug)]
pub enum DeviceError {
    /// None of the available queue families are suitable.
    #[error("None of the available queue families are suitable")]
    QueueNotFound,
    /// The physical device memory properties does not support the requested flags.
    #[error("{0:?}")]
    MemoryTypeNotFound(MemoryPropertyFlags),
    /// None of the available physical devices match the requested options.
    #[error("No suitable device was found")]
    NoDeviceSuitable,
}

#[derive(Clone, Copy)]
pub(crate) struct Queue {
    pub queue: vk::Queue,
    pub family: u32,
}

/// The Device is responsible for most of the vulkan operations.
pub struct Device {
    device: ash::Device,
    pub(crate) instance: Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) queue: Queue,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.destroy_device(None);
        }
    }
}

impl ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Device {
    /// Creates a Device and returns an [`Arc`] pointing to it.
    ///
    /// The Device takes ownership of an [`Instance`] and a [`Surface`]. The [`DeviceParameters`]
    /// are used to mach a physical device with the required features. If no device is from the
    /// preferred [`DeviceType`] will default to whatever is available.
    ///
    /// # Exmaple
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        params: &DeviceParameters,
        instance_params: &InstanceParameters,
        window: Option<&winit::window::Window>
    ) -> Result<Arc<Self>, Error> {
        let instance = Instance::new(window, instance_params)?;

        let devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = pick_device(&devices, &instance, params)?;

        let queue_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut graphics_family = None;
        queue_properties
            .iter()
            .enumerate()
            .map(|(i, properties)| {
                if graphics_family.is_none()
                    && properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    graphics_family = Some(i);
                }
                Ok(())
            })
            .collect::<Result<_, Error>>()?;

        let queue_family = graphics_family.ok_or(DeviceError::QueueNotFound)? as u32;

        let queue_infos = [*vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&[0.0])];

        let features = vk::PhysicalDeviceFeatures::builder();
        let extensions = [khr::Swapchain::name().as_ptr()];

        let mut draw_params = vk::PhysicalDeviceShaderDrawParametersFeatures::builder()
            .shader_draw_parameters(true);
        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_features(&features)
            .enabled_extension_names(&extensions)
            .push_next(&mut draw_params);

        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

        let queue = Queue {
            queue: unsafe { device.get_device_queue(queue_family, 0) },
            family: queue_family,
        };

        Ok(Arc::new(Self {
            device,
            instance,
            physical_device,
            queue,
        }))
    }

    /// Submit a [`CommandBuffer`] to be executed.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let cmd_pool = plate::CommandPool::new(&device)?;
    /// # let cmd_buffer = cmd_pool.alloc_cmd_buffer(plate::CommandBufferLevel::PRIMARY)?;
    /// # let fence = plate::Fence::new(&device, plate::FenceFlags::SIGNALED)?;
    /// # let acquire_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// # let present_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// device.queue_submit(
    ///     &cmd_buffer,
    ///     plate::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
    ///     Some(&acquire_sem),
    ///     Some(&present_sem),
    ///     Some(&fence),
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn queue_submit(
        &self,
        command_buffer: &CommandBuffer,
        wait_stage: PipelineStage,
        wait_semaphore: Option<&Semaphore>,
        signal_semaphore: Option<&Semaphore>,
        fence: Option<&Fence>
    ) -> Result<(), Error> {
        let wait_semaphores = match wait_semaphore {
            Some(s) => vec![**s],
            None => vec![],
        };
        let signal_semaphores = match signal_semaphore {
            Some(s) => vec![**s],
            None => vec![],
        };
        let fence = match fence {
            Some(f) => **f,
            None => vk::Fence::null(),
        };
        let wait_stages = match wait_stage {
            vk::PipelineStageFlags::NONE => vec![],
            _ => vec![wait_stage],
        };
        let command_buffers = [**command_buffer];
        let submit_infos = [*vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .command_buffers(&command_buffers)];

        Ok(unsafe { self.device.queue_submit(self.queue.queue, &submit_infos, fence)? })
    }

    /// Wait for all device queues to be executed.
    ///
    /// #Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// device.wait_idle()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn wait_idle(&self) -> Result<(), Error> {
        Ok(unsafe { self.device.device_wait_idle()? })
    }

    pub(crate) fn memory_type_index(&self, mem_requirements: vk::MemoryRequirements, memory_properties: MemoryPropertyFlags) -> Result<usize, Error> {
        let mem_properties = unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) };
        mem_properties
            .memory_types
            .iter()
            .enumerate()
            .find(|(i, ty)| {
                mem_requirements.memory_type_bits & (1 << i) > 0
                    && ty.property_flags.contains(memory_properties)
            })
            .map(|(i, _)| i)
            .ok_or(DeviceError::MemoryTypeNotFound(memory_properties).into())
    }
}

pub use vk::PhysicalDeviceType as DeviceType;
pub use vk::PipelineStageFlags as PipelineStage;

/// Parameters for physical device selection.
pub struct DeviceParameters {
    /// Will prefer devices of this type.
    pub preferred_type: DeviceType,
    /// What features the device should support.
    pub features: DeviceFeatures,
}

impl Default for DeviceParameters {
    fn default() -> Self {
        Self {
            preferred_type: DeviceType::DISCRETE_GPU,
            features: DeviceFeatures::empty(),
        }
    }
}

fn pick_device(
    devices: &Vec<vk::PhysicalDevice>,
    instance: &Instance,
    params: &DeviceParameters,
) -> Result<vk::PhysicalDevice, Error> {
    let devices = devices
        .iter()
        .filter(|device| {
            let features = unsafe { instance.get_physical_device_features(**device) };
            features.contains(&params.features)
        })
        .collect::<Vec<_>>();
    if devices.is_empty() {
        return Err(DeviceError::NoDeviceSuitable.into());
    };

    let preferred_device = devices
        .iter()
        .find(|device| {
            let properties = unsafe { instance.get_physical_device_properties(***device) };
            properties.device_type == params.preferred_type
        })
        .unwrap_or(&devices[0]);

    Ok(**preferred_device)
}

trait Contains<T> {
    fn contains(&self, _: &T) -> bool;
}

impl Contains<DeviceFeatures> for vk::PhysicalDeviceFeatures {
    fn contains(&self, other: &DeviceFeatures) -> bool {
        (self.robust_buffer_access == 1) >= other.contains(DeviceFeatures::ROBUST_BUFFER_ACCESS)
            && (self.full_draw_index_uint32 == 1)
                >= other.contains(DeviceFeatures::FULL_DRAW_INDEX_UINT32)
            && (self.image_cube_array == 1) >= other.contains(DeviceFeatures::IMAGE_CUBE_ARRAY)
            && (self.independent_blend == 1) >= other.contains(DeviceFeatures::INDEPENDENT_BLEND)
            && (self.geometry_shader == 1) >= other.contains(DeviceFeatures::GEOMETRY_SHADER)
            && (self.tessellation_shader == 1)
                >= other.contains(DeviceFeatures::TESSELLATION_SHADER)
            && (self.sample_rate_shading == 1)
                >= other.contains(DeviceFeatures::SAMPLE_RATE_SHADING)
            && (self.dual_src_blend == 1) >= other.contains(DeviceFeatures::DUAL_SRC_BLEND)
            && (self.logic_op == 1) >= other.contains(DeviceFeatures::LOGIC_OP)
            && (self.multi_draw_indirect == 1)
                >= other.contains(DeviceFeatures::MULTI_DRAW_INDIRECT)
            && (self.draw_indirect_first_instance == 1)
                >= other.contains(DeviceFeatures::DRAW_INDIRECT_FIRST_INSTANCE)
            && (self.depth_clamp == 1) >= other.contains(DeviceFeatures::DEPTH_CLAMP)
            && (self.depth_bias_clamp == 1) >= other.contains(DeviceFeatures::DEPTH_BIAS_CLAMP)
            && (self.fill_mode_non_solid == 1)
                >= other.contains(DeviceFeatures::FILL_MODE_NON_SOLID)
            && (self.depth_bounds == 1) >= other.contains(DeviceFeatures::DEPTH_BOUNDS)
            && (self.wide_lines == 1) >= other.contains(DeviceFeatures::WIDE_LINES)
            && (self.large_points == 1) >= other.contains(DeviceFeatures::LARGE_POINTS)
            && (self.alpha_to_one == 1) >= other.contains(DeviceFeatures::ALPHA_TO_ONE)
            && (self.multi_viewport == 1) >= other.contains(DeviceFeatures::MULTI_VIEWPORT)
            && (self.sampler_anisotropy == 1) >= other.contains(DeviceFeatures::SAMPLER_ANISOTROPY)
            && (self.texture_compression_etc2 == 1)
                >= other.contains(DeviceFeatures::TEXTURE_COMPRESSION_ETC2)
            && (self.texture_compression_astc_ldr == 1)
                >= other.contains(DeviceFeatures::TEXTURE_COMPRESSION_ASTC_LDR)
            && (self.texture_compression_bc == 1)
                >= other.contains(DeviceFeatures::TEXTURE_COMPRESSION_BC)
            && (self.occlusion_query_precise == 1)
                >= other.contains(DeviceFeatures::OCCLUSION_QUERY_PRECISE)
            && (self.pipeline_statistics_query == 1)
                >= other.contains(DeviceFeatures::PIPELINE_STATISTICS_QUERY)
            && (self.vertex_pipeline_stores_and_atomics == 1)
                >= other.contains(DeviceFeatures::VERTEX_PIPELINE_STORES_AND_ATOMICS)
            && (self.fragment_stores_and_atomics == 1)
                >= other.contains(DeviceFeatures::FRAGMENT_STORES_AND_ATOMICS)
            && (self.shader_tessellation_and_geometry_point_size == 1)
                >= other.contains(DeviceFeatures::SHADER_TESSELLATION_AND_GEOMETRY_POINT_SIZE)
            && (self.shader_image_gather_extended == 1)
                >= other.contains(DeviceFeatures::SHADER_IMAGE_GATHER_EXTENDED)
            && (self.shader_storage_image_extended_formats == 1)
                >= other.contains(DeviceFeatures::SHADER_STORAGE_IMAGE_EXTENDED_FORMATS)
            && (self.shader_storage_image_multisample == 1)
                >= other.contains(DeviceFeatures::SHADER_STORAGE_IMAGE_MULTISAMPLE)
            && (self.shader_storage_image_read_without_format == 1)
                >= other.contains(DeviceFeatures::SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT)
            && (self.shader_storage_image_write_without_format == 1)
                >= other.contains(DeviceFeatures::SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT)
            && (self.shader_uniform_buffer_array_dynamic_indexing == 1)
                >= other.contains(DeviceFeatures::SHADER_UNIFORM_BUFFER_ARRAY_DYNAMIC_INDEXING)
            && (self.shader_sampled_image_array_dynamic_indexing == 1)
                >= other.contains(DeviceFeatures::SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING)
            && (self.shader_storage_buffer_array_dynamic_indexing == 1)
                >= other.contains(DeviceFeatures::SHADER_STORAGE_BUFFER_ARRAY_DYNAMIC_INDEXING)
            && (self.shader_storage_image_array_dynamic_indexing == 1)
                >= other.contains(DeviceFeatures::SHADER_STORAGE_IMAGE_ARRAY_DYNAMIC_INDEXING)
            && (self.shader_clip_distance == 1)
                >= other.contains(DeviceFeatures::SHADER_CLIP_DISTANCE)
            && (self.shader_cull_distance == 1)
                >= other.contains(DeviceFeatures::SHADER_CULL_DISTANCE)
            && (self.shader_float64 == 1) >= other.contains(DeviceFeatures::SHADER_FLOAT64)
            && (self.shader_int64 == 1) >= other.contains(DeviceFeatures::SHADER_INT64)
            && (self.shader_int16 == 1) >= other.contains(DeviceFeatures::SHADER_INT16)
            && (self.shader_resource_residency == 1)
                >= other.contains(DeviceFeatures::SHADER_RESOURCE_RESIDENCY)
            && (self.shader_resource_min_lod == 1)
                >= other.contains(DeviceFeatures::SHADER_RESOURCE_MIN_LOD)
            && (self.sparse_binding == 1) >= other.contains(DeviceFeatures::SPARSE_BINDING)
            && (self.sparse_residency_buffer == 1)
                >= other.contains(DeviceFeatures::SPARSE_RESIDENCY_BUFFER)
            && (self.sparse_residency_image2_d == 1)
                >= other.contains(DeviceFeatures::SPARSE_RESIDENCY_IMAGE2_D)
            && (self.sparse_residency_image3_d == 1)
                >= other.contains(DeviceFeatures::SPARSE_RESIDENCY_IMAGE3_D)
            && (self.sparse_residency2_samples == 1)
                >= other.contains(DeviceFeatures::SPARSE_RESIDENCY2_SAMPLES)
            && (self.sparse_residency4_samples == 1)
                >= other.contains(DeviceFeatures::SPARSE_RESIDENCY4_SAMPLES)
            && (self.sparse_residency8_samples == 1)
                >= other.contains(DeviceFeatures::SPARSE_RESIDENCY8_SAMPLES)
            && (self.sparse_residency16_samples == 1)
                >= other.contains(DeviceFeatures::SPARSE_RESIDENCY16_SAMPLES)
            && (self.sparse_residency_aliased == 1)
                >= other.contains(DeviceFeatures::SPARSE_RESIDENCY_ALIASED)
            && (self.variable_multisample_rate == 1)
                >= other.contains(DeviceFeatures::VARIABLE_MULTISAMPLE_RATE)
            && (self.inherited_queries == 1) >= other.contains(DeviceFeatures::INHERITED_QUERIES)
    }
}

bitflags::bitflags! {
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceFeatures.html>
    pub struct DeviceFeatures: u64 {
        const ROBUST_BUFFER_ACCESS = 1 << 0;
        const FULL_DRAW_INDEX_UINT32 = 1 << 1;
        const IMAGE_CUBE_ARRAY = 1 << 2;
        const INDEPENDENT_BLEND = 1 << 3;
        const GEOMETRY_SHADER = 1 << 4;
        const TESSELLATION_SHADER = 1 << 5;
        const SAMPLE_RATE_SHADING = 1 << 6;
        const DUAL_SRC_BLEND = 1 << 7;
        const LOGIC_OP = 1 << 8;
        const MULTI_DRAW_INDIRECT = 1 << 9;
        const DRAW_INDIRECT_FIRST_INSTANCE = 1 << 10;
        const DEPTH_CLAMP = 1 << 11;
        const DEPTH_BIAS_CLAMP = 1 << 12;
        const FILL_MODE_NON_SOLID = 1 << 13;
        const DEPTH_BOUNDS = 1 << 14;
        const WIDE_LINES = 1 << 15;
        const LARGE_POINTS = 1 << 16;
        const ALPHA_TO_ONE = 1 << 17;
        const MULTI_VIEWPORT = 1 << 18;
        const SAMPLER_ANISOTROPY = 1 << 19;
        const TEXTURE_COMPRESSION_ETC2 = 1 << 20;
        const TEXTURE_COMPRESSION_ASTC_LDR = 1 << 21;
        const TEXTURE_COMPRESSION_BC = 1 << 22;
        const OCCLUSION_QUERY_PRECISE = 1 << 23;
        const PIPELINE_STATISTICS_QUERY = 1 << 24;
        const VERTEX_PIPELINE_STORES_AND_ATOMICS = 1 << 25;
        const FRAGMENT_STORES_AND_ATOMICS = 1 << 26;
        const SHADER_TESSELLATION_AND_GEOMETRY_POINT_SIZE = 1 << 27;
        const SHADER_IMAGE_GATHER_EXTENDED = 1 << 28;
        const SHADER_STORAGE_IMAGE_EXTENDED_FORMATS = 1 << 29;
        const SHADER_STORAGE_IMAGE_MULTISAMPLE = 1 << 30;
        const SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT = 1 << 31;
        const SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT = 1 << 32;
        const SHADER_UNIFORM_BUFFER_ARRAY_DYNAMIC_INDEXING = 1 << 33;
        const SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING = 1 << 34;
        const SHADER_STORAGE_BUFFER_ARRAY_DYNAMIC_INDEXING = 1 << 35;
        const SHADER_STORAGE_IMAGE_ARRAY_DYNAMIC_INDEXING = 1 << 36;
        const SHADER_CLIP_DISTANCE = 1 << 37;
        const SHADER_CULL_DISTANCE = 1 << 38;
        const SHADER_FLOAT64 = 1 << 39;
        const SHADER_INT64 = 1 << 40;
        const SHADER_INT16 = 1 << 41;
        const SHADER_RESOURCE_RESIDENCY = 1 << 42;
        const SHADER_RESOURCE_MIN_LOD = 1 << 43;
        const SPARSE_BINDING = 1 << 44;
        const SPARSE_RESIDENCY_BUFFER = 1 << 45;
        const SPARSE_RESIDENCY_IMAGE2_D = 1 << 46;
        const SPARSE_RESIDENCY_IMAGE3_D = 1 << 47;
        const SPARSE_RESIDENCY2_SAMPLES = 1 << 48;
        const SPARSE_RESIDENCY4_SAMPLES = 1 << 49;
        const SPARSE_RESIDENCY8_SAMPLES = 1 << 50;
        const SPARSE_RESIDENCY16_SAMPLES = 1 << 51;
        const SPARSE_RESIDENCY_ALIASED = 1 << 52;
        const VARIABLE_MULTISAMPLE_RATE = 1 << 53;
        const INHERITED_QUERIES = 1 << 54;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_contains() {
        let features_a = vk::PhysicalDeviceFeatures::builder().robust_buffer_access(true);
        let features_b = DeviceFeatures::ROBUST_BUFFER_ACCESS;
        let features_c = DeviceFeatures::empty();
        let features_d =
            DeviceFeatures::ROBUST_BUFFER_ACCESS | DeviceFeatures::FULL_DRAW_INDEX_UINT32;
        let features_e = DeviceFeatures::FULL_DRAW_INDEX_UINT32;
        assert!(features_a.contains(&features_b));
        assert!(features_a.contains(&features_c));
        assert!(!features_a.contains(&features_d));
        assert!(!features_a.contains(&features_e));

        let features_a = vk::PhysicalDeviceFeatures::builder();
        assert!(!features_a.contains(&features_b));
        assert!(features_a.contains(&features_c));
        assert!(!features_a.contains(&features_d));
        assert!(!features_a.contains(&features_e));
    }
}
