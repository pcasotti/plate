use std::sync::Arc;

use ash::{extensions::khr, vk};

use crate::{Device, sync::*, image::*, Format, Error, Surface};

/// Errors from the swapchain module.
#[derive(thiserror::Error, Debug)]
pub enum SwapchainError {
    /// None of the available image formats match the depth requirements.
    #[error("No suitable depth format is available")]
    NoSuitableDepthFormat,
}

/// The Swapchain is responsible for providing images to be rendered to the screen.
pub struct Swapchain {
    device: Arc<Device>,
    #[allow(dead_code)]
    surface: Surface,

    swapchain_loader: khr::Swapchain,
    swapchain: vk::SwapchainKHR,

    extent: vk::Extent2D,

    pub images: Vec<Image>,
    pub surface_format: Format,
    pub depth_format: Format,
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Swapchain {
    /// Creates a Swapchain.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # struct Vertex(f32);
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// let swapchain = plate::Swapchain::new(&device, &window)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        device: &Arc<Device>,
        window: &winit::window::Window,
    ) -> Result<Self, Error> {
        let surface = Surface::new(&device.instance, &window)?;

        let (
            swapchain_loader,
            swapchain,
            extent,
            images,
            surface_format,
            depth_format,
        ) = Self::create_swapchain(device, &surface, window, None)?;

        Ok(Self {
            device: Arc::clone(&device),
            surface,
            swapchain_loader,
            swapchain,
            extent,
            images,
            surface_format,
            depth_format,
        })
    }

    /// Recreates the swapchain.
    ///
    /// Sould be called if the window was resized or the surface format has changed.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # struct Vertex(f32);
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// let mut swapchain = plate::Swapchain::new(&device, &window)?;
    /// swapchain.recreate(&window)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn recreate(&mut self, window: &winit::window::Window) -> Result<(), Error> {
        self.device.wait_idle()?;
        
        let (
            swapchain_loader,
            swapchain,
            extent,
            images,
            surface_format,
            depth_format,
        ) = Self::create_swapchain(&self.device, &self.surface, window, Some(self.swapchain))?;

        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        self.swapchain_loader = swapchain_loader;
        self.swapchain = swapchain;
        self.extent = extent;
        self.images = images;
        self.surface_format = surface_format;
        self.depth_format = depth_format;

        Ok(())
    }

    /// Acquires the next available swapchain image.
    ///
    /// Returns the index of the next available image from the swapchain and whetherthe swapchain
    /// is suboptimal. Will signal the provided semaphore when done.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::Swapchain::new(&device, &window)?;
    /// # let acquire_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// let (image_index, _) = swapchain.next_image(&acquire_sem).unwrap();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn next_image(&self, semaphore: &Semaphore) -> Result<(u32, bool), Error> {
        Ok(unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                **semaphore,
                vk::Fence::null(),
            )?
        })
    }

    /// Present the image at `image_index` to the screen.
    ///
    /// Will wait on wait_semaphore.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::Swapchain::new(&device, &window)?;
    /// # let present_sem = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// let image_index = 0;
    /// swapchain.present(image_index, &present_sem).unwrap();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn present(&self, image_index: u32, wait_semaphore: &Semaphore) -> Result<bool, Error> {
        let swapchains = [self.swapchain];
        let wait_semaphores = [**wait_semaphore];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        Ok(unsafe { self.swapchain_loader.queue_present(self.device.queue.queue, &present_info)? })
    }

    /// Returns the aspect ration of the extent.
    ///
    /// # Examples
    /// 
    /// ```no_run
    /// # struct Vertex(f32);
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::Swapchain::new(&device, &window)?;
    /// let aspect_ratio = swapchain.aspect_ratio();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn aspect_ratio(&self) -> f32 {
        (self.extent.width as f32) / (self.extent.height as f32)
    }

    /// Returns the swapchain extent.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let mut swapchain = plate::Swapchain::new(&device, &window)?;
    /// let (width, height) = swapchain.extent();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    fn create_swapchain(
        device: &Arc<Device>,
        surface: &Surface,
        window: &winit::window::Window,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<(
        khr::Swapchain,
        vk::SwapchainKHR,
        vk::Extent2D,
        Vec<Image>,
        Format,
        Format,
    ), Error> {
        let surface_capabilities = unsafe {
            surface
                .surface_loader
                .get_physical_device_surface_capabilities(
                    device.physical_device,
                    surface.surface,
                )?
        };
        let surface_formats = unsafe {
            surface
                .surface_loader
                .get_physical_device_surface_formats(
                    device.physical_device,
                    surface.surface,
                )?
        };
        let present_modes = unsafe {
            surface
                .surface_loader
                .get_physical_device_surface_present_modes(
                    device.physical_device,
                    surface.surface,
                )?
        };

        let image_format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::R8G8B8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&surface_formats[0]);

        let present_mode = *present_modes
            .iter()
            .find(|mode| **mode == vk::PresentModeKHR::FIFO)
            .unwrap_or(&present_modes[0]);

        let window_extent = window.inner_size();
        let extent = vk::Extent2D {
            width: window_extent.width.clamp(
                surface_capabilities.min_image_extent.width,
                surface_capabilities.max_image_extent.width,
            ),
            height: window_extent.height.clamp(
                surface_capabilities.min_image_extent.height,
                surface_capabilities.max_image_extent.height,
            ),
        };

        let queue_families = [device.queue.family];

        let image_count = if surface_capabilities.max_image_count == 0 {
            surface_capabilities.min_image_count + 1
        } else {
            (surface_capabilities.min_image_count + 1).min(surface_capabilities.max_image_count)
        };

        let mut swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.surface)
            .min_image_count(image_count)
            .image_format(image_format.format)
            .image_color_space(image_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        match old_swapchain {
            Some(swapchain) => swapchain_info = swapchain_info.old_swapchain(swapchain),
            None => (),
        };

        let swapchain_loader = khr::Swapchain::new(&device.instance, &device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None)? };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? }.into_iter()
            .map(|i| Image::from_vk_image(device, i, extent.width, extent.height, image_format.format, ImageAspectFlags::COLOR))
            .collect::<Result<Vec<_>, _>>()?;

        let depth_format = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT
        ].into_iter()
            .find(|format| {
                let props = unsafe { device.instance.get_physical_device_format_properties(device.physical_device, *format) };
                props.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            })
            .ok_or(SwapchainError::NoSuitableDepthFormat)?;

        Ok((
            swapchain_loader,
            swapchain,
            extent,
            images,
            image_format.format,
            depth_format,
        ))
    }
}
