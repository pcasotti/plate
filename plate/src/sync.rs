use std::sync::Arc;

use ash::vk;

use crate::{Device, Error};

pub use vk::FenceCreateFlags as FenceFlags;
pub use vk::SemaphoreCreateFlags as SemaphoreFlags;

/// Used to synchronize the host with the GPU.
///
/// Some GPU operations can set the Fence to be signaled or unsignaled, the host can then wait on
/// for these operation to finish accordingly.
pub struct Fence {
    device: Arc<Device>,
    fence: vk::Fence,
}

impl Drop for Fence {
    fn drop(&mut self) { unsafe { self.device.destroy_fence(self.fence, None) }
    }
}

impl std::ops::Deref for Fence {
    type Target = vk::Fence;

    fn deref(&self) -> &Self::Target {
        &self.fence
    }
}

impl Fence {
    /// Creates a Fence.
    ///
    /// Eamples
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// let fence = plate::Fence::new(&device, plate::FenceFlags::SIGNALED)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, flags: FenceFlags) -> Result<Self, Error> {
        let info = vk::FenceCreateInfo::builder().flags(flags);
        let fence = unsafe { device.create_fence(&info, None)? };

        Ok(Self {
            device: Arc::clone(&device),
            fence,
        })
    }

    /// Block until the Fence is signaled.
    ///
    /// Eamples
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let fence = plate::Fence::new(&device, plate::FenceFlags::SIGNALED)?;
    /// fence.wait()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn wait(&self) -> Result<(), Error> {
        Ok(unsafe { self.device.wait_for_fences(&[self.fence], true, u64::MAX)? })
    }

    /// Resets the state of the Fence to unsignaled.
    ///
    /// Eamples
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// # let fence = plate::Fence::new(&device, plate::FenceFlags::SIGNALED)?;
    /// fence.wait()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reset(&self) -> Result<(), Error> {
        Ok(unsafe { self.device.reset_fences(&[self.fence])? })
    }
}

/// Used to synchronize the execution of GPU instructions.
///
/// The GPU executes instructions in parallel, to make sure these instructions run at the correct
/// order, some operations can wait on or set the state of Semaphores to be signaled or unsignaled.
pub struct Semaphore {
    device: Arc<Device>,
    semaphore: vk::Semaphore,
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self.device.destroy_semaphore(self.semaphore, None) }
    }
}

impl std::ops::Deref for Semaphore {
    type Target = vk::Semaphore;

    fn deref(&self) -> &Self::Target {
        &self.semaphore
    }
}

impl Semaphore {
    /// Creates a Semaphore.
    ///
    /// Eamples
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let device = plate::Device::new(&Default::default(), &Default::default(), Some(&window))?;
    /// let semaphore = plate::Semaphore::new(&device, plate::SemaphoreFlags::empty())?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(device: &Arc<Device>, flags: SemaphoreFlags) -> Result<Self, Error> {
        let info = vk::SemaphoreCreateInfo::builder().flags(flags);
        let semaphore = unsafe { device.create_semaphore(&info, None)? };

        Ok(Self {
            device: Arc::clone(&device),
            semaphore,
        })
    }
}
