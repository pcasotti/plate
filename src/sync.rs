use std::sync::Arc;

use ash::vk;

use crate::Device;

pub use vk::FenceCreateFlags as FenceFlags;
pub use vk::SemaphoreCreateFlags as SemaphoreFlags;

static NULL_FENCE: vk::Fence = vk::Fence::null();

pub enum Fence {
    Fence {
        device: Arc<Device>,
        fence: vk::Fence,
    },
    None,
}

impl Drop for Fence {
    fn drop(&mut self) {
        if let Self::Fence { device, fence } = self {
            unsafe { device.destroy_fence(*fence, None) }
        }
    }
}

impl std::ops::Deref for Fence {
    type Target = vk::Fence;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Fence { device: _, fence} => &fence,
            Self::None => &NULL_FENCE,
        }
    }
}

impl Fence {
    pub fn new(device: &Arc<Device>, flags: FenceFlags) -> Result<Self, vk::Result> {
        let info = vk::FenceCreateInfo::builder().flags(flags);
        let fence = unsafe { device.create_fence(&info, None)? };

        Ok(Self::Fence {
            device: Arc::clone(&device),
            fence,
        })
    }

    pub fn wait(&self) -> Result<(), vk::Result> {
        if let Self::Fence { device, fence } = self {
            return unsafe { device.wait_for_fences(&[*fence], true, u64::MAX) }
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<(), vk::Result> {
        if let Self::Fence { device, fence } = self {
            return unsafe { device.reset_fences(&[*fence]) }
        }
        Ok(())
    }
}

pub enum Semaphore {
    Semaphore {
        device: Arc<Device>,
        semaphore: vk::Semaphore,
    },
    None,
}

static NULL_SEMAPHORE: vk::Semaphore = vk::Semaphore::null();

impl Drop for Semaphore {
    fn drop(&mut self) {
        if let Self::Semaphore { device, semaphore } = self {
            unsafe { device.destroy_semaphore(*semaphore, None) }
        }
    }
}

impl std::ops::Deref for Semaphore {
    type Target = vk::Semaphore;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Semaphore { device: _, semaphore} => &semaphore,
            Self::None => &NULL_SEMAPHORE,
        }
    }
}

impl Semaphore {
    pub fn new(device: &Arc<Device>, flags: SemaphoreFlags) -> Result<Self, vk::Result> {
        let info = vk::SemaphoreCreateInfo::builder().flags(flags);
        let semaphore = unsafe { device.create_semaphore(&info, None)? };

        Ok(Self::Semaphore {
            device: Arc::clone(&device),
            semaphore,
        })
    }
}
