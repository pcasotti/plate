//TODO: change vk::Result to Error
pub mod buffer;
pub use buffer::*;
pub mod debug;
pub mod descriptor;
pub use descriptor::*;
pub mod device;
pub use device::*;
pub mod instance;
pub use instance::*;
pub mod pipeline;
pub use pipeline::*;
pub mod surface;
pub use surface::Surface;
pub mod swapchain;
pub use swapchain::*;
pub mod command;
pub use command::*;
pub mod sync;
pub use sync::*;
pub mod image;
pub use image::*;

pub use ash::vk::Format as Format;
pub use ash::vk::MemoryPropertyFlags as MemoryPropertyFlags;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("{0}")]
    VulkanError(#[from] ash::vk::Result),
    #[error("{0}")]
    DeviceError(#[from] DeviceError),
    #[error("{0}")]
    SwapchainError(#[from] SwapchainError),
}

#[cfg(feature = "macros")]
pub use plate_macros;
pub use memoffset;
