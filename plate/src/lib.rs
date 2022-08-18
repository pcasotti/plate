//! # plate
//!
//! Rust library for writing simpler Vulkan code
//!
//! ## Instalation
//!
//! Add the library to your Cargo.toml file:
//! ```toml
//! [dependencies]
//! plate = "0.3.0"
//! ```

pub mod buffer;
pub use buffer::*;
pub(crate) mod debug;
pub(crate) use debug::*;
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
pub mod rendering;
pub use rendering::*;

pub use ash::vk;

pub use ash::vk::Format;
pub use ash::vk::MemoryPropertyFlags;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("{0}")]
    VulkanError(#[from] ash::vk::Result),
    #[error("{0}")]
    DeviceError(#[from] DeviceError),
    #[error("{0}")]
    SwapchainError(#[from] SwapchainError),
    #[error("{0}")]
    InstanceError(#[from] InstanceError),
    #[error("{0}")]
    DescriptorError(#[from] DescriptorError),
}

#[cfg(feature = "macros")]
pub use plate_macros;
pub use memoffset;
