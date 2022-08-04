use ash::{extensions::khr, vk};

use crate::{Instance, Error};

/// An abstraction of the window to be used by Vulkan.
pub struct Surface {
    pub(crate) surface_loader: khr::Surface,
    pub(crate) surface: vk::SurfaceKHR,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

impl Surface {
    /// Creates a Surface used to present to a window.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let event_loop = winit::event_loop::EventLoop::new();
    /// # let window = winit::window::WindowBuilder::new().build(&event_loop)?;
    /// # let instance = plate::Instance::new(Some(&window), &Default::default())?;
    /// let surface = plate::Surface::new(&instance, &window)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        instance: &Instance,
        window: &winit::window::Window,
    ) -> Result<Self, Error> {
        let surface_loader = khr::Surface::new(&instance.entry, &instance);
        let surface = unsafe { ash_window::create_surface(&instance.entry, &instance, &window, None)? };

        Ok(Self {
            surface_loader,
            surface,
        })
    }
}
