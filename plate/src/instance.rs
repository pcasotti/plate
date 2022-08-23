use ash::vk;
use std::ffi;

use crate::{debug, Error, Debugger};

/// Errors from the instance module.
#[derive(thiserror::Error, Debug)]
pub enum InstanceError {
    /// Error trying to create a C string because of a nul byte.
    #[error("Error creating C string: {0}")]
    NulError(#[from] ffi::NulError),
}

/// Version of the Vulkan API.
#[derive(Clone, Copy)]
pub enum ApiVersion {
    /// API version 1.1.
    Type1_1,
    /// API version 1.2.
    Type1_2,
    /// API version 1.3.
    Type1_3,
}

impl Into<u32> for ApiVersion {
    fn into(self) -> u32 {
        use ApiVersion::*;
        match self {
            Type1_1 => vk::API_VERSION_1_1,
            Type1_2 => vk::API_VERSION_1_2,
            Type1_3 => vk::API_VERSION_1_3,
        }
    }
}

/// Optional parameters when creating an Instance.
pub struct InstanceParameters {
    /// Application name.
    pub app_name: String,
    /// Application version.
    pub app_version: (u32, u32, u32, u32),
    /// Engine name.
    pub engine_name: String,
    /// Engine version.
    pub engine_version: (u32, u32, u32, u32),
    /// API version to be used.
    pub api_version: ApiVersion,
    /// Aditional vulkan layers to be enabled.
    pub extra_layers: Vec<String>,
    /// Aditional vulkan extensions to be enabled.
    pub extra_extensions: Vec<String>,
    /// Whether to use validation layers or not.
    pub enable_validation_layers: bool,
}

impl Default for InstanceParameters {
    fn default() -> Self {
        Self {
            app_name: "wvk".into(),
            app_version: (0, 1, 0, 0),
            engine_name: "wvk".into(),
            engine_version: (0, 1, 0, 0),
            api_version: ApiVersion::Type1_2,
            extra_layers: vec![],
            extra_extensions: vec![],
            enable_validation_layers: true,
        }
    }
}

pub(crate) struct Instance {
    instance: ash::Instance,
    pub(crate) entry: ash::Entry,
    #[allow(dead_code)]
    debugger: Option<Debugger>,
}

impl std::ops::Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.debugger = None;
            self.destroy_instance(None);
        }
    }
}

impl Instance {
    pub fn new(
        window: Option<&winit::window::Window>,
        params: &InstanceParameters,
    ) -> Result<Self, Error> {
        let entry = ash::Entry::linked();

        let app_name = ffi::CString::new(params.app_name.clone()).map_err(|e| InstanceError::from(e))?;
        let engine_name = ffi::CString::new(params.engine_name.clone()).map_err(|e| InstanceError::from(e))?;
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(
                params.app_version.0,
                params.app_version.1,
                params.app_version.2,
                params.app_version.3,
            ))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(
                params.engine_version.0,
                params.engine_version.1,
                params.engine_version.2,
                params.engine_version.3,
            ))
            .api_version(params.api_version.into());

        let mut layers = if params.enable_validation_layers {
            vec!["VK_LAYER_KHRONOS_validation".into()]
        } else { vec![] };
        params
            .extra_layers
            .iter()
            .for_each(|layer| layers.push(layer.clone()));
        let layers = layers
            .into_iter()
            .map(|layer| ffi::CString::new(layer))
            .collect::<Result<Vec<_>, _>>().map_err(|e| InstanceError::from(e))?;
        let layers = layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<Vec<_>>();

        let mut extensions = match window {
            Some(window) => ash_window::enumerate_required_extensions(window)?.to_vec(),
            None => vec![],
        };
        if params.enable_validation_layers {
            extensions.push(ash::extensions::ext::DebugUtils::name().as_ptr());
        }

        let extra_extensions = params
            .extra_extensions
            .iter()
            .map(|extension| ffi::CString::new(extension.clone()))
            .collect::<Result<Vec<_>, _>>().map_err(|e| InstanceError::from(e))?;
        extra_extensions
            .into_iter()
            .for_each(|extension| extensions.push(extension.as_ptr()));

        let mut debug_messenger_info = debug::debug_messenger_info();

        let instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(extensions.as_slice())
            .enabled_layer_names(layers.as_slice());

        let instance_info = if params.enable_validation_layers {
            instance_info.push_next(&mut debug_messenger_info)
        } else { instance_info };

        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let debugger = if params.enable_validation_layers {
            Some(Debugger::new(&entry, &instance)?)
        } else { None };

        Ok(Self {
            instance,
            entry,
            debugger,
        })
    }
}
