use ash::{extensions::ext, vk};
use std::ffi;

use crate::debug;

#[derive(thiserror::Error, Debug)]
pub enum InstanceError {
    #[error("{0}")]
    VulkanError(#[from] vk::Result),
    #[error("Error creating C string: {0}")]
    NulError(#[from] ffi::NulError),
}

#[derive(Clone, Copy)]
pub enum ApiVersion {
    Type1_0,
    Type1_1,
    Type1_2,
    Type1_3,
}

impl Into<u32> for ApiVersion {
    fn into(self) -> u32 {
        use ApiVersion::*;
        match self {
            Type1_0 => vk::API_VERSION_1_0,
            Type1_1 => vk::API_VERSION_1_1,
            Type1_2 => vk::API_VERSION_1_2,
            Type1_3 => vk::API_VERSION_1_3,
        }
    }
}

pub struct InstanceParameters {
    pub app_name: String,
    pub app_version: (u32, u32, u32, u32),
    pub engine_name: String,
    pub engine_version: (u32, u32, u32, u32),
    pub api_version: ApiVersion,
    pub extra_layers: Vec<String>,
    pub extra_extensions: Vec<String>,
}

impl Default for InstanceParameters {
    fn default() -> Self {
        Self {
            app_name: "wvk".into(),
            app_version: (0, 1, 0, 0),
            engine_name: "wvk".into(),
            engine_version: (0, 1, 0, 0),
            api_version: ApiVersion::Type1_0,
            extra_layers: vec![],
            extra_extensions: vec![],
        }
    }
}

pub struct Instance {
    instance: ash::Instance,
    debug_utils: ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
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
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.destroy_instance(None);
        }
    }
}

impl Instance {
    pub fn new(
        entry: &ash::Entry,
        window: &winit::window::Window,
        params: &InstanceParameters,
    ) -> Result<Self, InstanceError> {
        let app_name = ffi::CString::new(params.app_name.clone())?;
        let engine_name = ffi::CString::new(params.engine_name.clone())?;
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

        let mut layers = vec!["VK_LAYER_KHRONOS_validation".into()];
        params
            .extra_layers
            .iter()
            .for_each(|layer| layers.push(layer.clone()));
        let layers = layers
            .into_iter()
            .map(|layer| ffi::CString::new(layer))
            .collect::<Result<Vec<_>, _>>()?;
        let layers = layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<Vec<_>>();

        let mut extensions = ash_window::enumerate_required_extensions(window)?.to_vec();
        extensions.push(ash::extensions::ext::DebugUtils::name().as_ptr());
        let extra_extensions = params
            .extra_extensions
            .iter()
            .map(|extension| ffi::CString::new(extension.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        extra_extensions
            .into_iter()
            .for_each(|extension| extensions.push(extension.as_ptr()));

        let mut debug_messenger_info = debug::debug_messenger_info();

        let instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(extensions.as_slice())
            .enabled_layer_names(layers.as_slice())
            .push_next(&mut debug_messenger_info);

        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let debug_utils = ext::DebugUtils::new(&entry, &instance);
        let debug_messenger_info = debug::debug_messenger_info();

        let debug_messenger = unsafe { debug_utils.create_debug_utils_messenger(&debug_messenger_info, None)? };

        Ok(Self {
            instance,
            debug_utils,
            debug_messenger,
        })
    }
}
