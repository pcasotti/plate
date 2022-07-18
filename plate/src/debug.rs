use std::ffi;

use ash::{extensions::ext, vk};

pub struct Debugger {
    debug_utils: ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl Drop for Debugger {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
        }
    }
}

impl Debugger {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self, vk::Result> {
        let debug_utils = ext::DebugUtils::new(entry, instance);

        let debug_messenger_info = debug_messenger_info();

        let debug_messenger =
            unsafe { debug_utils.create_debug_utils_messenger(&debug_messenger_info, None)? };

        Ok(Self {
            debug_utils,
            debug_messenger,
        })
    }
}

pub fn debug_messenger_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    *vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
}

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut ffi::c_void,
) -> vk::Bool32 {
    println!(
        "[Validation Layer][{:?}][{:?}] {:?}",
        message_severity,
        message_type,
        ffi::CStr::from_ptr((*p_callback_data).p_message)
    );

    vk::FALSE
}
