use crate::{c_lib, Result};
use std::marker::PhantomData;

/// A device attached to a [`super::PjRtClient`].
pub struct PjRtDevice<'a> {
    pub(super) device: c_lib::pjrt_device,
    pub(super) marker: PhantomData<&'a super::PjRtClient>,
}

impl PjRtDevice<'_> {
    /// The device unique identifier.
    pub fn id(&self) -> usize {
        (unsafe { c_lib::pjrt_device_id(self.device) }) as usize
    }

    pub fn process_index(&self) -> usize {
        (unsafe { c_lib::pjrt_device_process_index(self.device) }) as usize
    }

    pub fn local_hardware_id(&self) -> usize {
        (unsafe { c_lib::pjrt_device_local_hardware_id(self.device) }) as usize
    }

    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_device_to_string(self.device);
            super::c_ptr_to_string(ptr)
        }
    }

    pub fn kind(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_device_kind(self.device);
            super::c_ptr_to_string(ptr)
        }
    }

    pub fn debug_string(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_device_debug_string(self.device);
            super::c_ptr_to_string(ptr)
        }
    }

    pub fn transfer_to_infeed(&self, src: &super::Literal) -> Result<()> {
        let status = unsafe { c_lib::pjrt_device_transfer_to_infeed(self.device, src.0) };
        super::handle_status(status)?;
        Ok(())
    }

    /// Transfer and return a value for the given shape from the outfeed queue.
    pub fn transfer_from_outfeed(&self, dst: &mut super::Literal) -> Result<()> {
        let status = unsafe { c_lib::pjrt_device_transfer_from_outfeed(self.device, dst.0) };
        super::handle_status(status)?;
        Ok(())
    }
}
