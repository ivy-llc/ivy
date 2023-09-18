//! A view on a memory slice hosted on a device.
use super::{ArrayElement, ArrayShape, Literal, PjRtDevice, Shape};
use crate::{c_lib, Error, Result};
use pyo3::prelude::*;

/// A buffer represents a view on a memory slice hosted on a device.
#[derive(Clone)]
#[pyclass(unsendable)]
pub struct PjRtBuffer {
    pub(super) buffer: c_lib::pjrt_buffer,
    pub(super) client: super::PjRtClient,
}

impl PjRtBuffer {
    /// The client that owns this buffer.
    pub fn client(&self) -> &super::PjRtClient {
        &self.client
    }

    /// Copy the buffer to a different device.
    pub fn copy_to_device(&self, device: PjRtDevice) -> Result<PjRtBuffer> {
        let mut buffer: c_lib::pjrt_buffer = std::ptr::null_mut();
        let status =
            unsafe { c_lib::pjrt_buffer_copy_to_device(self.buffer, device.device, &mut buffer) };
        super::handle_status(status)?;
        Ok(Self { buffer, client: self.client.clone() })
    }

    /// Copy the buffer back to the host as a literal.
    pub fn to_literal_sync(&self) -> Result<Literal> {
        let mut result: c_lib::literal = std::ptr::null_mut();
        let status = unsafe { c_lib::pjrt_buffer_to_literal_sync(self.buffer, &mut result) };
        super::handle_status(status)?;
        Ok(Literal(result))
    }

    /// Retrieve the shape used by this buffer.
    pub fn on_device_shape(&self) -> Result<Shape> {
        let shape = unsafe { c_lib::pjrt_buffer_on_device_shape(self.buffer) };
        let c_shape = super::shape::CShape::from_ptr(shape);
        c_shape.shape()
    }

    /// Copy the data stored in a buffer to host memory in a blocking way.
    pub fn copy_raw_to_host_sync<T: ArrayElement>(
        &self,
        dst: &mut [T],
        offset: usize,
    ) -> Result<()> {
        let shape = ArrayShape::try_from(&self.on_device_shape()?)?;
        let on_host = T::TY;
        let on_device = shape.primitive_type().element_type()?;
        if on_device != on_host {
            Err(Error::ElementTypeMismatch { on_device, on_host })?
        }
        if offset + dst.len() > shape.element_count() {
            Err(Error::TargetBufferIsTooLarge { offset, shape, buffer_len: dst.len() })?
        }
        let status = unsafe {
            c_lib::pjrt_buffer_copy_raw_to_host_sync(
                self.buffer,
                dst.as_mut_ptr() as *mut libc::c_void,
                offset,
                dst.len() * T::ELEMENT_SIZE_IN_BYTES,
            )
        };
        super::handle_status(status)?;
        Ok(())
    }
}

impl Drop for PjRtBuffer {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_buffer_free(self.buffer) }
    }
}
