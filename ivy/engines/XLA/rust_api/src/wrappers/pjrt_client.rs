//! A device (CPUs, GPUs, TPUs) where computations can be run.
use super::{ArrayElement, Literal, PjRtBuffer, PjRtDevice, PjRtLoadedExecutable, XlaComputation};
use crate::{c_lib, Error, Result};
use std::marker::PhantomData;
use std::rc::Rc;
use pyo3::prelude::*;

pub(super) struct PjRtClientInternal(pub(self) c_lib::pjrt_client);

/// A client represents a device that can be used to run some computations. A computation graph is
/// compiled in a way that is specific to a device before it can be run.
#[derive(Clone)]
#[pyclass(unsendable)]
pub struct PjRtClient(Rc<PjRtClientInternal>);

impl PjRtClient {
    /// A CPU client, this can run computations on multiple CPUs at the same time.
    pub fn cpu() -> Result<Self> {
        let mut ptr: c_lib::pjrt_client = std::ptr::null_mut();
        let status = unsafe { c_lib::pjrt_cpu_client_create(&mut ptr) };
        super::handle_status(status)?;
        Ok(Self(Rc::new(PjRtClientInternal(ptr))))
    }

    /// A GPU client, the memory requirements are limited by the specified `memory_fraction` and
    /// this memory can either be allocated dynamically or pre-allocated depending on
    /// `preallocate`.
    pub fn gpu(memory_fraction: f64, preallocate: bool) -> Result<Self> {
        let mut ptr: c_lib::pjrt_client = std::ptr::null_mut();
        let status =
            unsafe { c_lib::pjrt_gpu_client_create(&mut ptr, memory_fraction, preallocate) };
        super::handle_status(status)?;
        Ok(Self(Rc::new(PjRtClientInternal(ptr))))
    }

    /// A TPU client.
    pub fn tpu(max_inflight_computations: usize) -> Result<Self> {
        let mut ptr: c_lib::pjrt_client = std::ptr::null_mut();
        let status =
            unsafe { c_lib::pjrt_tpu_client_create(&mut ptr, max_inflight_computations as i32) };
        super::handle_status(status)?;
        Ok(Self(Rc::new(PjRtClientInternal(ptr))))
    }

    fn ptr(&self) -> c_lib::pjrt_client {
        self.0 .0
    }

    /// Compile a computation for this device, and return the executable.
    pub fn compile(&self, c: &XlaComputation) -> Result<PjRtLoadedExecutable> {
        let mut exe: c_lib::pjrt_loaded_executable = std::ptr::null_mut();
        let status = unsafe { c_lib::compile(self.ptr(), c.0, &mut exe) };
        super::handle_status(status)?;
        Ok(PjRtLoadedExecutable { exe, client: self.clone() })
    }

    /// The number of devices that this client has detected, e.g. the number of GPUs.
    pub fn device_count(&self) -> usize {
        unsafe { c_lib::pjrt_client_device_count(self.ptr()) as usize }
    }

    /// The number of devices that this client can use.
    pub fn addressable_device_count(&self) -> usize {
        unsafe { c_lib::pjrt_client_addressable_device_count(self.ptr()) as usize }
    }

    /// The name of the platform.
    pub fn platform_name(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_client_platform_name(self.ptr());
            super::c_ptr_to_string(ptr)
        }
    }

    /// The version of the platform.
    pub fn platform_version(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_client_platform_version(self.ptr());
            super::c_ptr_to_string(ptr)
        }
    }

    /// A list of devices attached to this client.
    pub fn devices(&self) -> Vec<PjRtDevice> {
        let device_count = self.device_count();
        let mut device_ptrs = vec![std::ptr::null_mut(); device_count];
        unsafe { c_lib::pjrt_client_devices(self.ptr(), device_ptrs.as_mut_ptr()) };
        device_ptrs.into_iter().map(|device| PjRtDevice { device, marker: PhantomData }).collect()
    }

    /// A list of devices that can be used by this client.
    pub fn addressable_devices(&self) -> Vec<PjRtDevice> {
        let device_count = self.addressable_device_count();
        let mut device_ptrs = vec![std::ptr::null_mut(); device_count];
        unsafe { c_lib::pjrt_client_addressable_devices(self.ptr(), device_ptrs.as_mut_ptr()) };
        device_ptrs.into_iter().map(|device| PjRtDevice { device, marker: PhantomData }).collect()
    }

    /// Transfer some data from the host to a `PjRtBuffer` stored on the target device. If the
    /// device is not specified, the default device is used.
    /// The source data is passed as a slice of the specified primitive type, as well as the
    /// dimensions. The dimensions have to match the number of elements in the source data,
    /// otherwise an error is returned.
    pub fn buffer_from_host_buffer<T: ArrayElement>(
        &self,
        data: &[T],
        dims: &[usize],
        device: Option<&PjRtDevice>,
    ) -> Result<PjRtBuffer> {
        let mut buffer: c_lib::pjrt_buffer = std::ptr::null_mut();
        let element_count: usize = dims.iter().product();
        if element_count != data.len() {
            Err(Error::WrongElementCount { dims: dims.to_vec(), element_count })?
        }
        let device = device.map_or(std::ptr::null_mut(), |d| d.device);
        let dims: Vec<_> = dims.iter().map(|d| *d as i64).collect();
        let status = unsafe {
            c_lib::pjrt_buffer_from_host_buffer(
                self.ptr(),
                device,
                data.as_ptr() as *const libc::c_void,
                T::TY.primitive_type() as i32,
                dims.len() as i32,
                dims.as_ptr(),
                &mut buffer,
            )
        };
        super::handle_status(status)?;
        Ok(PjRtBuffer { buffer, client: self.clone() })
    }

    /// Transfer some data from the host to a `PjRtBuffer` stored on the target device. If the
    /// device is not specified, the default device is used.
    /// The source data is passed as a slice of raw bytes, as well as the dimensions. The
    /// dimensions have to match the number of bytes in the source data, otherwise an error
    /// is returned.
    pub fn buffer_from_host_raw_bytes(
        &self,
        ty: super::ElementType,
        data: &[u8],
        dims: &[usize],
        device: Option<&PjRtDevice>,
    ) -> Result<PjRtBuffer> {
        let mut buffer: c_lib::pjrt_buffer = std::ptr::null_mut();
        let element_count: usize = dims.iter().product();
        let element_size_in_bytes = ty.element_size_in_bytes();
        if element_count * element_size_in_bytes != data.len() {
            Err(Error::WrongElementCount { dims: dims.to_vec(), element_count })?
        }
        let device = device.map_or(std::ptr::null_mut(), |d| d.device);
        let dims: Vec<_> = dims.iter().map(|d| *d as i64).collect();
        let status = unsafe {
            c_lib::pjrt_buffer_from_host_buffer(
                self.ptr(),
                device,
                data.as_ptr() as *const libc::c_void,
                ty as i32,
                dims.len() as i32,
                dims.as_ptr(),
                &mut buffer,
            )
        };
        super::handle_status(status)?;
        Ok(PjRtBuffer { buffer, client: self.clone() })
    }

    /// Transfer some data from the host to a `PjRtBuffer` stored on the target device. If the
    /// device is not specified, the default device is used.
    /// The source data is passed as a literal.
    pub fn buffer_from_host_literal(
        &self,
        device: Option<&PjRtDevice>,
        literal: &Literal,
    ) -> Result<PjRtBuffer> {
        let mut buffer: c_lib::pjrt_buffer = std::ptr::null_mut();
        let device = device.map_or(std::ptr::null_mut(), |d| d.device);
        let status = unsafe {
            c_lib::pjrt_buffer_from_host_literal(self.ptr(), device, literal.0, &mut buffer)
        };
        super::handle_status(status)?;
        Ok(PjRtBuffer { buffer, client: self.clone() })
    }
}

impl Drop for PjRtClientInternal {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_client_free(self.0) }
    }
}
