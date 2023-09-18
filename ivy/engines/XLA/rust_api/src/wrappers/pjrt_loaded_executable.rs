use super::{Literal, PjRtBuffer};
use crate::{c_lib, Result};
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass(unsendable)]
pub struct PjRtLoadedExecutable {
    pub(super) exe: c_lib::pjrt_loaded_executable,
    pub(super) client: super::PjRtClient,
}

impl PjRtLoadedExecutable {
    /// The client that owns this executable.
    pub fn client(&self) -> &super::PjRtClient {
        &self.client
    }

    fn process_execute_outputs(
        &self,
        outputs: *mut *mut c_lib::pjrt_buffer,
    ) -> Vec<Vec<PjRtBuffer>> {
        unsafe {
            let mut vec = vec![];
            loop {
                let outputs = *outputs.add(vec.len());
                if outputs.is_null() {
                    break;
                }
                let mut replica_vec = vec![];
                loop {
                    let buffer = *outputs.add(replica_vec.len());
                    if buffer.is_null() {
                        break;
                    }
                    replica_vec.push(PjRtBuffer { buffer, client: self.client.clone() });
                }
                libc::free(outputs as *mut libc::c_void);
                vec.push(replica_vec);
            }
            libc::free(outputs as *mut libc::c_void);
            vec
        }
    }

    pub fn execute<L: std::borrow::Borrow<Literal>>(
        &self,
        args: &[L],
    ) -> Result<Vec<Vec<PjRtBuffer>>> {
        let mut outputs = std::ptr::null_mut();
        let args: Vec<_> = args.iter().map(|x| x.borrow().0).collect();
        let status =
            unsafe { c_lib::execute(self.exe, args.as_ptr(), args.len() as i32, &mut outputs) };
        super::handle_status(status)?;
        Ok(self.process_execute_outputs(outputs))
    }

    pub fn execute_b<L: std::borrow::Borrow<PjRtBuffer>>(
        &self,
        args: &[L],
    ) -> Result<Vec<Vec<PjRtBuffer>>> {
        let mut outputs = std::ptr::null_mut();
        let args: Vec<_> = args.iter().map(|x| x.borrow().buffer).collect();
        let status =
            unsafe { c_lib::execute_b(self.exe, args.as_ptr(), args.len() as i32, &mut outputs) };
        super::handle_status(status)?;
        Ok(self.process_execute_outputs(outputs))
    }
}

impl Drop for PjRtLoadedExecutable {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_loaded_executable_free(self.exe) }
    }
}
