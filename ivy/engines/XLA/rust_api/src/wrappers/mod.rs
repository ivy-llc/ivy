mod literal;
mod pjrt_buffer;
mod pjrt_client;
mod pjrt_device;
mod pjrt_loaded_executable;
mod shape;
mod xla_builder;
mod xla_op;

use crate::c_lib;
use crate::error::{Error, Result};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

pub use literal::Literal;
pub use pjrt_buffer::PjRtBuffer;
pub use pjrt_client::PjRtClient;
pub use pjrt_device::PjRtDevice;
pub use pjrt_loaded_executable::PjRtLoadedExecutable;
pub use shape::{ArrayShape, Shape};
pub use xla_builder::XlaBuilder;
pub use xla_op::XlaOp;
use pyo3::prelude::*;

pub(self) unsafe fn c_ptr_to_string(ptr: *const std::ffi::c_char) -> String {
    let str = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
    libc::free(ptr as *mut libc::c_void);
    str
}

/// The primitive types supported by XLA. `S8` is a signed 1 byte integer,
/// `U32` is an unsigned 4 bytes integer, etc.
#[derive(Clone, Copy, PartialEq, Eq, Debug, FromPrimitive)]
#[pyclass(unsendable)]
pub enum PrimitiveType {
    Invalid = 0,
    Pred = 1,
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    Bf16 = 16,
    F64 = 12,
    C64 = 15,
    C128 = 18,
    Tuple = 13,
    OpaqueType = 14,
    Token = 17,
}

impl PrimitiveType {
    pub fn element_type(self) -> Result<ElementType> {
        match self {
            Self::Pred => Ok(ElementType::Pred),
            Self::S8 => Ok(ElementType::S8),
            Self::S16 => Ok(ElementType::S16),
            Self::S32 => Ok(ElementType::S32),
            Self::S64 => Ok(ElementType::S64),
            Self::U8 => Ok(ElementType::U8),
            Self::U16 => Ok(ElementType::U16),
            Self::U32 => Ok(ElementType::U32),
            Self::U64 => Ok(ElementType::U64),
            Self::F16 => Ok(ElementType::F16),
            Self::F32 => Ok(ElementType::F32),
            Self::Bf16 => Ok(ElementType::Bf16),
            Self::F64 => Ok(ElementType::F64),
            Self::C64 => Ok(ElementType::C64),
            Self::C128 => Ok(ElementType::C128),
            Self::Invalid | Self::Tuple | Self::OpaqueType | Self::Token => {
                Err(Error::NotAnElementType { got: self })
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[pyclass(unsendable)]
pub enum ElementType {
    Pred,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    Bf16,
    F64,
    C64,
    C128,
}

impl ElementType {
    /// The size for this element type in bytes.
    pub fn element_size_in_bytes(&self) -> usize {
        match self {
            Self::Pred => 1,
            Self::S8 => 1,
            Self::S16 => 2,
            Self::S32 => 4,
            Self::S64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::Bf16 => 2,
            Self::F64 => 8,
            Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    pub fn primitive_type(&self) -> PrimitiveType {
        match self {
            Self::Pred => PrimitiveType::Pred,
            Self::S8 => PrimitiveType::S8,
            Self::S16 => PrimitiveType::S16,
            Self::S32 => PrimitiveType::S32,
            Self::S64 => PrimitiveType::S64,
            Self::U8 => PrimitiveType::U8,
            Self::U16 => PrimitiveType::U16,
            Self::U32 => PrimitiveType::U32,
            Self::U64 => PrimitiveType::U64,
            Self::F16 => PrimitiveType::F16,
            Self::F32 => PrimitiveType::F32,
            Self::Bf16 => PrimitiveType::Bf16,
            Self::F64 => PrimitiveType::F64,
            Self::C64 => PrimitiveType::C64,
            Self::C128 => PrimitiveType::C128,
        }
    }
}

pub trait ArrayElement: Copy {
    const TY: ElementType;
    const ELEMENT_SIZE_IN_BYTES: usize;
    const ZERO: Self;
}

#[allow(clippy::missing_safety_doc)]
/// A type implementing the `NativeType` trait can be directly converted to constant ops or
/// literals.
pub trait NativeType: Copy {
    unsafe fn constant_r0(b: c_lib::xla_builder, v: Self) -> c_lib::xla_op;
    unsafe fn constant_r1(b: c_lib::xla_builder, v: *const Self, l: usize) -> c_lib::xla_op;
    unsafe fn constant_r1c(b: c_lib::xla_builder, v: Self, l: usize) -> c_lib::xla_op;
    unsafe fn create_r0(v: Self) -> c_lib::literal;
    unsafe fn create_r1(v: *const Self, l: usize) -> c_lib::literal;
    unsafe fn literal_get_first_element(l: c_lib::literal) -> Self;
}

macro_rules! native_type {
    ($ty:ty, $cst0:ident, $cst1:ident, $cst1c:ident, $cre0:ident, $cre1:ident, $gf:ident) => {
        impl NativeType for $ty {
            unsafe fn constant_r0(b: c_lib::xla_builder, v: Self) -> c_lib::xla_op {
                c_lib::$cst0(b, v)
            }
            unsafe fn constant_r1(
                b: c_lib::xla_builder,
                v: *const Self,
                l: usize,
            ) -> c_lib::xla_op {
                c_lib::$cst1(b, v, l)
            }
            unsafe fn constant_r1c(b: c_lib::xla_builder, v: Self, l: usize) -> c_lib::xla_op {
                c_lib::$cst1c(b, v, l)
            }
            unsafe fn create_r0(v: Self) -> c_lib::literal {
                c_lib::$cre0(v)
            }
            unsafe fn create_r1(v: *const Self, l: usize) -> c_lib::literal {
                c_lib::$cre1(v, l)
            }
            unsafe fn literal_get_first_element(l: c_lib::literal) -> Self {
                c_lib::$gf(l)
            }
        }
    };
}

native_type!(
    bool,
    constant_r0_bool,
    constant_r1_bool,
    constant_r1c_bool,
    create_r0_bool,
    create_r1_bool,
    literal_get_first_element_bool
);

native_type!(
    i8,
    constant_r0_int8_t,
    constant_r1_int8_t,
    constant_r1c_int8_t,
    create_r0_int8_t,
    create_r1_int8_t,
    literal_get_first_element_int8_t
);

native_type!(
    i16,
    constant_r0_int16_t,
    constant_r1_int16_t,
    constant_r1c_int16_t,
    create_r0_int16_t,
    create_r1_int16_t,
    literal_get_first_element_int16_t
);

native_type!(
    i32,
    constant_r0_int32_t,
    constant_r1_int32_t,
    constant_r1c_int32_t,
    create_r0_int32_t,
    create_r1_int32_t,
    literal_get_first_element_int32_t
);

native_type!(
    i64,
    constant_r0_int64_t,
    constant_r1_int64_t,
    constant_r1c_int64_t,
    create_r0_int64_t,
    create_r1_int64_t,
    literal_get_first_element_int64_t
);

native_type!(
    u8,
    constant_r0_uint8_t,
    constant_r1_uint8_t,
    constant_r1c_uint8_t,
    create_r0_uint8_t,
    create_r1_uint8_t,
    literal_get_first_element_uint8_t
);

native_type!(
    u16,
    constant_r0_uint16_t,
    constant_r1_uint16_t,
    constant_r1c_uint16_t,
    create_r0_uint16_t,
    create_r1_uint16_t,
    literal_get_first_element_uint16_t
);

native_type!(
    u32,
    constant_r0_uint32_t,
    constant_r1_uint32_t,
    constant_r1c_uint32_t,
    create_r0_uint32_t,
    create_r1_uint32_t,
    literal_get_first_element_uint32_t
);

native_type!(
    u64,
    constant_r0_uint64_t,
    constant_r1_uint64_t,
    constant_r1c_uint64_t,
    create_r0_uint64_t,
    create_r1_uint64_t,
    literal_get_first_element_uint64_t
);

native_type!(
    f32,
    constant_r0_float,
    constant_r1_float,
    constant_r1c_float,
    create_r0_float,
    create_r1_float,
    literal_get_first_element_float
);

native_type!(
    f64,
    constant_r0_double,
    constant_r1_double,
    constant_r1c_double,
    create_r0_double,
    create_r1_double,
    literal_get_first_element_double
);

macro_rules! element_type {
    ($ty:ty, $v:ident, $sz:tt, $zero:expr) => {
        impl ArrayElement for $ty {
            const TY: ElementType = ElementType::$v;
            const ELEMENT_SIZE_IN_BYTES: usize = $sz;
            const ZERO: Self = $zero;
        }
    };
}

// Dummy F16 type.
#[derive(Copy, Clone, Debug)]
pub struct F16;

impl ArrayElement for F16 {
    const TY: ElementType = ElementType::F16;
    const ELEMENT_SIZE_IN_BYTES: usize = 2;
    const ZERO: Self = Self;
}

// Dummy BF16 type.
#[derive(Copy, Clone, Debug)]
pub struct Bf16;

impl ArrayElement for Bf16 {
    const TY: ElementType = ElementType::Bf16;
    const ELEMENT_SIZE_IN_BYTES: usize = 2;
    const ZERO: Self = Self;
}

element_type!(bool, Pred, 1, false);
element_type!(u8, U8, 1, 0);
element_type!(u16, U16, 2, 0);
element_type!(u32, U32, 4, 0);
element_type!(u64, U64, 8, 0);
element_type!(i8, S8, 1, 0);
element_type!(i16, S16, 2, 0);
element_type!(i32, S32, 4, 0);
element_type!(i64, S64, 8, 0);
element_type!(f32, F32, 4, 0.0f32);
element_type!(f64, F64, 8, 0.0f64);


/// A computation is built from a root [`XlaOp`]. Computations are device independent and can be
/// specialized to a given device through a compilation step.
#[derive(Clone)]
#[pyclass(unsendable)]
pub struct XlaComputation(c_lib::xla_computation);

pub(self) fn handle_status(status: c_lib::status) -> Result<()> {
    if status.is_null() {
        Ok(())
    } else {
        let msg = unsafe {
            let error_message_ptr = c_lib::status_error_message(status);
            let error_message = c_ptr_to_string(error_message_ptr);
            c_lib::status_free(status);
            error_message
        };
        let backtrace = std::backtrace::Backtrace::capture().to_string();
        Err(Error::XlaError { msg, backtrace })
    }
}

impl XlaComputation {
    pub fn from_proto(proto: &HloModuleProto) -> Self {
        let ptr = unsafe { c_lib::xla_computation_from_hlo_module_proto(proto.0) };
        Self(ptr)
    }

    /// The computation name.
    pub fn name(&self) -> String {
        unsafe {
            let ptr = c_lib::xla_computation_name(self.0);
            c_ptr_to_string(ptr)
        }
    }

    /// Compile this computation for the specified client.
    pub fn compile(&self, client: &PjRtClient) -> Result<PjRtLoadedExecutable> {
        client.compile(self)
    }

    /// Get the HloModuleProto for the computation.
    pub fn proto(&self) -> HloModuleProto {
        let ptr = unsafe { c_lib::xla_computation_proto(self.0) };
        HloModuleProto(ptr)
    }

}

impl Drop for XlaComputation {
    fn drop(&mut self) {
        unsafe { c_lib::xla_computation_free(self.0) }
    }
}

#[pyclass(unsendable)]
pub struct HloModuleProto(c_lib::hlo_module_proto);

impl HloModuleProto {
    /// Read a HLO module from a text file.
    pub fn from_text_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path.as_ref())?;
        let mut content = Vec::new();
        file.read_to_end(&mut content)?;
        Self::parse_and_return_unverified_module(&content)
    }

    /// Read a HLO module from a proto file, either in binary or pbtxt format.
    pub fn from_proto_file<P: AsRef<std::path::Path>>(path: P, binary: bool) -> Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path.as_ref())?;
        let mut content = Vec::new();
        file.read_to_end(&mut content)?;
        Self::parse_proto(&content, binary)
    }

    pub fn parse_and_return_unverified_module(data: &[u8]) -> Result<Self> {
        let mut ptr: c_lib::hlo_module_proto = std::ptr::null_mut();
        let status = unsafe {
            c_lib::hlo_module_proto_parse_and_return_unverified_module(
                data.as_ptr() as *const libc::c_char,
                data.len(),
                &mut ptr,
            )
        };
        handle_status(status)?;
        Ok(Self(ptr))
    }

    pub fn parse_proto(data: &[u8], binary: bool) -> Result<Self> {
        let mut ptr: c_lib::hlo_module_proto = std::ptr::null_mut();
        let status = unsafe {
            c_lib::hlo_module_proto_parse_proto(
                data.as_ptr() as *const libc::c_char,
                data.len(),
                binary,
                &mut ptr,
            )
        };
        handle_status(status)?;
        Ok(Self(ptr))
    }
}

impl Drop for HloModuleProto {
    fn drop(&mut self) {
        unsafe { c_lib::hlo_module_proto_free(self.0) }
    }
}

#[pyclass(unsendable)]
pub struct HloModule(c_lib::hlo_module);

impl HloModule {
    pub fn from_proto(proto: &HloModuleProto) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let status = unsafe {
            c_lib::hlo_module_from_proto(
                proto.0,
                &mut ptr,
            )};
        handle_status(status)?;
        Ok(Self(ptr))
    }

    pub fn to_string(&self) -> Result<String> {
        let str_ptr = unsafe {
            c_lib::hlo_module_to_string(self.0)
        };
        let module_str = unsafe {
            let c_str = std::ffi::CStr::from_ptr(str_ptr);
            let result = c_str.to_str()?.to_string();
            libc::free(str_ptr as *mut _);
            Ok(result)
        };
        module_str
    }


    pub fn get_entry_computation(&self) -> Result<HloComputation> {
        let entry_comp = unsafe {
            c_lib::hlo_module_entry_computation(self.0)
        };
        Ok(HloComputation(entry_comp))
    }

    pub fn computation_count(&self) -> Result<i64> {
        let comp_count = unsafe {
            c_lib::hlo_module_computation_count(self.0)
        };
        Ok(comp_count)
    }

    pub fn instruction_count(&self) -> Result<i64> {
        let instruct_count = unsafe {
            c_lib::hlo_module_instruction_count(self.0)
        };
        Ok(instruct_count)
    }
}


#[pyclass(unsendable)]
pub struct HloComputation(c_lib::hlo_computation);
