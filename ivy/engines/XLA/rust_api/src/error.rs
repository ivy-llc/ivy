use pyo3::prelude::*;
use pyo3::exceptions::{PyOSError};
use std::str::Utf8Error;

/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Incorrect number of elements.
    #[error("wrong element count {element_count} for dims {dims:?}")]
    WrongElementCount { dims: Vec<usize>, element_count: usize },

    /// Error from the xla C++ library.
    #[error("xla error {msg}\n{backtrace}")]
    XlaError { msg: String, backtrace: String },

    #[error("unexpected element type {0}")]
    UnexpectedElementType(i32),

    #[error("unexpected number of dimensions, expected: {expected}, got: {got} ({dims:?})")]
    UnexpectedNumberOfDims { expected: usize, got: usize, dims: Vec<i64> },

    #[error("not an element type, got: {got:?}")]
    NotAnElementType { got: crate::PrimitiveType },

    #[error("not an array, expected: {expected:?}, got: {got:?}")]
    NotAnArray { expected: Option<usize>, got: crate::Shape },

    #[error("cannot handle unsupported shapes {shape:?}")]
    UnsupportedShape { shape: crate::Shape },

    #[error("unexpected number of tuple elements, expected: {expected}, got: {got}")]
    UnexpectedNumberOfElemsInTuple { expected: usize, got: usize },

    #[error("element type mismatch, on-device: {on_device:?}, on-host: {on_host:?}")]
    ElementTypeMismatch { on_device: crate::ElementType, on_host: crate::ElementType },

    #[error("unsupported element type for {op}: {ty:?}")]
    UnsupportedElementType { ty: crate::PrimitiveType, op: &'static str },

    #[error(
    "target buffer is too large, offset {offset}, shape {shape:?}, buffer_len: {buffer_len}"
    )]
    TargetBufferIsTooLarge { offset: usize, shape: crate::ArrayShape, buffer_len: usize },

    #[error("binary buffer is too large, element count {element_count}, buffer_len: {buffer_len}")]
    BinaryBufferIsTooLarge { element_count: usize, buffer_len: usize },

    #[error("empty literal")]
    EmptyLiteral,

    #[error("index out of bounds {index}, rank {rank}")]
    IndexOutOfBounds { index: i64, rank: usize },

    #[error("npy/npz error {0}")]
    Npy(String),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Zip file format error.
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    #[error("cannot create literal with shape {ty:?} {dims:?} from bytes data with len {data_len_in_bytes}")]
    CannotCreateLiteralWithData {
        data_len_in_bytes: usize,
        ty: crate::PrimitiveType,
        dims: Vec<usize>,
    },

    #[error("invalid dimensions in matmul, lhs: {lhs_dims:?}, rhs: {rhs_dims:?}, {msg}")]
    MatMulIncorrectDims { lhs_dims: Vec<i64>, rhs_dims: Vec<i64>, msg: &'static str },

    #[error("Invalid UTF-8 data: {0}")]
    Utf8Error(#[from] Utf8Error),
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        PyOSError::new_err(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
