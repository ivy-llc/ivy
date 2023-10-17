mod c_lib;
mod error;
mod wrappers;

use std::rc::Rc;
pub use error::{Error, Result};
pub use wrappers::*;
use pyo3::prelude::*;
use ndarray::{ArrayD};
use numpy::{PyArrayDyn, ToPyArray};
use half::{f16, bf16};
use pyo3::exceptions::PyTypeError;
use pyo3::{exceptions, wrap_pyfunction};


#[derive(Debug, Copy, Clone)]
pub enum TfLogLevel {
    Info,
    Warning,
    Error,
    Fatal,
}

impl TfLogLevel {
    fn as_env_variable_str(&self) -> &'static str {
        match self {
            Self::Info => "0",
            Self::Warning => "1",
            Self::Error => "2",
            Self::Fatal => "3",
        }
    }
}

pub fn set_tf_min_log_level(log_level: TfLogLevel) {
    std::env::set_var("TF_CPP_MIN_LOG_LEVEL", log_level.as_env_variable_str())
}


#[derive(Debug)]
enum ArrayDyn {
    Pred(ArrayD<bool>),
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    Bf16(ArrayD<bf16>),
    F16(ArrayD<f16>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}

#[derive(Debug)]
#[pyclass(unsendable)]
pub struct Tensor {
    x: ArrayDyn
}

impl From<ArrayD<bool>> for Tensor {
    fn from(x: ArrayD<bool>) -> Self {
        Tensor {
            x: ArrayDyn::Pred(x),
        }
    }
}

impl From<ArrayD<i8>> for Tensor {
    fn from(x: ArrayD<i8>) -> Self {
        Tensor {
            x: ArrayDyn::I8(x),
        }
    }
}

impl From<ArrayD<i16>> for Tensor {
    fn from(x: ArrayD<i16>) -> Self {
        Tensor {
            x: ArrayDyn::I16(x),
        }
    }
}

impl From<ArrayD<i32>> for Tensor {
    fn from(x: ArrayD<i32>) -> Self {
        Tensor {
            x: ArrayDyn::I32(x),
        }
    }
}

impl From<ArrayD<i64>> for Tensor {
    fn from(x: ArrayD<i64>) -> Self {
        Tensor {
            x: ArrayDyn::I64(x),
        }
    }
}

impl From<ArrayD<u8>> for Tensor {
    fn from(x: ArrayD<u8>) -> Self {
        Tensor {
            x: ArrayDyn::U8(x),
        }
    }
}

impl From<ArrayD<u16>> for Tensor {
    fn from(x: ArrayD<u16>) -> Self {
        Tensor {
            x: ArrayDyn::U16(x),
        }
    }
}

impl From<ArrayD<u32>> for Tensor {
    fn from(x: ArrayD<u32>) -> Self {
        Tensor {
            x: ArrayDyn::U32(x),
        }
    }
}

impl From<ArrayD<u64>> for Tensor {
    fn from(x: ArrayD<u64>) -> Self {
        Tensor {
            x: ArrayDyn::U64(x),
        }
    }
}

impl From<ArrayD<bf16>> for Tensor {
    fn from(x: ArrayD<bf16>) -> Self {
        Tensor {
            x: ArrayDyn::Bf16(x),
        }
    }
}

impl From<ArrayD<f16>> for Tensor {
    fn from(x: ArrayD<f16>) -> Self {
        Tensor {
            x: ArrayDyn::F16(x),
        }
    }
}

impl From<ArrayD<f32>> for Tensor {
    fn from(x: ArrayD<f32>) -> Self {
        Tensor {
            x: ArrayDyn::F32(x),
        }
    }
}

impl From<ArrayD<f64>> for Tensor {
    fn from(x: ArrayD<f64>) -> Self {
        Tensor {
            x: ArrayDyn::F64(x),
        }
    }
}


#[pymethods]
impl Tensor {
    fn __repr__(&self) -> PyResult<String> {
        let desc = match &self.x {
            ArrayDyn::Pred(array) => format!("{:?}", array),
            ArrayDyn::I8(array) => format!("{:?}", array),
            ArrayDyn::I16(array) => format!("{:?}", array),
            ArrayDyn::I32(array) => format!("{:?}", array),
            ArrayDyn::I64(array) => format!("{:?}", array),
            ArrayDyn::U8(array) => format!("{:?}", array),
            ArrayDyn::U16(array) => format!("{:?}", array),
            ArrayDyn::U32(array) => format!("{:?}", array),
            ArrayDyn::U64(array) => format!("{:?}", array),
            ArrayDyn::Bf16(array) => format!("{:?}", array),
            ArrayDyn::F16(array) => format!("{:?}", array),
            ArrayDyn::F32(array) => format!("{:?}", array),
            ArrayDyn::F64(array) => format!("{:?}", array),
        };
        Ok(format!("Tensor({})", desc))
    }
}

#[derive(Clone, Debug)]
#[pyclass(unsendable)]
struct Bf16Array {
    x: Py<PyArrayDyn<f32>>
}
impl From<Py<PyArrayDyn<f32>>> for Bf16Array {
    fn from(x: Py<PyArrayDyn<f32>>) -> Self {
        Bf16Array {
            x
        }
    }
}

#[derive(Clone, Debug)]
#[pyclass(unsendable)]
struct F16Array {
    x: Py<PyArrayDyn<f32>>
}
impl From<Py<PyArrayDyn<f32>>> for F16Array {
    fn from(x: Py<PyArrayDyn<f32>>) -> Self {
        F16Array {
            x
        }
    }
}

#[pyfunction]
fn create_bf16_array(x: Py<PyArrayDyn<f32>>) -> PyResult<Bf16Array> {
    let x = Bf16Array{x};
    Ok(x)
}

#[pyfunction]
fn create_f16_array(x: Py<PyArrayDyn<f32>>) -> PyResult<F16Array> {
    let x = F16Array{x};
    Ok(x)
}

#[derive(Debug)]
enum DynamicPyArray {
    Pred(Py<PyArrayDyn<bool>>),
    I8(Py<PyArrayDyn<i8>>),
    I16(Py<PyArrayDyn<i16>>),
    I32(Py<PyArrayDyn<i32>>),
    I64(Py<PyArrayDyn<i64>>),
    U8(Py<PyArrayDyn<u8>>),
    U16(Py<PyArrayDyn<u16>>),
    U32(Py<PyArrayDyn<u32>>),
    U64(Py<PyArrayDyn<u64>>),
    Bf16(Bf16Array),
    F16(F16Array),
    F32(Py<PyArrayDyn<f32>>),
    F64(Py<PyArrayDyn<f64>>),
}

impl<'source> FromPyObject<'source> for DynamicPyArray {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        if let Ok(arr) = obj.extract::<Py<PyArrayDyn<bool>>>() {
            Ok(DynamicPyArray::Pred(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<i8>>>() {
            Ok(DynamicPyArray::I8(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<i16>>>() {
            Ok(DynamicPyArray::I16(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<i32>>>() {
            Ok(DynamicPyArray::I32(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<i64>>>() {
            Ok(DynamicPyArray::I64(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<u8>>>() {
            Ok(DynamicPyArray::U8(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<u16>>>() {
            Ok(DynamicPyArray::U16(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<u32>>>() {
            Ok(DynamicPyArray::U32(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<u64>>>() {
            Ok(DynamicPyArray::U64(arr))
        }
        else if let Ok(arr) = obj.extract::<Bf16Array>() {
            Ok(DynamicPyArray::Bf16(arr))
        }
        else if let Ok(arr) = obj.extract::<F16Array>() {
            Ok(DynamicPyArray::F16(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<f32>>>() {
            Ok(DynamicPyArray::F32(arr))
        }
        else if let Ok(arr) = obj.extract::<Py<PyArrayDyn<f64>>>() {
            Ok(DynamicPyArray::F64(arr))
        }
        else {
            Err(PyErr::from(PyTypeError::new_err(
                "Expected a numpy array of one of the valid types",
            )))
        }
    }
}

#[pyfunction]
fn constant_array(py: Python, array: DynamicPyArray, builder: XlaBuilder) -> PyResult<XlaOp> {
    match array {
        DynamicPyArray::Pred(py_array) => {
            let x = Literal::vec1(unsafe { py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice() });
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::I8(py_array) => {
            let x = Literal::vec1(unsafe { py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice() });
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::I16(py_array) => {
            let x = Literal::vec1(unsafe { py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice() });
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::I32(py_array) => {
            let x = Literal::vec1(unsafe { py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice() });
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::I64(py_array) => {
            let x = Literal::vec1(unsafe { py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice() });
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::U8(py_array) => {
            let x = Literal::vec1(unsafe { py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice() });
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::U16(py_array) => {
            let x = Literal::vec1(unsafe { py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice() });
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::U32(py_array) => {
            let x = Literal::vec1(unsafe { py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice() });
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::U64(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::Bf16(py_array) => {
            let x = Literal::vec1(unsafe {py_array.x.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()}).convert(PrimitiveType::Bf16)?;
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::F16(py_array) => {
            let x = Literal::vec1(unsafe {py_array.x.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()}).convert(PrimitiveType::F16)?;
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::F32(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
        DynamicPyArray::F64(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            let x = builder.constant_literal(&x)?;
            Ok(x)
        },
    }
}


#[pyfunction]
fn gather_params(py: Python, arrays: Vec<DynamicPyArray>) -> PyResult<Vec<Literal>> {
    let mut literals = Vec::with_capacity(arrays.len());
    for array in arrays {
        match array {
            DynamicPyArray::Pred(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::I8(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::I16(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::I32(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::I64(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::U8(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::U16(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::U32(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::U64(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::Bf16(py_array) => {
                let x = Literal::vec1(unsafe {py_array.x.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()}).convert(PrimitiveType::Bf16)?;
                literals.push(x);
            },
            DynamicPyArray::F16(py_array) => {
                let x = Literal::vec1(unsafe {py_array.x.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()}).convert(PrimitiveType::F16)?;
                literals.push(x);
            },
            DynamicPyArray::F32(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
            DynamicPyArray::F64(py_array) => {
                let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
                literals.push(x);
            },
        }
    }
    Ok(literals)
}

#[pyfunction]
fn new_input(py: Python, input: DynamicPyArray) -> PyResult<Literal> {
    match input {
        DynamicPyArray::Pred(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::I8(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::I16(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::I32(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::I64(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::U8(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::U16(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::U32(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::U64(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::Bf16(py_array) => {
            let x = Literal::vec1(unsafe {py_array.x.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()}).convert(PrimitiveType::Bf16)?;
            Ok(x)
        },
        DynamicPyArray::F16(py_array) => {
            let x = Literal::vec1(unsafe {py_array.x.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()}).convert(PrimitiveType::F16)?;
            Ok(x)
        },
        DynamicPyArray::F32(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
        DynamicPyArray::F64(py_array) => {
            let x = Literal::vec1(unsafe {py_array.as_ref(py).as_array().to_owned().into_raw_vec().as_slice()});
            Ok(x)
        },
    }
}

#[pyfunction]
fn swap_param(x: Literal, mut params: Vec<Literal>) -> PyResult<Vec<Literal>> {
    params[0] = x;
    Ok(params)
}

#[pyfunction]
fn to_tensor(literal: Literal) -> PyResult<Tensor> {
    let shape = literal.shape().unwrap();
    let shape = ArrayShape::try_from(&shape).unwrap();
    let shape: Vec<usize> = shape.dims().iter().map(|&x| x as usize).collect();

    match literal.ty().unwrap() {
        ElementType::Pred => {
            let data: Vec<bool> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::S8 => {
            let data: Vec<i8> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::S16 => {
            let data: Vec<i16> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::S32 => {
            let data: Vec<i32> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::S64 => {
            let data: Vec<i64> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::U8 => {
            let data: Vec<u8> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::U16 => {
            let data: Vec<u16> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::U32 => {
            let data: Vec<u32> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::U64 => {
            let data: Vec<u64> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::Bf16 => {
            let data: Vec<f32> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::F16 => {
            let data: Vec<f32> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::F32 => {
            let data: Vec<f32> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        ElementType::F64 => {
            let data: Vec<f64> = literal.to_vec().unwrap();
            let array = ArrayD::from_shape_vec(shape, data).unwrap();
            Ok(Tensor::from(array))
        }
        _ => Err(PyErr::from(PyTypeError::new_err(
            "Unsupported date type",
        )))

    }
}

#[pyfunction]
fn to_numpy(py: Python, literal: Literal) -> PyResult<PyObject> {
    let shape = literal.shape().unwrap();
    let shape = ArrayShape::try_from(&shape).unwrap();
    let shape: Vec<usize> = shape.dims().iter().map(|&x| x as usize).collect();

    match literal.ty().unwrap() {
        ElementType::Pred => {
            let data: Vec<bool> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::S8 => {
            let data: Vec<i8> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::S16 => {
            let data: Vec<i16> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::S32 => {
            let data: Vec<i32> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::S64 => {
            let data: Vec<i64> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::U8 => {
            let data: Vec<u8> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::U16 => {
            let data: Vec<u16> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::U32 => {
            let data: Vec<u32> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::U64 => {
            let data: Vec<u64> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::Bf16 | ElementType::F16 => {
            let literal = literal.convert(PrimitiveType::F32)?;
            let data: Vec<f32> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::F32 => {
            let data: Vec<f32> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        ElementType::F64 => {
            let data: Vec<f64> = literal.to_vec()?;
            let array = ArrayD::from_shape_vec(shape, data).unwrap().to_pyarray(py);
            Ok(array.to_object(py))
        }
        _ => Err(PyErr::from(PyTypeError::new_err(
            "Unsupported data type",
        )))
    }
}

#[pyfunction]
fn to_tuple(literal: Literal) -> PyResult<Vec<Literal>> {
    let y = literal.to_tuple()?;
    Ok(y)
}


macro_rules! param_gen {
    ($name:ident, $type:ty) => {
        #[pyfunction]
        fn $name(builder: XlaBuilder, param_number: i64, dims: Vec<i64>, name: &str) -> PyResult<XlaOp> {
            let shape = &Shape::array::<$type>(dims);
            let param = builder.parameter_s(param_number, shape, name)?;
            Ok(param)
        }
    }
}

param_gen!(param_pred, bool);
param_gen!(param_i8, i8);
param_gen!(param_i16, i16);
param_gen!(param_i32, i32);
param_gen!(param_i64, i64);
param_gen!(param_u8, u8);
param_gen!(param_u16, u16);
param_gen!(param_u32, u32);
param_gen!(param_u64, u64);
param_gen!(param_bf16, Bf16);
param_gen!(param_f16, F16);
param_gen!(param_f32, f32);
param_gen!(param_f64, f64);


macro_rules! constant {
    ($name:ident, $type:ty) => {
        #[pyfunction]
        fn $name(b: XlaBuilder, v: $type) -> PyResult<XlaOp> {
            let c = b.c0(v)?;
            Ok(c)
        }
    };
}

constant!(constant_bool, bool);
constant!(constant_i8, i8);
constant!(constant_i16, i16);
constant!(constant_i32, i32);
constant!(constant_i64, i64);
constant!(constant_u8, u8);
constant!(constant_u16, u16);
constant!(constant_u32, u32);
constant!(constant_u64, u64);
constant!(constant_f32, f32);
constant!(constant_f64, f64);


macro_rules! astype {
    ($name:ident, $primitive:ident) => {
        #[pyfunction]
        fn $name(x: XlaOp) -> PyResult<XlaOp> {
            let y = x.astype(PrimitiveType::$primitive)?;
            Ok(y)
        }
    };
}

astype!(astype_bool, Pred);
astype!(astype_i8, S8);
astype!(astype_i16, S16);
astype!(astype_i32, S32);
astype!(astype_i64, S64);
astype!(astype_u8, U8);
astype!(astype_u16, U16);
astype!(astype_u32, U32);
astype!(astype_u64, U64);
astype!(astype_bf16, Bf16);
astype!(astype_f16, F16);
astype!(astype_f32, F32);
astype!(astype_f64, F64);


#[pyfunction]
fn cpu_client() -> PyResult<PjRtClient> {
    let client = PjRtClient::cpu()?;
    Ok(client)
}

#[pyfunction]
fn gpu_client(memory_fraction: f64, preallocate: bool) -> PyResult<PjRtClient> {
    let client = PjRtClient::gpu(memory_fraction, preallocate)?;
    Ok(client)
}

#[pyfunction]
fn xla_builder(name: &str) -> PyResult<XlaBuilder> {
    let builder = XlaBuilder::new(name);
    Ok(builder)
}

#[pyfunction]
fn build(op: XlaOp) -> PyResult<XlaComputation> {
    let computation = op.build()?;
    Ok(computation)
}

#[pyfunction]
fn get_hlo_proto(comp: &XlaComputation) -> PyResult<HloModuleProto> {
    let hlo_proto = comp.proto();
    Ok(hlo_proto)
}

#[pyfunction]
fn hlo_module_from_proto(proto: &HloModuleProto) -> PyResult<HloModule> {
    let hlo_module = HloModule::from_proto(proto)?;
    Ok(hlo_module)
}

#[pyfunction]
fn hlo_module_to_string(module: &HloModule) -> PyResult<String> {
    let module_str = module.to_string()?;
    Ok(module_str)
}

#[pyfunction]
fn get_hlo_module_entry_computation(module: &HloModule) -> PyResult<HloComputation> {
    let hlo_comp = module.get_entry_computation()?;
    Ok(hlo_comp)
}

#[pyfunction]
fn computation_count(module: &HloModule) -> PyResult<i64> {
    let comp_count = module.computation_count()?;
    Ok(comp_count)
}

#[pyfunction]
fn instruction_count(module: &HloModule) -> PyResult<i64> {
    let instruct_count = module.instruction_count()?;
    Ok(instruct_count)
}

#[pyfunction]
fn compile(client: PjRtClient, computation: &XlaComputation) -> PyResult<PjRtLoadedExecutable> {
    let executable = client.compile(computation)?;
    Ok(executable)
}

#[pyfunction]
fn execute(executable: &PjRtLoadedExecutable, args: Vec<Literal>) -> PyResult<PjRtBuffer> {
    let buffer = executable.execute::<Literal>(args.as_slice())?[0].remove(0);
    Ok(buffer)
}

#[pyfunction]
fn to_literal(buffer: &PjRtBuffer) -> PyResult<Literal> {
    let literal = buffer.to_literal_sync()?;
    Ok(literal)
}

#[pyfunction]
fn add(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.add_(rhs)?;
    Ok(y)
}

#[pyfunction]
fn sub(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.sub_(rhs)?;
    Ok(y)
}

#[pyfunction]
fn mul(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.mul_(rhs)?;
    Ok(y)
}

#[pyfunction]
fn div(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.div_(rhs)?;
    Ok(y)
}

#[pyfunction]
fn rem(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.rem_(rhs)?;
    Ok(y)
}

#[pyfunction]
fn pow(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.pow(rhs)?;
    Ok(y)
}

#[pyfunction]
fn max(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.max(rhs)?;
    Ok(y)
}

#[pyfunction]
fn min(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.min(rhs)?;
    Ok(y)
}

#[pyfunction]
fn _and(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.and(rhs)?;
    Ok(y)
}

#[pyfunction]
fn _or(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.or(rhs)?;
    Ok(y)
}

#[pyfunction]
fn xor(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.xor(rhs)?;
    Ok(y)
}

#[pyfunction]
fn eq(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.eq(rhs)?;
    Ok(y)
}

#[pyfunction]
fn ne(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.ne(rhs)?;
    Ok(y)
}

#[pyfunction]
fn ge(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.ge(rhs)?;
    Ok(y)
}

#[pyfunction]
fn gt(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.gt(rhs)?;
    Ok(y)
}

#[pyfunction]
fn le(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.le(rhs)?;
    Ok(y)
}

#[pyfunction]
fn lt(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.lt(rhs)?;
    Ok(y)
}

#[pyfunction]
fn lshift(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.lshift(rhs)?;
    Ok(y)
}

#[pyfunction]
fn rshift(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.rshift_arith(rhs)?;
    Ok(y)
}

#[pyfunction]
fn atan2(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.atan2(rhs)?;
    Ok(y)
}

#[pyfunction]
fn dot(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.dot(rhs)?;
    Ok(y)
}

#[pyfunction]
fn matmul(lhs: XlaOp, rhs: &XlaOp) -> PyResult<XlaOp> {
    let y = lhs.matmul(rhs)?;
    Ok(y)
}

#[pyfunction]
fn population_count(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.population_count()?;
    Ok(y)
}

#[pyfunction]
fn _not(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.not()?;
    Ok(y)
}

#[pyfunction]
fn neg(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.neg()?;
    Ok(y)
}

#[pyfunction]
fn abs(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.abs()?;
    Ok(y)
}

#[pyfunction]
fn floor(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.floor()?;
    Ok(y)
}

#[pyfunction]
fn ceil(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.ceil()?;
    Ok(y)
}

#[pyfunction]
fn round(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.round()?;
    Ok(y)
}

#[pyfunction]
fn round_nearest_even(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.round_nearest_even()?;
    Ok(y)
}

#[pyfunction]
fn exp(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.exp()?;
    Ok(y)
}

#[pyfunction]
fn expm1(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.expm1()?;
    Ok(y)
}

#[pyfunction]
fn log(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.log()?;
    Ok(y)
}

#[pyfunction]
fn log1p(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.log1p()?;
    Ok(y)
}

#[pyfunction]
fn logistic(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.logistic()?;
    Ok(y)
}

#[pyfunction]
fn sign(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.sign()?;
    Ok(y)
}

#[pyfunction]
fn clz(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.clz()?;
    Ok(y)
}

#[pyfunction]
fn sin(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.sin()?;
    Ok(y)
}

#[pyfunction]
fn cos(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.cos()?;
    Ok(y)
}

#[pyfunction]
fn tanh(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.tanh()?;
    Ok(y)
}

#[pyfunction]
fn real(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.real()?;
    Ok(y)
}

#[pyfunction]
fn imag(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.imag()?;
    Ok(y)
}

#[pyfunction]
fn conj(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.conj()?;
    Ok(y)
}

#[pyfunction]
fn square(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.square()?;
    Ok(y)
}

#[pyfunction]
fn sqrt(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.sqrt()?;
    Ok(y)
}

#[pyfunction]
fn rsqrt(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.rsqrt()?;
    Ok(y)
}

#[pyfunction]
fn cbrt(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.cbrt()?;
    Ok(y)
}

#[pyfunction]
fn upper_triangle(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.upper_triangle()?;
    Ok(y)
}

#[pyfunction]
fn lower_triangle(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.lower_triangle()?;
    Ok(y)
}

#[pyfunction]
fn erf(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.erf()?;
    Ok(y)
}

#[pyfunction]
fn is_finite(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.is_finite()?;
    Ok(y)
}

#[pyfunction]
fn zeros_like(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.zeros_like()?;
    Ok(y)
}

#[pyfunction]
fn copy(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.copy()?;
    Ok(y)
}

#[pyfunction]
fn sigmoid(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.sigmoid()?;
    Ok(y)
}

#[pyfunction]
fn silu(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.silu()?;
    Ok(y)
}

#[pyfunction]
fn relu(x: XlaOp) -> PyResult<XlaOp> {
   let y = x.relu()?;
    Ok(y)
}

#[pyfunction]
fn gelu(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.gelu()?;
    Ok(y)
}

#[pyfunction]
fn gelu_approx(x: XlaOp) -> PyResult<XlaOp> {
    let y = x.gelu_approx()?;
    Ok(y)
}

#[pyfunction]
fn einsum1(x: XlaOp, config: &str) -> PyResult<XlaOp> {
    let y = x.einsum1(config)?;
    Ok(y)
}

#[pyfunction]
fn einsum2(x: XlaOp, rhs: &XlaOp, config: &str) -> PyResult<XlaOp> {
    let y = x.einsum2(rhs, config)?;
    Ok(y)
}

#[pyfunction]
fn reshape(x: XlaOp, dims: Vec<i64>) -> PyResult<XlaOp> {
    let dims = dims.as_slice();
    let y = x.reshape(dims)?;
    Ok(y)
}

#[pyfunction]
fn dynamic_reshape(
    x: XlaOp,
    dim_sizes: Vec<XlaOp>,
    new_size_bounds: Vec<i64>,
    dims_are_dynamic: Vec<bool>
) -> PyResult<XlaOp> {
    let dim_sizes = dim_sizes.as_slice();
    let new_size_bounds = new_size_bounds.as_slice();
    let y = x.dynamic_reshape(dim_sizes, new_size_bounds, dims_are_dynamic)?;
    Ok(y)
}

#[pyfunction]
fn broadcast(x: XlaOp, dims: Vec<i64>) -> PyResult<XlaOp> {
    let dims = dims.as_slice();
    let y = x.broadcast(dims)?;
    Ok(y)
}

#[pyfunction]
fn broadcast_in_dim(x: XlaOp, out_dims: Vec<i64>, broadcast_dims: Vec<i64>) -> PyResult<XlaOp> {
    let out_dims = out_dims.as_slice();
    let broadcast_dims = broadcast_dims.as_slice();
    let y = x.broadcast_in_dim(out_dims, broadcast_dims)?;
    Ok(y)
}

#[pyfunction]
fn collapse(x: XlaOp, dims: Vec<i64>) -> PyResult<XlaOp> {
    let dims = dims.as_slice();
    let y = x.collapse(dims)?;
    Ok(y)
}

#[pyfunction]
fn transpose(x: XlaOp, index_perm: Vec<i64>) -> PyResult<XlaOp> {
    let index_perm = index_perm.as_slice();
    let y = x.transpose(index_perm)?;
    Ok(y)
}

#[pyfunction]
fn swap_dims(x: XlaOp, index1: i64, index2: i64) -> PyResult<XlaOp> {
    let y = x.swap_dims(index1, index2)?;
    Ok(y)
}

#[pyfunction]
fn pad(x: XlaOp, padding_value: &XlaOp, padding_config:Vec<(i64, i64, i64)> ) -> PyResult<XlaOp> {
   let y = x.pad(padding_value, padding_config)?;
    Ok(y)
}

#[pyfunction]
fn pad_in_dim(x: XlaOp, padding_value: &XlaOp, dinmo: i64, pad_low: i64, pad_high: i64)  -> PyResult<XlaOp> {
    let y = x.pad_in_dim(padding_value, dinmo, pad_low, pad_high)?;
    Ok(y)
}

#[pyfunction]
fn slice(x: XlaOp, start_indices: Vec<i64>, limit_indices: Vec<i64>, strides: Vec<i64>) -> PyResult<XlaOp> {
    let start_indices = start_indices.as_slice();
    let limit_indices = limit_indices.as_slice();
    let strides = strides.as_slice();
    let y = x.slice(start_indices, limit_indices, strides)?;
    Ok(y)
}

#[pyfunction]
fn slice_in_dim(x: XlaOp, start_index: i64, stop_index: i64, stride: i64, dim: i64) -> PyResult<XlaOp> {
    let y = x.slice_in_dim(start_index, stop_index, stride, dim)?;
    Ok(y)
}

#[pyfunction]
fn dynamic_slice(x: XlaOp, start_indices: Vec<XlaOp>, slice_indices: Vec<i64>) -> PyResult<XlaOp> {
    let start_indices = start_indices.as_slice();
    let slice_indices = slice_indices.as_slice();
    let y = x.dynamic_slice(start_indices, slice_indices)?;
    Ok(y)
}

#[pyfunction]
fn dynamic_update_slice(x: XlaOp, update: &XlaOp, start_indices: Vec<XlaOp>) -> PyResult<XlaOp> {
    let start_indices = start_indices.as_slice();
    let y = x.dynamic_update_slice(update, start_indices)?;
    Ok(y)
}

#[pyfunction]
fn at(x: XlaOp, index_in_dim: i64, dim_index: i64) -> PyResult<XlaOp> {
    let y = x.at(index_in_dim, dim_index)?;
    Ok(y)
}

#[pyfunction]
fn squeeze(x: XlaOp, index: i64) -> PyResult<XlaOp> {
    let y = x.squeeze(index)?;
    Ok(y)
}

#[pyfunction]
fn clamp(x: XlaOp, min: &XlaOp, max: &XlaOp) -> PyResult<XlaOp> {
    let y = x.clamp(min, max)?;
    Ok(y)
}

#[pyfunction]
fn concat(x: XlaOp, args: Vec<XlaOp>, dim: i64) -> PyResult<XlaOp> {
    let args = args.as_slice();
    let y = x.concat_in_dim(args, dim)?;
    Ok(y)
}

#[pyfunction]
fn get_tuple_element(x: XlaOp, index: i64) -> PyResult<XlaOp> {
    let y = x.get_tuple_element(index)?;
    Ok(y)
}

#[pyfunction]
fn rng_uniform(min: &XlaOp, max: &XlaOp, shape: &ArrayShape) -> PyResult<XlaOp> {
    let y = XlaOp::rng_uniform(min, max, shape)?;
    Ok(y)
}

#[pyfunction]
fn rng_normal(mu: &XlaOp, sigma: &XlaOp, shape: &ArrayShape) -> PyResult<XlaOp> {
    let y = XlaOp::rng_normal(mu, sigma, shape)?;
    Ok(y)
}

#[pyfunction]
fn astype(x: XlaOp, ty: PrimitiveType) -> PyResult<XlaOp> {
    let y = x.astype(ty)?;
    Ok(y)
}

#[pyfunction]
fn dimension_size(x: XlaOp, index: i64) -> PyResult<XlaOp> {
    let y = x.dimensions_size(index)?;
    Ok(y)
}

#[pyfunction]
fn reduce(
    x: XlaOp,
    init_value: XlaOp,
    comp: &XlaComputation,
    dims: Vec<i64>,
    keep_dims: bool,
) -> PyResult<XlaOp> {
    let dims = dims.as_slice();
    let y = x.reduce(init_value, comp, dims, keep_dims)?;
    Ok(y)
}

#[pyfunction]
fn call(builder: XlaBuilder, computation: &XlaComputation, operands: Vec<XlaOp>) -> PyResult<XlaOp> {
    let operands = operands.as_slice();
    let y = builder.call(computation, operands)?;
    Ok(y)
}

#[pyfunction]
fn map(builder: XlaBuilder,
       operands: Vec<XlaOp>,
       computation: &XlaComputation,
       dims: Vec<i64>,
       static_operands: Vec<XlaOp>
) -> PyResult<XlaOp> {
    let operands = operands.as_slice();
    let dims = dims.as_slice();
    let static_operands = static_operands.as_slice();
    let y = builder.map(operands, computation, dims, static_operands)?;
    Ok(y)
}

#[pyfunction]
fn select(x: XlaOp, on_true: &XlaOp, on_false: &XlaOp) -> PyResult<XlaOp> {
    let y = x.select(on_true, on_false)?;
    Ok(y)
}

#[pyfunction]
fn while_loop(cond: &XlaComputation, body: &XlaComputation, init: XlaOp) -> PyResult<XlaOp> {
    let y = XlaOp::while_(cond, body, init)?;
    Ok(y)
}

#[pyfunction]
fn conditional(
    x: XlaOp,
    true_op: XlaOp,
    true_comp: &XlaComputation,
    false_op: XlaOp,
    false_comp: &XlaComputation,
) -> PyResult<XlaOp> {
    let y = x.conditional(true_op, true_comp,false_op, false_comp)?;
    Ok(y)
}

#[pyfunction]
fn conv(
    x: XlaOp,
    rhs: &XlaOp,
    window_strides: Vec<i64>,
    padding: &str,
    feature_group_count: i64,
    batch_group_count: i64,
) -> PyResult<XlaOp> {
    let window_strides = window_strides.as_slice();
    let y = x.conv(rhs, window_strides, padding, feature_group_count, batch_group_count)?;
    Ok(y)
}

#[pyfunction]
fn conv_general_dilated(
    x: XlaOp,
    rhs: &XlaOp,
    window_strides: Vec<i64>,
    padding: Vec<(i64, i64)>,
    lhs_dilations: Vec<i64>,
    rhs_dilations: Vec<i64>,
    input_batch_dim: i64,
    input_feature_dim: i64,
    input_spatial_dims: Vec<i64>,
    output_batch_dim: i64,
    output_feature_dim: i64,
    output_spatial_dims: Vec<i64>,
    kernel_input_feature_dim: i64,
    kernel_output_feature_dim: i64,
    kernel_spatial_dims: Vec<i64>,
    feature_group_count: i64,
    batch_group_count: i64
) -> PyResult<XlaOp> {
    let window_strides = window_strides.as_slice();
    let padding = padding.as_slice();
    let lhs_dilations = lhs_dilations.as_slice();
    let rhs_dilations = rhs_dilations.as_slice();
    let input_spatial_dims = input_spatial_dims.as_slice();
    let output_spatial_dims = output_spatial_dims.as_slice();
    let kernel_spatial_dims = kernel_spatial_dims.as_slice();
    let y = x.conv_general_dilated(
        rhs,
        window_strides,
        padding,
        lhs_dilations,
        rhs_dilations,
        &input_batch_dim,
        &input_feature_dim,
        input_spatial_dims,
        &output_batch_dim,
        &output_feature_dim,
        output_spatial_dims,
        &kernel_input_feature_dim,
        &kernel_output_feature_dim,
        kernel_spatial_dims,
        feature_group_count,
        batch_group_count,
    )?;
    Ok(y)
}

#[pyfunction]
fn batch_norm_inference(
    x: XlaOp,
    scale: &XlaOp,
    offset: &XlaOp,
    mean: &XlaOp,
    variance: &XlaOp,
    epsilon: f32,
    feature_index: i64,
) -> PyResult<XlaOp> {
    let y = x.batch_norm_inference(
        scale, offset, mean, variance, epsilon, feature_index
    )?;
    Ok(y)
}

#[pyfunction]
fn dot_general(
    x: XlaOp,
    rhs: &XlaOp,
    lhs_contracting_dims: Vec<i64>,
    rhs_contracting_dims: Vec<i64>,
    lhs_batch_dims: Vec<i64>,
    rhs_batch_dims: Vec<i64>,
) -> PyResult<XlaOp> {
    let lhs_contracting_dims = lhs_contracting_dims.as_slice();
    let rhs_contracting_dims = rhs_contracting_dims.as_slice();
    let lhs_batch_dims = lhs_batch_dims.as_slice();
    let rhs_batch_dims = rhs_batch_dims.as_slice();
    let y = x.dot_general(
        rhs,
        lhs_contracting_dims,
        rhs_contracting_dims,
        lhs_batch_dims,
        rhs_batch_dims
    )?;
    Ok(y)
}

#[pyfunction]
fn gather(
    x: XlaOp,
    start_indices: &XlaOp,
    offset_dims: Vec<i64>,
    collapsed_slice_dims: Vec<i64>,
    start_index_map: Vec<i64>,
    slice_sizes: Vec<i64>,
    set_index_vector_dim: Option<i64>,
) -> PyResult<XlaOp> {
    let offset_dims = offset_dims.as_slice();
    let collapsed_slice_dims = collapsed_slice_dims.as_slice();
    let start_index_map = start_index_map.as_slice();
    let slice_sizes = slice_sizes.as_slice();
    let y = x.gather(
        start_indices,
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        set_index_vector_dim,
        slice_sizes,
    )?;
    Ok(y)
}

#[pyfunction]
fn scatter(
    operands: Vec<XlaOp>,
    scatter_indices: &XlaOp,
    updates: Vec<XlaOp>,
    update_computation: &XlaComputation,
    update_window_dims: Vec<i64>,
    inserted_window_dims: Vec<i64>,
    scatter_dims_to_operand_dims: Vec<i64>,
    index_vector_dim: i64
) -> PyResult<XlaOp> {
    let operands = operands.as_slice();
    let updates = updates.as_slice();
    let update_window_dims = update_window_dims.as_slice();
    let inserted_window_dims = inserted_window_dims.as_slice();
    let scatter_dims_to_operand_dims = scatter_dims_to_operand_dims.as_slice();
    let y = XlaOp::scatter(
        operands,
        scatter_indices,
        updates,
        update_computation,
        update_window_dims,
        inserted_window_dims,
        scatter_dims_to_operand_dims,
        index_vector_dim
    )?;
    Ok(y)
}

#[pyfunction]
fn take(x: XlaOp, indices: &XlaOp, axis: i64) -> PyResult<XlaOp> {
    let y = x.take(indices, axis)?;
    Ok(y)
}

#[pyfunction]
fn reduce_sum(x: XlaOp, dims: Vec<i64>, keep_dims: bool) -> PyResult<XlaOp> {
    let dims = dims.as_slice();
    let y = x.reduce_sum(dims, keep_dims)?;
    Ok(y)
}

#[pyfunction]
fn reduce_mean(x: XlaOp, dims: Vec<i64>, keep_dims: bool) -> PyResult<XlaOp> {
    let dims = dims.as_slice();
    let y = x.reduce_mean(dims, keep_dims)?;
    Ok(y)
}

#[pyfunction]
fn reduce_max(x: XlaOp, dims: Vec<i64>, keep_dims: bool) -> PyResult<XlaOp> {
    let dims = dims.as_slice();
    let y = x.reduce_max(dims, keep_dims)?;
    Ok(y)
}

#[pyfunction]
fn reduce_min(x: XlaOp, dims: Vec<i64>, keep_dims: bool) -> PyResult<XlaOp> {
    let dims = dims.as_slice();
    let y = x.reduce_min(dims, keep_dims)?;
    Ok(y)
}

#[pyfunction]
fn softmax(x: XlaOp, axis: i64) -> PyResult<XlaOp> {
    let y = x.softmax(axis)?;
    Ok(y)
}

#[pyfunction]
fn layer_norm(x: XlaOp, dims: Vec<i64>, scale: &XlaOp, bias: &XlaOp, eps: f64) -> PyResult<XlaOp> {
    let dims = dims.as_slice();
    let y = x.layer_norm(dims, scale, bias, eps)?;
    Ok(y)
}

#[pyfunction]
fn primitive_type(x: XlaOp) -> PyResult<PrimitiveType> {
    let prim_type = x.primitive_type()?;
    Ok(prim_type)
}

#[pyfunction]
fn element_type(x: XlaOp) -> PyResult<ElementType> {
    let elem_type = PrimitiveType::element_type(x.ty()?)?;
    Ok(elem_type)
}

#[pyfunction]
fn dims(x: XlaOp) -> PyResult<Vec<usize>> {
    let dims = x.dims()?;
    Ok(dims)
}

#[pyfunction]
fn rank(x: XlaOp) -> PyResult<usize> {
    let rank = x.rank()?;
    Ok(rank)
}

#[pyfunction]
fn shape(x: XlaOp) -> PyResult<Vec<usize>> {
    let shape = x.shape()?;
    let shape = ArrayShape::try_from(&shape)?;
    let shape: Vec<usize> = shape.dims().iter().map(|&x| x as usize).collect();
    Ok(shape)
}

#[pyfunction]
fn array_shape(x: XlaOp) -> PyResult<ArrayShape> {
    let shape = x.array_shape()?;
    Ok(shape)
}

#[pyfunction]
fn create_array_shape(ty: ElementType, dims: Vec<i64>) -> PyResult<ArrayShape> {
    let shape = ArrayShape::new_with_type(ty, dims);
    Ok(shape)
}

#[pyfunction]
fn last_dim(x: XlaOp) -> PyResult<i64> {
    let shape = x.shape()?;
    let shape = ArrayShape::try_from(&shape)?;
    let last_dim = shape.last_dim().ok_or_else(|| PyErr::new::<exceptions::PyValueError, _>("Shape has no dimensions"))?;
    Ok(last_dim)
}

#[pyfunction]
fn tuple(builder: XlaBuilder, args: Vec<XlaOp>) -> PyResult<XlaOp> {
    let y = builder.tuple(&args)?;
    Ok(y)
}

#[pyfunction]
fn get_builder(x: XlaOp) -> PyResult<XlaBuilder> {
    let b = Rc::new(x.builder().clone());
    match Rc::try_unwrap(b) {
        Ok(builder) => Ok(builder),
        Err(_) => Err(PyErr::new::<exceptions::PyException, _>("Could not unwrap XlaBuilder")),
    }
}


#[pymodule]
#[pyo3(name="xlar")]
fn module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xla_builder, m)?)?;
    m.add_function(wrap_pyfunction!(constant_array, m)?)?;
    m.add_function(wrap_pyfunction!(gather_params, m)?)?;
    m.add_function(wrap_pyfunction!(swap_param, m)?)?;
    m.add_function(wrap_pyfunction!(new_input, m)?)?;
    m.add_function(wrap_pyfunction!(create_bf16_array, m)?)?;
    m.add_function(wrap_pyfunction!(create_f16_array, m)?)?;
    m.add_function(wrap_pyfunction!(to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(to_tuple, m)?)?;
    m.add_function(wrap_pyfunction!(param_pred, m)?)?;
    m.add_function(wrap_pyfunction!(param_i8, m)?)?;
    m.add_function(wrap_pyfunction!(param_i16, m)?)?;
    m.add_function(wrap_pyfunction!(param_i32, m)?)?;
    m.add_function(wrap_pyfunction!(param_i64, m)?)?;
    m.add_function(wrap_pyfunction!(param_u8, m)?)?;
    m.add_function(wrap_pyfunction!(param_u16, m)?)?;
    m.add_function(wrap_pyfunction!(param_u32, m)?)?;
    m.add_function(wrap_pyfunction!(param_u64, m)?)?;
    m.add_function(wrap_pyfunction!(param_bf16, m)?)?;
    m.add_function(wrap_pyfunction!(param_f16, m)?)?;
    m.add_function(wrap_pyfunction!(param_f32, m)?)?;
    m.add_function(wrap_pyfunction!(param_f64, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_client, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_client, m)?)?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add_function(wrap_pyfunction!(get_hlo_proto, m)?)?;
    m.add_function(wrap_pyfunction!(hlo_module_from_proto, m)?)?;
    m.add_function(wrap_pyfunction!(hlo_module_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(get_hlo_module_entry_computation, m)?)?;
    m.add_function(wrap_pyfunction!(computation_count, m)?)?;
    m.add_function(wrap_pyfunction!(instruction_count, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    m.add_function(wrap_pyfunction!(to_literal, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(rem, m)?)?;
    m.add_function(wrap_pyfunction!(pow, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(_and, m)?)?;
    m.add_function(wrap_pyfunction!(_or, m)?)?;
    m.add_function(wrap_pyfunction!(xor, m)?)?;
    m.add_function(wrap_pyfunction!(eq, m)?)?;
    m.add_function(wrap_pyfunction!(ne, m)?)?;
    m.add_function(wrap_pyfunction!(ge, m)?)?;
    m.add_function(wrap_pyfunction!(gt, m)?)?;
    m.add_function(wrap_pyfunction!(le, m)?)?;
    m.add_function(wrap_pyfunction!(lt, m)?)?;
    m.add_function(wrap_pyfunction!(lshift, m)?)?;
    m.add_function(wrap_pyfunction!(rshift, m)?)?;
    m.add_function(wrap_pyfunction!(atan2, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(population_count, m)?)?;
    m.add_function(wrap_pyfunction!(_not, m)?)?;
    m.add_function(wrap_pyfunction!(neg, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(round, m)?)?;
    m.add_function(wrap_pyfunction!(round_nearest_even, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(expm1, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(log1p, m)?)?;
    m.add_function(wrap_pyfunction!(logistic, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(clz, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(real, m)?)?;
    m.add_function(wrap_pyfunction!(imag, m)?)?;
    m.add_function(wrap_pyfunction!(conj, m)?)?;
    m.add_function(wrap_pyfunction!(square, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(rsqrt, m)?)?;
    m.add_function(wrap_pyfunction!(cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(upper_triangle, m)?)?;
    m.add_function(wrap_pyfunction!(lower_triangle, m)?)?;
    m.add_function(wrap_pyfunction!(erf, m)?)?;
    m.add_function(wrap_pyfunction!(is_finite, m)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(copy, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(silu, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu_approx, m)?)?;
    m.add_function(wrap_pyfunction!(einsum1, m)?)?;
    m.add_function(wrap_pyfunction!(einsum2, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_reshape, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_in_dim, m)?)?;
    m.add_function(wrap_pyfunction!(collapse, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(swap_dims, m)?)?;
    m.add_function(wrap_pyfunction!(pad, m)?)?;
    m.add_function(wrap_pyfunction!(pad_in_dim, m)?)?;
    m.add_function(wrap_pyfunction!(slice, m)?)?;
    m.add_function(wrap_pyfunction!(slice_in_dim, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_slice, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_update_slice, m)?)?;
    m.add_function(wrap_pyfunction!(at, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;
    m.add_function(wrap_pyfunction!(concat, m)?)?;
    m.add_function(wrap_pyfunction!(get_tuple_element, m)?)?;
    m.add_function(wrap_pyfunction!(rng_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(rng_normal, m)?)?;
    m.add_function(wrap_pyfunction!(astype, m)?)?;
    m.add_function(wrap_pyfunction!(dimension_size, m)?)?;
    m.add_function(wrap_pyfunction!(reduce, m)?)?;
    m.add_function(wrap_pyfunction!(call, m)?)?;
    m.add_function(wrap_pyfunction!(map, m)?)?;
    m.add_function(wrap_pyfunction!(select, m)?)?;
    m.add_function(wrap_pyfunction!(while_loop, m)?)?;
    m.add_function(wrap_pyfunction!(conditional, m)?)?;
    m.add_function(wrap_pyfunction!(conv, m)?)?;
    m.add_function(wrap_pyfunction!(conv_general_dilated, m)?)?;
    m.add_function(wrap_pyfunction!(batch_norm_inference, m)?)?;
    m.add_function(wrap_pyfunction!(dot_general, m)?)?;
    m.add_function(wrap_pyfunction!(gather, m)?)?;
    m.add_function(wrap_pyfunction!(scatter, m)?)?;
    m.add_function(wrap_pyfunction!(take, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_sum, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_mean, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_max, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_min, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(primitive_type, m)?)?;
    m.add_function(wrap_pyfunction!(element_type, m)?)?;
    m.add_function(wrap_pyfunction!(rank, m)?)?;
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(array_shape, m)?)?;
    m.add_function(wrap_pyfunction!(dims, m)?)?;
    m.add_function(wrap_pyfunction!(last_dim, m)?)?;
    m.add_function(wrap_pyfunction!(tuple, m)?)?;
    m.add_function(wrap_pyfunction!(get_builder, m)?)?;
    m.add_function(wrap_pyfunction!(constant_array, m)?)?;
    m.add_function(wrap_pyfunction!(create_array_shape, m)?)?;
    m.add_function(wrap_pyfunction!(constant_i32, m)?)?;
    m.add_function(wrap_pyfunction!(constant_bool, m)?)?;
    m.add_function(wrap_pyfunction!(constant_i8, m)?)?;
    m.add_function(wrap_pyfunction!(constant_i16, m)?)?;
    m.add_function(wrap_pyfunction!(constant_i32, m)?)?;
    m.add_function(wrap_pyfunction!(constant_i64, m)?)?;
    m.add_function(wrap_pyfunction!(constant_u8, m)?)?;
    m.add_function(wrap_pyfunction!(constant_u16, m)?)?;
    m.add_function(wrap_pyfunction!(constant_u32, m)?)?;
    m.add_function(wrap_pyfunction!(constant_u64, m)?)?;
    m.add_function(wrap_pyfunction!(constant_f32, m)?)?;
    m.add_function(wrap_pyfunction!(constant_f64, m)?)?;
    m.add_function(wrap_pyfunction!(astype_bool, m)?)?;
    m.add_function(wrap_pyfunction!(astype_i8, m)?)?;
    m.add_function(wrap_pyfunction!(astype_i16, m)?)?;
    m.add_function(wrap_pyfunction!(astype_i32, m)?)?;
    m.add_function(wrap_pyfunction!(astype_i64, m)?)?;
    m.add_function(wrap_pyfunction!(astype_u8, m)?)?;
    m.add_function(wrap_pyfunction!(astype_u16, m)?)?;
    m.add_function(wrap_pyfunction!(astype_u32, m)?)?;
    m.add_function(wrap_pyfunction!(astype_u64, m)?)?;
    m.add_function(wrap_pyfunction!(astype_bf16, m)?)?;
    m.add_function(wrap_pyfunction!(astype_f16, m)?)?;
    m.add_function(wrap_pyfunction!(astype_f32, m)?)?;
    m.add_function(wrap_pyfunction!(astype_f64, m)?)?;
    Ok(())
}
