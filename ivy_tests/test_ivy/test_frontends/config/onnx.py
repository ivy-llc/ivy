import onnx
import onnx.helper as h
import numpy as np

# ToDo check device support again when onnx 1.15.0 releases
valid_devices = "cpu"
invalid_devices = ("gpu", "tpu")

valid_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
]
invalid_dtypes = ["bfloat16"]

valid_numeric_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
invalid_numeric_dtypes = ["bfloat16"]

valid_int_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
invalid_int_dtypes = []

valid_uint_dtypes = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
invalid_uint_dtypes = []

valid_float_dtypes = [
    "float16",
    "float32",
    "float64",
]
invalid_float_dtypes = ["bfloat16"]

valid_complex_dtypes = [
    "complex64",
    "complex128",
]
invalid_complex_dtypes = []


# Helpers for function testing


Dtype = int
Device = str


def native_array(x, name="native_x"):
    dtype = h.np_dtype_to_tensor_dtype(x.dtype)
    return h.make_tensor(name=name, vals=x, data_type=dtype, dims=x.shape)


def is_native_array(x):
    return isinstance(x, onnx.TensorProto)


def to_numpy(x):
    return onnx.numpy_helper.to_array(x)


def as_native_dtype(dtype: str):
    return h.np_dtype_to_tensor_dtype(np.dtype(dtype))


def as_native_dev(device: str):
    # ToDo: uncomment when onnx 1.15.0 is officially released
    # device = onnx.backend.base.Device(device)
    return device


def isscalar(x):
    return np.isscalar(x)
