import scipy
import numpy as np

valid_devices = "cpu"
invalid_devices = ("tpu", "gpu")

valid_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
]

invalid_dtypes = [
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
]

valid_numeric_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

invalid_numeric_dtypes = [
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
]

valid_int_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
]

invalid_int_dtypes = [
    "uint16",
    "uint32",
    "uint64",
]

valid_uint_dtypes = [
    "uint8",
]

invalid_uint_dtypes = [
    "uint16",
    "uint32",
    "uint64",
]

valid_float_dtypes = [
    "float16",
    "float32",
    "float64",
]

invalid_float_dtypes = [
    "bfloat16",
]

valid_complex_dtypes = [
    "complex64",
    "complex128",
]

invalid_complex_dtypes = []

# Helpers for function testing

Dtype = scipy.dtype
Device = str


def native_array(x):
    return scipy.array(x)


def is_native_array(x):
    return isinstance(x, scipy.ndarray)


def to_numpy(x):
    return x


def as_native_dtype(dtype: str):
    return scipy.dtype(dtype)


def as_native_dev(device: str):
    return device


def isscalar(x):
    return np.isscalar(x)
