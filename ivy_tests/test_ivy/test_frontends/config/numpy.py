import numpy as np

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


Dtype = np.dtype
Device = str


def native_array(x):
    return np.array(x)


def is_native_array(x):
    return isinstance(x, np.ndarray)


def to_numpy(x):
    return x


def as_native_dtype(dtype: str):
    return np.dtype(dtype)


def as_native_dev(device: str):
    return device


def isscalar(x):
    return np.isscalar(x)
