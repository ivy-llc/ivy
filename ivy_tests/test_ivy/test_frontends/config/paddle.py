import paddle
from paddle.device import core
import numpy as np

valid_devices = ("cpu", "gpu")
invalid_devices = ("tpu",)


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


Dtype = paddle.dtype
Device = core.Place


def native_array(x):
    return paddle.to_tensor(x)


def is_native_array(x):
    return isinstance(x, (paddle.Tensor, paddle.fluid.framework.EagerParamBase))


def to_numpy(x):
    return np.array(x)


def as_native_dtype(dtype: str):
    return paddle.to_tensor([], dtype=dtype).dtype


def as_native_dev(device: str):
    if isinstance(device, core.Place):
        return device
    native_dev = core.Place()
    if "cpu" in device:
        native_dev.set_place(paddle.device.core.CPUPlace())

    elif "gpu" in device:
        if ":" in device:
            gpu_idx = int(device.split(":")[-1])
        else:
            gpu_idx = 0
        native_dev.set_place(paddle.device.core.CUDAPlace(gpu_idx))
    return native_dev


def isscalar(x):
    return is_native_array(x) and x.ndim == 0
