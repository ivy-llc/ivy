# global
import numpy as np
import mxnet as mx
from typing import Union
import numpy as _np

# local
import ivy
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out


as_ivy_dtype = {
    _np.dtype("int8"): "int8",
    _np.dtype("int16"): "int16",
    _np.dtype("int32"): "int32",
    _np.dtype("int64"): "int64",
    _np.dtype("uint8"): "uint8",
    _np.dtype("uint16"): "uint16",
    _np.dtype("uint32"): "uint32",
    _np.dtype("uint64"): "uint64",
    "bfloat16": "bfloat16",
    _np.dtype("float16"): "float16",
    _np.dtype("float32"): "float32",
    _np.dtype("float64"): "float64",
    _np.dtype("bool"): "bool",
    _np.int8: "int8",
    _np.int16: "int16",
    _np.int32: "int32",
    _np.int64: "int64",
    _np.uint8: "uint8",
    _np.uint16: "uint16",
    _np.uint32: "uint32",
    _np.uint64: "uint64",
    _np.float16: "float16",
    _np.float32: "float32",
    _np.float64: "float64",
    _np.bool_: "bool",
}

as_native_dtype = {
    "int8": _np.int8,
    "int16": _np.int16,
    "int32": _np.int32,
    "int64": _np.int64,
    "uint8": _np.uint8,
    "uint16": _np.uint16,
    "uint32": _np.uint32,
    "uint64": _np.uint64,
    "bfloat16": "bfloat16",
    "float16": _np.float16,
    "float32": _np.float32,
    "float64": _np.float64,
    "bool": _np.bool_,
}


# noinspection PyShadowingBuiltins
def iinfo(type: Union[type, str, mx.ndarray.ndarray.NDArray]) -> np.iinfo:
    return np.iinfo(ivy.as_native_dtype(type))


class Finfo:
    def __init__(self, mx_finfo):
        self._mx_finfo = mx_finfo

    @property
    def bits(self):
        return self._mx_finfo.bits

    @property
    def eps(self):
        return float(self._mx_finfo.eps)

    @property
    def max(self):
        return float(self._mx_finfo.max)

    @property
    def min(self):
        return float(self._mx_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._mx_finfo.tiny)


# noinspection PyShadowingBuiltins
def finfo(type: Union[type, str, mx.ndarray.ndarray.NDArray]) -> Finfo:
    return Finfo(np.finfo(ivy.as_native_dtype(type)))


def broadcast_to(x, new_shape):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    num_shape_dims = len(new_shape)
    diff = num_shape_dims - num_x_dims
    if diff == 0:
        return mx.nd.broadcast_to(x, new_shape)
    x = mx.nd.reshape(x, [1] * diff + x_shape)
    return mx.nd.broadcast_to(x, new_shape)


@_handle_flat_arrays_in_out
def astype(x, dtype):
    return x.astype(dtype)


def dtype_bits(dtype_in):
    dtype_str = as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("<class 'numpy.", "")
        .replace("'>", "")
        .replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
    )


def dtype(x, as_str=False):
    dt = x.dtype
    if as_str:
        return as_ivy_dtype(dt)
    return x.dtype


def as_ivy_dtype(dtype_in):
    if isinstance(dtype_in, str):
        return ivy.Dtype(dtype_in)
    return ivy.Dtype(as_ivy_dtype[dtype_in])


def as_native_dtype(dtype_in):
    if not isinstance(dtype_in, str):
        return dtype_in
    return as_native_dtype[ivy.Dtype(dtype_in)]
