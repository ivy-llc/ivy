# global
import numpy as np
import mxnet as mx
from typing import Union, Sequence, Optional

# local
import ivy
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out


ivy_dtype_dict = {
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    "bfloat16": "bfloat16",
    np.dtype("float16"): "float16",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("bool"): "bool",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.bool_: "bool",
}

native_dtype_dict = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bfloat16": "bfloat16",
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
}


class Finfo:
    def __init__(self, mx_finfo: np.finfo):
        self._mx_finfo = mx_finfo

    def __repr__(self):
        return repr(self._mx_finfo)

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


def iinfo(type: Union[type, str, mx.nd.NDArray]) -> np.iinfo:
    return np.iinfo(ivy.as_native_dtype(type))


def finfo(type: Union[type, str, mx.nd.NDArray]) -> Finfo:
    return Finfo(np.finfo(ivy.as_native_dtype(type)))


def broadcast_to(
    x: mx.nd.NDArray,
    new_shape: Union[ivy.NativeShape, Sequence[int]],
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    num_shape_dims = len(new_shape)
    diff = num_shape_dims - num_x_dims
    if diff == 0:
        return mx.nd.broadcast_to(x, new_shape)
    x = mx.nd.reshape(x, [1] * diff + x_shape)
    return mx.nd.broadcast_to(x, new_shape, out=out)


@_handle_flat_arrays_in_out
def astype(x: mx.nd.NDArray, dtype: type) -> mx.nd.NDArray:
    dtype = ivy.as_native_dtype(dtype)
    return x.astype(dtype)


def dtype_bits(dtype_in: Union[type, str]) -> int:
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


def dtype(x: mx.nd.NDArray, as_native: bool = False) -> ivy.Dtype:
    dt = x.dtype
    if as_native:
        return x.dtype
    return as_ivy_dtype(dt)


def as_ivy_dtype(dtype_in: Union[type, str]) -> ivy.Dtype:
    if isinstance(dtype_in, str):
        return ivy.Dtype(dtype_in)
    return ivy.Dtype(ivy_dtype_dict[dtype_in])


def as_native_dtype(dtype_in: Union[type, str]) -> type:
    if not isinstance(dtype_in, str):
        return dtype_in
    return native_dtype_dict[ivy.Dtype(dtype_in)]
