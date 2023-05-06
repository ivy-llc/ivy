import mxnet as mx
from typing import Optional, Union, Sequence, List
import numpy as np
import ivy
from ivy.functional.ivy.data_type import _handle_nestable_dtype_info

ivy_dtype_dict = {
    None: "bool",
}
native_dtype_dict = {
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
}

char_rep_dtype_dict = {
    "?": "bool",
    "i": int,
    "i1": "int8",
    "i4": "int32",
    "i8": "int64",
    "f": float,
    "f2": "float16",
    "f4": "float32",
    "f8": "float64",
    "u1": "uint8",
}


class Finfo:
    def __init__(self, mx_finfo: mx.np.finfo):
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


class Bfloat16Finfo:
    def __init__(self, mx_finfo: mx.np.finfo):
        self._mx_finfo = mx_finfo

    def __repr__(self):
        return repr(self._mx_finfo)


def astype(
    x: Union[(None, mx.ndarray.NDArray)],
    dtype: Union[(None, str)],
    /,
    *,
    copy: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.astype Not Implemented")


def broadcast_arrays(
    *arrays: Union[(None, mx.ndarray.NDArray)]
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.broadcast_arrays Not Implemented")


def broadcast_to(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    shape: Union[(ivy.NativeShape, Sequence[int])],
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.broadcast_to Not Implemented")


@_handle_nestable_dtype_info
def finfo(type: Union[str, mx.ndarray.NDArray, np.dtype], /) -> Finfo:
    if isinstance(type, mx.ndarray.NDArray):
        type = type.dtype
    return Finfo(mx.np.finfo(ivy.as_native_dtype(type)))


@_handle_nestable_dtype_info
def iinfo(type: Union[str, mx.ndarray.NDArray, np.dtype], /) -> np.iinfo:
    # using np.iinfo as mx use np dtypes and mxnet iinfo not provided
    if isinstance(type, mx.ndarray.NDArray):
        type = type.asnumpy().dtype
    return np.iinfo(ivy.as_native_dtype(type))


def result_type(
    *arrays_and_dtypes: Union[(None, mx.ndarray.NDArray, None)]
) -> ivy.Dtype:
    raise NotImplementedError("mxnet.result_type Not Implemented")


def as_ivy_dtype(
    dtype_in: Union[(None, str, int, float, complex, bool, np.dtype)], /
) -> ivy.Dtype:
    raise NotImplementedError("mxnet.as_ivy_dtype Not Implemented")


def as_native_dtype(dtype_in: Union[(None, str, bool, int, float, np.dtype)]) -> None:
    if dtype_in is int:
        return ivy.default_int_dtype(as_native=True)
    if dtype_in is float:
        return ivy.default_float_dtype(as_native=True)
    if dtype_in is complex:
        return ivy.default_complex_dtype(as_native=True)
    if dtype_in is bool:
        return np.dtype("bool")
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in char_rep_dtype_dict:
        return as_native_dtype(char_rep_dtype_dict[dtype_in])
    if dtype_in in native_dtype_dict:
        return native_dtype_dict[ivy.Dtype(dtype_in)]
    else:
        raise ivy.utils.exceptions.IvyException(
            f"Cannot convert to numpy dtype. {dtype_in} is not supported by NumPy."
        )


def dtype(
    x: Union[(None, mx.ndarray.NDArray, np.ndarray)], *, as_native: bool = False
) -> ivy.Dtype:
    raise NotImplementedError("mxnet.dtype Not Implemented")


def dtype_bits(dtype_in: Union[(None, str, np.dtype)], /) -> int:
    raise NotImplementedError("mxnet.dtype_bits Not Implemented")


def is_native_dtype(dtype_in: Union[(None, str)], /) -> bool:
    raise NotImplementedError("mxnet.is_native_dtype Not Implemented")
