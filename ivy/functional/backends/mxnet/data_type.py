import mxnet as mx
from typing import Optional, Union, Sequence, List
import numpy as np
import ivy
from ivy.functional.ivy.data_type import _handle_nestable_dtype_info

ivy_dtype_dict = {
    None: "bool",
}
native_dtype_dict = {
    "int8": None,
    "int16": None,
    "int32": None,
    "int64": None,
    "uint8": None,
    "uint16": None,
    "uint32": None,
    "uint64": None,
    "bfloat16": None,
    "float16": None,
    "float32": None,
    "float64": None,
    "complex64": None,
    "complex128": None,
    "bool": None,
}


class Finfo:
    def __init__(self, mx_finfo: np.finfo):
        raise NotImplementedError("mxnet.__init__ Not Implemented")

    def __repr__(self):
        raise NotImplementedError("mxnet.__repr__ Not Implemented")

    @property
    def bits(self):
        raise NotImplementedError("mxnet.bits Not Implemented")

    @property
    def eps(self):
        raise NotImplementedError("mxnet.eps Not Implemented")

    @property
    def max(self):
        raise NotImplementedError("mxnet.max Not Implemented")

    @property
    def min(self):
        raise NotImplementedError("mxnet.min Not Implemented")

    @property
    def smallest_normal(self):
        raise NotImplementedError("mxnet.smallest_normal Not Implemented")


class Bfloat16Finfo:
    def __init__(self):
        raise NotImplementedError("mxnet.__init__ Not Implemented")

    def __repr__(self):
        raise NotImplementedError("mxnet.__repr__ Not Implemented")


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
def finfo(type: Union[(None, str, None, mx.ndarray.NDArray, np.ndarray)], /) -> Finfo:
    raise NotImplementedError("mxnet.finfo Not Implemented")


@_handle_nestable_dtype_info
def iinfo(
    type: Union[(None, str, None, mx.ndarray.NDArray, np.ndarray)], /
) -> np.iinfo:
    raise NotImplementedError("mxnet.iinfo Not Implemented")


def result_type(
    *arrays_and_dtypes: Union[(None, mx.ndarray.NDArray, None)]
) -> ivy.Dtype:
    raise NotImplementedError("mxnet.result_type Not Implemented")


def as_ivy_dtype(
    dtype_in: Union[(None, str, int, float, complex, bool, np.dtype)], /
) -> ivy.Dtype:
    raise NotImplementedError("mxnet.as_ivy_dtype Not Implemented")


def as_native_dtype(dtype_in: Union[(None, str, bool, int, float, np.dtype)]) -> None:
    raise NotImplementedError("mxnet.as_native_dtype Not Implemented")


def dtype(
    x: Union[(None, mx.ndarray.NDArray, np.ndarray)], *, as_native: bool = False
) -> ivy.Dtype:
    raise NotImplementedError("mxnet.dtype Not Implemented")


def dtype_bits(dtype_in: Union[(None, str, np.dtype)], /) -> int:
    raise NotImplementedError("mxnet.dtype_bits Not Implemented")


def is_native_dtype(dtype_in: Union[(None, str)], /) -> bool:
    raise NotImplementedError("mxnet.is_native_dtype Not Implemented")
