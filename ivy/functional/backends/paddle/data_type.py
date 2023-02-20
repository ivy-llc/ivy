# global
from typing import Optional, Union, Sequence, List

import paddle

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.ivy.data_type import _handle_nestable_dtype_info
from . import backend_version
from ivy.exceptions import IvyNotImplementedException


ivy_dtype_dict = {
    paddle.int8: "int8",
    paddle.int16: "int16",
    paddle.int32: "int32",
    paddle.int64: "int64",
    paddle.uint8: "uint8",
    paddle.bfloat16: "bfloat16",
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
    paddle.complex64: "complex64",
    paddle.complex128: "complex128",
    paddle.bool: "bool",
}

native_dtype_dict = {
    "int8": paddle.int8,
    "int16": paddle.int16,
    "int32": paddle.int32,
    "int64": paddle.int64,
    "uint8": paddle.uint8,
    "bfloat16": paddle.bfloat16,
    "float16": paddle.float16,
    "float32": paddle.float32,
    "float64": paddle.float64,
    "complex64": paddle.complex64,
    "complex128": paddle.complex128,
    "bool": paddle.bool,
}


class Finfo:
    def __init__(self, paddle_finfo: None):
        self._paddle_finfo = paddle_finfo

    def __repr__(self):
        return repr(self._paddle_finfo)

    @property
    def bits(self):
        return self._paddle_finfo.bits

    @property
    def eps(self):
        return self._paddle_finfo.eps

    @property
    def max(self):
        return self._paddle_finfo.max

    @property
    def min(self):
        return self._paddle_finfo.min

    @property
    def smallest_normal(self):
        return self._paddle_finfo.tiny


class Iinfo:
    def __init__(self, paddle_iinfo: None):
        self._paddle_iinfo = paddle_iinfo

    def __repr__(self):
        return repr(self._paddle_iinfo)

    @property
    def bits(self):
        return self._paddle_iinfo.bits

    @property
    def max(self):
        return self._paddle_iinfo.max

    @property
    def min(self):
        return self._paddle_iinfo.min


# Array API Standard #
# -------------------#


def astype(
    x: paddle.Tensor,
    dtype: paddle.dtype,
    /,
    *,
    copy: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def broadcast_arrays(*arrays: paddle.Tensor) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def broadcast_to(
    x: paddle.Tensor,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def finfo(type: Union[paddle.dtype, str, paddle.Tensor], /) -> Finfo:
    raise IvyNotImplementedException()


def iinfo(type: Union[paddle.dtype, str, paddle.Tensor], /) -> Iinfo:
    raise IvyNotImplementedException()


def result_type(*arrays_and_dtypes: Union[paddle.Tensor, paddle.dtype]) -> ivy.Dtype:
    raise IvyNotImplementedException()


# Extra #
# ------#


def as_ivy_dtype(dtype_in: Union[paddle.dtype, str, bool, int, float], /) -> ivy.Dtype:
    raise IvyNotImplementedException()


def as_native_dtype(dtype_in: Union[paddle.dtype, str, bool, int, float]) -> paddle.dtype:
    raise IvyNotImplementedException()


def dtype(x: paddle.Tensor, *, as_native: bool = False) -> ivy.Dtype:
    raise IvyNotImplementedException()


def dtype_bits(dtype_in: Union[paddle.dtype, str], /) -> int:
    raise IvyNotImplementedException()
