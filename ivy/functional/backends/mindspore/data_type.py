# global
from typing import Optional, Union, Sequence, List

import mindspore as ms
import mindspore.numpy as np
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import Type

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.ivy.data_type import _handle_nestable_dtype_info
from . import backend_version

ivy_dtype_dict = {
    ms.int8: "int8",
    ms.int16: "int16",
    ms.int32: "int32",
    ms.int64: "int64",
    ms.uint8: "uint8",
    ms.uint16: "uint16",
    ms.uint32: "uint32",
    ms.uint64: "uint64",
    ms.float16: "float16",
    ms.float32: "float32",
    ms.float64: "float64",
    ms.complex64: "complex64",
    ms.complex128: "complex128",
    ms.bool_: "bool",
}

native_dtype_dict = {
    "int8": ms.int8,
    "int16": ms.int16,
    "int32": ms.int32,
    "int64": ms.int64,
    "uint8": ms.uint8,
    "uint16": ms.uint16,
    "uint32": ms.uint32,
    "uint64": ms.uint64,
    "float16": ms.float16,
    "float32": ms.float32,
    "float64": ms.float64,
    "complex64": ms.complex64,
    "complex128": ms.complex128,
    "bool": ms.bool_
}


class Finfo:
    def __init__(self, ms_finfo: None):
        self._ms_finfo = ms_finfo

    def __repr__(self):
        return repr(self._ms_finfo)

    @property
    def bits(self):
        return self._ms_finfo.bits

    @property
    def eps(self):
        return float(self._ms_finfo.eps)

    @property
    def max(self):
        return float(self._ms_finfo.max)

    @property
    def min(self):
        return float(self._ms_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._ms_finfo.tiny)


class Bfloat16Finfo:
    def __init__(self):
        self.resolution = 0.01
        self.bits = 16
        self.eps = 0.0078125
        self.max = 3.38953e38
        self.min = -3.38953e38
        self.tiny = 1.17549e-38

    def __repr__(self):
        return "finfo(resolution={}, min={}, max={}, dtype={})".format(
            self.resolution, self.min, self.max, "bfloat16"
        )


# Array API Standard #
# -------------------#


class Iinfo:
    def __init__(self, mindspore_iinfo: None):
        self._paddle_iinfo = mindspore_iinfo

    def __repr__(self):
        return repr(self.mindspore_iinfo)

    @property
    def bits(self):
        return self.mindspore_iinfo.bits

    @property
    def max(self):
        return self.mindspore_iinfo.max

    @property
    def min(self):
        return self.mindspore_iinfo.min

def astype(
    x: ms.Tensor,
    dtype: Type,
    /,
    *,
    copy: bool = True,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if x.dtype == dtype:
        return np.copy(x) if copy else x
    return F.cast(x, dtype)


def broadcast_arrays(*arrays: ms.Tensor) -> List[ms.Tensor]:
    return np.broadcast_arrays(*arrays)

@with_unsupported_dtypes({"2.0.0a0 and below": ("bfloat16",)}, backend_version)
def broadcast_to(
    x: ms.Tensor,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if x.ndim > len(shape):
        return np.broadcast_to(x.reshape(-1), shape)
    return np.broadcast_to(x, shape)

def finfo(type: Union[Type, str, ms.Tensor], /) -> Finfo:
    raise IvyNotImplementedException()


def iinfo(type: Union[Type, str, ms.Tensor], /) -> Iinfo:
    print('failing here')
    raise IvyNotImplementedException()


@with_unsupported_dtypes({"2.0.0a0 and below": ("bfloat16",)}, backend_version)
def result_type(
    *arrays_and_dtypes: Union[ms.Tensor, str, Type],
) -> ivy.Dtype:
    if len(arrays_and_dtypes) <= 1:
        return np.result_type(arrays_and_dtypes)

    result = np.result_type(
        arrays_and_dtypes[0], arrays_and_dtypes[1]
    )
    for i in range(2, len(arrays_and_dtypes)):
        result = np.result_type(result, arrays_and_dtypes[i])
    return as_ivy_dtype(result)

# Extra #
# ------#

@with_unsupported_dtypes({"2.0.0a0 and below": ("bfloat16",)}, backend_version)
def as_ivy_dtype(dtype_in: Union[Type, str, bool, int, float], /) -> ivy.Dtype:
    if dtype_in is int:
        return ivy.default_int_dtype()
    if dtype_in is float:
        return ivy.default_float_dtype()
    if dtype_in is complex:
        return ivy.default_complex_dtype()
    if dtype_in is bool:
        return ivy.Dtype("bool")
    if isinstance(dtype_in, str):
        if dtype_in in native_dtype_dict:
            return ivy.Dtype(dtype_in)
        else:
            raise ivy.utils.exceptions.IvyException(
                "Cannot convert to ivy dtype."
                f" {dtype_in} is not supported by mindspore backend."
            )
    return ivy.Dtype(ivy_dtype_dict[dtype_in])


@with_unsupported_dtypes({"2.0.0a0 and below": ("bfloat16",)}, backend_version)
def as_native_dtype(dtype_in: Union[Type, str, bool, int, float]) -> Type:
    if dtype_in is int:
        return ivy.default_int_dtype(as_native=True)
    if dtype_in is float:
        return ivy.default_float_dtype(as_native=True)
    if dtype_in is complex:
        return ivy.default_complex_dtype(as_native=True)
    if dtype_in is bool:
        return ms.bool_
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict.keys():
        return native_dtype_dict[ivy.Dtype(dtype_in)]
    else:
        raise ivy.utils.exceptions.IvyException(
            "Cannot convert to mindspore dtype."
            f" {dtype_in} is not supported by mindspore."
        )


def dtype(x: ms.Tensor, *, as_native: bool = False) -> ivy.Dtype:
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def dtype_bits(dtype_in: Union[Type, str], /) -> int:
    dtype_str = as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("mindspore.", "")
        .replace("uint", "")
        .replace("int", "")
        .replace("float", "")
        .replace("complex", "")
    )
