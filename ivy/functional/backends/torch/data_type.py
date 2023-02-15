# global
from typing import Optional, Union, Sequence, List

import torch

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.ivy.data_type import _handle_nestable_dtype_info
from . import backend_version

ivy_dtype_dict = {
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.bool: "bool",
}

native_dtype_dict = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "bool": torch.bool,
}


class Finfo:
    def __init__(self, torch_finfo: torch.finfo):
        self._torch_finfo = torch_finfo

    def __repr__(self):
        return repr(self._torch_finfo)

    @property
    def bits(self):
        return self._torch_finfo.bits

    @property
    def eps(self):
        return self._torch_finfo.eps

    @property
    def max(self):
        return self._torch_finfo.max

    @property
    def min(self):
        return self._torch_finfo.min

    @property
    def smallest_normal(self):
        return self._torch_finfo.tiny


# Array API Standard #
# -------------------#


def astype(
    x: torch.Tensor,
    dtype: torch.dtype,
    /,
    *,
    copy: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if x.dtype == dtype:
        return x.clone() if copy else x
    return x.to(dtype)


def broadcast_arrays(*arrays: torch.Tensor) -> List[torch.Tensor]:
    return list(torch.broadcast_tensors(*arrays))


@with_unsupported_dtypes(
    {"1.11.0 and below": ("uint8", "uint16", "uint32", "uint64")}, backend_version
)
def broadcast_to(
    x: torch.Tensor,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.ndim > len(shape):
        return torch.broadcast_to(x.reshape(-1), shape)
    return torch.broadcast_to(x, shape)


@_handle_nestable_dtype_info
def finfo(type: Union[torch.dtype, str, torch.Tensor], /) -> Finfo:
    if isinstance(type, torch.Tensor):
        type = type.dtype
    return Finfo(torch.finfo(ivy.as_native_dtype(type)))


@_handle_nestable_dtype_info
def iinfo(type: Union[torch.dtype, str, torch.Tensor], /) -> torch.iinfo:
    if isinstance(type, torch.Tensor):
        type = type.dtype
    return torch.iinfo(ivy.as_native_dtype(type))


def result_type(*arrays_and_dtypes: Union[torch.tensor, torch.dtype]) -> ivy.Dtype:
    input = []
    for val in arrays_and_dtypes:
        torch_val = as_native_dtype(val)
        if isinstance(torch_val, torch.dtype):
            torch_val = torch.tensor(1, dtype=torch_val)
        input.append(torch_val)

    result = torch.tensor(1, dtype=torch.result_type(input[0], input[1]))

    for i in range(2, len(input)):
        result = torch.tensor(1, dtype=torch.result_type(result, input[i]))
    return as_ivy_dtype(result.dtype)


# Extra #
# ------#


def as_ivy_dtype(dtype_in: Union[torch.dtype, str, bool, int, float], /) -> ivy.Dtype:
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
            raise ivy.exceptions.IvyException(
                "Cannot convert to ivy dtype."
                f" {dtype_in} is not supported by PyTorch backend."
            )
    return ivy.Dtype(ivy_dtype_dict[dtype_in])


@with_unsupported_dtypes({"1.11.0 and below": ("uint16",)}, backend_version)
def as_native_dtype(dtype_in: Union[torch.dtype, str, bool, int, float]) -> torch.dtype:
    if dtype_in is int:
        return ivy.default_int_dtype(as_native=True)
    if dtype_in is float:
        return ivy.default_float_dtype(as_native=True)
    if dtype_in is complex:
        return ivy.default_complex_dtype(as_native=True)
    if dtype_in is bool:
        return torch.bool
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict.keys():
        return native_dtype_dict[ivy.Dtype(dtype_in)]
    else:
        raise ivy.exceptions.IvyException(
            "Cannot convert to PyTorch dtype."
            f" {dtype_in} is not supported by PyTorch."
        )


def dtype(x: torch.tensor, *, as_native: bool = False) -> ivy.Dtype:
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def dtype_bits(dtype_in: Union[torch.dtype, str], /) -> int:
    dtype_str = as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("torch.", "")
        .replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
        .replace("complex", "")
    )
