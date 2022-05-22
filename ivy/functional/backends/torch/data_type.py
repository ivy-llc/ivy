# global
import torch
from typing import Union, Tuple, List

# local
import ivy


def can_cast(from_: Union[torch.dtype, torch.Tensor], to: torch.dtype) -> bool:
    if isinstance(from_, torch.Tensor):
        from_ = from_.dtype
    from_str = str(from_)
    to_str = str(to)
    if ivy.dtype_bits(to) < ivy.dtype_bits(from_):
        return False
    if ".int" in from_str and "uint" in to_str:
        return False
    if "uint" in from_str and ".int" in to_str:
        if ivy.dtype_bits(to) <= ivy.dtype_bits(from_):
            return False
        else:
            return True
    if "bool" in from_str and (("int" in to_str) or ("float" in to_str)):
        return False
    if "int" in from_str and "float" in to_str:
        return False
    return torch.can_cast(from_, to)


# noinspection PyShadowingBuiltins
def iinfo(type: Union[torch.dtype, str, torch.Tensor]) -> torch.iinfo:
    return torch.iinfo(ivy.as_native_dtype(type))


class Finfo:
    def __init__(self, torch_finfo):
        self._torch_finfo = torch_finfo

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


# noinspection PyShadowingBuiltins
def finfo(type: Union[torch.dtype, str, torch.Tensor]) -> Finfo:
    return Finfo(torch.finfo(ivy.as_native_dtype(type)))


def result_type(*arrays_and_dtypes: Union[torch.tensor, torch.dtype]) -> torch.dtype:
    arrays_and_dtypes = list(arrays_and_dtypes)
    for i in range(len(arrays_and_dtypes)):
        if type(arrays_and_dtypes[i]) == torch.dtype:
            arrays_and_dtypes[i] = torch.tensor([], dtype=arrays_and_dtypes[i])
    if len(arrays_and_dtypes) == 1:
        return arrays_and_dtypes[0].dtype
    result = torch.result_type(arrays_and_dtypes[0], arrays_and_dtypes[1])
    for i in range(2, len(arrays_and_dtypes)):
        result = torch.result_type(torch.tensor([], dtype=result), arrays_and_dtypes[i])
    return result


def broadcast_to(x: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    return torch.broadcast_to(x, shape)


def broadcast_arrays(*arrays: torch.Tensor) -> List[torch.Tensor]:
    return torch.broadcast_tensors(*arrays)


def astype(x: torch.Tensor, dtype: torch.dtype, copy: bool = True) -> torch.Tensor:
    if isinstance(dtype, str):
        dtype = ivy.as_native_dtype(dtype)
    if copy:
        if x.dtype == dtype:
            new_tensor = x.clone().detach()
            return new_tensor
    else:
        if x.dtype == dtype:
            return x
        else:
            new_tensor = x.clone().detach()
            return new_tensor.to(dtype)
    return x.to(dtype)


def dtype_bits(dtype_in):
    dtype_str = as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("torch.", "")
        .replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
    )


def dtype(x, as_native=False):
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def as_ivy_dtype(dtype_in):
    if isinstance(dtype_in, str):
        return ivy.Dtype(dtype_in)
    return ivy.Dtype(
        {
            torch.int8: "int8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.uint8: "uint8",
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
            torch.float64: "float64",
            torch.bool: "bool",
        }[dtype_in]
    )


def as_native_dtype(dtype_in: str) -> torch.dtype:
    if not isinstance(dtype_in, str):
        return dtype_in
    return {
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bool": torch.bool,
    }[ivy.Dtype(dtype_in)]
