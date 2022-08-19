# global
import torch
from typing import Union, Sequence, List

# local
import ivy

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


def astype(x: torch.Tensor, dtype: torch.dtype, *, copy: bool = True) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
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


def broadcast_arrays(*arrays: torch.Tensor) -> List[torch.Tensor]:
    return list(torch.broadcast_tensors(*arrays))


def broadcast_to(
    x: torch.Tensor, shape: Union[ivy.NativeShape, Sequence[int]]
) -> torch.Tensor:
    return torch.broadcast_to(x, shape)


broadcast_to.unsupported_dtypes = ("uint8", "uint16", "uint32", "uint64")


def can_cast(from_: Union[torch.dtype, torch.Tensor], to: torch.dtype) -> bool:
    if isinstance(from_, torch.Tensor):
        from_ = from_.dtype
    from_str = str(from_)
    to_str = str(to)
    if ivy.dtype_bits(to) < ivy.dtype_bits(from_):
        return False
    if ("int" in from_str and "u" not in from_str) and "uint" in to_str:
        return False
    if "bool" in from_str and (("int" in to_str) or ("float" in to_str)):
        return False
    if "int" in from_str and (("float" in to_str) or ("bool" in to_str)):
        return False
    if "float" in from_str and "bool" in to_str:
        return False
    if "float" in from_str and "int" in to_str:
        return False
    if "uint" in from_str and ("int" in to_str and "u" not in to_str):
        if ivy.dtype_bits(to) <= ivy.dtype_bits(from_):
            return False
    return True


def finfo(type: Union[torch.dtype, str, torch.Tensor]) -> Finfo:
    if isinstance(type, torch.Tensor):
        type = type.dtype
    return Finfo(torch.finfo(ivy.as_native_dtype(type)))


def iinfo(type: Union[torch.dtype, str, torch.Tensor]) -> torch.iinfo:
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


def as_ivy_dtype(dtype_in: Union[torch.dtype, str]) -> ivy.Dtype:
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


def as_native_dtype(dtype_in: Union[torch.dtype, str]) -> torch.dtype:
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict.keys():
        return native_dtype_dict[ivy.Dtype(dtype_in)]
    else:
        raise TypeError(
            f"Cannot convert to PyTorch dtype. {dtype_in} is not supported by PyTorch."
        )


as_native_dtype.unsupported_dtypes = ("uint16",)


def dtype(x: torch.tensor, as_native: bool = False) -> ivy.Dtype:
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def dtype_bits(dtype_in: Union[torch.dtype, str]) -> int:
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
