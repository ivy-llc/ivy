# global
torch_scatter = None
from typing import Union, Optional, Sequence

import torch

# local
import ivy
from ivy.functional.ivy.statistical import _get_promoted_type_of_operands
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Array API Standard #
# -------------------#


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def min(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis == ():
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        else:
            return x
    if not keepdims and not axis and axis != 0:
        return torch.amin(input=x, out=out)
    return torch.amin(input=x, dim=axis, keepdim=keepdims, out=out)


min.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def max(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis == ():
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        else:
            return x
    if not keepdims and not axis and axis != 0:
        return torch.amax(input=x, out=out)
    return torch.amax(input=x, dim=axis, keepdim=keepdims, out=out)


max.support_native_out = True


def mean(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if axis == () or axis == []:
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        else:
            return x
    return torch.mean(x, dim=axis, keepdim=keepdims, out=out)


mean.support_native_out = True


def _infer_dtype(dtype: torch.dtype) -> torch.dtype:
    default_dtype = ivy.infer_default_dtype(dtype)
    if default_dtype in ivy.valid_dtypes:
        if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
            return ivy.as_native_dtype(default_dtype)
    return ivy.as_native_dtype(dtype)


# Function does support uint8, but allowing support for unsigned will cause
# the function to break the upcasting rule defined in the Array API Standard
@with_unsupported_dtypes(
    {
        "2.0.1 and below": ("uint8", "float16", "bfloat16"),
    },
    backend_version,
)
def prod(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    if axis == ():
        return x.type(dtype)
    if axis is None:
        return torch.prod(input=x, dtype=dtype)
    if isinstance(axis, tuple) or isinstance(axis, list):
        for i in axis:
            x = torch.prod(x, i, keepdim=keepdims, dtype=dtype)
        return x
    return torch.prod(x, axis, keepdim=keepdims, dtype=dtype)


@with_unsupported_dtypes(
    {"2.0.1 and below": ("int8", "int16", "int32", "int64", "float16")},
    backend_version,
)
def std(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = list(range(len(x.shape)))
    if axis == ():
        return x
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if correction == 0:
        return torch.std(x, dim=axis, unbiased=False, keepdim=keepdims)
    elif correction == 1:
        return torch.std(x, dim=axis, unbiased=True, keepdim=keepdims)
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size - correction <= 0:
        ret = torch.std(x, dim=axis, unbiased=False, keepdim=keepdims)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    ret = torch.mul(
        torch.std(x, dim=axis, unbiased=False, keepdim=keepdims),
        (size / (size - correction)) ** 0.5,
    )
    return ret


# Function does support uint8, but allowing support for unsigned will cause
# the function to break the upcasting rule defined in the Array API Standard
@with_unsupported_dtypes({"2.0.1": ("uint8",)}, backend_version)
def sum(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None and not ivy.is_bool_dtype(x):
        dtype = x.dtype
    if axis == ():
        return x.type(dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    if axis is None:
        return torch.sum(input=x, dim=(), dtype=dtype, keepdim=keepdims)
    return torch.sum(input=x, dim=axis, dtype=dtype, keepdim=keepdims)


def var(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = list(range(len(x.shape)))
    if axis == ():
        return x
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if correction == 0:
        return torch.var(x, dim=axis, unbiased=False, keepdim=keepdims)
    elif correction == 1:
        return torch.var(x, dim=axis, unbiased=True, keepdim=keepdims)
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size - correction <= 0:
        ret = torch.var(x, dim=axis, unbiased=False, keepdim=keepdims)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    else:
        return torch.mul(
            torch.var(x, dim=axis, unbiased=False, keepdim=keepdims),
            (size / (size - correction)),
        ).to(x.dtype)


# Extra #
# ----- #


# Function does support uint8, but allowing support for unsigned will cause
# the function to break the upcasting rule defined in the Array API Standard
# TODO: bfloat16 support is added in PyTorch 1.12.1
@with_unsupported_dtypes(
    {
        "2.0.1 and below": ("uint8", "float16", "bfloat16"),
    },
    backend_version,
)
def cumprod(
    x: torch.Tensor,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)

    if not (exclusive or reverse):
        return torch.cumprod(x, axis, dtype=dtype, out=out)
    elif exclusive and reverse:
        x = torch.cumprod(torch.flip(x, dims=(axis,)), axis, dtype=dtype)
        x = torch.transpose(x, axis, -1)
        x = torch.concat((torch.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = torch.transpose(x, axis, -1)
        ret = torch.flip(x, dims=(axis,))
    elif exclusive:
        x = torch.transpose(x, axis, -1)
        x = torch.cat((torch.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = torch.cumprod(x, -1, dtype=dtype)
        ret = torch.transpose(x, axis, -1)
    else:
        x = torch.cumprod(torch.flip(x, dims=(axis,)), axis, dtype=dtype)
        ret = torch.flip(x, dims=(axis,))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


cumprod.support_native_out = True


# Function does support uint8, but allowing support for unsigned will cause
# the function to break the upcasting rule defined in the Array API Standard
# TODO: bfloat16 support is added in PyTorch 1.12.1
@with_unsupported_dtypes(
    {
        "1.12.1 and below": ("uint8", "float16", "bfloat16"),
        "1.12.1 and above": ("uint8", "float16"),
    },
    backend_version,
)
def cumsum(
    x: torch.Tensor,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        if ivy.is_int_dtype(x.dtype):
            dtype = ivy.promote_types(x.dtype, ivy.default_int_dtype(as_native=True))
        dtype = _infer_dtype(x.dtype)
    if exclusive or reverse:
        if exclusive and reverse:
            x = torch.cumsum(torch.flip(x, dims=(axis,)), axis, dtype=dtype)
            x = torch.transpose(x, axis, -1)
            x = torch.concat((torch.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = torch.transpose(x, axis, -1)
            res = torch.flip(x, dims=(axis,))
        elif exclusive:
            x = torch.transpose(x, axis, -1)
            x = torch.cat((torch.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = torch.cumsum(x, -1, dtype=dtype)
            res = torch.transpose(x, axis, -1)
        else:
            x = torch.cumsum(torch.flip(x, dims=(axis,)), axis, dtype=dtype)
            res = torch.flip(x, dims=(axis,))
        if ivy.exists(out):
            return ivy.inplace_update(out, res)
        return res
    return torch.cumsum(x, axis, dtype=dtype, out=out)


cumsum.support_native_out = True


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16",)},
    backend_version,
)
def einsum(
    equation: str,
    *operands: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = _get_promoted_type_of_operands(operands)
    return ivy.astype(torch.einsum(equation, *operands), dtype, copy=False)
