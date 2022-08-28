# global
torch_scatter = None
import torch
from typing import Tuple, Union, Optional, Sequence

# local
import ivy


# Array API Standard #
# -------------------#


def max(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
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
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if axis == ():
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        else:
            return x
    return torch.mean(x, dim=axis, keepdim=keepdims, out=out)


mean.support_native_out = True


def min(
    x: torch.Tensor,
    /,
    *,
    axis: Union[int, Tuple[int]] = None,
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


def _infer_dtype(x_dtype: torch.dtype):
    default_dtype = ivy.infer_default_dtype(x_dtype, as_native=True)
    if ivy.dtype_bits(x_dtype) < ivy.dtype_bits(default_dtype):
        dtype = default_dtype
    else:
        dtype = x_dtype
    return dtype


def prod(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    keepdims: Optional[bool] = False,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    if axis is None:
        if keepdims:
            return torch.prod(x, 0, keepdim=keepdims, dtype=dtype)
        return torch.prod(x, dtype=dtype)
    elif isinstance(axis, tuple):
        if len(axis) == 0:
            if keepdims:
                return torch.prod(x, 0, keepdim=keepdims, dtype=dtype)
            ret = torch.unsqueeze(torch.prod(x, dtype=dtype), dim=0)
            return ret
        for i in axis:
            x = torch.prod(x, i, keepdim=keepdims, dtype=dtype)
        return x
    else:
        return torch.prod(x, axis, keepdim=keepdims, dtype=dtype)


prod.unsupported_dtypes = ("float16", "uint16")


def std(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = tuple(range(len(x.shape)))
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if correction == 0:
        ret = torch.std(x, dim=axis, unbiased=False, keepdim=keepdims)
        return ret
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


std.unsupported_dtypes = ("int8", "int16", "int32", "int64", "float16")


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
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    if axis is None:
        return torch.sum(input=x, dtype=dtype)
    return torch.sum(input=x, dim=axis, dtype=dtype, keepdim=keepdims)


def var(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = tuple(range(len(x.shape)))
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if correction == 0:
        return torch.var(x, dim=axis, unbiased=False, keepdims=keepdims)
    elif correction == 1:
        return torch.var(x, dim=axis, unbiased=True, keepdims=keepdims)
    size = 1
    for a in axis:
        size *= x.shape[a]
    return (size / (size - correction)) * torch.var(
        x, dim=axis, unbiased=False, keepdims=keepdims
    )


# Extra #
# ------#


def einsum(
    equation: str,
    *operands: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.einsum(equation, *operands)
