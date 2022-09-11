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


def _infer_dtype(dtype: torch.dtype) -> torch.dtype:
    default_dtype = ivy.infer_default_dtype(dtype)
    if default_dtype in ivy.valid_dtypes:
        if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
            return ivy.as_native_dtype(default_dtype)
    return ivy.as_native_dtype(dtype)


def prod(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    keepdims: bool = False,
) -> torch.Tensor:
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    if axis is None:
        axis = 0
    if axis == ():
        return x.type(dtype)
    if isinstance(axis, tuple) or isinstance(axis, list):
        for i in axis:
            x = torch.prod(x, i, keepdim=keepdims, dtype=dtype)
        return x
    return torch.prod(x, axis, keepdim=keepdims, dtype=dtype)


prod.unsupported_dtypes = ("float16",)


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


std.unsupported_dtypes = ("int8", "int16", "int32", "int64", "float16")


def sum(
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
            (size / (size - correction)) ** 0.5,
        )


# Extra #
# ----- #


def cumprod(
    x: torch.Tensor,
    axis: int = 0,
    exclusive: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        if dtype is torch.bool:
            dtype = ivy.default_int_dtype(as_native=True)
        else:
            dtype = _infer_dtype(x.dtype)
    if exclusive:
        x = torch.transpose(x, axis, -1)
        x = torch.cat((torch.ones_like(x[..., -1:]), x[..., :-1]), -1, out=out)
        res = torch.cumprod(x, -1, dtype=dtype, out=out)
        return torch.transpose(res, axis, -1)
    return torch.cumprod(x, axis, dtype=dtype, out=out)


cumprod.support_native_out = True
cumprod.unsupported_dtypes = ("bfloat16",)  # TODO Fixed in PyTorch 1.12.1


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
        dtype = _infer_dtype(x.dtype)
    if exclusive or reverse:
        if exclusive and reverse:
            x = torch.cumsum(torch.flip(x, dims=(axis,)), axis=axis, dtype=dtype)
            x = torch.transpose(x, axis, -1)
            x = torch.concat((torch.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = torch.transpose(x, axis, -1)
            res = torch.flip(x, dims=(axis,))
        elif exclusive:
            x = torch.transpose(x, axis, -1)
            x = torch.cat((torch.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = torch.cumsum(x, -1, dtype=dtype)
            res = torch.transpose(x, axis, -1)
        elif reverse:
            x = torch.cumsum(torch.flip(x, dims=(axis,)), axis=axis, dtype=dtype)
            res = torch.flip(x, dims=(axis,))
        return res
    return torch.cumsum(x, axis, dtype=dtype, out=out)


cumsum.support_native_out = True
cumsum.unsupported_dtypes = ("bfloat16",)  # TODO Fixed in PyTorch 1.12.1


def einsum(
    equation: str,
    *operands: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.einsum(equation, *operands)
