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
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    *,
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
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    *,
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
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    *,
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


def prod(
    x: torch.Tensor,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if dtype is None:
        if x.dtype in [torch.int8, torch.int16]:
            dtype = torch.int32
        elif x.dtype == torch.uint8:
            dtype = torch.uint8
        elif x.dtype in [torch.int64, torch.int32]:
            dtype = torch.int64
        elif x.dtype == torch.bfloat16:
            dtype = torch.float16

    dtype = ivy.as_native_dtype(dtype)

    if axis is None:
        axis = x.dim() - 1
    elif type(axis) == tuple:
        if len(axis) == 0:
            axis = x.dim() - 1
        else:
            return torch.prod(
                torch.Tensor(
                    [
                        torch.prod(input=x, dim=i, dtype=dtype, keepdim=keepdims)
                        for i in axis
                    ]
                ),
                dtype=dtype,
                out=out,
            )
    return torch.prod(input=x, dim=axis, dtype=dtype, keepdim=keepdims, out=out)


prod.support_native_out = True


def std(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    if isinstance(axis, int):
        return torch.std(x, dim=axis, keepdim=keepdims, unbiased=False, out=out)
    dims = len(x.shape)
    axis = tuple([i % dims for i in axis])
    for i, a in enumerate(axis):
        if i == len(axis) - 1:
            x = torch.std(
                x,
                dim=a if keepdims else a - i,
                keepdim=keepdims,
                unbiased=False,
                out=out,
            )
        else:
            x = torch.std(
                x,
                dim=a if keepdims else a - i,
                keepdim=keepdims,
                unbiased=False,
                out=out,
            )
    return x


std.support_native_out = True


def sum(
    x: torch.Tensor,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: torch.dtype = None,
    keepdims: bool = False,
) -> torch.Tensor:
    if dtype is None:
        if x.dtype in [torch.int8, torch.int16]:
            dtype = torch.int32
        elif x.dtype == torch.uint8:
            dtype = torch.uint8
        elif x.dtype in [torch.int32, torch.int64]:
            dtype = torch.int64

    dtype = ivy.as_native_dtype(dtype)

    if axis is None:
        return torch.sum(input=x, dtype=dtype)
    elif type(axis) == list:
        return torch.sum(input=x, dim=axis)
    elif type(axis) == tuple:
        if len(axis) == 0:
            axis = 0
        else:
            return torch.sum(
                torch.Tensor(
                    [
                        torch.sum(input=x, dim=i, dtype=dtype, keepdim=keepdims)
                        for i in axis
                    ]
                ),
                dtype=dtype,
            )
    return torch.sum(input=x, dim=axis, dtype=dtype, keepdim=keepdims)


def var(
    x: torch.Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: Optional[bool] = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    return torch.var(x, dim=axis, keepdim=keepdims, unbiased=False, out=out)


var.support_native_out = True


# Extra #
# ------#


def einsum(
    equation: str, *operands: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.einsum(equation, *operands)
