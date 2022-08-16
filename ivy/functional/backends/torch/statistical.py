# global
torch_scatter = None
import torch
from typing import Tuple, Union, Optional, Sequence

# local
import ivy


def _new_var_fun(x, *, axis, correction, dtype):
    output = x.to(dtype)
    length = output.shape[axis]
    divisor = length - correction
    mean = torch.sum(output, dim=axis) / length
    output = torch.abs(output.to(dtype=dtype) - torch.unsqueeze(mean, dim=axis))
    output = output**2
    output = torch.sum(output, axis=axis) / divisor
    return output


def _new_std_fun(x, *, axis, correction, dtype):
    output = torch.sqrt(_new_var_fun(x, axis=axis, correction=correction, dtype=dtype))
    output = output.to(dtype=dtype)
    return output


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


def prod(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    keepdims: Optional[bool] = False,
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

    axis = tuple(axis) if isinstance(axis, list) else axis
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
            )
    return torch.prod(input=x, dim=axis, dtype=dtype, keepdim=keepdims)


prod.unsupported_dtypes = ("float16",)


def std(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    axis = tuple(axis) if isinstance(axis, list) else axis
    dtype = x.dtype
    if isinstance(axis, tuple):
        ret = []
        for i in axis:
            ret.append(_new_std_fun(x, axis=i, correction=correction, dtype=dtype))
        ret = torch.tensor(ret, dtype=dtype)
    elif isinstance(axis, int):
        ret = _new_std_fun(x, axis=axis, correction=correction, dtype=dtype)
    else:
        num = torch.numel(x)
        ret = _new_std_fun(
            torch.reshape(x, (num,)), axis=0, correction=correction, dtype=dtype
        )

    if keepdims:
        shape = tuple(
            [1 if ret.shape.numel() <= 1 else ret.shape[0]]
            + [1 for i in range(len(x.shape) - 1)]
        )
        ret = torch.reshape(ret, shape)
    return ret


std.unsupported_dtypes = ("int8", "int16", "int32", "int64", "float16")


def sum(
    x: torch.Tensor,
    /,
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
        elif x.dtype == torch.float16:
            dtype = torch.float16

    dtype = ivy.as_native_dtype(dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    if axis is None:
        return torch.sum(input=x, dtype=dtype)
    return torch.sum(input=x, dim=axis, dtype=dtype, keepdim=keepdims)


def var(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    axis = tuple(axis) if isinstance(axis, list) else axis
    dtype = x.dtype
    if isinstance(axis, tuple):
        ret = []
        for i in axis:
            ret.append(_new_var_fun(x, axis=i, correction=correction, dtype=dtype))
        ret = torch.tensor(ret, dtype=dtype)
    elif isinstance(axis, int):
        ret = _new_var_fun(x, axis=axis, correction=correction, dtype=dtype)
    else:
        num = torch.numel(x)
        ret = _new_var_fun(
            torch.reshape(x, (num,)), axis=0, correction=correction, dtype=dtype
        )

    if keepdims:
        shape = tuple(
            [1 if ret.shape.numel() <= 1 else ret.shape[0]]
            + [1 for i in range(len(x.shape) - 1)]
        )
        ret = torch.reshape(ret, shape)
    return ret


# Extra #
# ------#


def einsum(
    equation: str,
    *operands: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.einsum(equation, *operands)um(equation, *operands)

