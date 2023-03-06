# global

torch_scatter = None
from typing import Union, Optional, Sequence


import paddle

# local
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from . import backend_version
from ivy.func_wrapper import with_unsupported_dtypes

# Array API Standard #
# -------------------#


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def min(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.min(x, axis=axis, keepdim=keepdims)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def max(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.max(x, axis=axis, keepdim=keepdims)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("int8", "int16", "int32", "int64", "uint8", "uint16", "bfloat16", "float16", "complex64", "complex128")},
    backend_version,
)
def mean(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.mean(x, axis=axis, keepdim=keepdims)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def prod(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[paddle.dtype] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.prod(x, axis=axis, keepdim=keepdims, dtype=dtype)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def std(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if axis is None:
        axis = list(range(len(x.shape)))
    if axis == ():
        return x
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if correction == 0:
        return paddle.std(x, axis=axis, unbiased=False, keepdim=keepdims)
    elif correction == 1:
        return paddle.std(x, axis=axis, unbiased=True, keepdim=keepdims)
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size - correction <= 0:
        ret = paddle.std(x, axis=axis, unbiased=False, keepdim=keepdims)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    ret = paddle.mul(
        paddle.std(x, axis=axis, unbiased=False, keepdim=keepdims),
        (size / (size - correction)) ** 0.5,
    )
    return ret


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def sum(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[paddle.dtype] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.sum(x, axis=axis, dtype=dtype, keepdim=keepdims)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def var(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if axis is None:
        axis = list(range(len(x.shape)))
    if axis == ():
        return x
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if correction == 0:
        return paddle.var(x, axis=axis, unbiased=False, keepdim=keepdims)
    elif correction == 1:
        return paddle.var(x, axis=axis, unbiased=True, keepdim=keepdims)
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size - correction <= 0:
        ret = paddle.var(x, axis=axis, unbiased=False, keepdim=keepdims)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    else:
        ret = paddle.mul(
            paddle.var(x, axis=axis, unbiased=False, keepdim=keepdims),
            (size / (size - correction)) ** 0.5,
        )
    return ret


# Extra #
# ----- #
@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "bool",
        )
    },
    backend_version,
)
def cumprod(
    x: paddle.Tensor,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = paddle.as_native(dtype)
    if dtype is None:
        dtype = x.dtype

    if not (exclusive or reverse):
        return paddle.cumprod(x, dim, dtype=dtype)
    elif exclusive and reverse:
        x = paddle.cumprod(paddle.flip(x, dims=(axis,)), dim, dtype=dtype)
        x = paddle.transpose(x, axis, -1)
        x = paddle.concat((paddle.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = paddle.transpose(x, axis, -1)
        ret = paddle.flip(x, dims=(axis,))
    elif exclusive:
        x = paddle.transpose(x, axis, -1)
        x = paddle.cat((paddle.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = paddle.cumprod(x, -1, dtype=dtype)
        ret = paddle.transpose(x, axis, -1)
    else:
        x = paddle.cumprod(paddle.flip(x, dims=(axis,)), dim, dtype=dtype)
        ret = paddle.flip(x, dims=(axis,))
    
    return ret


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def cumsum(
    x: paddle.Tensor,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
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
        if paddle.exists(out):
            return paddle.inplace_update(out, res)
        return res
    return torch.cumsum(x, dim=dim, dtype=dtype, out=out)


def einsum(
    equation: str,
    *operands: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = paddle.promote_types_of_inputs(operands)
    operands = (
        paddle.astype(operand, paddle.float32, copy=False).to_native()
        for operand in operands
    )
    return paddle.astype(paddle.einsum(equation, *operands), dtype, copy=False)
