# global

torch_scatter = None
from typing import Union, Optional, Sequence


import paddle

# local
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from . import backend_version

# Array API Standard #
# -------------------#

def min(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.min(x, axis=axis, keepdim=keepdims)


def max(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.max(x, axis=axis, keepdims=keepdims)


def mean(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def _infer_dtype(dtype: paddle.dtype) -> paddle.dtype:
    raise IvyNotImplementedException()


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
        ret = paddle.std(x, axis==axis, unbiased=False, keepdim=keepdims)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    ret = torch.mul(
        torch.std(x, axis=axis, unbiased=False, keepdim=keepdims),
        (size / (size - correction)) ** 0.5,
    )
    return ret

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
        return paddle.std(x, axis=axis, unbiased=False, keepdim=keepdims)
    elif correction == 1:
        return paddle.std(x, axis=axis, unbiased=True, keepdim=keepdims)
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size - correction <= 0:
        ret = paddle.std(x, axis==axis, unbiased=False, keepdim=keepdims)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    ret = torch.mul(
        torch.std(x, axis=axis, unbiased=False, keepdim=keepdims),
        (size / (size - correction)) ** 0.5,
    )
    return ret


# Extra #
# ----- #

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
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)

    if not (exclusive or reverse):
        return paddle.cumprod(x, dim, dtype=dtype)
    elif exclusive and reverse:
        x = paddle.cumprod(paddle.flip(x, dim), dim, dtype=dtype)
        x = paddle.transpose(x, axis, -1)
        x = paddle.concat((paddle.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = paddle.transpose(x, axis, -1)
        ret = paddle.flip(x, dim)
    elif exclusive:
        x = paddle.transpose(x, axis, -1)
        x = paddle.cat((torch.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = paddle.cumprod(x, -1, dtype=dtype)
        ret = paddle.transpose(x, axis, -1)
    else:
        x = paddle.cumprod(torch.flip(x, dim), dim, dtype=dtype)
        ret = paddle.flip(x, dim)
    
    return ret


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
        if dtype is paddle.bool_:
            dtype = ivy.default_int_dtype(as_native=True)
        elif ivy.is_int_dtype(x.dtype):
            dtype = ivy.promote_types(x.dtype, ivy.default_int_dtype(as_native=True))
        else:
            dtype = _infer_dtype(x.dtype)
    if exclusive or reverse:
        if exclusive and reverse:
            x = paddle.cumsum(paddle.flip(x, axis=axis), axis=axis, dtype=dtype)
            x = paddle.transpose(x, axis, -1)
            x = paddle.concatenate((paddle.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = paddle.transpose(x, axis, -1)
            res = paddle.flip(x, axis=axis)
        elif exclusive:
            x = paddle.transpose(x, axis, -1)
            x = paddle.concatenate((paddle.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x = paddle.cumsum(x, -1, dtype=dtype)
            res = paddle.transpose(x, axis, -1)
        elif reverse:
            x = paddle.cumsum(paddle.flip(x, axis=axis), axis=axis, dtype=dtype)
            res = paddle.flip(x, axis=axis)
        return res
    return paddle.cumsum(x, axis, dtype=dtype)

def einsum(
    equation: str,
    *operands: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = _get_promoted_type_of_operands(operands)
    operands = (
        ivy.astype(operand, paddle.float32, copy=False).to_native()
        for operand in operands
    )
    return ivy.astype(paddle.einsum(equation, *operands), dtype, copy=False)

