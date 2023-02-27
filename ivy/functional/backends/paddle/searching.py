from numbers import Number
from typing import Optional, Tuple, Union

import paddle


import ivy
# from ivy.utils.exceptions import IvyNotImplementedException

# from . import backend_version

# Array API Standard #
# ------------------ #


def argmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    select_last_index: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if select_last_index:
        x = paddle.flip(x, axis=axis)
        ret = paddle.argmax(x, axis=axis, keepdim=keepdims)
        if axis is not None:
            ret = x.shape[axis] - ret - 1
        else:
            ret = paddle.size(x, out_dtype=paddle.int64) - ret - 1
    else:
        ret = paddle.argmax(x, axis=axis, keepdim=keepdims)
    if dtype is not None:
        dtype = ivy.as_native_dtype(dtype)
        return ret.astype(dtype)
    return ret


def argmin(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[paddle.dtype] = None,
    select_last_index: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if select_last_index:
        x = paddle.flip(x, axis=axis)
        ret = paddle.argmin(x, axis=axis, keepdim=keepdims)
        if axis is not None:
            ret = x.shape[axis] - ret - 1
        else:
            ret = paddle.size(x).numpy() - ret - 1
    else:
        ret = paddle.argmin(x, axis=axis, keepdim=keepdims)
    if output_dtype:
        output_dtype = ivy.as_native_dtype(output_dtype)
        return ret.astype(output_dtype)
    return ret


def nonzero(
    x: paddle.Tensor,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[paddle.Tensor, Tuple[paddle.Tensor]]:
    res = paddle.nonzero(x)

    if size is not None:
        if isinstance(fill_value, float):
            res = paddle.cast(res, dtype="float64")

        diff = size - res.shape[0]
        if diff > 0:
            pad_shape = [(0, diff)] + [(0, 0)] * (res.ndim - 1)
            res = paddle.pad(res, pad_shape, value=fill_value)
        elif diff < 0:
            res = res[:size]

    if as_tuple:
        return tuple(res.unbind())
    return res


def where(
    condition: paddle.Tensor,
    x1: Union[float, int, paddle.Tensor],
    x2: Union[float, int, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ivy.astype(paddle.where(condition, x1, x2), x1.dtype, copy=False)


# Extra #
# ----- #


def argwhere(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.argwhere(x)
