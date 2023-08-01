# global

torch_scatter = None

from typing import Union, Optional, Sequence

import paddle
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_unsupported_device_and_dtypes
import ivy.functional.backends.paddle as paddle_backend

# local
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
    ret_dtype = x.dtype
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            real = paddle.amin(x.real(), axis=axis, keepdim=keepdims)
            masked_x = paddle_backend.greater_equal(x, paddle.amin(x.real())) * x
            imag = paddle.amin(masked_x.imag(), axis=axis, keepdim=keepdims)
            ret = paddle.complex(real, imag)
        else:
            ret = paddle.amin(x.cast("float32"), axis=axis, keepdim=keepdims)
    else:
        ret = paddle.amin(x, axis=axis, keepdim=keepdims)
    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret.astype(ret_dtype)


def max(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = x.dtype
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            real = paddle.amax(x.real(), axis=axis, keepdim=keepdims)
            masked_x = paddle_backend.greater_equal(x, paddle.amax(x.real())) * x
            imag = paddle.amax(masked_x.imag(), axis=axis, keepdim=keepdims)
            ret = paddle.complex(real, imag)
        else:
            ret = paddle.amax(x.cast("float32"), axis=axis, keepdim=keepdims)
    else:
        ret = paddle.amax(x, axis=axis, keepdim=keepdims)

    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret.astype(ret_dtype)


def mean(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = x.dtype
    if x.dtype not in [
        paddle.float32,
        paddle.float64,
    ]:
        if paddle.is_complex(x):
            ret = paddle.complex(
                paddle.mean(x.real(), axis=axis, keepdim=keepdims),
                paddle.mean(x.imag(), axis=axis, keepdim=keepdims),
            )
        else:
            ret = paddle.mean(x.cast("float32"), axis=axis, keepdim=keepdims)
    else:
        ret = paddle.mean(x, axis=axis, keepdim=keepdims)

    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret.astype(ret_dtype)


def prod(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[paddle.dtype] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
    # TODO:prod causes segmentation fault
    return paddle.prod(x, axis=axis, keepdim=keepdims, dtype=dtype)


def _std(x, axis, correction, keepdim):
    u = paddle_backend.mean(x, axis=axis, keepdims=True)
    out = paddle_backend.sum(
        paddle_backend.pow(paddle_backend.subtract(x, u), 2),
        axis=axis,
        keepdims=keepdim,
    )
    num_elm_in = paddle.prod(paddle.to_tensor(x.shape)).item()
    num_elm_out = paddle.prod(paddle.to_tensor(out.shape)).item()
    n = num_elm_out / num_elm_in
    out = paddle_backend.sqrt(paddle_backend.multiply(out, n))
    if correction:
        n = paddle_backend.sqrt(
            paddle_backend.divide(num_elm_in, (num_elm_in - correction * num_elm_out))
        )
        out = paddle_backend.multiply(out, n)
    return out


def std(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return _std(x, axis, correction, keepdims).cast(x.dtype)


def sum(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[paddle.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = x.dtype if dtype is None else dtype
    dtype = ivy.as_ivy_dtype(dtype)
    if x.dtype in [paddle.int8, paddle.uint8]:
        ret = paddle.sum(x.cast("float32"), axis=axis, dtype=dtype, keepdim=keepdims)
    else:
        ret = paddle.sum(x.cast(dtype), axis=axis, dtype=dtype, keepdim=keepdims)
    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = paddle_backend.squeeze(ret, axis=-1)
    return ret


def var(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret = paddle_backend.pow(_std(x, axis, correction, keepdims), 2).cast(x.dtype)
    return ret


# Extra #
# ----- #
@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("uint8", "int16")}},
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
    dtype = dtype if dtype is not None else x.dtype
    x = paddle.cast(x, dtype)
    if ivy.as_native_dtype(dtype) in [
        paddle.uint8,
        paddle.int8,
        paddle.int16,
        paddle.float16,
    ]:
        x = paddle.cast(x, "float32")
    if not (exclusive or reverse):
        return paddle.cumprod(x, dim=axis).cast(dtype)
    elif exclusive and reverse:
        x = paddle.cumprod(paddle_backend.flip(x, axis=(axis,)), dim=axis)
        x = paddle_backend.swapaxes(x, axis, -1)
        x = paddle_backend.concat(
            [
                paddle.ones_like(
                    paddle_backend.get_item(x, (..., slice(-1, None, None)))
                ),
                paddle_backend.get_item(x, (..., slice(None, -1, None))),
            ],
            axis=-1,
        )
        x = paddle_backend.swapaxes(x, axis, -1)
        return paddle_backend.flip(x, axis=(axis,)).cast(dtype)
    elif exclusive:
        x = paddle_backend.swapaxes(x, axis, -1)
        x = paddle_backend.concat(
            [
                paddle.ones_like(
                    paddle_backend.get_item(x, (..., slice(-1, None, None)))
                ),
                paddle_backend.get_item(x, (..., slice(None, -1, None))),
            ],
            axis=-1,
        )
        x = paddle.cumprod(x, -1)
        return paddle_backend.swapaxes(x, axis, -1).cast(dtype)
    else:
        x = paddle.cumprod(paddle_backend.flip(x, axis=(axis,)), dim=axis)
        return paddle_backend.flip(x, axis=axis).cast(dtype)


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("complex64", "complex128")}},
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
    dtype = dtype if dtype is not None else x.dtype
    x = paddle.cast(x, dtype)
    if ivy.as_native_dtype(dtype) in [
        paddle.uint8,
        paddle.int8,
        paddle.float16,
        paddle.bool,
    ]:
        x = paddle.cast(x, "float32")
    if not (exclusive or reverse):
        return paddle.cumsum(x, axis=axis).cast(dtype)
    elif exclusive and reverse:
        x = paddle.cumsum(paddle_backend.flip(x, axis=(axis,)), axis=axis)
        x = paddle_backend.swapaxes(x, axis, -1)
        x = paddle_backend.concat(
            [
                paddle.zeros_like(
                    paddle_backend.get_item(x, (..., slice(-1, None, None)))
                ),
                paddle_backend.get_item(x, (..., slice(None, -1, None))),
            ],
            axis=-1,
        )
        x = paddle_backend.swapaxes(x, axis, -1)
        return paddle_backend.flip(x, axis=(axis,)).cast(dtype)
    elif exclusive:
        x = paddle_backend.swapaxes(x, axis, -1)
        x = paddle_backend.concat(
            [
                paddle.zeros_like(
                    paddle_backend.get_item(x, (..., slice(-1, None, None)))
                ),
                paddle_backend.get_item(x, (..., slice(None, -1, None))),
            ],
            axis=-1,
        )
        x = paddle.cumsum(x, -1)
        return paddle_backend.swapaxes(x, axis, -1).cast(dtype)
    else:
        x = paddle.cumsum(paddle_backend.flip(x, axis=(axis,)), axis=axis)
        return paddle_backend.flip(x, axis=axis).cast(dtype)


def einsum(
    equation: str,
    *operands: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
