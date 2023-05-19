from numbers import Number
from typing import Optional, Tuple, Union

import paddle
import ivy.functional.backends.paddle as paddle_backend
import ivy
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version
from .elementwise import _elementwise_helper

# Array API Standard #
# ------------------ #


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "complex64",
                "complex128",
            )
        }
    },
    backend_version,
)
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
    dtype = dtype if dtype is not None else paddle.int64
    if x.dtype in [paddle.int8, paddle.float16, paddle.bool]:
        x = x.cast("float32")
    if select_last_index:
        x = paddle_backend.flip(x, axis=axis)
        ret = paddle.argmax(x, axis=axis)
        if axis is not None:
            ret = paddle.to_tensor(x.shape[axis] - ret - 1)
        else:
            ret = paddle.to_tensor(x.size - ret - 1)
    else:
        ret = paddle.argmax(x, axis=axis)

    if keepdims:
        shape = [1] * x.ndim
        ret = paddle_backend.reshape(ret, shape)
    elif axis is None or x.ndim == 1:
        ret = paddle_backend.squeeze(ret, axis=-1)

    return ret.astype(dtype)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "complex64",
                "complex128",
            )
        }
    },
    backend_version,
)
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
    dtype = output_dtype if output_dtype is not None else paddle.int64
    if x.dtype in [paddle.int8, paddle.float16, paddle.bool]:
        x = x.cast("float32")
    if select_last_index:
        x = paddle_backend.flip(x, axis=axis)
        ret = paddle.argmin(x, axis=axis)
        if axis is not None:
            ret = paddle.to_tensor(x.shape[axis] - ret - 1)
        else:
            ret = paddle.to_tensor(x.size - ret - 1)
    else:
        ret = paddle.argmin(x, axis=axis)

    if keepdims:
        shape = [1] * x.ndim
        ret = paddle_backend.reshape(ret, shape)
    elif axis is None or x.ndim == 1:
        ret = paddle_backend.squeeze(ret, axis=-1)

    return ret.astype(dtype)


def nonzero(
    x: paddle.Tensor,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[paddle.Tensor, Tuple[paddle.Tensor]]:
    if x.dtype in [
        paddle.int8,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
    ]:
        if paddle.is_complex(x):
            real_idx = paddle.nonzero(x.real())
            imag_idx = paddle.nonzero(x.imag())
            idx = paddle.concat([real_idx, imag_idx], axis=0)
            res = paddle.unique(idx, axis=0)
        else:
            res = paddle.nonzero(x.cast("float32"))
    else:
        res = paddle.nonzero(x)

    res = res.T
    if size is not None:
        if isinstance(fill_value, float):
            res = res.cast(paddle.float64)
        diff = size - res[0].shape[0]
        if diff > 0:
            res = paddle.nn.functional.pad(
                res.unsqueeze(0),
                [0, diff],
                mode="constant",
                value=fill_value,
                data_format="NCL",
            ).squeeze(0)
        elif diff < 0:
            res = res[:, :size]

    if as_tuple:
        return tuple(res)
    return res.T


def where(
    condition: paddle.Tensor,
    x1: Union[float, int, paddle.Tensor],
    x2: Union[float, int, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    arrays = [condition, x1, x2]
    scalar_out = all(map(lambda x: x.ndim == 0, arrays))
    for i, array in enumerate(arrays):
        if array.ndim == 0:
            arrays[i] = paddle_backend.expand_dims(array, axis=0)
    condition, x1, x2 = arrays

    if ret_dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        x1 = x1.cast("float32")
        x2 = x2.cast("float32")
        result = paddle.where(condition, x1, x2)
    elif ret_dtype in [paddle.complex64, paddle.complex128]:
        result_real = paddle.where(condition, paddle.real(x1), paddle.real(x2))
        result_imag = paddle.where(condition, paddle.imag(x1), paddle.imag(x2))
        result = paddle.complex(result_real, result_imag)
    else:
        result = paddle.where(condition, x1, x2)

    return result.squeeze().cast(ret_dtype) if scalar_out else result.cast(ret_dtype)


# Extra #
# ----- #


def argwhere(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.ndim == 0:
        return paddle.zeros(shape=[int(bool(x.item())), 0], dtype="int64")
    if x.dtype in [
        paddle.int8,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
    ]:
        if paddle.is_complex(x):
            real_idx = paddle.nonzero(x.real())
            imag_idx = paddle.nonzero(x.imag())
            idx = paddle.concat([real_idx, imag_idx], axis=0)
            return paddle.unique(idx, axis=0)
        return paddle.nonzero(x.cast("float32"))
    return paddle.nonzero(x)
