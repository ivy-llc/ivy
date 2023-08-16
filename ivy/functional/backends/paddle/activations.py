"""
Paddle activation functions.

Collection of Paddle activation functions, wrapped to fit Ivy syntax and
signature.
"""
from typing import Optional, Union

# global
import paddle
import paddle.nn.functional as F

# local
import ivy.functional.backends.paddle as paddle_backend
import ivy
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


unsupported_dtypes = [
    paddle.int8,
    paddle.int16,
    paddle.int32,
    paddle.int64,
    paddle.uint8,
    paddle.float16,
    paddle.complex64,
    paddle.complex128,
    paddle.bool,
]


def relu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in unsupported_dtypes:
        if paddle.is_complex(x):
            return paddle.complex(F.relu(x.real()), F.relu(x.imag()))
        return F.relu(x.cast("float32")).cast(x.dtype)
    return F.relu(x)


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("bfloat16",)}}, backend_version
)
def leaky_relu(
    x: paddle.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in unsupported_dtypes:
        if paddle.is_complex(x):
            return paddle.complex(
                F.leaky_relu(x.real(), negative_slope=alpha),
                F.leaky_relu(x.imag(), negative_slope=alpha),
            )
        return F.leaky_relu(x.cast("float32"), negative_slope=alpha).cast(x.dtype)
    return F.leaky_relu(x, negative_slope=alpha)


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("bfloat16",)}}, backend_version
)
def gelu(
    x: paddle.Tensor,
    /,
    *,
    approximate: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if paddle.is_complex(x):
        sqrt_2_over_pi = 0.7978845608
        # the other magic number comes directly from the formula in
        # https://doi.org/10.48550/arXiv.1606.08415
        return (
            0.5
            * x
            * (1 + paddle_backend.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))
        )
    if x.dtype in unsupported_dtypes:
        return F.gelu(x.cast("float32"), approximate=approximate).cast(x.dtype)
    return F.gelu(x, approximate=approximate)


def sigmoid(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype in unsupported_dtypes:
        if paddle.is_complex(x):
            return 1 / (1 + paddle_backend.exp(-x))
        return F.sigmoid(x.cast("float32")).cast(x.dtype)
    return F.sigmoid(x)


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("float16",)}}, backend_version
)
def softmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if axis is None:
        axis = -1
    exp_x = paddle_backend.exp(
        paddle_backend.subtract(x, paddle_backend.max(x, axis=axis, keepdims=True))
    )
    return paddle_backend.divide(
        exp_x, paddle_backend.sum(exp_x, axis=axis, keepdims=True)
    )


def softplus(
    x: paddle.Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if beta is not None and beta != 1:
        x_beta = x * beta
        res = (
            ivy.add(
                ivy.log1p(ivy.exp(-ivy.abs(x_beta))),
                ivy.maximum(x_beta, 0),
            )
        ) / beta
    else:
        x_beta = x
        res = ivy.add(
            ivy.log1p(ivy.exp(-ivy.abs(x_beta))),
            ivy.maximum(x_beta, 0),
        )
    if threshold is not None:
        return ivy.where(x_beta > threshold, x, res).astype(x.dtype)
    return res.astype(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("float16",)}}, backend_version
)
def log_softmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
):
    if axis is None:
        axis = -1
    x_max = paddle_backend.max(x, axis=axis, keepdims=True)
    x_max = paddle_backend.where(
        paddle_backend.isfinite(x_max),
        x_max,
        paddle.zeros(shape=x_max.shape).astype(x_max.dtype),
    )
    exp_tmp = paddle_backend.exp(paddle_backend.subtract(x, x_max))

    s = paddle_backend.sum(exp_tmp, axis=axis, keepdims=True)
    ret = paddle_backend.log(s)
    ret = paddle_backend.subtract(paddle_backend.subtract(x, x_max), ret)
    return ret


def mish(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in unsupported_dtypes:
        if paddle.is_complex(x):
            return x * paddle_backend.tanh(paddle_backend.log1p(paddle_backend.exp(x)))
        return F.mish(x.cast("float32")).cast(x.dtype)
    return F.mish(x)


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("float16",)}}, backend_version
)
def hardswish(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return F.hardswish(x)
