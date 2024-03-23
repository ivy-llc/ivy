"""Paddle activation functions.

Collection of Paddle activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional, Union, Literal

# global
import paddle
import paddle.nn.functional as F

# local
import ivy.functional.backends.paddle as paddle_backend
import ivy
from ivy.func_wrapper import (
    with_unsupported_device_and_dtypes,
    with_supported_dtypes,
    with_unsupported_dtypes,
    with_supported_device_and_dtypes,
)
from . import backend_version


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "float32",
            "float64",
            "complex64",
        )
    },
    backend_version,
)
def relu(
    x: paddle.Tensor, /, *, complex_mode="jax", out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if paddle.is_complex(x):
        return paddle.complex(F.relu(x.real()), F.relu(x.imag()))
    return F.relu(x)


@with_supported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float32", "float64", "complex")}},
    backend_version,
)
def leaky_relu(
    x: paddle.Tensor,
    /,
    *,
    alpha: float = 0.2,
    complex_mode="jax",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if paddle.is_complex(x):
        return paddle.complex(
            F.leaky_relu(x.real(), negative_slope=alpha),
            F.leaky_relu(x.imag(), negative_slope=alpha),
        )
    return F.leaky_relu(x, negative_slope=alpha)


@with_supported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float32", "float64", "complex")}},
    backend_version,
)
def gelu(
    x: paddle.Tensor,
    /,
    *,
    approximate: bool = False,
    complex_mode="jax",
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
    return F.gelu(x, approximate=approximate)


@with_supported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float32", "float64", "complex")}},
    backend_version,
)
def sigmoid(
    x: paddle.Tensor, /, *, complex_mode="jax", out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if paddle.is_complex(x):
        return 1.0 / (1.0 + paddle_backend.exp(-x))
    return F.sigmoid(x)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("bfloat16", "float16", "complex128")}, backend_version
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

    if paddle.is_complex(x):
        amax = paddle_backend.max(x, axis=axis, keepdims=True)
    else:
        amax = paddle.max(x, axis, keepdim=True)
    exp_x = paddle_backend.exp(paddle.subtract(x, amax))
    return paddle.divide(exp_x, paddle.sum(exp_x, axis=axis, keepdim=True))


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float16", "bfloat16")}}, backend_version
)
def softplus(
    x: paddle.Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    complex_mode="jax",
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


# Softsign
@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float16", "bfloat16")}}, backend_version
)
def softsign(
    x: paddle.Tensor,
    /,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return F.softsign(x)


softsign.support_native_out = True


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float16", "bfloat16")}}, backend_version
)
def log_softmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = -1,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[paddle.Tensor] = None,
):
    x_max = paddle_backend.max(x, axis=axis, keepdims=True)
    sub_tmp = paddle_backend.subtract(x, x_max)
    ret = paddle_backend.sum(paddle_backend.exp(sub_tmp), axis=axis, keepdims=True)
    ret = paddle_backend.log(ret)
    ret = paddle_backend.subtract(sub_tmp, ret)
    return ret


@with_supported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float32", "float64", "complex")}},
    backend_version,
)
def mish(
    x: paddle.Tensor,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if paddle.is_complex(x):
        return x * paddle_backend.tanh(paddle_backend.log1p(paddle_backend.exp(x)))
    return F.mish(x)


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float16",)}}, backend_version
)
def hardswish(
    x: paddle.Tensor,
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return F.hardswish(x)
