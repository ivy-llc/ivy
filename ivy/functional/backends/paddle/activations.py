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
import ivy
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version

unsupported_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "float16",
    "complex64",
    "complex128",
    "bool",
]


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def relu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if ivy.as_ivy_dtype(x.dtype) in unsupported_dtypes:
        if paddle.is_complex(x):
            return F.relu(x.real()) + 1j * F.relu(x.imag())
        return F.relu(x.cast("float32")).cast(x.dtype)
    return F.relu(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def leaky_relu(
    x: paddle.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if ivy.as_ivy_dtype(x.dtype) in unsupported_dtypes:
        if paddle.is_complex(x):
            return F.leaky_relu(x.real(), negative_slope=alpha) + 1j * F.leaky_relu(
                x.imag(), negative_slope=alpha
            )
        return F.leaky_relu(x.cast("float32"), negative_slope=alpha).cast(x.dtype)
    return F.leaky_relu(x, negative_slope=alpha)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def gelu(
    x: paddle.Tensor,
    /,
    *,
    approximate: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if ivy.as_ivy_dtype(x.dtype) in unsupported_dtypes:
        if paddle.is_complex(x):
            if approximate:
                return (
                    0.5 * x * (1 + ivy.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
                )
            return 0.5 * x * (1 + ivy.erf(x / ivy.sqrt(2)))
        return F.gelu(x.cast("float32"), approximate=approximate).cast(x.dtype)
    return F.gelu(x, approximate=approximate)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def sigmoid(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if ivy.as_ivy_dtype(x.dtype) in unsupported_dtypes:
        if paddle.is_complex(x):
            return 1 / (1 + ivy.exp(x))
        return F.sigmoid(x.cast("float32")).cast(x.dtype)
    return F.sigmoid(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
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
    exp_x = ivy.exp(ivy.array(x) - ivy.max(x, axis=axis, keepdims=True))
    return ivy.divide(exp_x, ivy.sum(exp_x, axis=axis, keepdims=True))


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
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
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
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
    with ivy.ArrayMode(False):
        x_max = ivy.max(x, axis=axis, keepdims=True)
        x_max = ivy.where(ivy.isfinite(x_max), x_max, ivy.zeros_like(x_max))
        exp_tmp = ivy.exp(ivy.subtract(x, x_max))

        s = ivy.sum(exp_tmp, axis=axis, keepdims=True)
        ret = ivy.log(s)
        ret = ivy.subtract(ivy.subtract(x, x_max), ret)
        return ret


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def mish(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if ivy.as_ivy_dtype(x.dtype) in unsupported_dtypes:
        if paddle.is_complex(x):
            return x * ivy.tanh(ivy.log1p(ivy.exp(x)))
        return F.mish(x.cast("float32")).cast(x.dtype)
    return F.mish(x)
