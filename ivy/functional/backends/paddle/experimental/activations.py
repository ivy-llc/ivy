# global
from typing import Optional, Union, Literal
import paddle
import paddle.nn.functional as F

# local
import ivy.functional.backends.paddle as paddle_backend
from ivy.func_wrapper import (
    with_unsupported_device_and_dtypes,
    with_supported_dtypes,
    with_supported_device_and_dtypes,
)
from . import backend_version


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float16", "bfloat16")}}, backend_version
)
def logit(
    x: paddle.Tensor,
    /,
    *,
    eps: Optional[float] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out=None,
):
    if x.dtype in [paddle.float32, paddle.float64]:
        return paddle.logit(x, eps)
    if eps is None:
        nan = paddle_backend.squeeze(
            paddle.to_tensor(float("nan"), dtype=x.dtype), axis=-1
        )
        x = paddle_backend.where(
            paddle_backend.logical_or(
                paddle_backend.greater(x, 1), paddle_backend.less(x, 0)
            ),
            nan,
            x,
        )
    else:
        x = paddle_backend.minimum(paddle_backend.maximum(x, eps), 1 - eps)
    return paddle_backend.log(
        paddle_backend.divide(x, paddle_backend.subtract(1, x))
    ).cast(x.dtype)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, backend_version)
def thresholded_relu(
    x: paddle.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return F.thresholded_relu(x, threshold=threshold)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64")}, backend_version
)
def relu6(
    x: paddle.Tensor, /, *, complex_mode="jax", out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if paddle.is_complex(x):
        if x.real > 0 and x.real <= 6:
            return x.astype(x.dtype)
        else:
            return paddle_backend.zeros_like(x).astype(x.dtype)
    return F.relu6(x)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64")}, backend_version
)
def logsigmoid(
    input: paddle.Tensor, /, *, complex_mode="jax", out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if paddle.is_complex(input):
        return paddle_backend.log(
            paddle_backend.divide(
                1.0, (paddle_backend.add(1.0, paddle_backend.exp(-input)))
            )
        )
    return F.log_sigmoid(input)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64")}, backend_version
)
def selu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if paddle.is_complex(x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        ret = paddle_backend.multiply(
            scale,
            paddle_backend.where(
                paddle_backend.greater(x, 0),
                x,
                paddle_backend.multiply(alpha, paddle_backend.expm1(x)),
            ),
        )
        return ret
    return F.selu(x)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64")}, backend_version
)
def silu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if paddle.is_complex(x):
        return x * (1.0 / (1.0 + paddle_backend.exp(-x)))
    return F.silu(x)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64")}, backend_version
)
def elu(
    x: paddle.Tensor, /, *, alpha: float = 1.0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if paddle.is_complex(x):
        ret = (
            paddle_backend.where(
                paddle_backend.greater(x, 0),
                x,
                paddle_backend.multiply(alpha, paddle_backend.expm1(x)),
            ),
        )
        return ret
    return F.elu(x, alpha=alpha)


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("bfloat16", "float16")}}, backend_version
)
def hardtanh(
    x: paddle.Tensor,
    /,
    *,
    max_val: float = 1.0,
    min_val: float = -1.0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.hardtanh(x, min=min_val, max=max_val)

    if paddle.is_complex(x):
        ret = (
            paddle_backend.where(
                paddle_backend.greater(x, max_val),
                max_val,
                paddle_backend.where(paddle_backend.less(x, min_val), min_val, x),
            ),
        )
        return ret
    return F.hardtanh(x.cast("float32"), min=min_val, max=max_val).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("bfloat16", "float16")}}, backend_version
)
def tanhshrink(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.tanhshrink(x)
    if paddle.is_complex(x):
        return paddle.complex(F.tanhshrink(x.real()), F.tanhshrink(x.imag()))
    return F.tanhshrink(x.cast("float32")).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("bfloat16", "float16")}}, backend_version
)
def threshold(
    x: paddle.Tensor,
    /,
    *,
    threshold: float,
    value: float,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return paddle_backend.where(paddle_backend.greater(x, threshold), x, value)
    if paddle.is_complex(x):
        return paddle_backend.where(paddle_backend.greater(x, threshold), x, value)
    x = x.cast("float32")
    return paddle_backend.where(paddle_backend.greater(x, threshold), x, value).cast(
        x.dtype
    )


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("bfloat16", "float16")}}, backend_version
)
def softshrink(
    x: paddle.Tensor, /, *, lambd: float = 0.5, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.softshrink(x, threshold=lambd)
    if paddle.is_complex(x):
        return paddle.complex(
            F.softshrink(x.real(), threshold=lambd),
            F.softshrink(x.img(), threshold=lambd),
        )
    return F.softshrink(x.cast("float32"), threshold=lambd).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("bfloat16", "float16")}}, backend_version
)
def celu(
    x: paddle.Tensor,
    /,
    *,
    alpha: float = 1.0,
    complex_mode="jax",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return F.celu(x, alpha=alpha)


@with_supported_device_and_dtypes(
    {
        "2.6.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("uint16", "float16", "float32", "float64"),
        }
    },
    backend_version,
)
def scaled_tanh(
    x: paddle.Tensor,
    /,
    *,
    alpha: float = 1.7159,
    beta: float = 0.67,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.stanh(x, scale_a=beta, scale_b=alpha)


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float16", "bfloat16")}},
    backend_version,
)
def hardshrink(
    x: paddle.Tensor, /, *, lambd: float = 0.5, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.hardshrink(x, threshold=lambd)
    if paddle.is_complex(x):
        return paddle.complex(
            F.hardshrink(x.real(), threshold=lambd),
            F.hardshrink(x.img(), threshold=lambd),
        )
    return F.hardshrink(x.cast("float32"), threshold=lambd).cast(x.dtype)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, backend_version)
def hardsilu(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return F.hardswish(x)
