# global
from typing import Optional, Union, Literal
import paddle
import paddle.nn.functional as F

# local
import ivy.functional.backends.paddle as paddle_backend
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "uint16", "float32", "float64")},
    backend_version,
)
def logit(
    x: paddle.Tensor,
    /,
    *,
    eps: Optional[float] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out=None,
):
    return paddle.logit(x, eps)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "uint16", "float32", "float64")},
    backend_version,
)
def thresholded_relu(
    x: paddle.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return F.thresholded_relu(x, threshold=threshold)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "uint16", "float32", "float64", "complex")},
    backend_version,
)
def relu6(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if paddle.is_complex(x):
        return paddle.complex(F.relu6(x.real()), F.relu6(x.imag()))
    return F.relu6(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float", "complex")},
    backend_version,
)
def logsigmoid(
    x: paddle.Tensor, /, *, complex_mode="jax", out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if paddle.is_complex(x):
        return paddle_backend.log(
            paddle_backend.divide(
                1.0, (paddle_backend.add(1.0, paddle_backend.exp(-x)))
            )
        )
    return F.log_sigmoid(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float", "complex")},
    backend_version,
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
    {"2.5.1 and below": ("float", "uint16", "complex")},
    backend_version,
)
def silu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if paddle.is_complex(x):
        return x * (1.0 / (1.0 + paddle_backend.exp(-x)))
    return F.silu(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float", "uint16", "complex")},
    backend_version,
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
