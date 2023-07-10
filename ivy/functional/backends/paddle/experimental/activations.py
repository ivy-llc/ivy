# global
from typing import Optional, Union
import paddle
import paddle.nn.functional as F

# local
import ivy.functional.backends.paddle as paddle_backend
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("float16",)}}, backend_version
)
def logit(x: paddle.Tensor, /, *, eps: Optional[float] = None, out=None):
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


def thresholded_relu(
    x: paddle.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.thresholded_relu(x, threshold=threshold)
    return paddle_backend.where(paddle_backend.greater(x, threshold), x, 0).cast(
        x.dtype
    )


def relu6(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.relu6(x)
    if paddle.is_complex(x):
        return paddle.complex(F.relu6(x.real()), F.relu6(x.imag()))
    return F.relu6(x.cast("float32")).cast(x.dtype)


def logsigmoid(
    input: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if input.dtype in [paddle.float32, paddle.float64]:
        return F.log_sigmoid(input)
    if paddle.is_complex(input):
        return paddle_backend.log(
            paddle_backend.divide(
                1.0, (paddle_backend.add(1.0, paddle_backend.exp(input)))
            )
        )
    return F.log_sigmoid(input.cast("float32")).cast(input.dtype)


def selu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.selu(x)
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
    return F.selu(x.cast("float32")).cast(x.dtype)


def silu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.silu(x)
    if paddle.is_complex(x):
        return x * (1.0 / (1.0 + paddle_backend.exp(-x)))
    return F.silu(x.cast("float32")).cast(x.dtype)


def elu(
    x: paddle.Tensor, /, *, alpha: float = 1.0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.elu(x, alpha=alpha)

    if paddle.is_complex(x):
        ret = (
            paddle_backend.where(
                paddle_backend.greater(x, 0),
                x,
                paddle_backend.multiply(alpha, paddle_backend.expm1(x)),
            ),
        )
        return ret
    return F.elu(x.cast("float32"), alpha).cast(x.dtype)
