# local
import ivy
from ivy.utils.exceptions import handle_exceptions
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
from ivy.functional.frontends.paddle.tensor.math import (
    log_softmax as paddle_log_softmax,
)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("bfloat16", "uint32", "uint16", "uint64")}, "paddle"
)
@to_ivy_arrays_and_back
@handle_exceptions
def selu(
    x,
    /,
    *,
    alpha=1.6732632423543772848170429916717,
    scale=1.0507009873554804934193349852946,
    name=None,
):
    if scale <= 1.0:
        raise ValueError(f"The scale must be greater than 1.0. Received: {scale}.")

    if alpha < 0:
        raise ValueError(f"The alpha must be no less than zero. Received: {alpha}.")
    mask = (x > 0).astype(float)
    return scale * ((x * mask) + (alpha * (ivy.exp(x) - 1) * (1 - mask)))


tanh = paddle_tanh
log_softmax = paddle_log_softmax
