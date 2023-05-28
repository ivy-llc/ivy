# local
import ivy
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


tanh = paddle_tanh


@with_unsupported_dtypes(
    {"2.4.2 and below": ("bfloat16", "uint32", "uint16", "uint64")}, "paddle"
)
@to_ivy_arrays_and_back
def selu(
    x,
    alpha=1.6732632423543772848170429916717,
    scale=1.0507009873554804934193349852946,
    name=None,
):
    mask = (x > 0).astype(float)
    return scale * ((x * mask) + (alpha * (ivy.exp(x) - 1) * (1 - mask)))
