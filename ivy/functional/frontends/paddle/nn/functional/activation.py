# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)

# local
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
from ivy.functional.frontends.paddle.tensor.math import (
    log_softmax as paddle_log_softmax,
)

tanh = paddle_tanh
log_softmax = paddle_log_softmax


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def gelu(x, approximate=False, name=None):
    return ivy.gelu(x, approximate=approximate)
