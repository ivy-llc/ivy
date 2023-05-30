# local
import ivy
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
from ivy.functional.frontends.paddle.tensor.math import (
    log_softmax as paddle_log_softmax,
)
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


tanh = paddle_tanh
log_softmax = paddle_log_softmax


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def hardswish(x, name=None):
    relu6_val = ivy.relu6(ivy.add(x, 3))
    ret = ivy.multiply(x, ivy.divide(relu6_val, 6))
    return ret
