# local
import ivy
from ivy.utils.exceptions import handle_exceptions
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
from ivy.functional.frontends.paddle.tensor.math import (
    log_softmax as paddle_log_softmax,
)


def _batch_promotion(*args, default_dtype="float64"):
    # Promote all types
    promote_types = set()

    for arg in args:
        if args is None:
            continue
        if isinstance(arg, (float, int)):
            continue
        promote_types.add(ivy.dtype(arg))

    if "float64" in promote_types:
        return "float64"

    if "float32" in promote_types:
        return "float32"

    if "float16" in promote_types:
        return "float32" if "bfloat16" in promote_types else "float16"

    if "bfloat16" in promote_types:
        return "bfloat16"

    if "int64" in promote_types or "uint64" in promote_types:
        return "float64"

    ints = ["int8", "int16", "int32"]
    if "uint32" in promote_types and any(d in promote_types for d in ints):
        return "float64"

    return default_dtype


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
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

    ret = ivy.where(x > 0, x, alpha * ivy.expm1(x))
    dtype = _batch_promotion(x, alpha, default_dtype="float64")
    arr = ivy.asarray(ret, dtype=dtype)
    return scale * arr


tanh = paddle_tanh
log_softmax = paddle_log_softmax
