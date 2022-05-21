# global
import mxnet as mx
import math
from typing import Optional

# local
import ivy
from ivy.functional.backends.mxnet import (
    _handle_flat_arrays_in_out,
    _scalar_or_flat_array_to_scalar,
)


@_handle_flat_arrays_in_out
def add(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
) -> mx.ndarray.ndarray.NDArray:
    return mx.nd.add(x1, x2)


@_handle_flat_arrays_in_out
def bitwise_and(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.nd.ndarray.NDArray:
    ret = mx.numpy.bitwise_and(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def ceil(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.ceil(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def floor(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.floor(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def divide(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.divide(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def greater(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.greater(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def greater_equal(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.greater_equal(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def isfinite(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    # ToDo: remove float32 conversion once int8 and uint8 work correctly.
    #  Currently 0 returns 0 for these types.
    ret = mx.nd.contrib.isfinite(x.astype("float32")).astype("bool")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def isinf(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.contrib.isinf(x.astype("float32")).astype("bool")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sqrt(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    if isinstance(x, float):
        ret = math.sqrt(x)
    else:
        ret = mx.nd.sqrt(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def isnan(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.contrib.isnan(x).astype("bool")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def less(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.lesser(x1, x2).astype("bool")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def logical_xor(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    dtype: ["bool"],
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.logical_xor(x1, x2, dtype).astype("bool")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def logical_not(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.logical_not(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def acos(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    if isinstance(x, float):
        ret = math.acos(x)
    else:
        ret = mx.nd.arccos(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def logical_and(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    dtype: ["bool"],
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.logical_and(x1, x2, dtype).astype("bool")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def logical_or(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    dtype: ["bool"],
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.logical_or(x1, x2, dtype).astype("bool")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def multiply(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.multiply(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def acosh(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    if isinstance(x, float):
        ret = math.acosh(x)
    else:
        ret = mx.nd.arccosh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def sin(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    if isinstance(x, float):
        ret = math.sin(x)
    else:
        ret = mx.nd.sin(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def negative(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.np.negative(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def tanh(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    if isinstance(x, float):
        ret = math.tanh(x)
    else:
        ret = mx.nd.tanh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def bitwise_or(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.nd.ndarray.NDArray:
    ret = mx.numpy.bitwise_or(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def sinh(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    if isinstance(x, float):
        ret = math.sinh(x)
    else:
        ret = mx.nd.sinh(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def square(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.square(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def round(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.round(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def trunc(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.nd.ndarray.NDArray:
    ret = mx.np.trunc(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in_out
def subtract(
    x1: mx.ndarray.ndarray.NDArray,
    x2: mx.ndarray.ndarray.NDArray,
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.subtract(x1, x2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


# noinspection PyShadowingBuiltins
@_handle_flat_arrays_in_out
def abs(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.nd.ndarray.NDArray:
    ret = mx.nd.abs(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def cos(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    if isinstance(x, float):
        ret = math.cos(x)
    else:
        ret = mx.nd.cos(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def exp(
    x: mx.ndarray.ndarray.NDArray, out: Optional[mx.ndarray.ndarray.NDArray] = None
) -> mx.ndarray.ndarray.NDArray:
    if isinstance(x, float):
        ret = math.exp(x)
    else:
        ret = mx.nd.exp(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


tan = lambda x: math.tan(x) if isinstance(x, float) else mx.nd.tan(x)
asin = lambda x: math.asin(x) if isinstance(x, float) else mx.nd.arcsin(x)
atan = lambda x: math.atan(x) if isinstance(x, float) else mx.nd.arctan(x)
atan2 = (
    lambda x, y: math.atan2(x, y)
    if isinstance(x, float)
    else mx.np.arctan2(x.as_np_ndarray(), y.as_np_ndarray()).as_nd_ndarray()
)
cosh = lambda x: math.cosh(x) if isinstance(x, float) else mx.nd.cosh(x)
asinh = lambda x: math.asinh(x) if isinstance(x, float) else mx.nd.arcsinh(x)
atanh = lambda x: math.atanh(x) if isinstance(x, float) else mx.nd.arctanh(x)
log = lambda x: math.log(x) if isinstance(x, float) else mx.nd.log(x)
equal = lambda x1, x2: x1 == x2
equal.__name__ = "equal"

# Extra #
# ------#


minimum = lambda x, y: mx.nd.array(
    mx.nd.minimum(
        _scalar_or_flat_array_to_scalar(x), _scalar_or_flat_array_to_scalar(y)
    )
)
maximum = lambda x, y: mx.nd.array(
    mx.nd.maximum(
        _scalar_or_flat_array_to_scalar(x), _scalar_or_flat_array_to_scalar(y)
    )
)
erf = lambda x: math.erf(x) if isinstance(x, float) else mx.nd.erf(x)
