# global
import mxnet as mx
import math
from typing import Union

# local
from ivy.functional.backends.mxnet import (
    _handle_flat_arrays_in_out,
    _scalar_or_flat_array_to_scalar,
)


@_handle_flat_arrays_in_out
def add(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.add(x1, x2)


@_handle_flat_arrays_in_out
def bitwise_and(
    x1: Union[int, mx.nd.NDArray],
    x2: Union[int, mx.nd.NDArray],
) -> mx.nd.ndarray.NDArray:
    return mx.numpy.bitwise_and(x1, x2)


@_handle_flat_arrays_in_out
def ceil(x: mx.nd.NDArray) -> mx.nd.NDArray:
    ret = mx.nd.ceil(x)
    return ret


@_handle_flat_arrays_in_out
def floor(x: mx.nd.NDArray) -> mx.nd.NDArray:
    ret = mx.nd.floor(x)
    return ret


@_handle_flat_arrays_in_out
def divide(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.divide(x1, x2)


@_handle_flat_arrays_in_out
def greater(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.greater(x1, x2)


@_handle_flat_arrays_in_out
def greater_equal(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.greater_equal(x1, x2)


@_handle_flat_arrays_in_out
def isfinite(x: mx.nd.NDArray) -> mx.nd.NDArray:
    # ToDo: remove float32 conversion once int8 and uint8 work correctly.
    #  Currently 0 returns 0 for these types.
    ret = mx.nd.contrib.isfinite(x.astype("float32")).astype("bool")
    return ret


@_handle_flat_arrays_in_out
def isinf(x: mx.nd.NDArray) -> mx.nd.NDArray:
    ret = mx.nd.contrib.isinf(x.astype("float32")).astype("bool")
    return ret


def sqrt(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.sqrt(x)
    else:
        ret = mx.nd.sqrt(x)
    return ret


@_handle_flat_arrays_in_out
def isnan(x: mx.nd.NDArray) -> mx.nd.NDArray:
    ret = mx.nd.contrib.isnan(x).astype("bool")
    return ret


@_handle_flat_arrays_in_out
def less(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.lesser(x1, x2).astype("bool")


@_handle_flat_arrays_in_out
def logical_xor(x1: mx.nd.NDArray, x2: mx.nd.NDArray, dtype: ["bool"]) -> mx.nd.NDArray:
    ret = mx.nd.logical_xor(x1, x2, dtype).astype("bool")
    return ret


@_handle_flat_arrays_in_out
def logical_not(x: mx.nd.NDArray) -> mx.nd.NDArray:
    ret = mx.nd.logical_not(x)
    return ret


@_handle_flat_arrays_in_out
def acos(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.acos(x)
    else:
        ret = mx.nd.arccos(x)
    return ret


@_handle_flat_arrays_in_out
def asin(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.asin(x)
    else:
        ret = mx.nd.arcsin(x)
    return ret


@_handle_flat_arrays_in_out
def logical_and(x1: mx.nd.NDArray, x2: mx.nd.NDArray, dtype: ["bool"]) -> mx.nd.NDArray:
    ret = mx.nd.logical_and(x1, x2, dtype).astype("bool")
    return ret


@_handle_flat_arrays_in_out
def logical_or(x1: mx.nd.NDArray, x2: mx.nd.NDArray, dtype: ["bool"]) -> mx.nd.NDArray:
    ret = mx.nd.logical_or(x1, x2, dtype).astype("bool")
    return ret


@_handle_flat_arrays_in_out
def multiply(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.multiply(x1, x2)


@_handle_flat_arrays_in_out
def acosh(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.acosh(x)
    else:
        ret = mx.nd.arccosh(x)
    return ret


@_handle_flat_arrays_in_out
def sin(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.sin(x)
    else:
        ret = mx.nd.sin(x)
    return ret


@_handle_flat_arrays_in_out
def negative(x: Union[float, mx.nd.NDArray]) -> mx.nd.NDArray:
    return mx.np.negative(x)


@_handle_flat_arrays_in_out
def tanh(
    x: mx.nd.NDArray,
) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.tanh(x)
    else:
        ret = mx.nd.tanh(x)
    return ret


@_handle_flat_arrays_in_out
def bitwise_or(
    x1: Union[int, mx.nd.NDArray],
    x2: Union[int, mx.nd.NDArray],
) -> mx.nd.ndarray.NDArray:
    return mx.numpy.bitwise_or(x1, x2)


@_handle_flat_arrays_in_out
def sinh(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.sinh(x)
    else:
        ret = mx.nd.sinh(x)
    return ret


@_handle_flat_arrays_in_out
def square(x: mx.nd.NDArray) -> mx.nd.NDArray:
    ret = mx.nd.square(x)
    return ret


@_handle_flat_arrays_in_out
def round(x: mx.nd.NDArray) -> mx.nd.NDArray:
    ret = mx.nd.round(x)
    return ret


@_handle_flat_arrays_in_out
def trunc(x: mx.nd.NDArray) -> mx.nd.ndarray.NDArray:
    ret = mx.np.trunc(x)
    return ret


@_handle_flat_arrays_in_out
def subtract(
    x1: Union[float, mx.nd.NDArray],
    x2: Union[float, mx.nd.NDArray],
) -> mx.nd.NDArray:
    return mx.nd.subtract(x1, x2)


@_handle_flat_arrays_in_out
def abs(x: Union[float, mx.nd.NDArray]) -> mx.nd.ndarray.NDArray:
    return mx.nd.abs(x)


def cos(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.cos(x)
    else:
        ret = mx.nd.cos(x)
    return ret


def exp(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.exp(x)
    else:
        ret = mx.nd.exp(x)
    return ret


def tan(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.tan(x)
    else:
        ret = mx.nd.tan(x)
    return ret


@_handle_flat_arrays_in_out
def atan(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        return math.atan(x)
    else:
        return mx.nd.arctan(x)


@_handle_flat_arrays_in_out
def atanh(x: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        return math.atanh(x)
    else:
        return mx.nd.arctanh(x)


@_handle_flat_arrays_in_out
def atan2(x: mx.nd.NDArray, y: mx.nd.NDArray) -> mx.nd.NDArray:
    if isinstance(x, float):
        ret = math.atan2(x, y)
    else:
        ret = mx.np.arctan2(x.as_np_ndarray(), y.as_np_ndarray()).as_nd_ndarray()
    return ret


cosh = lambda x: math.cosh(x) if isinstance(x, float) else mx.nd.cosh(x)
asinh = lambda x: math.asinh(x) if isinstance(x, float) else mx.nd.arcsinh(x)
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
