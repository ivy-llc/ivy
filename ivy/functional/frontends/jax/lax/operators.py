# global
from typing import Any

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.jax.numpy import can_cast


@to_ivy_arrays_and_back
def abs(x):
    return ivy.abs(x)


@to_ivy_arrays_and_back
def acos(x):
    return ivy.acos(x)


@to_ivy_arrays_and_back
def add(x, y):
    return ivy.add(x, y)


@to_ivy_arrays_and_back
def argmax(operand, axis, index_dtype):
    return ivy.astype(ivy.argmax(operand, axis=axis), index_dtype)


@to_ivy_arrays_and_back
def argmin(operand, axis, index_dtype):
    return ivy.astype(ivy.argmin(operand, axis=axis), index_dtype)


@to_ivy_arrays_and_back
def asin(x):
    return ivy.asin(x)


@to_ivy_arrays_and_back
def atan(x):
    return ivy.atan(x)


@to_ivy_arrays_and_back
def atan2(x, y):
    return ivy.atan2(x, y)


@to_ivy_arrays_and_back
def bitwise_and(x, y):
    return ivy.bitwise_and(x, y)


@to_ivy_arrays_and_back
def bitwise_not(x):
    return ivy.bitwise_invert(x)


@to_ivy_arrays_and_back
def bitwise_or(x, y):
    return ivy.bitwise_or(x, y)


@to_ivy_arrays_and_back
def bitwise_xor(x, y):
    return ivy.bitwise_xor(x, y)


@to_ivy_arrays_and_back
def broadcast(operand, sizes):
    ret = ivy.zeros(tuple(sizes) + tuple(ivy.shape(operand)), dtype=ivy.dtype(operand))
    return ret + operand


@to_ivy_arrays_and_back
def ceil(x):
    return ivy.ceil(x)


@to_ivy_arrays_and_back
def clamp(min, x, max):
    return ivy.clip(x, min, max)


@to_ivy_arrays_and_back
def concatenate(operands, dimension):
    return ivy.concat(operands, axis=dimension)


@to_ivy_arrays_and_back
def conv(
    lhs, rhs, window_strides, padding, precision=None, preferred_element_type=None
):
    if preferred_element_type:
        lhs = ivy.astype(lhs, dtype=preferred_element_type)
        rhs = ivy.astype(rhs, dtype=preferred_element_type)
    return ivy.conv2d(lhs, rhs, window_strides, padding)


@to_ivy_arrays_and_back
def conv_transpose(
    lhs,
    rhs,
    strides,
    padding,
    rhs_dilation=None,
    dimension_numbers=None,
    transpose_kernel=False,
    precision=None,
    preferred_element_type=None,
):
    if preferred_element_type:
        lhs = ivy.astype(lhs, dtype=preferred_element_type)
        rhs = ivy.astype(rhs, dtype=preferred_element_type)
    return ivy.conv2d_transpose(lhs, rhs, strides, padding)


@to_ivy_arrays_and_back
def convert_element_type(operand, new_dtype):
    assert can_cast(ivy.dtype(operand), new_dtype), "Cannot cast from {} to {}".format(
        ivy.dtype(operand), new_dtype
    )
    return ivy.astype(operand, new_dtype, copy=False)


@to_ivy_arrays_and_back
def cos(x):
    return ivy.cos(x)


@to_ivy_arrays_and_back
def cosh(x):
    return ivy.cosh(x)


@to_ivy_arrays_and_back
def cumprod(operand, axis=None, reverse=False):
    dtype = ivy.dtype(operand)
    return ivy.cumprod(operand, axis=axis, reverse=reverse).astype(dtype)


@to_ivy_arrays_and_back
def cumsum(operand, axis=None, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumsum(ivy.flip(operand), axis=axis, dtype=operand.dtype))
    return ivy.cumsum(operand, axis=axis, dtype=operand.dtype)


@to_ivy_arrays_and_back
def div(x, y):
    return ivy.astype(ivy.divide(x, y), x.dtype)


@to_ivy_arrays_and_back
def dot(lhs, rhs, precision=None, preferred_element_type=None):
    ret = ivy.matmul(lhs, rhs)
    if preferred_element_type:
        ret = ivy.astype(ret, preferred_element_type, copy=False)
    return ret


@to_ivy_arrays_and_back
def eq(x, y):
    return ivy.equal(x, y)


@to_ivy_arrays_and_back
def erf(x):
    return ivy.erf(x)


@to_ivy_arrays_and_back
def exp(x):
    return ivy.exp(x)


@to_ivy_arrays_and_back
def expand_dims(array, dimensions):
    return ivy.expand_dims(array, axis=dimensions)


@to_ivy_arrays_and_back
def expm1(x):
    return ivy.expm1(x)


@to_ivy_arrays_and_back
def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


@to_ivy_arrays_and_back
def full_like(x, fill_value, dtype=None, shape=None):
    if shape is None:
        return ivy.full_like(x, fill_value, dtype=dtype)
    return ivy.full(shape, fill_value, dtype=dtype)


@to_ivy_arrays_and_back
def ge(x, y):
    return ivy.greater_equal(x, y)


@to_ivy_arrays_and_back
def gt(x, y):
    return ivy.greater(x, y)


@to_ivy_arrays_and_back
def int16(x):
    return ivy.int16(x)


@to_ivy_arrays_and_back
def le(x, y):
    return ivy.less_equal(x, y)


@to_ivy_arrays_and_back
def log(x):
    return ivy.log(x)


@to_ivy_arrays_and_back
def log1p(x):
    return ivy.log1p(x)


@to_ivy_arrays_and_back
def lt(x, y):
    return ivy.less(x, y)


@to_ivy_arrays_and_back
def max(x: Any, y: Any):
    return ivy.maximum(x, y)


@to_ivy_arrays_and_back
def min(x, y):
    return ivy.minimum(x, y)


@to_ivy_arrays_and_back
def mul(x, y):
    return ivy.multiply(x, y)


@to_ivy_arrays_and_back
def ne(x, y):
    return ivy.not_equal(x, y)


@to_ivy_arrays_and_back
def neg(x):
    return ivy.negative(x)


@to_ivy_arrays_and_back
def pow(x, y):
    return ivy.pow(x, y)


@to_ivy_arrays_and_back
def reciprocal(x):
    return ivy.reciprocal(x)


@to_ivy_arrays_and_back
def rem(x, y):
    return ivy.remainder(ivy.abs(x), ivy.abs(y)) * ivy.sign(x)


@to_ivy_arrays_and_back
def reshape(operand, new_sizes, dimensions=None):
    if dimensions:
        operand = ivy.permute_dims(operand, dimensions)
    return ivy.reshape(operand, new_sizes)


@to_ivy_arrays_and_back
def rev(operand, dimensions):
    return ivy.flip(operand, axis=dimensions)


@to_ivy_arrays_and_back
def round(x, rounding_method=1):
    if rounding_method == 0:
        ret = ivy.where(
            ivy.less(x, 0),
            ivy.ceil(x) - (ivy.ceil(x) - ivy.floor(x)),
            ivy.ceil(x),
        )
    elif rounding_method == 1:
        ret = ivy.ceil(x)
        ret = ivy.where(ivy.remainder(ret, 2) == 0, ret, ret - 1)
    return ivy.where(ivy.abs(x - ivy.floor(x) - 0.5) < 1e-7, ret, ivy.round(x))


@to_ivy_arrays_and_back
def rsqrt(x):
    return ivy.reciprocal(ivy.sqrt(x))


@to_ivy_arrays_and_back
def shift_left(x, y):
    return ivy.bitwise_left_shift(x, y)


@to_ivy_arrays_and_back
def sign(x):
    return ivy.sign(x)


@to_ivy_arrays_and_back
def sin(x):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def sinh(x):
    return ivy.sinh(x)


@to_ivy_arrays_and_back
def sort(operand, dimension=-1, is_stable=True, num_keys=1):
    return ivy.sort(operand, axis=dimension, stable=is_stable)


@to_ivy_arrays_and_back
def sqrt(x):
    return ivy.sqrt(x)


@to_ivy_arrays_and_back
def square(x):
    return ivy.square(x)


@to_ivy_arrays_and_back
def sub(x, y):
    return ivy.subtract(x, y)


@to_ivy_arrays_and_back
def tan(x):
    return ivy.tan(x)


@to_ivy_arrays_and_back
def transpose(operand, permutation):
    return ivy.permute_dims(operand, permutation)


@to_ivy_arrays_and_back
def shift_right_logical(x, y):
    return ivy.bitwise_right_shift(x, y)


@to_ivy_arrays_and_back
def asinh(x):
    return ivy.asinh(x)


@to_ivy_arrays_and_back
def atanh(x):
    return ivy.atanh(x)


@to_ivy_arrays_and_back
def select(pred, on_true, on_false):
    return ivy.where(pred, on_true, on_false)
