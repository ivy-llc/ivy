# global
from typing import Any

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import inputs_to_ivy_arrays


@inputs_to_ivy_arrays
def abs(x):
    return ivy.abs(x)


@inputs_to_ivy_arrays
def acos(x):
    return ivy.acos(x)


@inputs_to_ivy_arrays
def add(x, y):
    return ivy.add(x, y)


@inputs_to_ivy_arrays
def argmax(operand, axis, index_dtype):
    return ivy.astype(ivy.argmax(operand, axis=axis), index_dtype)


@inputs_to_ivy_arrays
def argmin(operand, axis, index_dtype):
    return ivy.astype(ivy.argmin(operand, axis=axis), index_dtype)


@inputs_to_ivy_arrays
def asin(x):
    return ivy.asin(x)


@inputs_to_ivy_arrays
def atan(x):
    return ivy.atan(x)


@inputs_to_ivy_arrays
def atan2(x, y):
    return ivy.atan2(x, y)


@inputs_to_ivy_arrays
def bitwise_and(x, y):
    return ivy.bitwise_and(x, y)


@inputs_to_ivy_arrays
def bitwise_not(x):
    return ivy.bitwise_invert(x)


@inputs_to_ivy_arrays
def bitwise_or(x, y):
    return ivy.bitwise_or(x, y)


@inputs_to_ivy_arrays
def bitwise_xor(x, y):
    return ivy.bitwise_xor(x, y)


@inputs_to_ivy_arrays
def broadcast(operand, sizes):
    ret = ivy.zeros(tuple(sizes) + tuple(ivy.shape(operand)), dtype=ivy.dtype(operand))
    return ret + operand


@inputs_to_ivy_arrays
def ceil(x):
    return ivy.ceil(x)


@inputs_to_ivy_arrays
def clamp(min, x, max):
    return ivy.clip(x, min, max)


@inputs_to_ivy_arrays
def concatenate(operands, dimension):
    return ivy.concat(operands, axis=dimension)


@inputs_to_ivy_arrays
def conv(
    lhs, rhs, window_strides, padding, precision=None, preferred_element_type=None
):
    if preferred_element_type:
        lhs = ivy.astype(lhs, dtype=preferred_element_type)
        rhs = ivy.astype(rhs, dtype=preferred_element_type)
    return ivy.conv2d(lhs, rhs, window_strides, padding)


@inputs_to_ivy_arrays
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


@inputs_to_ivy_arrays
def convert_element_type(operand, new_dtype):
    return ivy.astype(operand, new_dtype)


@inputs_to_ivy_arrays
def cos(x):
    return ivy.cos(x)


@inputs_to_ivy_arrays
def cosh(x):
    return ivy.cosh(x)


@inputs_to_ivy_arrays
def cumprod(operand, axis=0, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumprod(ivy.flip(operand), axis, dtype=operand.dtype))
    return ivy.cumprod(operand, axis, dtype=operand.dtype)


@inputs_to_ivy_arrays
def cumsum(operand, axis=0, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumsum(ivy.flip(operand), axis, dtype=operand.dtype))
    return ivy.cumsum(operand, axis, dtype=operand.dtype)


@inputs_to_ivy_arrays
def div(x, y):
    return ivy.astype(ivy.divide(x, y), x.dtype)


@inputs_to_ivy_arrays
def dot(lhs, rhs, precision=None, preferred_element_type=None):
    if preferred_element_type:
        lhs = ivy.astype(lhs, dtype=preferred_element_type)
        rhs = ivy.astype(rhs, dtype=preferred_element_type)
    return ivy.tensordot(lhs, rhs)


@inputs_to_ivy_arrays
def eq(x, y):
    return ivy.equal(x, y)


@inputs_to_ivy_arrays
def erf(x):
    return ivy.erf(x)


@inputs_to_ivy_arrays
def exp(x):
    return ivy.exp(x)


@inputs_to_ivy_arrays
def expand_dims(array, dimensions):
    return ivy.expand_dims(array, axis=dimensions)


@inputs_to_ivy_arrays
def expm1(x):
    return ivy.expm1(x)


def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


@inputs_to_ivy_arrays
def full_like(x, fill_value, dtype=None, shape=None):
    if shape is None:
        return ivy.full_like(x, fill_value, dtype=dtype)
    return ivy.full(shape, fill_value, dtype=dtype)


@inputs_to_ivy_arrays
def ge(x, y):
    return ivy.greater_equal(x, y)


@inputs_to_ivy_arrays
def gt(x, y):
    return ivy.greater(x, y)


@inputs_to_ivy_arrays
def le(x, y):
    return ivy.less_equal(x, y)


@inputs_to_ivy_arrays
def log(x):
    return ivy.log(x)


@inputs_to_ivy_arrays
def log1p(x):
    return ivy.log1p(x)


@inputs_to_ivy_arrays
def lt(x, y):
    return ivy.less(x, y)


@inputs_to_ivy_arrays
def max(x: Any, y: Any):
    return ivy.maximum(x, y)


@inputs_to_ivy_arrays
def min(x, y):
    return ivy.minimum(x, y)


@inputs_to_ivy_arrays
def mul(x, y):
    return ivy.multiply(x, y)


@inputs_to_ivy_arrays
def ne(x, y):
    return ivy.not_equal(x, y)


@inputs_to_ivy_arrays
def neg(x):
    return ivy.negative(x)


@inputs_to_ivy_arrays
def pow(x, y):
    return ivy.pow(x, y)


@inputs_to_ivy_arrays
def reciprocal(x):
    return ivy.reciprocal(x)


@inputs_to_ivy_arrays
def rem(x, y):
    return ivy.remainder(ivy.abs(x), ivy.abs(y)) * ivy.sign(x)


@inputs_to_ivy_arrays
def reshape(operand, new_sizes, dimensions=None):
    if dimensions:
        operand = ivy.permute_dims(operand, dimensions)
    return ivy.reshape(operand, new_sizes)


@inputs_to_ivy_arrays
def rev(operand, dimensions):
    return ivy.flip(operand, axis=dimensions)


@inputs_to_ivy_arrays
def round(x):
    return ivy.round(x)


@inputs_to_ivy_arrays
def rsqrt(x):
    return ivy.reciprocal(ivy.sqrt(x))


@inputs_to_ivy_arrays
def shift_left(x, y):
    return ivy.bitwise_left_shift(x, y)


@inputs_to_ivy_arrays
def sign(x):
    return ivy.sign(x)


@inputs_to_ivy_arrays
def sin(x):
    return ivy.sin(x)


@inputs_to_ivy_arrays
def sinh(x):
    return ivy.sinh(x)


@inputs_to_ivy_arrays
def sort(operand, dimension=-1, is_stable=True, num_keys=1):
    return ivy.sort(operand, axis=dimension, stable=is_stable)


@inputs_to_ivy_arrays
def sqrt(x):
    return ivy.sqrt(x)


@inputs_to_ivy_arrays
def square(x):
    return ivy.square(x)


@inputs_to_ivy_arrays
def sub(x, y):
    return ivy.subtract(x, y)


@inputs_to_ivy_arrays
def tan(x):
    return ivy.tan(x)


@inputs_to_ivy_arrays
def transpose(operand, permutation):
    return ivy.permute_dims(operand, permutation)


@inputs_to_ivy_arrays
def shift_right_logical(x, y):
    return ivy.bitwise_right_shift(x, y)
