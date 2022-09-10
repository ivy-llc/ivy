# global
from typing import Any

# local
import ivy


def abs(x):
    return ivy.abs(x)


def acos(x):
    return ivy.acos(x)


def add(x, y):
    return ivy.add(x, y)


def argmax(operand, axis, index_dtype):
    return ivy.astype(ivy.argmax(operand, axis=axis), index_dtype)


def argmin(operand, axis, index_dtype):
    return ivy.astype(ivy.argmin(operand, axis=axis), index_dtype)


def asin(x):
    return ivy.asin(x)


def atan(x):
    return ivy.atan(x)


def atan2(x, y):
    return ivy.atan2(x, y)


def bitwise_and(x, y):
    return ivy.bitwise_and(x, y)


def bitwise_not(x):
    return ivy.bitwise_invert(x)


def bitwise_or(x, y):
    return ivy.bitwise_or(x, y)


def bitwise_xor(x, y):
    return ivy.bitwise_xor(x, y)


def broadcast(operand, sizes):
    ret = ivy.zeros(tuple(sizes) + tuple(ivy.shape(operand)), dtype=ivy.dtype(operand))
    return ret + operand


def ceil(x):
    return ivy.ceil(x)


def clamp(min, x, max):
    return ivy.clip(x, min, max)


def concatenate(operands, dimension):
    return ivy.concat(operands, axis=dimension)


def conv(
    lhs, rhs, window_strides, padding, precision=None, preferred_element_type=None
):
    if preferred_element_type:
        lhs = ivy.astype(lhs, dtype=preferred_element_type)
        rhs = ivy.astype(rhs, dtype=preferred_element_type)
    return ivy.conv2d(lhs, rhs, window_strides, padding)


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


def convert_element_type(operand, new_dtype):
    return ivy.astype(operand, new_dtype)


def cos(x):
    return ivy.cos(x)


def cosh(x):
    return ivy.cosh(x)


def cumprod(operand, axis=0, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumprod(ivy.flip(operand), axis, dtype=operand.dtype))
    return ivy.cumprod(operand, axis, dtype=operand.dtype)


def cumsum(operand, axis=0, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumsum(ivy.flip(operand), axis, dtype=operand.dtype))
    return ivy.cumsum(operand, axis, dtype=operand.dtype)


def div(x, y):
    return ivy.astype(ivy.divide(x, y), x.dtype)


def dot(lhs, rhs, precision=None, preferred_element_type=None):
    if preferred_element_type:
        lhs = ivy.astype(lhs, dtype=preferred_element_type)
        rhs = ivy.astype(rhs, dtype=preferred_element_type)
    return ivy.tensordot(lhs, rhs)


def eq(x, y):
    return ivy.equal(x, y)


def exp(x):
    return ivy.exp(x)


def expm1(x):
    return ivy.expm1(x)


def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None, shape=None):
    if shape is None:
        return ivy.full_like(x, fill_value, dtype=dtype)
    return ivy.full(shape, fill_value, dtype=dtype)


def ge(x, y):
    return ivy.greater_equal(x, y)


def gt(x, y):
    return ivy.greater(x, y)


def le(x, y):
    return ivy.less_equal(x, y)


def log(x):
    return ivy.log(x)


def log1p(x):
    return ivy.log1p(x)


def lt(x, y):
    return ivy.less(x, y)


def max(x: Any, y: Any):
    return ivy.maximum(x, y)


def min(x, y):
    return ivy.minimum(x, y)


def mul(x, y):
    return ivy.multiply(x, y)


def ne(x, y):
    return ivy.not_equal(x, y)


def neg(x):
    return ivy.negative(x)


def pow(x, y):
    return ivy.pow(x, y)


def reciprocal(x):
    return ivy.reciprocal(x)


def reshape(operand, new_sizes, dimensions=None):
    if dimensions:
        operand = ivy.permute_dims(operand, dimensions)
    return ivy.reshape(operand, new_sizes)


def rev(operand, dimensions):
    return ivy.flip(operand, axis=dimensions)


def round(x):
    return ivy.round(x)


def rsqrt(x):
    return ivy.reciprocal(ivy.sqrt(x))


def sign(x):
    return ivy.sign(x)


def sin(x):
    return ivy.sin(x)


def sinh(x):
    return ivy.sinh(x)


def sort(operand, dimension=-1, is_stable=True, num_keys=1):
    return ivy.sort(operand, axis=dimension, stable=is_stable)


def sqrt(x):
    return ivy.sqrt(x)


def tan(x):
    return ivy.tan(x)


def transpose(operand, permutation):
    return ivy.permute_dims(operand, permutation)
