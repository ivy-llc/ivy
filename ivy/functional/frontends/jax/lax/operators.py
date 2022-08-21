# global
from typing import Any
import ivy


def add(x, y):
    return ivy.add(x, y)


def tan(x):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16",)}


def concatenate(operands, dimension):
    return ivy.concat(operands, axis=dimension)


def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


def max(x: Any, y: Any):
    return ivy.maximum(x, y)


def abs(x):
    return ivy.abs(x)


def sqrt(x):
    return ivy.sqrt(x)


sqrt.unsupported_dtypes = {"torch": ("float16",)}


def acos(x):
    return ivy.acos(x)


acos.unsupported_dtypes = {"torch": ("float16",)}


def sin(x):
    return ivy.sin(x)


sin.unsupported_dtypes = {"torch": ("float16",)}


def sign(x):
    return ivy.sign(x)


def asin(x):
    return ivy.asin(x)


def sinh(x):
    return ivy.sinh(x)


def atan2(x, y):
    return ivy.atan2(x, y)


def min(x, y):
    return ivy.minimum(x, y)


def mul(x, y):
    return ivy.multiply(x, y)


def eq(x, y):
    return ivy.equal(x, y)


eq.unsupported_dtypes = {"torch": ("bfloat16",)}


def atan(x):
    return ivy.atan(x)


def ceil(x):
    return ivy.ceil(x)


ceil.unsupported_dtypes = {"torch": ("float16",)}


def bitwise_and(x, y):
    return ivy.bitwise_and(x, y)


def bitwise_or(x, y):
    return ivy.bitwise_or(x, y)


def bitwise_not(x, y):
    return ivy.bitwise_invert(x, y)


def neg(x):
    return ivy.negative(x)


neg.unsupported_dtypes = {"torch": ("bfloat16",)}


def argmax(operand, axis, index_dtype):
    return ivy.astype(ivy.argmax(operand, axis=axis), index_dtype)


argmax.unsupported_dtypes = {"torch": ("bfloat16",)}


def argmin(operand, axis, index_dtype):
    return ivy.astype(ivy.argmin(operand, axis=axis), index_dtype)


argmin.unsupported_dtypes = {"torch": ("bfloat16",)}


def bitwise_xor(x, y):
    return ivy.bitwise_xor(x, y)


def full_like(x, fill_value, dtype=None, shape=None):
    if shape is None:
        return ivy.full_like(x, fill_value, dtype=dtype)
    return ivy.full(shape, fill_value, dtype=dtype)


def exp(x):
    return ivy.exp(x)


exp.unsupported_dtypes = {"torch": ("float16",)}


def convert_element_type(operand, new_dtype):
    return ivy.astype(operand, new_dtype)


def cumprod(operand, axis=0, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumprod(ivy.flip(operand), axis))
    return ivy.cumprod(operand, axis)


cumprod.unsupported_dtypes = {"torch": ("float16",)}


def cumsum(operand, axis=0, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumsum(ivy.flip(operand), axis))
    return ivy.cumsum(operand, axis)


cumsum.unsupported_dtypes = {"torch": ("float16",)}


def ge(x, y):
    return ivy.greater(x, y)
