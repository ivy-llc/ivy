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


asin.unsupported_dtypes = {"torch": ("float16",)}


def sinh(x):
    return ivy.sinh(x)


sinh.unsupported_dtypes = {"torch": ("float16",)}


def atan2(x, y):
    return ivy.atan2(x, y)


atan2.unsupported_dtypes = {"torch": ("float16",)}


def min(x, y):
    return ivy.minimum(x, y)


def mul(x, y):
    return ivy.multiply(x, y)


def eq(x, y):
    return ivy.equal(x, y)


def atan(x):
    return ivy.atan(x)


atan.unsupported_dtypes = {"torch": ("float16",)}


def ceil(x):
    return ivy.ceil(x)


ceil.unsupported_dtypes = {"torch": ("float16",)}


def bitwise_and(x, y):
    return ivy.bitwise_and(x, y)


def bitwise_or(x, y):
    return ivy.bitwise_or(x, y)


def neg(x):
    return ivy.negative(x)


def argmax(operand, axis, index_dtype):
    return ivy.astype(ivy.argmax(operand, axis=axis), index_dtype)


def argmin(operand, axis, index_dtype):
    return ivy.astype(ivy.argmin(operand, axis=axis), index_dtype)


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
    return ivy.greater_equal(x, y)


def reshape(operand, new_sizes, dimensions=None):
    if dimensions:
        operand = ivy.permute_dims(operand, dimensions)
    return ivy.reshape(operand, new_sizes)


def reciprocal(x):
    return ivy.reciprocal(x)


reciprocal.unsupported_dtypes = {
    "torch": ("float16",),
    "tensorflow": (
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
    ),
}


def broadcast(operand, sizes):
    ret = ivy.zeros(tuple(sizes) + tuple(ivy.shape(operand)), dtype=ivy.dtype(operand))
    return ret + operand


def sort(operand, dimension=-1, is_stable=True, num_keys=1):
    return ivy.sort(operand, axis=dimension, stable=is_stable)


def le(x, y):
    return ivy.less_equal(x, y)


def ne(x, y):
    return ivy.not_equal(x, y)


def cosh(x):
    return ivy.cosh(x)


cosh.unsupported_dtypes = {"torch": ("float16",)}


def round(x):
    return ivy.round(x)


round.unsupported_dtypes = {"torch": ("float16",)}


def lt(x, y):
    return ivy.less(x, y)
