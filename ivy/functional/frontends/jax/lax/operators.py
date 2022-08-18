# global
from typing import Any
import ivy


def add(x, y):
    return ivy.add(x, y)


def tan(x):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16",)}


def concatenate(operands, dimension):
    return ivy.concat(operands, dimension)


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


def atan(x):
    return ivy.atan(x)


def ceil(x):
    return ivy.ceil(x)


ceil.unsupported_dtypes = {"torch": ("float16",)}


def bitwise_and(x, y):
    return ivy.bitwise_and(x, y)


def bitwise_or(x, y):
    return ivy.bitwise_or(x, y)


def bitwise_xor(x, y):
    return ivy.bitwise_xor(x, y)
