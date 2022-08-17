# global
from typing import Any
import ivy


def add(x, y):
    return ivy.add(x, y)


add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def tan(x):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def concatenate(operands, dimension):
    return ivy.concat(operands, dimension)


def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


full.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def max(x: Any, y: Any):
    return ivy.maximum(x, y)


max.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def abs(x):
    return ivy.abs(x)


abs.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def sqrt(x):
    return ivy.sqrt(x)


sqrt.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def acos(x):
    return ivy.acos(x)


acos.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def sin(x):
    return ivy.sin(x)


sin.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def sign(x):
    return ivy.sign(x)


def asin(x):
    return ivy.asin(x)


def sinh(x):
    return ivy.sinh(x)


def atan2(x, y):
    return ivy.atan2(x, y)


def atan(x):
    return ivy.atan(x)
