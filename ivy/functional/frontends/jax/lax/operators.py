# global
import ivy
from typing import Sequence, Any


def add(x, y):
    return ivy.add(x, y)


add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def tan(x):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def concatenate(operands: Sequence[Any], dimension: int) -> Any:
    return ivy.concat(operands, dimension)


def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


full.unsupported_dtypes = {"torch": ("float16", "bfloat16")}

def abs(x):
    return ivy.abs(x)

abs.ubsupported_dtypes = {"torch": ("float16", "bfloat16")}