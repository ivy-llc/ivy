# global
import ivy
from typing import Any


def ones(shape, dtype=None, order="C", *, like=None) -> Any:
    return ivy.ones(shape, dtype=dtype)


ones.unsupported_dtypes = {"torch": ("float16",)}
