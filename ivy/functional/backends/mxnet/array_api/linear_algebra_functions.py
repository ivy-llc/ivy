# global
import mxnet as mx
from typing import Union, Tuple, Optional


def cross(x1: mx.ndarray.ndarray.NDArray, x2: mx.ndarray.ndarray.NDArray, /, *,
          axis: int = -1) -> mx.ndarray.ndarray.NDArray:
    return mx.linalg.cross(x1, x2, axis=axis)
