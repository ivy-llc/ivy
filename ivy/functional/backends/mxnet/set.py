# global
from typing import Tuple
import mxnet as mx


def unique_inverse(x: mx.ndarray.ndarray.NDArray) \
        -> Tuple[mx.ndarray.ndarray.NDArray, mx.ndarray.ndarray.NDArray]:
    values, inverse_indices = mx.np.unique(x, return_inverse=True)
    return values, inverse_indices
