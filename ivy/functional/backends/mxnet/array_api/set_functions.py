# global
from typing import Tuple
import mxnet as mx


def unique_inverse(x: mx.ndarray) \
        -> Tuple[mx.ndarray, mx.ndarray]:
    values, inverse_indices = mx.np.unique(x, return_inverse=True)
    return values, inverse_indices
