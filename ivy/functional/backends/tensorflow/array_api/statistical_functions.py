# global
_round = round
import tensorflow as _tf
from typing import Tuple, Union


def min(x: _tf.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> _tf.Tensor:
    return _tf.math.reduce_min(x, axis = axis, keepdims = keepdims)