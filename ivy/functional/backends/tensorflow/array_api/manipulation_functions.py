# global
import tensorflow as tf
from typing import Union, Tuple, Optional, List
from tensorflow.python.types.core import Tensor


def flip(x: Tensor,
         axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
         -> Tensor:
    num_dims = len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        new_axis = list(range(num_dims))
    else:
        new_axis = axis
    if type(new_axis) is int:
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return tf.reverse(x, new_axis)

def squeeze(x: Tensor,
             axis: Union[int, Tuple[int], List[int]])\
             -> Tensor:

    if axis is ():
        return x
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        if x.shape[ax] > 1:
            raise ValueError
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise Exception('tried to squeeze a zero-dimensional input by axis {}'.format(axis))
    return tf.squeeze(x, axis)