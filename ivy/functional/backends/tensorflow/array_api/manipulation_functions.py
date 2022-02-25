# global
from typing import Union, Tuple
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local

def roll(x: Tensor, shift: Union[int, Tuple[int]], axis: Union[int, Tuple[int]]=None) -> Tensor:
    return tf.roll(x, shift, axis)