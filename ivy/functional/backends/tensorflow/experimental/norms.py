import tensorflow as tf
from typing import Union, Tuple


def l2_normalize(x: Union[tf.Tensor, tf.Variable],
                 axis: int = None,
                 out=None
                 ) -> tf.Tensor:
    return tf.nn.l2_normalize(x, axis=axis)

