import tensorflow as tf
from typing import Union, Tuple


def l2_normalize(x: Union[tf.Tensor, tf.Variable],
                 axis: int = None,
                 out=None
                 ) -> tf.Tensor:
    if axis is None:
        axis = tuple(range(x.ndim))
    return tf.nn.l2_normalize(x, axis=axis)

