from typing import Union, Optional
import tensorflow as tf


def diagflat(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    k: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.diagflat(x, k=k)
