# global
from typing import Optional, Union

import tensorflow as tf


# msort
def msort(
    a: Union[tf.Tensor, tf.Variable, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.sort(a, axis=0)
