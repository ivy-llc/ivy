#global
from typing import Union, Tuple
from collections import namedtuple

import tensorflow as tf

def unique_counts(
    x: Union[tf.Tensor, tf.Variable],
    /,
) -> Tuple[Union[tf.Tensor, tf.Variable], Union[tf.Tensor, tf.Variable]]:
    Results = namedtuple("Results", ["values", "counts"])
    v, _, c = tf.unique_with_counts(tf.sort(tf.reshape(x, [-1])))
    v = tf.cast(v, dtype=x.dtype)
    c = tf.cast(c, dtype=tf.int64)
    return Results(v, c)
