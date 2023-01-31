# global
import tensorflow as tf
from typing import Union, Optional


# msort
def msort(
        a: Union[tf.Tensor, tf.Variable, list, tuple],
        /,
        *,
        out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.sort(a, axis=0)


# lexsort
def lexsort(
        keys: Union[tf.Tensor, tf.Variable],
        /,
        *,
        axis: int = -1,
) -> Union[tf.Tensor, tf.Variable]:
    size = keys.shape.as_list()[0]
    if size == 0:
        raise TypeError('need sequence of keys with len > 0 in lexsort')
    result = tf.argsort(keys[0], axis=axis, stable=True)
    if size == 1:
        return result
    for i in range(1, size):
        key = keys[i]
        ind = key[result]
        temp = tf.argsort(ind, axis=axis, stable=True)
        result = result[temp]
    return result
