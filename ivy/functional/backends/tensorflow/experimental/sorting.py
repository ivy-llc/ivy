# global
import tensorflow as tf
from typing import Union, Optional


# invert_permutation
def invert_permutation(
    x: Union[tf.Tensor, tf.Variable, list, tuple],
    /,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.invert_permutation(x)


# lexsort
def lexsort(
    keys: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = -1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    shape = keys.shape.as_list()
    if len(shape) == 1:
        return tf.cast(tf.argsort(keys, axis=axis, stable=True), tf.int64)
    if shape[0] == 0:
        raise TypeError("need sequence of keys with len > 0 in lexsort")
    if len(shape) == 2 and shape[1] == 1:
        return tf.cast(tf.convert_to_tensor([0]), tf.int64)
    result = tf.argsort(keys[0], axis=axis, stable=True)
    if shape[0] == 1:
        return tf.cast(result, tf.int64)
    for i in range(1, shape[0]):
        key = keys[i]
        ind = tf.gather(key, result)
        temp = tf.argsort(ind, axis=axis, stable=True)
        result = tf.gather(result, temp)
    return tf.cast(result, tf.int64)
