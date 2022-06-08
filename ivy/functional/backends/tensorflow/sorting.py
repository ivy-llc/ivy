# global
import tensorflow as tf
from typing import Union

# local


def argsort(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Union[tf.Tensor, tf.Variable]:
    if tf.convert_to_tensor(x).dtype.is_bool:
        if descending:
            ret = tf.argsort(
                tf.cast(x, dtype=tf.int32),
                axis=axis,
                direction="DESCENDING",
                stable=stable,
            )
        else:
            ret = tf.argsort(
                tf.cast(x, dtype=tf.int32),
                axis=axis,
                direction="ASCENDING",
                stable=stable,
            )
    else:
        if descending:
            ret = tf.argsort(
                tf.convert_to_tensor(x),
                axis=axis,
                direction="DESCENDING",
                stable=stable,
            )
        else:
            ret = tf.argsort(
                tf.convert_to_tensor(x),
                axis=axis,
                direction="ASCENDING",
                stable=stable
            )
    return ret


def sort(
    x: Union[tf.Tensor, tf.Variable],
    axis: int = -1,
    descending: bool = False,
    stable: bool = True
) -> Union[tf.Tensor, tf.Variable]:
    if tf.convert_to_tensor(x).dtype.is_bool:
        if descending:
            res = tf.sort(tf.cast(x, dtype=tf.int32),
                          axis=axis,
                          direction="DESCENDING")
            ret = tf.cast(res, tf.bool)
        else:
            res = tf.sort(tf.cast(x, dtype=tf.int32),
                          axis=axis,
                          direction="ASCENDING")
            ret = tf.cast(res, tf.bool)
    else:
        if descending:
            ret = tf.sort(tf.convert_to_tensor(x),
                          axis=axis,
                          direction="DESCENDING")
        else:
            ret = tf.sort(tf.convert_to_tensor(x),
                          axis=axis,
                          direction="ASCENDING")
    return ret
