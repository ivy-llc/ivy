from typing import Optional
import tensorflow as tf


def isin(
    elements: tf.Tensor,
    test_elements: tf.Tensor,
    /,
    *,
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
) -> tf.Tensor:
    if not assume_unique:
        test_elements = tf.unique(tf.reshape(test_elements, [-1]))[0]
    return tf.reduce_any(tf.equal(elements[..., tf.newaxis], test_elements), axis=-1) ^ invert
