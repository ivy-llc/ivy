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
    input_shape = elements.shape

    if tf.rank(elements) == 0:
        elements = tf.reshape(elements, [1])
    if tf.rank(test_elements) == 0:
        test_elements = tf.reshape(test_elements, [1])
    if not assume_unique:
        test_elements = tf.unique(tf.reshape(test_elements, [-1]))[0]

    elements = tf.reshape(elements, [-1])
    test_elements = tf.reshape(test_elements, [-1])

    output = tf.reduce_any(
        tf.equal(tf.expand_dims(elements, -1), test_elements), axis=-1
    )
    return tf.reshape(output, input_shape) ^ invert
