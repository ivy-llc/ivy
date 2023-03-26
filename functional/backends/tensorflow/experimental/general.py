# global
<<<<<<< HEAD
from typing import Optional
=======
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
import tensorflow as tf

# local
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16",)}, backend_version)
def isin(
    elements: tf.Tensor,
    test_elements: tf.Tensor,
    /,
    *,
<<<<<<< HEAD
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
=======
    assume_unique: bool = False,
    invert: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
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
