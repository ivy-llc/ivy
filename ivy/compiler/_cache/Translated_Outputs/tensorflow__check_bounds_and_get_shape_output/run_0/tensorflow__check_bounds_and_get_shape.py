import tensorflow
import tensorflow as tf

from .tensorflow__helpers import tensorflow_check_all_or_any_fn
from .tensorflow__helpers import tensorflow_check_equal
from .tensorflow__helpers import tensorflow_shape

backend_stack = []


def tensorflow__check_bounds_and_get_shape(low, high, shape):
    if shape is not None:
        tensorflow_check_all_or_any_fn(
            low,
            high,
            fn=lambda x: isinstance(x, (int, float)),
            type="all",
            message="low and high bounds must be numerics when shape is specified",
        )
        return tuple(shape)
    valid_types = (tensorflow.Tensor,)
    if len(backend_stack) == 0:
        valid_types = valid_types + (tf.Tensor,)
    else:
        valid_types = valid_types + (tf.Tensor,)
    if isinstance(low, valid_types):
        if isinstance(high, valid_types):
            tensorflow_check_equal(
                tensorflow_shape(low), tensorflow_shape(high), as_array=False
            )
        return tensorflow_shape(low)
    if isinstance(high, valid_types):
        return tensorflow_shape(high)
    return tuple(())
