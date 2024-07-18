import tensorflow as tf

from .tensorflow__helpers import tensorflow_empty_1
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_to_scalar_2


def tensorflow_empty(
    *args,
    size=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
    memory_format=None,
):
    if args and size:
        raise TypeError("empty() got multiple values for argument 'shape'")
    if size is None:
        size = (
            args[0]
            if isinstance(args[0], (tuple, list, tuple, tf.TensorShape))
            else args
        )
    if isinstance(size, (tuple, list)):
        size = tuple(
            tensorflow_to_scalar_2(s) if tensorflow_is_array(s) else s for s in size
        )
    return tensorflow_empty_1(shape=size, dtype=dtype, device=device, out=out)
