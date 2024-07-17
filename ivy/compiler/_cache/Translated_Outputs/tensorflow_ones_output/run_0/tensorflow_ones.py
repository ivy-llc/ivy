import tensorflow as tf

from .tensorflow__helpers import tensorflow_ones_1


def tensorflow_ones(
    *args, size=None, out=None, dtype=None, device=None, requires_grad=False
):
    if args and size:
        raise TypeError("ones() got multiple values for argument 'shape'")
    if size is None:
        size = (
            args[0]
            if isinstance(args[0], (tuple, list, tuple, tf.TensorShape))
            else args
        )
    return tensorflow_ones_1(shape=size, dtype=dtype, device=device, out=out)
