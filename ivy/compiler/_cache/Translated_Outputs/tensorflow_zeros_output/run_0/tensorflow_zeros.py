import tensorflow as tf

from .tensorflow__helpers import tensorflow_zeros_1


def tensorflow_zeros(
    *args, size=None, out=None, dtype=None, device=None, requires_grad=False
):
    if args and size:
        raise TypeError("zeros() got multiple values for argument 'shape'")
    if size is None:
        size = (
            args[0]
            if isinstance(args[0], (tuple, list, tuple, tf.TensorShape))
            else args
        )
    return tensorflow_zeros_1(shape=size, dtype=dtype, device=device, out=out)
