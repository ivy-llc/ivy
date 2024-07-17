from .tensorflow__helpers import tensorflow__numel
from .tensorflow__helpers import tensorflow_broadcast_to
from .tensorflow__helpers import tensorflow_expand_dims
from .tensorflow__helpers import tensorflow_reshape


def tensorflow__broadcast_to(input, target_shape):
    if tensorflow__numel(tuple(input.shape)) == tensorflow__numel(tuple(target_shape)):
        return tensorflow_reshape(input, target_shape)
    else:
        input = input if len(input.shape) else tensorflow_expand_dims(input, axis=0)
        return tensorflow_broadcast_to(input, target_shape)
