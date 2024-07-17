from .tensorflow__helpers import tensorflow_prod
from .tensorflow__helpers import tensorflow_to_scalar


def tensorflow__numel(shape):
    shape = tuple(shape)
    return tensorflow_to_scalar(tensorflow_prod(shape)) if shape != () else 1
