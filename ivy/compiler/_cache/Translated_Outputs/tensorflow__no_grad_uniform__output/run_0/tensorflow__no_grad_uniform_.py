from .tensorflow__helpers import tensorflow_uniform_


def tensorflow__no_grad_uniform_(tensor, a, b, generator=None):
    return tensorflow_uniform_(tensor, a, b, generator=generator)
