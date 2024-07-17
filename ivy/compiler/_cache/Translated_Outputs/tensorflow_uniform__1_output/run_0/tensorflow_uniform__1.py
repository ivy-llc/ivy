from .tensorflow__helpers import tensorflow__no_grad_uniform_


def tensorflow_uniform__1(tensor, a=0.0, b=1.0, generator=None):
    return tensorflow__no_grad_uniform_(tensor, a, b, generator)
