from .tensorflow__helpers import tensorflow_fill_


def tensorflow__no_grad_fill_(tensor, val):
    return tensorflow_fill_(tensor, val)
