from .tensorflow__helpers import tensorflow__no_grad_fill_


def tensorflow_ones_(tensor):
    return tensorflow__no_grad_fill_(tensor, 1.0)
