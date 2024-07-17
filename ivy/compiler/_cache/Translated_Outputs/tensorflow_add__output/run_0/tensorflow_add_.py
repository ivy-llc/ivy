from .tensorflow__helpers import tensorflow_add_2


def tensorflow_add_(arr, other, *, alpha=1):
    arr = tensorflow_add_2(arr, other, alpha=alpha)
    return arr
