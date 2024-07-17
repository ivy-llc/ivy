from .tensorflow__helpers import tensorflow_inplace_update
from .tensorflow__helpers import tensorflow_zeros_like_1


def tensorflow_zero_(arr):
    ret = tensorflow_zeros_like_1(arr)
    arr = tensorflow_inplace_update(arr, ret)
    return arr
