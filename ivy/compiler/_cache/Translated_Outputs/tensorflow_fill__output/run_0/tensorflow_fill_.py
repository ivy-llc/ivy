from .tensorflow__helpers import tensorflow_full_like_1
from .tensorflow__helpers import tensorflow_inplace_update


def tensorflow_fill_(arr, value):
    ret = tensorflow_full_like_1(arr, value, dtype=arr.dtype, device=arr.device)
    arr = tensorflow_inplace_update(arr, ret)
    return arr
