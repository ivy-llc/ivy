import tensorflow

from typing import Union

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_to_numpy


@tensorflow_handle_array_like_without_promotion
def tensorflow_to_scalar(x: Union[tensorflow.Tensor, tensorflow.Variable], /):
    ret = tensorflow_to_numpy(x).item()
    if x.dtype == tensorflow.bfloat16:
        return float(ret)
    return ret
