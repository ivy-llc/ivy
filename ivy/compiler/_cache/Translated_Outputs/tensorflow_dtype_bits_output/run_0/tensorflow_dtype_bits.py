import tensorflow
import numpy as np

from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1


def tensorflow_dtype_bits(dtype_in: Union[tensorflow.DType, str, np.dtype], /):
    dtype_str = tensorflow_as_ivy_dtype_1(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("tf.", "")
        .replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
        .replace("complex", "")
    )
