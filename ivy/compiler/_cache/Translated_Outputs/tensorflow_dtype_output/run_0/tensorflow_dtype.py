import tensorflow
import numpy as np

from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1
from .tensorflow__helpers import tensorflow_as_native_dtype


def tensorflow_dtype(
    x: Union[tensorflow.Tensor, tensorflow.Variable, np.ndarray],
    *,
    as_native: bool = False,
):
    if as_native:
        return tensorflow_as_native_dtype(x.dtype)
    return tensorflow_as_ivy_dtype_1(x.dtype)
