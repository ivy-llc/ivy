import tensorflow
import tensorflow as tf
import numpy as np

from typing import Union
from numbers import Number

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1
from .tensorflow__helpers import tensorflow_dtype
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_nested_argwhere


def tensorflow_is_bool_dtype(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "bool" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (bool, np.bool_)) and not isinstance(dtype_in, bool)
    elif isinstance(dtype_in, (list, tuple, dict)):
        return bool(
            tensorflow_nested_argwhere(
                dtype_in, lambda x: isinstance(x, (bool, np.bool_)) and x is not int
            )
        )
    return "bool" in tensorflow_as_ivy_dtype_1(dtype_in)
