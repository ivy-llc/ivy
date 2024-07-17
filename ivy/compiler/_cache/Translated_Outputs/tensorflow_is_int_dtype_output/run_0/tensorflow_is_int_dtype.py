import tensorflow
import tensorflow as tf
import numpy as np

from numbers import Number
from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1
from .tensorflow__helpers import tensorflow_default_int_dtype
from .tensorflow__helpers import tensorflow_dtype
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_nested_argwhere


def tensorflow_is_int_dtype(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "int" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (int, np.integer)) and not isinstance(
            dtype_in, bool
        )
    elif isinstance(dtype_in, (list, tuple, dict)):

        def nested_fun(x):
            return (
                isinstance(x, (int, np.integer))
                or tensorflow_is_array(x)
                and "int" in tensorflow_dtype(x)
            ) and x is not bool

        return bool(tensorflow_nested_argwhere(dtype_in, nested_fun))
    return "int" in tensorflow_as_ivy_dtype_1(dtype_in)
