import tensorflow
import tensorflow as tf
import numpy as np

from numbers import Number
from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype
from .tensorflow__helpers import tensorflow_default_int_dtype
from .tensorflow__helpers import tensorflow_dtype
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_nested_argwhere


def tensorflow_is_uint_dtype(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "uint" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, np.unsignedinteger)
    elif isinstance(dtype_in, (list, tuple, dict)):
        return tensorflow_nested_argwhere(
            dtype_in,
            lambda x: isinstance(x, np.unsignedinteger)
            or tensorflow_is_array(x)
            and "uint" in tensorflow_dtype(x),
        )
    return "uint" in tensorflow_as_ivy_dtype(dtype_in)
