import tensorflow
import tensorflow as tf
import numpy as np

from typing import Any

from .tensorflow__helpers import tensorflow_is_native_array


def tensorflow__to_ivy(x: Any):
    if isinstance(x, tensorflow.Tensor):
        return x
    elif isinstance(x, tf.TensorShape):
        return tuple(x)
    elif isinstance(x, dict):
        return x.to_ivy()
    if tensorflow_is_native_array(x) or isinstance(x, np.ndarray):
        return tensorflow.convert_to_tensor(x)
    return x
