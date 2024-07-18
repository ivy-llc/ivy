import tensorflow
import tensorflow as tf

from typing import Union

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_to_scalar_1


@tensorflow_handle_array_like_without_promotion
def tensorflow_to_scalar(x: Union[tensorflow.Tensor, tf.Tensor], /):
    return tensorflow_to_scalar_1(x)
