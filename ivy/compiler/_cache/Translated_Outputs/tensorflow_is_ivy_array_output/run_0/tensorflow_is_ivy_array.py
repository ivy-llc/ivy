import tensorflow
import tensorflow as tf

from typing import Union
from typing import Optional

from .tensorflow__helpers import tensorflow_is_native_array


def tensorflow_is_ivy_array(
    x: Union[tensorflow.Tensor, tf.Tensor], /, *, exclusive: Optional[bool] = False
):
    return isinstance(x, tensorflow.Tensor) and tensorflow_is_native_array(
        x, exclusive=exclusive
    )
