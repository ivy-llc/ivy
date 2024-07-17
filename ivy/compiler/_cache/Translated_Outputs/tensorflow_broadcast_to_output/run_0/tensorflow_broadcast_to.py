import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Union
from typing import Sequence

from .tensorflow__helpers import tensorflow_check_shapes_broadcastable


def tensorflow_broadcast_to(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shape: Union[tf.TensorShape, Sequence[int]],
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    tensorflow_check_shapes_broadcastable(x.shape, shape)
    if tensorflow.rank(x) > len(shape):
        return tensorflow.broadcast_to(tensorflow.reshape(x, -1), shape)
    return tensorflow.broadcast_to(x, shape)
