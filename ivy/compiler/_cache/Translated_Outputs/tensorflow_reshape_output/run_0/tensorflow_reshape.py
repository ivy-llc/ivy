import tensorflow
import tensorflow as tf

from typing import Union
from typing import Optional
from typing import Sequence

from .tensorflow__helpers import tensorflow_reshape_1


def tensorflow_reshape(
    self: tensorflow.Tensor,
    /,
    shape: Union[tuple, tf.TensorShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[tensorflow.Tensor] = None,
):
    return tensorflow_reshape_1(
        self, shape, copy=copy, allowzero=allowzero, out=out, order=order
    )
