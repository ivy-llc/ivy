import tensorflow
import tensorflow as tf

from typing import Tuple
from typing import Union
from typing import List
from typing import Optional

from .tensorflow__helpers import tensorflow_stack_1


def tensorflow_stack(
    self: tensorflow.Tensor,
    /,
    arrays: Union[
        Tuple[Union[tensorflow.Tensor, tf.Tensor]],
        List[Union[tensorflow.Tensor, tf.Tensor]],
    ],
    *,
    axis: int = 0,
    out: Optional[tensorflow.Tensor] = None,
):
    if not isinstance(arrays, (tuple, list)):
        arrays = [arrays]
    if isinstance(arrays, tuple):
        x = (self,) + arrays
    else:
        x = [self] + arrays
    return tensorflow_stack_1(x, axis=axis, out=out)
