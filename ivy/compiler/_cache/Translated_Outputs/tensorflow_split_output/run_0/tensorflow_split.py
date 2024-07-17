import tensorflow
import tensorflow as tf

from typing import Union
from typing import Sequence
from typing import Optional

from .tensorflow__helpers import tensorflow_split_1


def tensorflow_split(
    self: tensorflow.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[
        Union[int, Sequence[int], tensorflow.Tensor, tf.Tensor]
    ] = None,
    axis: int = 0,
    with_remainder: bool = False,
):
    return tensorflow_split_1(
        self,
        copy=copy,
        num_or_size_splits=num_or_size_splits,
        axis=axis,
        with_remainder=with_remainder,
    )
