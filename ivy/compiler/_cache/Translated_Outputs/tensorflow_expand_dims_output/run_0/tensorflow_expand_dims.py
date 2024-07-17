import tensorflow
import numpy as np

from typing import Union
from typing import Sequence
from typing import Optional

from .tensorflow__helpers import tensorflow__calculate_out_shape


def tensorflow_expand_dims(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    try:
        out_shape = tensorflow__calculate_out_shape(axis, tensorflow.shape(x))
        ret = tensorflow.reshape(x, shape=out_shape)
        return ret
    except (tensorflow.errors.InvalidArgumentError, np.AxisError) as error:
        raise Exception(error) from error
