import tensorflow
import numpy as np

from typing import Optional
from typing import Union
from typing import Sequence

from .tensorflow__helpers import tensorflow__calculate_out_shape_bknd
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_expand_dims(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    try:
        out_shape = tensorflow__calculate_out_shape_bknd(axis, tensorflow.shape(x))
        ret = tensorflow.reshape(x, shape=out_shape)
        return ret
    except (tensorflow.errors.InvalidArgumentError, np.AxisError) as error:
        raise Exception(error) from error
