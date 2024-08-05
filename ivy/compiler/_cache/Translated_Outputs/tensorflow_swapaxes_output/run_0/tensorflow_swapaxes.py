import tensorflow

from typing import Union
from typing import Optional

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_swapaxes(
    x,
    axis0,
    axis1,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    x_shape = x.shape
    num_dims = len(x_shape)
    axis0 %= num_dims
    axis1 %= num_dims
    config = list(range(num_dims))
    config.pop(axis0)
    config.insert(axis0, axis1)
    config.pop(axis1)
    config.insert(axis1, axis0)
    return tensorflow.transpose(x, config)
