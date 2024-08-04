import tensorflow

from typing import Union
from typing import Optional
from numbers import Number

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_nonzero(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
):
    res = tensorflow.experimental.numpy.nonzero(x)
    if size is not None:
        dtype = tensorflow.int64
        if isinstance(fill_value, float):
            dtype = tensorflow.float64
        res = tensorflow.cast(res, dtype)
        diff = size - res[0].shape[0]
        if diff > 0:
            res = tensorflow.pad(res, [[0, 0], [0, diff]], constant_values=fill_value)
        elif diff < 0:
            res = tensorflow.slice(res, [0, 0], [-1, size])
    if as_tuple:
        return tuple(res)
    return tensorflow.stack(res, axis=1)
