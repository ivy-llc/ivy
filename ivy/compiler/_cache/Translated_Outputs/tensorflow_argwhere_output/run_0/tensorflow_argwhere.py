import tensorflow

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_ndim_bknd_


@tensorflow_handle_array_like_without_promotion
def tensorflow_argwhere(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if isinstance(x, tensorflow.Variable):
        x_ndim = x.shape.rank
    else:
        x_ndim = tensorflow_ndim_bknd_(x)
    if x_ndim == 0:
        return tensorflow.zeros(shape=[int(bool(x)), 0], dtype="int64")
    where_x = tensorflow.experimental.numpy.nonzero(x)
    ag__result_list_0 = []
    for item in where_x:
        res = tensorflow.expand_dims(item, -1)
        ag__result_list_0.append(res)
    res = tensorflow.experimental.numpy.concatenate(ag__result_list_0, -1)
    return res
