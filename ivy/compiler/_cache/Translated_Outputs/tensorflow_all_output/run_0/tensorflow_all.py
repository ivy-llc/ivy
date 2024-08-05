import tensorflow

from typing import Sequence
from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_all(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    try:
        return tensorflow.reduce_all(
            tensorflow.cast(x, tensorflow.bool), axis=axis, keepdims=keepdims
        )
    except tensorflow.errors.InvalidArgumentError as e:
        raise Exception(e) from e
