import tensorflow

from typing import Union
from typing import Optional
from typing import Sequence


def tensorflow_any(
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
        return tensorflow.reduce_any(
            tensorflow.cast(x, tensorflow.bool), axis=axis, keepdims=keepdims
        )
    except tensorflow.errors.InvalidArgumentError as e:
        raise Exception(e) from e
