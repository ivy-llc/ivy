import tensorflow

from typing import Sequence
from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_roll(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if axis is None:
        originalShape = x.shape
        axis = 0
        x = tensorflow.reshape(x, [-1])
        roll = tensorflow.roll(x, shift, axis)
        ret = tensorflow.reshape(roll, originalShape)
    else:
        if isinstance(shift, int) and type(axis) in [list, tuple]:
            shift = [shift for _ in range(len(axis))]
        ret = tensorflow.roll(x, shift, axis)
    return ret
