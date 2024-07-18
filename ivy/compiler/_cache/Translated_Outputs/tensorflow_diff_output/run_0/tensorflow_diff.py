import tensorflow

from typing import Optional
from typing import Union


def tensorflow_diff(
    x: Union[tensorflow.Tensor, tensorflow.Variable, list, tuple],
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[
        Union[tensorflow.Tensor, tensorflow.Variable, int, float, list, tuple]
    ] = None,
    append: Optional[
        Union[tensorflow.Tensor, tensorflow.Variable, int, float, list, tuple]
    ] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if n == 0:
        return x
    if prepend is not None:
        x = tensorflow.experimental.numpy.append(
            prepend, x, axis=axis if axis != -1 else None
        )
    if append is not None:
        x = tensorflow.experimental.numpy.append(
            x, append, axis=axis if axis != -1 else None
        )
    return tensorflow.experimental.numpy.diff(x, n=n, axis=axis)
