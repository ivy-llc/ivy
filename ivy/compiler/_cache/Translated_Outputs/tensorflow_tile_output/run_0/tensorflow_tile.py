import tensorflow

from typing import Sequence
from numbers import Number
from typing import Optional
from typing import Union


def tensorflow_tile(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    repeats: Sequence[int],
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if x.shape == ():
        x = tensorflow.reshape(x, (-1,))
    if isinstance(repeats, Number):
        repeats = [repeats]
    if isinstance(repeats, tensorflow.Tensor) and repeats.shape == ():
        repeats = tensorflow.reshape(repeats, (-1,))
    if len(x.shape) < len(repeats):
        while len(x.shape) != len(repeats):
            x = tensorflow.expand_dims(x, 0)
    elif len(x.shape) > len(repeats):
        repeats = list(repeats)
        while len(x.shape) != len(repeats):
            repeats = [1] + repeats
    return tensorflow.tile(x, repeats)
