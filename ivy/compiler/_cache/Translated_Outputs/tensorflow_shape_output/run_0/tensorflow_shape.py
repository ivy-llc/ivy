import tensorflow

from typing import Union

from .tensorflow__helpers import tensorflow_asarray
from .tensorflow__helpers import tensorflow_default_int_dtype


def tensorflow_shape(
    x: Union[tensorflow.Tensor, tensorflow.Variable], /, *, as_array: bool = False
):
    if as_array:
        return tensorflow_asarray(
            tensorflow.shape(x), dtype=tensorflow_default_int_dtype()
        )
    else:
        return tuple(x.shape)
