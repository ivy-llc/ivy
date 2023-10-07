# global
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf

from . import backend_version
from ivy.func_wrapper import with_supported_dtypes

# local


@with_supported_dtypes(
    {
        "2.14.0 and below": (
            "int32",
            "int64",
        )
    },
    backend_version,
)
def unravel_index(
    indices: Union[tf.Tensor, tf.Variable],
    shape: Tuple[int],
    /,
    *,
    out: Optional[Tuple[Union[tf.Tensor, tf.Variable]]] = None,
) -> Tuple[tf.Tensor]:
    temp = indices
    output = []
    for dim in reversed(shape):
        output.append(temp % dim)
        temp = temp // dim
    output.reverse()
    ret = tf.convert_to_tensor(output, dtype=tf.int32)
    return tuple(ret)
