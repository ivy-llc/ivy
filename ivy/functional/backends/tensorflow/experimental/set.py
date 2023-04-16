from typing import Optional
import tensorflow as tf

from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes(
    {"2.9.1 and below": ("int", "float16", "bfloat16", "float32", "float64")}, backend_version
)
def difference(
    x1: tf.Tensor,
    x2: tf.Tensor = None,
    /,
    *,
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    return tf.sets.difference(x1, x2)
