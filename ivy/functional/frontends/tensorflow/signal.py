from typing import Union, Optional, Literal
import tensorflow as tf

import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
)


@handle_tf_dtype
@to_ivy_arrays_and_back
def kaiser_window(window_length, beta=12.0, dtype=ivy.float32, name=None):
    return ivy.kaiser_window(window_length, periodic=False, beta=beta, dtype=dtype)


kaiser_window.supported_dtypes = ("float32", "float64", "float16", "bfloat16")

def idct(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> tf.Tensor:
    if x.dtype not in (tf.float32, tf.float64):
        x = tf.cast(x, tf.float32)
    if axis != -1:
        new_dims = list(range(len(x.shape)))
        new_dims[axis], new_dims[-1] = new_dims[-1], axis
        x = tf.transpose(x, new_dims)
        idct_out = tf.signal.idct(x, type=type, n=n, axis=-1, norm=norm)
        idct_out = tf.transpose(idct_out, new_dims)
    else:
        idct_out = tf.signal.idct(x, type=type, n=n, axis=-1, norm=norm)
    return idct_out