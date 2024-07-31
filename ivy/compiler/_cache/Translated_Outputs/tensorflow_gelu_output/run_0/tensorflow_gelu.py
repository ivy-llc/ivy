import tensorflow
import tensorflow as tf

from typing import Optional

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_gelu(
    x: tf.Tensor,
    /,
    *,
    approximate: bool = False,
    complex_mode="jax",
    out: Optional[tf.Tensor] = None,
):
    if x.dtype in [tensorflow.complex64, tensorflow.complex128]:
        return (
            0.5
            * x
            * (1 + tensorflow.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
        )
    return tensorflow.nn.gelu(x, approximate)
