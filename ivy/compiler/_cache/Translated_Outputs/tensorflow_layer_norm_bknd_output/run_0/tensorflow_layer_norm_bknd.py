import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Union
from typing import List

from .tensorflow__helpers import tensorflow_add
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_mean
from .tensorflow__helpers import tensorflow_multiply
from .tensorflow__helpers import tensorflow_var


@tensorflow_handle_array_like_without_promotion
def tensorflow_layer_norm_bknd(
    x: Union[tensorflow.Tensor, tf.Tensor],
    normalized_idxs: List[int],
    /,
    *,
    scale: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    offset: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    eps: float = 1e-05,
    new_std: float = 1.0,
    out: Optional[tensorflow.Tensor] = None,
):
    mean = tensorflow_mean(x, axis=normalized_idxs, keepdims=True)
    var = tensorflow_var(x, axis=normalized_idxs, keepdims=True)
    x = (x - mean) / (var + eps) ** 0.5
    if scale is not None:
        if offset is not None:
            return tensorflow_multiply(
                tensorflow_add(tensorflow_multiply(x, scale), offset), new_std, out=out
            )
        return tensorflow_multiply(tensorflow_multiply(x, scale), new_std, out=out)
    return tensorflow_multiply(x, new_std, out=out)
