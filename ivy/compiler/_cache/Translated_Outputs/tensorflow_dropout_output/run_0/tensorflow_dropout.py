import tensorflow

from typing import Sequence
from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_astype
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_dropout(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    prob: float,
    /,
    *,
    scale: bool = True,
    dtype: tensorflow.DType = None,
    training: bool = True,
    seed: Optional[int] = None,
    noise_shape: Optional[Sequence[int]] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    x = tensorflow_astype(x, dtype) if dtype and x.dtype != dtype else x
    if prob == 0 or not training:
        return x
    res = tensorflow.nn.dropout(x, prob, noise_shape=noise_shape, seed=seed)
    res = tensorflow.multiply(res, 1.0 - prob) if not scale else res
    return res
