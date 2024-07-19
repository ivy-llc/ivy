import tensorflow

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_log2(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.math.log(x) / tensorflow.math.log(
        tensorflow.constant(2.0, x.dtype)
    )
