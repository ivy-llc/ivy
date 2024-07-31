import tensorflow

from typing import Union
from typing import Optional

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_sign(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    np_variant: Optional[bool] = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.math.sign(x)
