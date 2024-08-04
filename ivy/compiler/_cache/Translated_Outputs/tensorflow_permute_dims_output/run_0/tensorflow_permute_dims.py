import tensorflow

from typing import Union
from typing import Tuple
from typing import Optional

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_permute_dims(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.transpose(x, perm=axes)
