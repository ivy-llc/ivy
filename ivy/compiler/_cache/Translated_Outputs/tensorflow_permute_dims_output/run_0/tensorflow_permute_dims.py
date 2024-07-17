import tensorflow

from typing import Tuple
from typing import Optional
from typing import Union


def tensorflow_permute_dims(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.transpose(x, perm=axes)
