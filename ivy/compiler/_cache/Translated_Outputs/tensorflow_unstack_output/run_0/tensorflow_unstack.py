import tensorflow

from typing import Optional

from .tensorflow__helpers import tensorflow_unstack_1


def tensorflow_unstack(
    self: tensorflow.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
):
    return tensorflow_unstack_1(self, copy=copy, axis=axis, keepdims=keepdims)
