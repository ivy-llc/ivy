# global
from typing import Optional
import paddle
from ivy.utils.exceptions import IvyNotImplementedException

# local
from paddle.fluid import layers as pfl


def union(
        x1: paddle.Tensor,
        x2: paddle.Tensor,
        /,
        *,
        out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if isinstance(x1, list):
        x1 = paddle.concat(x1, axis=0)
    if isinstance(x2, list):
        x2 = paddle.concat(x2, axis=0)
    if out is not None:
        raise IvyNotImplementedException(
            'paddle set difference does not support out kwarg.')
    return pfl.union1d(x1, x2)