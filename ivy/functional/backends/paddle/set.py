# global
import paddle
from typing import Tuple, Optional
from collections import namedtuple
import paddle.fluid as fluid
# local

from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException


def unique_all(
    x: paddle.Tensor,
    /,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_counts(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    unique, counts = paddle.unique(x, return_counts=True)
    nan_count = paddle.count_nonzero(paddle.where(paddle.isnan(x) > 0)).numpy()[0]

    if nan_count > 0:
        unique_nan = paddle.full(shape=[1, nan_count], fill_value=float('nan'), dtype=x.dtype)
        counts_nan = paddle.full(shape=[1, nan_count], fill_value=1, dtype=x.dtype)
        unique = fluid.layers.concat(input=[unique.astype(x.dtype), paddle.reshape(unique_nan, [nan_count])], axis=0)
        counts = fluid.layers.concat(input=[counts.astype(x.dtype), paddle.reshape(counts_nan, [nan_count])], axis=0)

    Results = namedtuple("Results", ["values", "counts"])
    return Results(unique, counts)


def unique_inverse(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_values(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
