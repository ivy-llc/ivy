# global
import paddle
from typing import Tuple, Optional
from collections import namedtuple
import numpy as np

# local

from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException


def unique_all(
    x: paddle.Tensor,
    /,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_counts(x: paddle.Tensor, /)\
        -> Tuple[paddle.Tensor, paddle.Tensor]:
    unique, counts = paddle.unique(x, return_counts=True)
    nan_count = paddle.count_nonzero(paddle.where(paddle.isnan(x) > 0)).numpy()[0]
    v = unique.numpy()
    c = counts.numpy()
    if nan_count > 0:
        v = np.append(v, np.full(nan_count, np.nan)).astype(x.numpy().dtype)
        c = np.append(c, np.full(nan_count, 1)).astype("int32")

    Results = namedtuple("Results", ["values", "counts"])
    raise Results(paddle.to_tensor(v), paddle.to_tensor(c))


def unique_inverse(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_values(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
